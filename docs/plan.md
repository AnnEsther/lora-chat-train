# Cloud Deployment Plan — AWS EC2 + HF vLLM Inference Endpoint

This document is a step-by-step guide to deploying the entire LoRA Chat & Train stack
fully to the cloud with **no local GPU required for inference**.

## Architecture overview

```
AWS EC2 (t3.large — no GPU)                  HuggingFace Cloud
┌────────────────────────────┐               ┌──────────────────────────────┐
│  postgres      (Docker)    │               │  Private vLLM Endpoint       │
│  redis         (Docker)    │  HTTPS/SSE    │  (Nvidia L4, your model,     │
│  backend       (Docker)    │ ────────────► │   HF_TOKEN auth only)        │
│  worker        (Docker)    │               └──────────────────────────────┘
│  hf_serve.py   (Docker)    │
│  frontend      (Docker)    │               AWS S3
└────────────────────────────┘               (dataset + adapter artifacts)
```

- **EC2** runs the application stack (Postgres, Redis, FastAPI backend, Celery worker,
  `hf_serve.py` proxy, Next.js frontend). A GPU is **not** needed here.
- **HF vLLM Endpoint** hosts the model privately. Only your `HF_TOKEN` can access it.
  All chat traffic is encrypted in transit (TLS). No third party sees your prompts.
- **Training** (`/sleep` pipeline) submits jobs to `HF_TRAINING_ENDPOINT` via Celery,
  or can be run locally using `local_gpu_serve.py` when needed.
- **S3** stores training datasets and LoRA adapter artifacts.

---

## Prerequisites checklist

Collect all of these before starting:

| Item | Where to get it |
|------|----------------|
| AWS account with IAM access | [aws.amazon.com](https://aws.amazon.com) |
| HuggingFace token (`hf_...`, **write** scope) | HF → Settings → Access Tokens |
| HF model access approved | HF model page → accept licence (Llama 3.2) |
| HF Dedicated Inference Endpoint URL | Created in Step 1 below |
| S3 bucket name | Created in Step 3 below |
| Slack webhook URL (optional) | Slack app → Incoming Webhooks |
| Domain name (optional) | Route 53 or any registrar |

---

## Step 1 — Create the private HF vLLM Inference Endpoint

> Do this **first** — you need the endpoint URL before configuring `.env`.

1. Go to [huggingface.co/inference-endpoints](https://huggingface.co/inference-endpoints)
   and click **New Endpoint**.

2. Fill in the settings:

   | Field | Value |
   |-------|-------|
   | Model | `meta-llama/Llama-3.2-1B-Instruct` (or your `BASE_MODEL`) |
   | Framework | **vLLM** |
   | Hardware | **Nvidia L4** (24 GB) — sufficient for 1B models |
   | Region | Same as your EC2 region (e.g. `us-east-1`) to minimise latency |
   | Visibility | **Private** |
   | Scale-to-zero | Enable (scales down after 1 h of inactivity — saves cost) |

   > **Note:** TGI is in maintenance mode on HF. Always select **vLLM** for new endpoints.

3. Click **Create Endpoint** and wait 3–5 minutes for it to reach `Running` state.

4. Copy the endpoint URL — it looks like:
   ```
   https://your-endpoint-name.us-east-1.aws.endpoints.huggingface.cloud
   ```
   You will paste this into `.env` as `HF_ENDPOINT_URL` in Step 5.

### Endpoint cost

| Hardware | VRAM | Price | When to use |
|----------|------|-------|------------|
| Nvidia L4 | 24 GB | ~$1.80/hr | 1B–3B models |
| Nvidia A10G | 24 GB | ~$2.40/hr | 7B models |

With **scale-to-zero** enabled the endpoint costs nothing while idle. It cold-starts
in ~60 seconds on first request after sleeping.

---

## Step 2 — Launch an EC2 instance (no GPU required)

### 2.1 Choose instance type

The EC2 instance only runs Postgres, Redis, the FastAPI backend, Celery worker,
`hf_serve.py` (a lightweight HTTP proxy), and the Next.js frontend.
No GPU is needed.

| Use case | Instance type | vCPU | RAM | Est. monthly cost |
|----------|--------------|------|-----|-------------------|
| Single user / dev | `t3.medium` | 2 | 4 GB | ~$30/mo |
| Normal use (recommended) | `t3.large` | 2 | 8 GB | ~$60/mo |
| Heavy Celery workloads | `t3.xlarge` | 4 | 16 GB | ~$120/mo |

> Use a Spot instance for ~70% savings if you can tolerate occasional interruptions.
> Pair with an Elastic IP so the public address stays fixed across restarts.

### 2.2 Launch the instance

```bash
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.large \
  --key-name YOUR_KEY_PAIR \
  --security-group-ids sg-XXXXXXXX \
  --subnet-id subnet-XXXXXXXX \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":40,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=lora-chat-train}]'
```

Find the latest Ubuntu 22.04 AMI ID for your region:

```bash
aws ec2 describe-images \
  --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64*" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
  --output text
```

### 2.3 Configure Security Group

| Port | Protocol | Source | Purpose |
|------|----------|--------|---------|
| 22 | TCP | Your IP only | SSH |
| 3000 | TCP | 0.0.0.0/0 | Frontend |
| 8000 | TCP | 0.0.0.0/0 | Backend API |
| 80 | TCP | 0.0.0.0/0 | HTTP (nginx, optional) |
| 443 | TCP | 0.0.0.0/0 | HTTPS (nginx + TLS, optional) |

> Keep ports 5432 (postgres), 6379 (redis), and 8001 (hf_serve proxy) closed to the
> internet — they communicate only within Docker's internal network.

### 2.4 Attach an IAM role for S3 access

```bash
aws iam create-role --role-name lora-ec2-role --assume-role-policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Principal\":{\"Service\":\"ec2.amazonaws.com\"},\"Action\":\"sts:AssumeRole\"}]}"

aws iam attach-role-policy --role-name lora-ec2-role --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam create-instance-profile --instance-profile-name lora-ec2-profile

aws iam add-role-to-instance-profile --instance-profile-name lora-ec2-profile --role-name lora-ec2-role

aws ec2 associate-iam-instance-profile --instance-id i-xxxxxxxxxxxxxxxxxx --iam-instance-profile Name=lora-ec2-profile --region us-east-1
```

---

## Step 3 — Create an S3 bucket

```bash
# Create bucket (replace region and name):
aws s3 mb s3://your-lora-bucket --region us-east-1

# Block all public access:
aws s3api put-public-access-block --bucket your-lora-bucket --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

---

## Step 4 — Connect and prepare the instance

### 4.1 SSH in

```bash
ssh -i ~/.ssh/YOUR_KEY_PAIR.pem ubuntu@YOUR_INSTANCE_PUBLIC_IP
```

### 4.2 Install Docker and Docker Compose

```bash
docker --version       # Should be 24+ if already installed
docker compose version # Should be 2.x

# If not installed:
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
newgrp docker
```

> No NVIDIA drivers or GPU toolkit needed — inference runs on the HF endpoint.

---

## Step 5 — Clone the repository

```bash
git clone https://github.com/YOUR_ORG/lora-chat-train.git
cd lora-chat-train
```

For private repositories:

```bash
git clone https://YOUR_GITHUB_TOKEN@github.com/YOUR_ORG/lora-chat-train.git
```

---

## Step 6 — Configure environment variables

### 6.1 Create `.env`

```bash
cp .env.example .env
nano .env
```

### 6.2 Complete `.env` for production

```ini
# ── Hugging Face ───────────────────────────────────────────────────────────────
HF_TOKEN=hf_your_actual_token
HF_USERNAME=your_hf_username

# ── HF vLLM Inference Endpoint (inference) ────────────────────────────────────
# URL from Step 1:
HF_ENDPOINT_URL=https://your-endpoint-name.us-east-1.aws.endpoints.huggingface.cloud
# Must match the model deployed on the endpoint:
HF_ENDPOINT_MODEL=meta-llama/Llama-3.2-1B-Instruct

# ── HF Training Endpoint (fine-tuning, optional) ──────────────────────────────
# If you have a separate HF endpoint for training jobs:
HF_TRAINING_ENDPOINT=https://api.endpoints.huggingface.co/v2/endpoint/your_training_endpoint
# Leave blank to fall back to local GPU training (requires local_gpu_serve.py)

# ── AWS / S3 ───────────────────────────────────────────────────────────────────
# Leave blank if using IAM instance role (recommended):
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
S3_BUCKET=your-lora-bucket

# ── Slack (optional) ───────────────────────────────────────────────────────────
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T.../B.../...

# ── Database ───────────────────────────────────────────────────────────────────
DATABASE_URL=postgresql+asyncpg://lora:lora@postgres:5432/lora
POSTGRES_USER=lora
POSTGRES_PASSWORD=CHANGE_THIS_STRONG_PASSWORD
POSTGRES_DB=lora

# ── Redis / Celery ─────────────────────────────────────────────────────────────
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1

# ── Model ──────────────────────────────────────────────────────────────────────
BASE_MODEL=meta-llama/Llama-3.2-1B-Instruct
# hf_serve.py listens on 8001 inside Docker; backend reaches it by service name:
MODEL_SERVER_URL=http://model_server:8001
ADAPTER_DIR=/adapters/current

# ── Session budget ─────────────────────────────────────────────────────────────
MAX_SESSION_TOKENS=4096
PRE_SLEEP_THRESHOLD=512

# ── API ────────────────────────────────────────────────────────────────────────
BACKEND_PORT=8000
SECRET_KEY=CHANGE_THIS_TO_A_64_CHAR_RANDOM_STRING

# ── Frontend ───────────────────────────────────────────────────────────────────
# The browser calls this URL — use the public IP or domain:
NEXT_PUBLIC_API_URL=http://YOUR_INSTANCE_PUBLIC_IP:8000

# ── Training hyperparameters ───────────────────────────────────────────────────
MIN_TRAINING_SAMPLES=10
EXTRACTION_WINDOW_SIZE=4
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES=q_proj,v_proj
TRAIN_EPOCHS=3
TRAIN_BATCH_SIZE=4
TRAIN_GRAD_ACCUM=4
TRAIN_LR=2e-4
MAX_SEQ_LENGTH=512
```

> **Security:** `.env` is in `.gitignore`. Never commit it.

---

## Step 7 — Update `docker-compose.yml` for production

Three changes are needed before deploying:

### 7.1 Switch model_server to `hf_serve.py`

The default `docker-compose.yml` runs `serve.py` (local GPU). Replace with the vLLM
proxy and remove the GPU resource reservation:

```yaml
model_server:
  build:
    context: ./backend
    dockerfile: Dockerfile.model
  env_file: .env
  ports:
    - "8001:8001"
  volumes:
    - adapter_store:/adapters
    - ./outputs:/app/outputs
  environment:
    ADAPTER_DIR: /adapters/current
    ADAPTER_HISTORY_DIR: /adapters/history
  command: python -m model_server.hf_serve   # ← was model_server.serve
  # deploy.resources.reservations (GPU) block removed — not needed
  restart: unless-stopped
```

### 7.2 Add `restart: unless-stopped` to every service

```yaml
services:
  postgres:
    restart: unless-stopped
  redis:
    restart: unless-stopped
  backend:
    restart: unless-stopped
  worker:
    restart: unless-stopped
  model_server:
    restart: unless-stopped
  frontend:
    restart: unless-stopped
```

### 7.3 Disable hot-reload on the backend

```yaml
backend:
  command: uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
  # was: --reload  (dev only)
```

### 7.4 Fix the frontend API URL

The `frontend` service in `docker-compose.yml` currently hardcodes
`NEXT_PUBLIC_API_URL: http://localhost:8000`. This is overridden by the value in
`.env`, but only if the compose file reads from `env_file`. Either add `env_file: .env`
to the frontend service, or set the variable explicitly:

```yaml
frontend:
  environment:
    NEXT_PUBLIC_API_URL: ${NEXT_PUBLIC_API_URL}
```

Then rebuild the frontend image after changing this value.

---

## Step 8 — Build and start all services

```bash
# From the project root on the EC2 instance:
docker compose up --build -d
```

This builds four custom images (backend, worker, model_server, frontend), pulls
postgres and redis, and starts all six containers.

**Monitor startup:**

```bash
docker compose logs -f
# Or a specific service:
docker compose logs -f model_server
```

`hf_serve.py` starts in a few seconds (it is just a proxy — no model to load).
You should see:

```
model_server | INFO hf_endpoint_configured url=https://your-endpoint...
model_server | INFO hf_endpoint_reachable   url=https://your-endpoint...
```

If the HF endpoint is cold (scaled to zero), the first chat request will take
~60 seconds while it warms up. Subsequent requests are fast.

---

## Step 9 — Initialise the database

The schema is applied automatically via the `initdb.d` volume mount on first start.
Verify it worked:

```bash
docker compose exec postgres psql -U lora -d lora -c "\dt"
# Should list: sessions, turns, training_candidates, datasets, training_runs, ...
```

If the schema was not applied (e.g. postgres data volume already existed from a
previous run), apply it manually:

```bash
docker compose exec postgres psql -U lora -d lora \
  -f /docker-entrypoint-initdb.d/01_schema.sql
```

---

## Step 10 — Verify every service

```bash
# 1. All containers running:
docker compose ps

# 2. Backend health:
curl http://localhost:8000/health
# Expected: {"status": "ok"}

# 3. Model server (hf_serve.py) health:
curl http://localhost:8001/health
# Expected: {"status":"ok","model_loaded":true,"remote_healthy":true,...}
# If remote_healthy is false, check HF_ENDPOINT_URL and HF_TOKEN in .env

# 4. Frontend:
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000
# Expected: 200

# 5. Database connectivity:
curl http://localhost:8000/sessions
# Expected: [] (empty list — no 500 error)

# 6. End-to-end chat smoke test:
curl -N http://localhost:8000/sessions \
  -X POST -H "Content-Type: application/json" -d '{}' | python3 -m json.tool
# Creates a session; note the id

SESSION_ID=<id from above>
curl -N "http://localhost:8000/sessions/$SESSION_ID/chat" \
  -X POST -H "Content-Type: application/json" \
  -d '{"message":"Hello, are you there?"}' \
  --no-buffer
# Expected: SSE stream of {"type":"chunk","text":"..."} events
```

---

## Step 11 — (Optional) Set up a reverse proxy with HTTPS

If you want a domain and HTTPS instead of raw `IP:port`:

### 11.1 Install nginx and Certbot

```bash
sudo apt-get install -y nginx certbot python3-certbot-nginx
```

### 11.2 Create nginx config

```bash
sudo nano /etc/nginx/sites-available/lora-chat
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # Backend API + SSE streaming
    location /api/ {
        rewrite ^/api(/.*)$ $1 break;
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;       # required for SSE
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/lora-chat /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### 11.3 Obtain TLS certificate

```bash
sudo certbot --nginx -d your-domain.com
```

### 11.4 Update `.env` and redeploy frontend

```ini
NEXT_PUBLIC_API_URL=https://your-domain.com/api
```

```bash
docker compose up --build -d frontend
```

---

## Step 12 — Configure log rotation

```bash
sudo nano /etc/docker/daemon.json
```

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "50m",
    "max-file": "5"
  }
}
```

```bash
sudo systemctl restart docker
docker compose up -d
```

---

## Step 13 — Enable Docker on boot

```bash
sudo systemctl enable docker
```

---

## Ongoing operations

### Updating the application

```bash
cd lora-chat-train
git pull origin main
docker compose up --build -d
```

### Viewing logs

```bash
docker compose logs -f                   # all services
docker compose logs -f backend worker    # specific services
docker compose logs --tail=100 worker    # last 100 lines
```

### Stopping everything

```bash
docker compose down        # stop + remove containers (volumes preserved)
docker compose down -v     # WARNING: also deletes postgres_data and adapter_store
```

### Checking training status

```bash
# Via worker logs (most reliable):
docker compose logs -f worker

# Via model server stub (always returns idle for hf_serve.py):
curl http://localhost:8001/train/status
```

### Database access

```bash
docker compose exec postgres psql -U lora -d lora
```

### Waking the HF endpoint from scale-to-zero

The endpoint wakes automatically on the first chat request. If you want to pre-warm it:

```bash
curl -X POST \
  "${HF_ENDPOINT_URL}/v1/chat/completions" \
  -H "Authorization: Bearer ${HF_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.2-1B-Instruct","messages":[{"role":"user","content":"ping"}],"max_tokens":1}'
```

### Pausing / deleting the HF endpoint

To stop paying for the HF endpoint when not in use:
- HF console → your endpoint → **Pause** (free while paused, wakes in ~5 min)
- Or **Delete** to remove it entirely

---

## Cost summary (approximate, us-east-1)

| Component | Service | Cost model | Est. monthly |
|-----------|---------|-----------|-------------|
| App stack | EC2 `t3.large` On-Demand | ~$0.083/hr | ~$60 |
| App storage | 40 GB gp3 EBS | ~$0.08/GB/mo | ~$3 |
| Inference | HF Nvidia L4 Endpoint | ~$1.80/hr (active only) | varies |
| Artifacts | S3 (1 GB) | ~$0.023/GB/mo | ~$0.02 |
| Network | Data transfer out | ~$0.09/GB | ~$1–5 |

**Inference cost examples (scale-to-zero enabled):**

| Daily active hours | Monthly HF endpoint cost |
|-------------------|--------------------------|
| 1 hr/day | ~$54 |
| 4 hr/day | ~$216 |
| 8 hr/day | ~$432 |
| Always-on (24/7) | ~$1,296 |

> **Tip:** Keep scale-to-zero enabled and pause the endpoint when not needed.
> The EC2 instance (`t3.large`) is the only fixed cost at ~$60/mo.

---

## Troubleshooting

### `hf_serve.py` logs `hf_endpoint_unreachable` on startup

The HF endpoint may be cold (scaled to zero) or still initialising.

```bash
# Check endpoint status in HF console, or hit it directly:
curl -I "${HF_ENDPOINT_URL}/health" -H "Authorization: Bearer ${HF_TOKEN}"
# 200 = ready, 503 = cold-starting, 401 = bad token
```

Wait 60–90 seconds for a cold endpoint to warm up, then restart hf_serve:

```bash
docker compose restart model_server
```

### Chat returns `[Could not reach HF endpoint: ...]`

1. Check `HF_ENDPOINT_URL` is set correctly in `.env` (no trailing slash)
2. Check `HF_TOKEN` has **write** scope and access to the model
3. Verify the endpoint is **Running** (not Paused) in the HF console
4. Check `docker compose logs model_server` for detailed error

### Chat returns `[Endpoint error 401: ...]`

`HF_TOKEN` is invalid or expired. Generate a new token at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with write scope,
update `.env`, and restart:

```bash
docker compose restart model_server
```

### Frontend shows "Failed to fetch" for API calls

`NEXT_PUBLIC_API_URL` is baked into the Next.js bundle at build time.
After changing it in `.env`, always rebuild the frontend:

```bash
docker compose up --build -d frontend
```

### `ModuleNotFoundError` in worker or backend

A new Python dependency was added. Rebuild the affected image:

```bash
docker compose build backend worker
docker compose up -d backend worker
```

### Postgres not ready errors on first start

`depends_on` healthchecks handle this, but on a slow instance the first build
can time out. Re-run:

```bash
docker compose up -d
```

### `No space left on device` during build

Docker layer cache can fill a 40 GB volume. Prune unused layers:

```bash
docker system prune -f
docker compose up --build -d
```

---

## Change Log
| Date | Change | Author |
|------|--------|--------|
| 2026-04-29 | Full rewrite: EC2 now t3.large (no GPU); inference moved to private HF vLLM Endpoint; model download step removed; HF endpoint setup added as Step 1; cost table updated; troubleshooting updated for vLLM proxy | opencode |
| 2026-04-29 | Initial cloud deployment plan created | opencode |


Check
Backend Health
curl http://localhost:8000/health
# Expected: {"status":"ok"}
Public Endpoint health
curl https://train.anratelier.com/api/health
# Expected: {"status":"ok"}
4. Model server reachable
curl http://localhost:8001/health
5. Database tables exist
docker compose exec postgres psql -U lora -d lora -c "\dt"
6. Worker connected
docker compose logs --tail=20 worker
7. Frontend loads
Open https://train.anratelier.com in your browser — you should see the chat UI.
8. Full end-to-end test

Create a session in the browser
Send a chat message
You should get a streamed response back


Option 1 — Nginx Basic Auth (simplest, no code changes)
Just adds a username/password prompt at the browser level.
bash# Install apache2-utils for htpasswd
sudo apt install apache2-utils

# Create a password file (replace 'ann' with your username)
sudo htpasswd -c /etc/nginx/.htpasswd ann
# It will prompt you to enter and confirm a password
Then update your Nginx config:
bashsudo nano /etc/nginx/sites-available/lora-chat
Add these two lines inside the location / block:
nginxlocation / {
    auth_basic "Restricted";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://localhost:3000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
Reload Nginx:
bashsudo nginx -t && sudo systemctl reload nginx
Now visiting https://train.anratelier.com will show a browser login prompt.

sasquatch