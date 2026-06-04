# Setup on EC2 — LoRA Chat & Train

Complete step-by-step guide for deploying the full stack on an AWS EC2 GPU instance.
Covers everything from launching the instance to verifying a training run end-to-end.
Based on real deployment experience — every gotcha is documented.

---

## Instance Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Instance type | `g4dn.xlarge` | `g4dn.xlarge` |
| GPU | Tesla T4 (16 GB VRAM) | Tesla T4 (16 GB VRAM) |
| vCPU | 4 | 4 |
| RAM | 16 GB | 16 GB |
| Root volume | 30 GB gp3 | 30 GB gp3 |
| **Data volume** | **100 GB gp3** | **120 GB gp3** |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |

> **Critical:** Always attach a **separate data volume** (100 GB+) at launch.
> The 30 GB root volume fills fast with Docker images, CUDA layers, and model weights.
> Forgetting this causes cryptic `No space left on device` errors mid-build.

---

## Step 1 — Launch the Instance

### 1.1 AWS Console

1. EC2 → Launch Instance
2. AMI: **Ubuntu Server 22.04 LTS (HVM), SSD Volume Type**
3. Instance type: **g4dn.xlarge**
4. Key pair: create or select
5. Security group — open these ports:

| Port | Protocol | Source | Purpose |
|------|----------|--------|---------|
| 22 | TCP | Your IP | SSH |
| 80 | TCP | 0.0.0.0/0 | HTTP → HTTPS redirect |
| 443 | TCP | 0.0.0.0/0 | HTTPS (nginx) |

6. Storage:
   - Root volume (`/dev/sda1`): **30 GB gp3**
   - Add a second EBS volume: **120 GB gp3** — this is where Docker lives

7. Launch.

### 1.2 Allocate an Elastic IP

```bash
# AWS Console: EC2 → Elastic IPs → Allocate → Associate to instance
# Or CLI:
aws ec2 allocate-address --domain vpc
aws ec2 associate-address --instance-id i-XXXXX --allocation-id eipalloc-XXXXX
```

### 1.3 Point your domain

Add an A record in your DNS pointing `your-domain.com` to the Elastic IP.
Wait for DNS to propagate before running Certbot (Step 10).

---

## Step 2 — First SSH and System Prep

```bash
ssh -i ~/.ssh/your-key.pem ubuntu@YOUR_ELASTIC_IP
```

### 2.1 Update the system

```bash
sudo apt-get update && sudo apt-get upgrade -y
```

### 2.2 Mount the data volume

Find the device name:

```bash
lsblk
# nvme0n1   30G   ← root (has MOUNTPOINT /)
# nvme1n1  120G   ← data volume (no MOUNTPOINT — unformatted)
```

Format, mount, make permanent:

```bash
sudo mkfs.ext4 /dev/nvme1n1
sudo mkdir -p /mnt/data
sudo mount /dev/nvme1n1 /mnt/data
echo '/dev/nvme1n1 /mnt/data ext4 defaults 0 2' | sudo tee -a /etc/fstab
df -h /mnt/data   # verify ~116 GB available
```

---

## Step 3 — Move ALL System Storage to the Data Volume

Do this **before** pulling any Docker images or building anything. On Ubuntu 22.04
GPU instances, three things silently eat the 30 GB root volume:

| What | Default location | Post-build size | Fix |
|------|-----------------|-----------------|-----|
| Docker images/layers | `/var/lib/docker` | 5–20 GB | Move via `daemon.json` |
| containerd snapshots | `/var/lib/containerd` | **~16 GB** | Symlink to data volume |
| snapd packages | `/var/lib/snapd` | ~200 MB | Remove entirely |

### 3.1 Move Docker

```bash
sudo systemctl stop docker

sudo mkdir -p /mnt/data/docker
sudo tee /etc/docker/daemon.json << 'EOF'
{
  "data-root": "/mnt/data/docker"
}
EOF

sudo systemctl start docker
docker info | grep "Docker Root Dir"
# Must show: Docker Root Dir: /mnt/data/docker
```

### 3.2 Move containerd (the one most people miss)

containerd manages its own snapshot store at `/var/lib/containerd` **independently**
of Docker's `data-root`. After a few CUDA image builds it silently accumulates 16 GB+
on the root volume even though Docker is pointing at the data volume.

```bash
sudo systemctl stop docker
sudo systemctl stop containerd

sudo mv /var/lib/containerd /mnt/data/containerd
sudo ln -s /mnt/data/containerd /var/lib/containerd

ls -la /var/lib/containerd
# Must be: lrwxrwxrwx ... /var/lib/containerd -> /mnt/data/containerd

sudo systemctl start containerd
sudo systemctl start docker

docker info | grep "Docker Root Dir"   # /mnt/data/docker
sudo du -sh /mnt/data/containerd       # size here, not on root
```

### 3.3 Remove snapd

```bash
sudo snap remove --purge amazon-ssm-agent 2>/dev/null || true
sudo snap remove --purge snapd 2>/dev/null || true
sudo apt-get purge -y snapd
sudo rm -rf /var/lib/snapd /snap
```

### 3.4 Clean apt cache and journals

```bash
sudo apt-get clean && sudo apt-get autoremove --purge -y
sudo journalctl --vacuum-size=100M
```

### 3.5 Verify root volume usage

```bash
df -h /
# Expected: ~5–6 GB used out of 28 GB
```

---

## Step 4 — Install NVIDIA Drivers and Container Toolkit

### 4.1 NVIDIA drivers

```bash
nvidia-smi   # if this works, drivers already installed — skip to 4.2

# Otherwise:
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot
# SSH back in
nvidia-smi   # must show Tesla T4
```

### 4.2 NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
# Must show Tesla T4
```

---

## Step 5 — Install nginx and Certbot

```bash
sudo apt-get install -y nginx certbot python3-certbot-nginx
sudo systemctl enable nginx
```

---

## Step 6 — Clone the Repository

```bash
cd /home/ubuntu
git clone https://github.com/YourOrg/lora-chat-train.git
cd lora-chat-train
```

---

## Step 7 — Configure Environment Variables

```bash
cp .env.example .env
nano .env
```

Required values:

```ini
# ── Database ───────────────────────────────────────────────────────────────────
DATABASE_URL=postgresql+asyncpg://lora:lora@postgres:5432/lora
POSTGRES_USER=lora
POSTGRES_PASSWORD=lora        # change this
POSTGRES_DB=lora

# ── Redis / Celery ─────────────────────────────────────────────────────────────
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1

# ── HuggingFace ────────────────────────────────────────────────────────────────
HF_TOKEN=hf_your_actual_token
BASE_MODEL=meta-llama/Llama-3.2-1B-Instruct

# ── Frontend URLs (baked into the Next.js bundle at build time) ────────────────
NEXT_PUBLIC_API_URL=https://your-domain.com/api
NEXT_PUBLIC_MODEL_SERVER_URL=https://your-domain.com/model

# ── CORS (must match the public domain) ───────────────────────────────────────
EXTERNAL_SITE_ORIGIN=https://your-domain.com

# ── Internal service URLs ──────────────────────────────────────────────────────
MODEL_SERVER_URL=http://model_server:8001

# ── AWS S3 (optional — falls back to local filesystem) ────────────────────────
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET=

# ── Slack notifications (optional) ────────────────────────────────────────────
SLACK_WEBHOOK_URL=

# ── Training defaults ──────────────────────────────────────────────────────────
MIN_TRAINING_SAMPLES=10
MAX_SESSION_TOKENS=4096
PRE_SLEEP_THRESHOLD=512
TRAIN_BATCH_SIZE=4
MAX_SEQ_LENGTH=512
TRAIN_EPOCHS=3
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES=q_proj,v_proj
```

> **NEXT_PUBLIC_* are build-time.** They are baked into the Next.js JavaScript bundle
> when `docker compose build frontend` runs. If you change them later, you must rebuild
> the frontend image.

---

## Step 8 — Configure nginx

```bash
sudo cp infra/nginx/lora-chat /etc/nginx/sites-available/lora-chat
# Edit the domain name if needed:
sudo nano /etc/nginx/sites-available/lora-chat

sudo ln -sf /etc/nginx/sites-available/lora-chat /etc/nginx/sites-enabled/lora-chat
sudo rm -f /etc/nginx/sites-enabled/default

sudo nginx -t          # must say "syntax is ok"
sudo systemctl reload nginx
```

The nginx config proxies:

| Path | Service | Notes |
|------|---------|-------|
| `/` | `localhost:3000` | Frontend; basic auth via htpasswd |
| `/api/` | `localhost:8000` | Backend; strips `/api` prefix; `proxy_buffering off` for SSE |
| `/model/` | `localhost:8001` | Model server; strips `/model` prefix |
| `/glyph` | `localhost:3001` | Glyph chat static app |

### Set up basic auth (optional but recommended)

```bash
sudo apt-get install -y apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd your-username
# Enter password when prompted
```

---

## Step 9 — Issue TLS Certificate

Make sure DNS has propagated (your domain resolves to the Elastic IP) before running this:

```bash
sudo certbot --nginx -d your-domain.com
# Follow prompts; select "Redirect" to force HTTPS
# Auto-renews via systemd timer — no action needed
```

Verify:

```bash
sudo certbot renew --dry-run   # test renewal works
```

---

## Step 10 — Build and Start All Services

```bash
cd /home/ubuntu/lora-chat-train
docker compose up --build -d

# Watch startup logs
docker compose logs -f --tail=50
```

Expected healthy state after ~3–5 minutes (model download on first start):

```bash
docker compose ps
# NAME          STATUS
# postgres      running (healthy)
# redis         running (healthy)
# backend       running
# worker        running
# model_server  running
# frontend      running
```

### Verify GPU access in containers

```bash
docker compose exec worker nvidia-smi
# Must show Tesla T4

docker compose exec worker python3 -c \
  "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: CUDA: True Tesla T4
```

---

## Step 11 — Initialize the Database

The schema is applied automatically on first start via Docker's `initdb.d/` volume mount.
Verify:

```bash
docker compose exec postgres psql -U lora -d lora -c "\dt"
# Must list: sessions, turns, training_candidates, synthesized_qa, datasets, ...
```

If tables are missing (e.g. the volume already existed from a previous run):

```bash
docker compose exec postgres psql -U lora -d lora \
  -f /docker-entrypoint-initdb.d/01_schema.sql
```

---

## Step 12 — Smoke Test

```bash
# Backend health
curl http://localhost:8000/health
# {"status":"ok"}

# Model server health
curl http://localhost:8001/health
# {"status":"ok","model_loaded":true,...}

# Frontend accessible
curl -sI http://localhost:3000 | head -3
# HTTP/1.1 200 OK (or 401 if basic auth is on — that's correct)

# Through nginx with HTTPS
curl https://your-domain.com/api/health
# {"status":"ok"}
```

---

## Step 13 — Test the Full Training Pipeline

1. Open `https://your-domain.com` in a browser
2. Send several passages to generate Q&A pairs (aim for 10+ pairs)
3. Use the inline deck below each message to validate pairs:
   - Click **Mark validated** (auto-advances to next card)
   - Or **Validate all** to approve all pairs at once
4. Once the **Start Training** button shows `(10/10)` or more, click it
5. Monitor training:

```bash
watch -n 5 docker compose logs worker --tail=20
watch -n 3 nvidia-smi   # expect 60–80% GPU utilization during training
```

6. After training completes (~5–15 min), the session transitions to `READY`
7. Verify the new adapter was saved:

```bash
docker compose exec model_server ls /adapters/current/
# adapter_model.safetensors  adapter_config.json  manifest.json
```

---

## Partial Updates (day-to-day)

When you push code changes, rebuild only the affected service:

```bash
# Backend (e.g. backend/main.py changes)
git pull && docker compose build backend && docker compose up -d --no-deps backend

# Frontend (e.g. frontend/app/page.tsx changes)
git pull && docker compose build frontend && docker compose up -d --no-deps frontend

# Worker (e.g. worker/tasks.py or training/ changes)
git pull && docker compose build worker && docker compose up -d --no-deps worker
```

`--no-deps` restarts only the named container; all other services keep running.

---

## Troubleshooting

### `No space left on device` during Docker build

```bash
docker info | grep "Docker Root Dir"
# Must be /mnt/data/docker — if not, redo Step 3.1

ls -la /var/lib/containerd
# Must be a symlink → /mnt/data/containerd — if a real dir, redo Step 3.2

df -h /mnt/data     # check data volume space
df -h /             # check root volume space

docker system df    # Docker breakdown: images, containers, volumes, cache
docker system prune -f   # free up dangling images and stopped containers
```

### Worker keeps restarting

```bash
docker compose logs worker --tail=50
# Look for: import errors, missing env vars, Redis connection refused, CUDA OOM
```

### Training fails with "An error occurred while generating the dataset"

This was a known bug — the inline chat flow stored QA data in `synthesized_qa` but the
training pipeline read from `training_candidates`. **Fixed in backend/main.py**: the
`_promote_qa_to_candidates()` helper now converts validated QA pairs into
`TrainingCandidate` rows before Phase 2 is enqueued. Ensure you have pulled the latest
code and rebuilt the backend image.

### Glyph Chat still using adapter after switching to Base Model

This was a known bug — `POST /chat/direct` was not sending the unload call when
`adapter_id == "base"`. **Fixed in backend/main.py**: an explicit
`POST /reload_adapter {"adapter_dir": "base"}` is now sent, triggering
`merge_and_unload()` on the model server. Rebuild the backend image.

### GPU not visible in worker or model_server container

```bash
# Test toolkit is installed
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# Check docker-compose.yml for deploy.resources block
# Check worker command has --concurrency=1
```

### Session stuck in TRAINING state

```bash
docker compose exec postgres psql -U lora -d lora -c \
  "UPDATE sessions SET state='ACTIVE' WHERE id='your-session-uuid';"
```

### CORS errors in browser (`Access-Control-Allow-Origin`)

Check `EXTERNAL_SITE_ORIGIN` in `.env` matches the public domain exactly (including
`https://`). Rebuild the backend image after changing it.

### nginx 502 Bad Gateway

```bash
docker compose ps   # check all services are running
curl http://localhost:8000/health   # backend reachable?
curl http://localhost:8001/health   # model server reachable?
sudo nginx -t && sudo systemctl reload nginx
```

### Model server OOM (CUDA out of memory)

Only one of `worker` and `model_server` can hold the model in VRAM at a time.
Training is handled by the worker which loads its own model instance.
Ensure `--concurrency=1` in the worker command and that no other GPU processes
are running:

```bash
nvidia-smi   # check all processes and VRAM usage
```

---

## Maintenance

### Check disk usage

```bash
df -h                            # root and data volumes
sudo du -sh /mnt/data/*          # breakdown of data volume
docker system df                 # Docker-specific breakdown
```

### Free up disk space

```bash
docker system prune -f           # stopped containers + dangling images
docker system prune -af          # ALL unused images (more aggressive)
sudo journalctl --vacuum-size=100M
sudo apt-get clean
```

### Back up the database

```bash
docker compose exec postgres pg_dump -U lora lora > backup_$(date +%Y%m%d).sql
```

### Restart all services after a reboot

```bash
cd /home/ubuntu/lora-chat-train
docker compose up -d
```

Add to `/etc/rc.local` or a systemd service to auto-start on boot:

```bash
sudo tee /etc/systemd/system/lora-chat.service << 'EOF'
[Unit]
Description=LoRA Chat & Train
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/lora-chat-train
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
User=ubuntu

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable lora-chat
```

### Renew TLS certificate

Certbot installs a systemd timer that renews automatically. Check it:

```bash
sudo systemctl status certbot.timer
sudo certbot renew --dry-run   # test
```

---

## Change Log
| Date | Change | Author |
|------|--------|--------|
| 2026-05-20 | Full rewrite: added Steps 5 (nginx install), 8 (nginx config), 9 (TLS), 12 (smoke test), 13 (pipeline test); updated training test to reflect inline deck + Start Training button flow; added Partial Updates section; expanded Troubleshooting with dataset bug, base model bug, CORS, OOM; added auto-start systemd service; added Certbot renewal note | opencode |
| 2026-05-08 | Add containerd symlink step; add snapd removal; add disk verification step; expand env var examples; add training pipeline test section | opencode |
| 2026-04-28 | Initial documentation created | opencode |
