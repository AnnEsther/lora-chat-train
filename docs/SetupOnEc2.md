# Setup on EC2 — LoRA Chat & Train

Complete step-by-step guide for deploying the stack on an AWS EC2 GPU instance
(Tesla T4 / g4dn.xlarge). Based on real deployment experience — includes all the
gotchas that aren't in the original docs.

---

## Instance Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Instance type | `g4dn.xlarge` | `g4dn.xlarge` |
| GPU | Tesla T4 (16GB VRAM) | Tesla T4 (16GB VRAM) |
| vCPU | 4 | 4 |
| RAM | 16GB | 16GB |
| Root volume | 30GB | 30GB |
| **Data volume** | **100GB** | **120GB** |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |

> ⚠️ **Critical:** Always attach a **separate data volume** (100GB+) at launch time.
> The root volume fills up fast with Docker images, CUDA layers, and model weights.
> Forgetting this causes cryptic `No space left on device` errors mid-build.

---

## Step 1 — Launch the EC2 Instance

### 1.1 In the AWS Console

1. Go to EC2 → Launch Instance
2. Choose **Ubuntu Server 22.04 LTS (HVM), SSD Volume Type**
3. Instance type: **g4dn.xlarge**
4. Key pair: create or select an existing one
5. Security group — open these ports:

| Port | Protocol | Source | Purpose |
|------|----------|--------|---------|
| 22 | TCP | Your IP only | SSH |
| 80 | TCP | 0.0.0.0/0 | HTTP |
| 443 | TCP | 0.0.0.0/0 | HTTPS |
| 3000 | TCP | 0.0.0.0/0 | Frontend (if no nginx) |
| 8000 | TCP | 0.0.0.0/0 | Backend API (if no nginx) |

6. Storage — **this is the critical part**:
   - Root volume (`/dev/sda1`): **30 GB gp3**
   - Add a second volume: **120 GB gp3** — this is where Docker will live

7. Launch.

### 1.2 Allocate and associate an Elastic IP

So the instance IP doesn't change on restart:

```bash
# In AWS Console: EC2 → Elastic IPs → Allocate → Associate to instance
# Or via CLI:
aws ec2 allocate-address --domain vpc
aws ec2 associate-address --instance-id i-XXXXX --allocation-id eipalloc-XXXXX
```

---

## Step 2 — First SSH and System Setup

```bash
ssh -i ~/.ssh/your-key.pem ubuntu@YOUR_ELASTIC_IP
```

### 2.1 Update the system

```bash
sudo apt-get update && sudo apt-get upgrade -y
```

### 2.2 Mount the data volume

**Do this before installing anything.** Find the data volume device name:

```bash
lsblk
# Look for the large unformatted disk — usually nvme1n1
# Example output:
# nvme0n1   30G  ← root volume
# nvme1n1  120G  ← data volume (unformatted, no MOUNTPOINTS)
```

Format and mount it:

```bash
sudo mkfs.ext4 /dev/nvme1n1
sudo mkdir -p /mnt/data
sudo mount /dev/nvme1n1 /mnt/data

# Make permanent across reboots
echo '/dev/nvme1n1 /mnt/data ext4 defaults 0 2' | sudo tee -a /etc/fstab

# Verify
df -h /mnt/data
# Should show ~116GB available
```

---

## Step 3 — Move ALL System Storage to the Data Volume

**Do all of this before pulling any Docker images or building anything.**

On Ubuntu 22.04 GPU instances, three things silently eat the 30GB root volume:

| What | Default location | Size after builds | Fix |
|---|---|---|---|
| Docker images/volumes | `/var/lib/docker` | 5-20GB | Move via `daemon.json` |
| containerd snapshots | `/var/lib/containerd` | **~16GB** | Symlink to data volume |
| snapd packages | `/var/lib/snapd` | ~200MB | Remove entirely |

### 3.1 Move Docker to the data volume

```bash
sudo systemctl stop docker

sudo mkdir -p /mnt/data/docker
sudo tee /etc/docker/daemon.json << 'EOF'
{
  "data-root": "/mnt/data/docker"
}
EOF

sudo systemctl start docker

# Verify
docker info | grep "Docker Root Dir"
# Expected: Docker Root Dir: /mnt/data/docker
```

### 3.2 Move containerd to the data volume

> ⚠️ This is the one most people miss. containerd manages its own snapshot store
> at `/var/lib/containerd` **independently** of Docker's `data-root`. After a few
> builds of a CUDA image it will silently accumulate 16GB+ on the root volume
> even though Docker is correctly pointing at the data volume.

```bash
sudo systemctl stop docker
sudo systemctl stop containerd

sudo mv /var/lib/containerd /mnt/data/containerd
sudo ln -s /mnt/data/containerd /var/lib/containerd

# Verify symlink
ls -la /var/lib/containerd
# lrwxrwxrwx ... /var/lib/containerd -> /mnt/data/containerd

sudo systemctl start containerd
sudo systemctl start docker

# Verify both are on data volume
docker info | grep "Docker Root Dir"   # /mnt/data/docker
sudo du -sh /mnt/data/containerd       # size shown here, not on root
```

### 3.3 Remove snapd

Snap packages are not needed for this stack:

```bash
sudo snap remove --purge amazon-ssm-agent 2>/dev/null || true
sudo snap remove --purge snapd 2>/dev/null || true
sudo apt-get purge -y snapd
sudo rm -rf /var/lib/snapd /snap
```

### 3.4 Clean up apt cache and old packages

```bash
sudo apt-get clean
sudo apt-get autoremove --purge -y
sudo journalctl --vacuum-size=100M
```

### 3.5 Verify root volume usage

After all moves, root should be well under 30% used:

```bash
df -h /
# Expected: ~5-6GB used out of 28GB

sudo du -sh /var/* 2>/dev/null | sort -rh | head -5
# /var/lib should now be small (no docker, no containerd, no snapd)
```

---

## Step 4 — Install NVIDIA Drivers and Container Toolkit

### 4.1 Install NVIDIA drivers

```bash
# Check if already installed
nvidia-smi

# If not installed:
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot
# SSH back in after reboot
nvidia-smi  # should show Tesla T4
```

### 4.2 Install NVIDIA Container Toolkit

This allows Docker containers to access the GPU:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU is accessible in Docker
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
# Should show Tesla T4
```

---

## Step 5 — Clone the Repository

```bash
cd /home/ubuntu
git clone https://github.com/AnnEsther/lora-chat-train.git
cd lora-chat-train
```

---

## Step 6 — Configure Environment Variables

```bash
cp .env.example .env
nano .env
```

Fill in every value. Key variables:

```ini
# ── Database ───────────────────────────────────────────────────────────────────
DATABASE_URL=postgresql+asyncpg://lora:lora@postgres:5432/lora
POSTGRES_USER=lora
POSTGRES_PASSWORD=lora
POSTGRES_DB=lora

# ── Redis / Celery ─────────────────────────────────────────────────────────────
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1

# ── HuggingFace ────────────────────────────────────────────────────────────────
HF_TOKEN=hf_your_actual_token
BASE_MODEL=meta-llama/Llama-3.2-1B-Instruct

# ── HF vLLM Inference Endpoint ─────────────────────────────────────────────────
HF_ENDPOINT_URL=https://your-endpoint.us-east-1.aws.endpoints.huggingface.cloud

# ── Training (leave empty to train locally on the T4) ─────────────────────────
HF_TRAINING_ENDPOINT=

# ── AWS S3 ─────────────────────────────────────────────────────────────────────
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET=your-lora-bucket

# ── Slack ──────────────────────────────────────────────────────────────────────
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx/yyy/zzz

# ── Frontend URLs ──────────────────────────────────────────────────────────────
NEXT_PUBLIC_API_URL=https://your-domain.com/api
NEXT_PUBLIC_MODEL_SERVER_URL=https://your-domain.com/model

# ── Training tuning ────────────────────────────────────────────────────────────
MAX_FACTS_PER_CANDIDATE=3
QA_BATCH_SIZE=5
QA_SYNTHESIS_TIMEOUT=20
TRAIN_BATCH_SIZE=4
MAX_SEQ_LENGTH=512
TRAIN_EPOCHS=3
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES=q_proj,v_proj
```

---

## Step 7 — Verify docker-compose.yml GPU Config

The `worker` service must have the GPU reservation, adapter volume, and concurrency=1:

```yaml
worker:
  build:
    context: ./worker
    dockerfile: Dockerfile
  <<: *common-env
  depends_on:
    postgres:
      condition: service_healthy
    redis:
      condition: service_healthy
  volumes:
    - ./worker:/app
    - ./shared:/app/shared
    - ./training:/app/training
    - ./outputs:/app/outputs
    - ./backend:/app/backend
    - adapter_store:/adapters          # ← required for saving trained adapters
  deploy:
    resources:
      reservations:
        devices:
          - driver: "nvidia"
            count: 1
            capabilities: ["gpu"]
  restart: unless-stopped
  command: celery -A tasks worker --loglevel=info --concurrency=1  # ← 1, not 2
```

---

## Step 8 — Build and Start

```bash
docker compose up --build -d

# Watch startup logs
docker compose logs -f
```

Expected healthy state:

```bash
docker compose ps
# postgres     running (healthy)
# redis        running (healthy)
# backend      running
# worker       running
# model_server running
# frontend     running
```

### Verify GPU is accessible in the worker

```bash
docker compose exec worker nvidia-smi
# Should show Tesla T4

docker compose exec worker python3 -c \
  "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: CUDA: True Tesla T4
```

---

## Step 9 — Initialize the Database

The schema is applied automatically on first start via Docker's `initdb.d` mount.
Verify:

```bash
docker compose exec postgres psql -U lora -d lora -c "\dt"
# Should list all tables: sessions, turns, training_candidates, etc.
```

If tables are missing (e.g. volume already existed from a previous run):

```bash
docker compose exec postgres psql -U lora -d lora \
  -f /docker-entrypoint-initdb.d/01_schema.sql
```

---

## Step 10 — Set Up nginx + HTTPS (Optional but Recommended)

### 10.1 Install nginx and Certbot

```bash
sudo apt-get install -y nginx certbot python3-certbot-nginx
```

### 10.2 Create nginx config

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
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        # SSE — disable buffering
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 3600s;
        chunked_transfer_encoding on;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/lora-chat /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 10.3 Issue TLS certificate

```bash
sudo certbot --nginx -d your-domain.com
# Follow prompts — auto-renews via systemd timer
```

---

## Step 11 — Health Checks

```bash
# All services running
docker compose ps

# Backend
curl http://localhost:8000/health
# {"status":"ok"}

# Model server (HF proxy)
curl http://localhost:8001/health
# {"status":"ok","model_loaded":true,"remote_healthy":true}

# GPU utilization baseline (0% when idle)
nvidia-smi

# Database has tables
docker compose exec postgres psql -U lora -d lora -c "\dt"
```

---

## Step 12 — Test the Training Pipeline

1. Open the frontend at `https://your-domain.com`
2. Start a chat and teach the model something for 5-10 turns
3. Type `/sleep` to trigger training
4. Watch Slack for notifications — expected sequence:
   - ✅ Curation Completed
   - ℹ️ Knowledge Extracted
   - ℹ️ QA Synthesized
   - ✅ Training Data Ready ← review modal opens here
5. Review the QA pairs in the modal, click **Validate All & Start Training**
6. Watch GPU spin up: `watch -n 3 nvidia-smi` — expect 60-80% utilization
7. Training completes in ~5-10 minutes for a small session
8. Slack notifications: Training Started → Training Succeeded → Evaluation → Deployed

---

## Troubleshooting

### `No space left on device` during Docker build

Check which disk Docker and containerd are actually using:

```bash
docker info | grep "Docker Root Dir"
# Must show /mnt/data/docker — if not, redo Step 3.1

ls -la /var/lib/containerd
# Must be a symlink → /mnt/data/containerd — if a real dir, redo Step 3.2

df -h /mnt/data   # check space on data volume
df -h /           # check root volume

# If containerd is still a real directory eating space:
sudo systemctl stop docker containerd
sudo mv /var/lib/containerd /mnt/data/containerd
sudo ln -s /mnt/data/containerd /var/lib/containerd
sudo systemctl start containerd docker
```

### Worker keeps restarting

```bash
docker compose logs worker --tail=50
# Check for import errors, missing env vars, or Redis connection issues
```

### `AttributeError: 'NoneType' object has no attribute 'Redis'`

Redis package conflict on the CUDA base image. Pin versions in `worker/requirements.txt`:

```
redis==5.1.1
kombu==5.3.4
```

### GPU not visible in worker container

```bash
# Check toolkit is installed
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# Check docker-compose.yml has the deploy.resources block
# Check concurrency=1 (not 2) in the worker command
```

### Training hangs in `synthesize_qa` for 30+ minutes

Facts are being processed one at a time. Check `.env`:

```ini
QA_BATCH_SIZE=5
QA_SYNTHESIS_TIMEOUT=20
MAX_FACTS_PER_CANDIDATE=3
```

### Session stuck in TRAINING state after a failed run

```bash
docker compose exec postgres psql -U lora -d lora -c \
  "UPDATE sessions SET state='VALIDATING' WHERE id='your-session-id';"
```

### `train_local` returns None / TypeError on `upload_adapter`

The `train_local` function body is missing from `hf_launcher.py`. Check:

```bash
wc -l training/trainer/hf_launcher.py
# Should be 300+ lines — if ~220, the training body is truncated
tail -20 training/trainer/hf_launcher.py
# Should end with: return output_dir
```

---

## Maintenance

### Restart all services after a code change

```bash
git pull
docker compose up --build -d
```

### Restart just the worker (picks up code changes instantly via volume mount)

```bash
docker compose restart worker
```

### Check disk usage

```bash
df -h                              # overall — root and data volume
sudo du -sh /mnt/data/*            # breakdown of everything on data volume
docker system df                   # Docker-specific: images, containers, volumes, cache
```

### Free up disk space

```bash
docker system prune -f             # stopped containers + dangling images
docker system prune -af            # ALL unused images (more aggressive)
sudo journalctl --vacuum-size=100M # trim system logs
sudo apt-get clean                 # clear apt package cache
```

### Backup the database

```bash
docker compose exec postgres pg_dump -U lora lora > backup_$(date +%Y%m%d).sql
```