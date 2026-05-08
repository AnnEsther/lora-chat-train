# Infrastructure

## Overview
The application is containerised with Docker Compose. Six services cover the full stack: database, cache/queue, API, background worker, model server, and frontend.

## Key Files
- `docker-compose.yml` — Service definitions, volumes, networks
- `infra/schema.sql` — PostgreSQL DDL (applied via Docker initdb)
- `backend/Dockerfile` — FastAPI backend image
- `backend/Dockerfile.model` — Model server image (GPU-enabled)
- `worker/Dockerfile` — Celery worker image
- `frontend/Dockerfile` — Next.js frontend image
- `Makefile` — Developer shortcuts

## Services

### `postgres`
- Image: `postgres:16-alpine`
- Port: `5432`
- Schema applied automatically via Docker `initdb.d/` volume mount
- Persistent data volume: `postgres_data`

### `redis`
- Image: `redis:7-alpine`
- Port: `6379`
- Used as Celery broker (`CELERY_BROKER_URL`) and result backend (`CELERY_RESULT_BACKEND`)

### `backend`
- Built from `./backend/Dockerfile`
- Port: `8000`
- Hot-reload via `uvicorn --reload`
- Volume mounts: `backend/` and `shared/` for live code changes
- Depends on: `postgres`, `redis`, `model_server`

### `worker`
- Built from `./worker/Dockerfile`
- No exposed port
- Concurrency: **1** (`--concurrency=1`) — single-threaded to avoid GPU contention during training
- Volume mounts: `worker/`, `shared/`, `training/`, `outputs/`, `backend/` (for model imports), `adapter_store` (at `/adapters`)
- GPU reservation: `capabilities: [gpu]` — required for local training tasks
- Depends on: `postgres`, `redis`

### `model_server`
- Built from `./backend/Dockerfile.model.gpu`
- Port: `8001`
- **GPU reservation:** `capabilities: [gpu]` — requires NVIDIA container toolkit
- Named volume `adapter_store` mounted at `/adapters` (shared with worker; persists adapter files across container restarts)
- Also mounts `outputs/` directory
- Runs `backend/model_server/serve.py` (Docker/production); swap to `local_gpu_serve.py` for local RTX 4060 dev or `hf_serve.py` for HF-hosted inference

### `frontend`
- Built from `./frontend/Dockerfile`
- Port: `3000`
- `NEXT_PUBLIC_API_URL` env var points to backend at `http://backend:8000`

## Named Volumes
| Volume | Used by | Purpose |
|--------|---------|---------|
| `postgres_data` | postgres | Persistent database storage |
| `adapter_store` | model_server, worker | Persistent LoRA adapter storage; shared so the worker can write new adapters and the model server can hot-swap them |

## Makefile Commands
| Command | Description |
|---------|-------------|
| `make up` | `docker compose up --build` — build and start all services |
| `make down` | `docker compose down` — stop all services |
| `make db` | Start only `postgres` + `redis` |
| `make init-db` | Run `scripts/init_db.py` to apply schema |
| `make reset-all` | Run `scripts/reset_all.py` — wipe all DB data + clear adapter/output files |
| `make backend` | Start FastAPI backend locally (no Docker) |
| `make worker` | Start Celery worker locally (no Docker) |
| `make frontend` | Start Next.js dev server locally |
| `make test` | Run `pytest` unit tests |
| `make lint` | Run `ruff` linter over all Python code |
| `make clean` | Remove `__pycache__` directories |

## Environment Variables
All services read from `.env` (see `.env.example`).

| Var | Description |
|-----|-------------|
| `DATABASE_URL` | PostgreSQL DSN with `+asyncpg` for backend; plain `postgresql://` for scripts/Celery |
| `REDIS_URL` | Redis connection URL |
| `CELERY_BROKER_URL` | Redis URL for Celery broker |
| `CELERY_RESULT_BACKEND` | Redis URL for Celery result storage (default `redis://localhost:6379/1`) |
| `HF_TOKEN` | HuggingFace API token |
| `HF_ENDPOINT_URL` | Full URL of your private vLLM inference endpoint (only for `hf_serve.py`) |
| `HF_TRAINING_ENDPOINT` | HF endpoint URL for training; if absent, uses local model server |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | AWS credentials for S3 (optional; falls back to local storage) |
| `S3_BUCKET` | S3 bucket for artifacts (optional; falls back to local storage) |
| `AWS_REGION` | AWS region (default `us-east-1`) |
| `SLACK_WEBHOOK_URL` | Slack notifications webhook (optional; silent if absent) |
| `BASE_MODEL` | Base LLM model ID (e.g., `meta-llama/Llama-3.2-1B-Instruct`) |
| `MODEL_SERVER_URL` | URL backend uses to reach model server (e.g., `http://model_server:8001`) |
| `MAX_SESSION_TOKENS` | Token budget per session (default `4096`) |
| `PRE_SLEEP_THRESHOLD` | Warning threshold in remaining tokens (default `512`) |
| `MIN_TRAINING_SAMPLES` | Minimum curated samples to start training (default `10`) |
| `LOCAL_OUTPUT_DIR` | Override for local artifact storage directory (default `<project_root>/outputs/`) |
| `NEXT_PUBLIC_API_URL` | Frontend: backend API base URL (build arg; default `http://localhost:8000`) |
| `NEXT_PUBLIC_MODEL_SERVER_URL` | Frontend: model server base URL (build arg; default `http://localhost:8001`) |

## GPU Requirements
The model server requires an NVIDIA GPU with CUDA support. Tested on RTX 4060 (8 GB VRAM) with:
- 4-bit NF4 quantization via `bitsandbytes`
- bf16 compute dtype
- `device_map="auto"` for automatic layer distribution

For CPU-only or cloud deployment: use `backend/model_server/serve.py` instead and remove the GPU reservation from `docker-compose.yml`.

## Developer Scripts
| Script | Description |
|--------|-------------|
| `scripts/init_db.py` | Apply schema; safe to run multiple times |
| `scripts/reset_all.py` | Wipe all DB rows + clear adapter files |
| `scripts/resume_training.py` | Re-queue training from a known `run_id` |
| `scripts/test_train.py` | Standalone training smoke test (no full pipeline) |

## Change Log
<!-- Agents: append an entry here after every change -->
| Date | Change | Author |
|------|--------|--------|
| 2026-05-08 | Fix worker concurrency to 1; fix model_server Dockerfile to Dockerfile.model.gpu; note adapter_store shared with worker; add reset-all to Makefile table; expand env vars table with HF_ENDPOINT_URL, HF_TRAINING_ENDPOINT, AWS_REGION, LOCAL_OUTPUT_DIR, NEXT_PUBLIC_MODEL_SERVER_URL | opencode |
| 2026-04-28 | Initial documentation created | opencode |
