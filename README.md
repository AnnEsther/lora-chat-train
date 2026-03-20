# LoRA Chat & Train

A single-user private web application where you chat with a small model (Llama 3.2:1b). When you type `/sleep` — or the session token budget runs low — the conversation automatically closes and triggers a LoRA fine-tuning cycle using that session's data. The new adapter is evaluated, promoted to production, and ready for your next chat.

Built on top of the patterns from [EndToEndLoRA](https://github.com/nicknochnack/EndToEndLoRA), refactored into a production-structured application.

---

## Architecture overview

```
Browser (Next.js)
  └─ FastAPI backend  ──── Postgres (state)
       │                ── Redis / Celery (jobs)
       └─ Model server (Llama 3.2:1b + current LoRA adapter)

  On /sleep or token threshold:
  Celery → Extractor → Curator → Dataset → Trainer (HF A100)
                                            └─ Eval → Deploy → adapter swap
                                                       └─ Slack + S3 at every stage
```

---

## Quickstart

### Prerequisites

- Docker + Docker Compose
- Python 3.11+
- Node.js 20+
- AWS account with an S3 bucket
- Hugging Face account with API token
- Slack app with incoming webhook URL

### 1. Clone and configure

```bash
git clone https://github.com/yourname/lora-chat-train
cd lora-chat-train
cp .env.example .env
# Edit .env — fill in every value
```

### 2. Start infrastructure

```bash
docker compose up -d postgres redis
```

### 3. Initialise the database

```bash
docker compose run --rm backend python -m scripts.init_db
```

### 4. Start the backend and worker

```bash
docker compose up backend worker model_server
```

### 5. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 — start chatting.

---

## Environment variables

See `.env.example` for the full list. Key variables:

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face API token |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | S3 credentials |
| `S3_BUCKET` | Bucket name |
| `SLACK_WEBHOOK_URL` | Incoming webhook URL |
| `DATABASE_URL` | Postgres connection string |
| `REDIS_URL` | Redis connection string |
| `BASE_MODEL` | e.g. `meta-llama/Llama-3.2-1B-Instruct` |
| `MAX_SESSION_TOKENS` | Hard token budget per session (default 4096) |
| `PRE_SLEEP_THRESHOLD` | Tokens remaining that trigger warning (default 512) |

---

## Repository structure

```
/
  frontend/           Next.js chat UI
  backend/            FastAPI API + model serving
  worker/             Celery task definitions
  training/
    extractor/        Transcript → candidate segments
    curator/          Scoring and filtering
    datasets/         JSONL dataset writer
    trainer/          HF SFTTrainer wrapper
    eval/             Evaluation suite
    deployment/       Promote / rollback logic
  shared/             S3, Slack, logging utilities
  infra/              Docker, Compose, Postgres schema
  scripts/            DB init, seed helpers
```

---

## Session states

```
ACTIVE → PRE_SLEEP_WARNING → SLEEPING → TRAINING → EVALUATING → DEPLOYING → READY
                                                                           ↘ FAILED (rollback)
```

---

## Training cycle

1. Session locked → raw transcript uploaded to S3
2. Extractor pulls user/assistant turn pairs
3. Curator scores and filters candidates (removes low-quality / secrets)
4. Dataset writer builds JSONL in SFTTrainer chat format
5. HF training job launched (A100, LoRA on Llama 3.2:1b)
6. Evaluator runs domain-specific eval suite
7. If pass → new adapter promoted; model server hot-swaps adapter
8. If smoke test fails → instant rollback to previous production adapter
9. Slack notification at every stage

---

## Running tests

```bash
cd backend
pytest tests/ -v
```

---

## Adapter versioning

Adapters are stored in S3 under:
- `s3://<bucket>/production/current/` — live adapter
- `s3://<bucket>/production/history/<run_id>/` — previous adapters

The model server reads the current adapter at startup and after each successful deployment.
