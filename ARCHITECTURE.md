# Architecture reference — repo tree and API contracts

## Final repo tree

```
lora-chat-train/
├── README.md
├── .env.example
├── docker-compose.yml
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx                   # Chat UI (Next.js App Router)
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── next.config.js
│   ├── tailwind.config.ts
│   ├── package.json
│   └── Dockerfile
│
├── backend/
│   ├── main.py                        # FastAPI app, chat endpoint, session lifecycle
│   ├── models.py                      # SQLAlchemy ORM models
│   ├── schemas.py                     # Pydantic request/response schemas
│   ├── database.py                    # Async SQLAlchemy engine + session factory
│   ├── model_client.py                # Async model inference wrapper
│   ├── token_counter.py               # Token counting utility
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_core.py               # Unit tests for all core behaviors
│   ├── requirements.txt
│   ├── Dockerfile
│   └── Dockerfile.model               # Model server image
│
├── worker/
│   ├── tasks.py                       # Celery tasks — full pipeline orchestration
│   ├── requirements.txt
│   └── Dockerfile
│
├── training/
│   ├── extractor/
│   │   ├── __init__.py
│   │   └── transcript_extractor.py    # Raw transcript → clean candidate pairs
│   ├── curator/
│   │   ├── __init__.py
│   │   └── curator.py                 # Quality scoring and filtering
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── dataset_writer.py          # JSONL dataset writer (SFTTrainer format)
│   ├── trainer/
│   │   ├── __init__.py
│   │   └── hf_launcher.py             # HF job launcher + local SFTTrainer wrapper
│   ├── eval/
│   │   ├── __init__.py
│   │   └── evaluator.py               # Domain eval suite
│   └── deployment/
│       ├── __init__.py
│       └── deploy.py                  # Promote, smoke test, rollback
│
├── shared/
│   ├── __init__.py
│   ├── s3_uploader.py                 # S3 upload utility with retries
│   └── slack_notifier.py              # Slack webhook notifier with retries
│
├── infra/
│   └── schema.sql                     # Postgres schema (all 7 tables)
│
└── scripts/
    └── init_db.py                     # DB initialisation script
```

---

## FastAPI API contracts

### Sessions

| Method | Path | Description |
|---|---|---|
| POST | `/sessions` | Create a new session → returns `SessionResponse` |
| GET | `/sessions/{session_id}` | Get session metadata and state |
| GET | `/sessions` | List recent sessions |

**SessionResponse**
```json
{
  "id": "uuid",
  "state": "ACTIVE",
  "total_tokens": 120,
  "max_tokens": 4096,
  "created_at": "2025-01-01T00:00:00Z",
  "closed_at": null
}
```

### Chat

| Method | Path | Description |
|---|---|---|
| POST | `/sessions/{session_id}/chat` | Send a message, receive SSE stream |

**ChatRequest**
```json
{ "message": "Hello!" }
```

**SSE stream event types:**
```
data: {"type": "start"}
data: {"type": "chunk", "text": "Hello"}
data: {"type": "chunk", "text": " there!"}
data: {"type": "status", "remaining_tokens": 3976, "session_state": "ACTIVE"}
data: {"type": "sleep_warning", "message": "Session closing…"}
data: {"type": "sleeping", "reason": "token_threshold"}
data: {"type": "end"}
```

If `/sleep` is sent:
```
data: {"type": "sleep_ack", "message": "Going to sleep — see you after fine-tuning!"}
data: {"type": "end"}
data: {"type": "sleeping", "reason": "user_command"}
```

### Health

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Returns `{"status": "ok"}` |

### Model server (internal, port 8001)

| Method | Path | Description |
|---|---|---|
| POST | `/generate` | `{"prompt": str, "max_new_tokens": int}` → `{"response": str}` |
| POST | `/reload_adapter` | `{"adapter_dir": str}` — hot-swap adapter without restart |
| GET | `/health` | Health check |

---

## Celery task chain

```
enqueue_training_pipeline(session_id)
  → extract_candidates(session_id, run_id)
  → curate_candidates(session_id, run_id)
  → build_dataset(session_id, run_id)
  → launch_training(session_id, run_id)
  → poll_training(session_id, run_id)    # retries every 60s, up to 60×
  → run_evaluation(session_id, run_id)
  → deploy_or_rollback(session_id, run_id)
```

Each task:
- Receives the output dict of the previous task as its first argument
- Updates `training_runs.status` and `sessions.state` in Postgres
- Sends a Slack notification at start and completion
- Uploads artifacts to S3

---

## S3 prefix layout

```
s3://<bucket>/
  sessions/{session_id}/
    raw/transcript.json
    candidates/candidates.json
    curated/curated.json
    dataset/dataset.jsonl
  training_runs/{run_id}/
    config/config.json
    logs/training.log
    artifacts/           ← adapter weights and config
    eval/eval_report.json
  production/
    current/             ← live production adapter
      manifest.json
      adapter_model.bin
      adapter_config.json
    history/{run_id}/    ← previous adapters for rollback
```

---

## Slack notification matrix

| Stage | Status | Trigger |
|---|---|---|
| session_started | info | Session POST |
| pre_sleep_warning | warn | Token budget < threshold |
| session_sleeping | info | /sleep or budget exhausted |
| extraction_started | info | Pipeline start |
| extraction_completed | ok | Candidates extracted |
| curation_started | info | Curation task start |
| curation_completed | ok/warn | After scoring |
| dataset_built | ok | JSONL written |
| artifact_uploaded | info | Any S3 upload |
| training_started | info | HF job launched |
| training_succeeded | ok | Job complete |
| training_failed | error | Job error |
| evaluation_started | info | Eval begin |
| evaluation_completed | ok/warn | Eval done |
| deployment_approved | ok | Eval passed |
| deployment_rejected | warn | Eval failed |
| adapter_switch_succeeded | ok | Promotion complete |
| adapter_switch_failed | error | Reload failed |
| rollback_triggered | warn | Smoke test fail |
| rollback_completed | ok | Rollback done |
