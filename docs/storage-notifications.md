# Storage & Notifications

## Overview
All pipeline artifacts are stored in S3. When S3 credentials are not configured, the system transparently falls back to a local `outputs/` directory. Slack webhook notifications are sent at every pipeline stage.

## Key Files
- `shared/s3_uploader.py` — S3 upload with local fallback + all named stage functions
- `shared/local_storage.py` — Local filesystem fallback using `outputs/` directory
- `shared/slack_notifier.py` — Slack webhook client + 25 event convenience wrappers

---

## S3 Storage (`shared/s3_uploader.py`)

### Detection
`_s3_available()` checks for both `S3_BUCKET` **and** `AWS_ACCESS_KEY_ID`. If either is absent, all uploads go to `shared/local_storage.py` instead.

### Core Upload
`upload_bytes(data, key, content_type, retries=3, backoff=1.5)`
- Exponential-backoff retry (default 3 attempts, 1.5× factor)
- All other functions wrap this

### Type-Specific Wrappers
| Function | Input | Content-Type |
|----------|-------|-------------|
| `upload_text(text, key)` | `str` | `text/plain` |
| `upload_json(obj, key)` | `dict/list` | `application/json` |
| `upload_file(path, key)` | local file path | `application/octet-stream` |
| `upload_directory(dir, prefix)` | directory path | per-file |

### Named Stage Functions
One function per pipeline stage, called by Celery tasks:

| Function | S3 Key Pattern |
|----------|---------------|
| `upload_raw_transcript` | `sessions/{session_id}/raw/transcript.json` |
| `upload_candidates` | `sessions/{session_id}/candidates/candidates.json` |
| `upload_curated` | `sessions/{session_id}/curated/curated.json` |
| `upload_dataset_jsonl` | `sessions/{session_id}/dataset/dataset.jsonl` |
| `upload_training_config` | `training_runs/{run_id}/config/config.json` |
| `upload_training_logs` | `training_runs/{run_id}/logs/training.log` |
| `upload_adapter` | `training_runs/{run_id}/artifacts/` (directory upload) |
| `upload_eval_report` | `training_runs/{run_id}/eval/eval_report.json` |
| `upload_deployment_manifest` | `production/current/manifest.json` |
| `upload_rollback_manifest` | `production/history/{run_id}/rollback_manifest.json` |
| `sync_adapter_to_production` | `production/current/` + `production/history/{run_id}/` |

### Configuration
| Env Var | Description |
|---------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials |
| `S3_BUCKET` | Target bucket name |

---

## Local Filesystem Fallback (`shared/local_storage.py`)

Used automatically when S3 is not configured.

**`OUTPUT_DIR`:** `LOCAL_OUTPUT_DIR` env var; defaults to `<project_root>/outputs/`

All return values are `local://{absolute_path}` URIs, so calling code treats them uniformly.

| Function | Mirrors |
|----------|---------|
| `save_bytes(data, key)` | `upload_bytes` |
| `save_text(text, key)` | `upload_text` |
| `save_json(obj, key)` | `upload_json` |
| `save_file(path, key)` | `upload_file` |
| `save_directory(dir, prefix)` | `upload_directory` |

---

## Slack Notifications (`shared/slack_notifier.py`)

### `SlackEvent` Dataclass
| Field | Type | Required |
|-------|------|---------|
| `stage` | `str` | Yes |
| `status` | `"ok" \| "warn" \| "error" \| "info"` | Yes |
| `summary` | `str` | Yes |
| `session_id` | `str` | No |
| `run_id` | `str` | No |
| `model_version` | `str` | No |
| `adapter_version` | `str` | No |
| `s3_path` | `str` | No |
| `extra` | `dict` | No |

### `send(event, retries=3, backoff=1.5)`
- Builds a Slack attachment payload with:
  - Color-coded sidebar: green (`ok`), yellow (`warn`), red (`error`), blue (`info`)
  - Emoji status indicator
  - Field rows for all optional fields that are set
- Silently skips (no-op) if `SLACK_WEBHOOK_URL` is not configured

### 22 Convenience Wrappers
Covering every named pipeline event. Call these from Celery tasks or backend code:

`session_started`, `pre_sleep_warning`, `session_sleeping`, `extraction_completed`, `curation_completed`, `knowledge_extracted`, `qa_synthesized`, `training_data_ready`, `dataset_built`, `artifact_uploaded`, `training_started`, `training_succeeded`, `training_failed`, `evaluation_started`, `evaluation_completed`, `deployment_approved`, `deployment_rejected`, `adapter_switch_succeeded`, `adapter_switch_failed`, `rollback_triggered`, `rollback_completed`, `insufficient_data_warning`

### Configuration
| Env Var | Description |
|---------|-------------|
| `SLACK_WEBHOOK_URL` | Slack incoming webhook URL; if absent, notifications are silently skipped |
| `LOCAL_OUTPUT_DIR` | Override for local storage fallback directory |

---

## Mattermost Professional Notifier (`shared/mattermost_notifier.py`)

A second notifier that posts **one message per training run** to a dedicated Mattermost channel and **edits it in-place** as each phase completes, giving a live checklist view. Unlike `slack_notifier.py` (which fires 22 separate events), this emits 6–8 targeted updates per run and requires the Mattermost REST API (not just a webhook).

### How it works

When Phase 1 starts, one message is created via `POST /api/v4/posts`. Its `post_id` is stored in `training_runs.config["mm_post_id"]`. Every subsequent call reads that `post_id` and calls `PUT /api/v4/posts/{post_id}` to update the message in-place.

The message shows a Markdown table of pipeline stages:

```
#### :robot_face: LoRA Training Run — in progress…
Session `a3f2…` · Run `b7c1…`

| Stage | Status |
|---|---|
| :white_check_mark: Phase 1 — Extraction, curation & QA synthesis | Done — 14 Q&A pairs synthesised |
| :hourglass_spinning: QA review | Awaiting human review… |
| :grey_question: Model training | Pending |
| :grey_question: Evaluation | Pending |
| :grey_question: Deployment | Pending |
```

Clicking the ℹ icon next to the post opens a side panel (`props.card`) with full detail: run ID, session ID, dataset sample count, S3 paths, eval score, and timestamps — without cluttering the main message.

### Configuration
| Env Var | Required | Description |
|---------|----------|-------------|
| `MATTERMOST_BOT_TOKEN` | Yes | Bot personal access token (`post:create` + `post:write` permissions) |
| `MATTERMOST_API_URL` | Yes | Root URL of your Mattermost server, no trailing slash |
| `MATTERMOST_CHANNEL_ID` | Yes | Channel ID (not name) of the professional updates channel |

All three must be set or all calls are silently skipped. `SLACK_WEBHOOK_URL` is independent.

### Functions
| Function | REST call | When called |
|----------|-----------|-------------|
| `pipeline_started(session_id, run_id) -> str \| None` | `POST /api/v4/posts` | `enqueue_phase1_pipeline` — creates message, returns `post_id` |
| `qa_ready(session_id, run_id, post_id, qa_count, validated_count)` | `PUT` | End of `validate_qa` — Phase 1 done, QA awaiting review |
| `insufficient_data(session_id, run_id, post_id, kept, required)` | `PUT` | `curate_candidates` — not enough samples, pipeline stopped |
| `training_launched(session_id, run_id, post_id, job_id, sample_count)` | `PUT` | `build_dataset` — dataset built, training chain queued |
| `training_done(session_id, run_id, post_id, artifact_s3)` | `PUT` | `launch_training` (local) or `poll_training` (HF) — training succeeded |
| `training_failed(session_id, run_id, post_id, error)` | `PUT` | `launch_training` / `poll_training` on failure |
| `eval_result(session_id, run_id, post_id, passed, score, eval_s3)` | `PUT` | End of `run_evaluation` |
| `pipeline_finished(session_id, run_id, post_id, status, version, reason)` | `PUT` | `deploy_or_rollback` — `status` is `deployed`, `rolled_back`, or `failed` |

### `post_id` persistence
`pipeline_started()` returns the Mattermost `post_id` (a string). The caller (`enqueue_phase1_pipeline`) stores it in `training_runs.config["mm_post_id"]`. Every subsequent function reads it from the DB — meaning it survives worker restarts and no function signatures needed changing.

When Phase 2 creates a new `TrainingRun` row (`enqueue_phase2_pipeline`), it looks up the Phase 1 run's `mm_post_id` and copies it into the Phase 2 run's `config` so the same message continues to be updated.

### Setup (Mattermost Free)
1. Go to **Product menu → Integrations → Bot Accounts** → Create a bot named `lora-train`
2. Copy the generated personal access token → `MATTERMOST_BOT_TOKEN`
3. Set `MATTERMOST_API_URL` to your server root (e.g. `https://mattermost.example.com`)
4. Right-click the target channel → **Copy Link** → the last segment is the `MATTERMOST_CHANNEL_ID`

## Change Log
<!-- Agents: append an entry here after every change -->
| Date | Change | Author |
|------|--------|--------|
| 2026-05-08 | Add shared/mattermost_notifier.py: live-editing Mattermost notifier with 8 functions, post_id threading via training_runs.config, props.card for detail; wire into worker/tasks.py at all 6 phase boundaries; add MATTERMOST_* env vars to .env.example | opencode |
| 2026-05-08 | Fix S3 key paths: upload_adapter now artifacts/ (not adapter/), upload_eval_report now eval_report.json, upload_deployment_manifest now production/current/manifest.json, upload_rollback_manifest now production/history/{run_id}/rollback_manifest.json; fix wrapper count to 22 | opencode |
| 2026-04-28 | Initial documentation created | opencode |
| 2026-05-05 | Remove extraction_started, curation_started; add knowledge_extracted, qa_synthesized, training_data_ready | opencode |
