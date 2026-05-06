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
| `upload_adapter` | `training_runs/{run_id}/adapter/` |
| `upload_eval_report` | `training_runs/{run_id}/eval/report.json` |
| `upload_deployment_manifest` | `training_runs/{run_id}/deployment/manifest.json` |
| `upload_rollback_manifest` | `training_runs/{run_id}/rollback/manifest.json` |
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

### 26 Convenience Wrappers
Covering every named pipeline event. Call these from Celery tasks or backend code:

`session_started`, `pre_sleep_warning`, `session_sleeping`, `extraction_completed`, `curation_completed`, `knowledge_extracted`, `qa_synthesized`, `training_data_ready`, `dataset_built`, `artifact_uploaded`, `training_started`, `training_succeeded`, `training_failed`, `evaluation_started`, `evaluation_completed`, `deployment_approved`, `deployment_rejected`, `adapter_switch_succeeded`, `adapter_switch_failed`, `rollback_triggered`, `rollback_completed`, `insufficient_data_warning`

### Configuration
| Env Var | Description |
|---------|-------------|
| `SLACK_WEBHOOK_URL` | Slack incoming webhook URL; if absent, notifications are silently skipped |
| `LOCAL_OUTPUT_DIR` | Override for local storage fallback directory |

## Change Log
<!-- Agents: append an entry here after every change -->
| Date | Change | Author |
|------|--------|--------|
| 2026-04-28 | Initial documentation created | opencode |
| 2026-05-05 | Remove extraction_started, curation_started; add knowledge_extracted, qa_synthesized, training_data_ready | opencode |
