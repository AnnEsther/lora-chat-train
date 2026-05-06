# Training Pipeline

## Overview
The training pipeline is an asynchronous Celery-based workflow that transforms a completed chat session into a fine-tuned LoRA adapter. It is split into two phases to allow a human QA review step between curation and training.

## Key Files
- `worker/tasks.py` — All Celery task definitions (902 lines)
- `training/extractor/transcript_extractor.py` — Turn pair extraction + PII redaction
- `training/curator/curator.py` — Multi-factor quality scoring and filtering
- `training/datasets/dataset_writer.py` — JSONL SFTTrainer format writer
- `training/trainer/hf_launcher.py` — HF endpoint launcher + local SFTTrainer fallback
- `training/eval/evaluator.py` — Domain eval suite + pass/fail scoring
- `training/deployment/deploy.py` — Promote adapter, smoke test, rollback

## Pipeline Entry Points

### Phase 1 — `enqueue_phase1_pipeline(session_id)`
Triggered by `_force_sleep()` in the backend when a session transitions to `VALIDATING`.

```
extract_candidates
    → curate_candidates
        → extract_knowledge
            → synthesize_qa
                → validate_qa
                    → [session enters VALIDATING; user reviews QA modal]
```

If curation finds fewer than `MIN_TRAINING_SAMPLES` (default 10), the session transitions to `INSUFFICIENT_DATA` and the pipeline stops. The user can continue chatting and trigger `/sleep` again.

### Phase 2 — `enqueue_phase2_pipeline(session_id, run_id)`
Triggered by `POST /sessions/{id}/start-training` after the user validates QA.

```
build_dataset
    → launch_training
        → poll_training
            → run_evaluation
                → deploy_or_rollback
```

## Individual Tasks

### `extract_candidates`
- Input: raw transcript from DB (`turns` table)
- Runs `TranscriptExtractor.extract()` — produces `(user, assistant)` candidate pairs
- Applies PII/secret redaction (API keys, passwords, credit cards, emails, cloud secrets)
- Rejects: too-short turns (< 20 chars), too-long assistant (> 4000 chars), `/sleep` commands
- Persists `TrainingCandidate` rows to DB
- Uploads raw transcript + candidate JSON to S3

### `curate_candidates`
- Input: candidate IDs from DB
- Runs `Curator.score_and_filter()` — 4-dimensional scoring (see [Curation doc](./curation.md))
- Persists scores + `included` flag to DB
- If `included_count < MIN_TRAINING_SAMPLES`: transitions session to `INSUFFICIENT_DATA`, stops pipeline
- Uploads curated set to S3

### `extract_knowledge`
- Input: included candidates from DB
- Runs `KnowledgeExtractor` + `KnowledgeNormalizer`
- Persists `KnowledgeRecord` rows (topic, facts as JSONB, source_turn_id) to DB
- Sends `knowledge_extracted` Slack notification

### `synthesize_qa`
- Input: knowledge records from DB
- Runs `QASynthesizer` — calls model server `/generate` to produce 2–3 Q&A pairs per fact
- Falls back to raw candidates if synthesis fails
- Persists `SynthesizedQA` rows to DB

### `validate_qa`
- Input: synthesized Q&A from DB
- Runs `QAValidator` — scores on relevance, grammar, completeness, accuracy
- Marks `validated = True` or increments `retry_count`
- Auto-marks for human review after 3 consecutive failures

### `merge_corpus`
- Input: validated Q&A + knowledge records
- Runs `CorpusManager.merge()` — deduplicates across sessions by `(type, content[:100])` key
- Persists `KnowledgeCorpus` entries

### `build_dataset`
- Input: included candidates from DB
- Runs `DatasetWriter.write_jsonl()` — writes chat-template-compatible JSONL
- Uploads JSONL to S3
- Persists `Dataset` record to DB

### `launch_training`
- Input: dataset S3 path
- Calls `HFTrainingLauncher.launch()`:
  - If `HF_TRAINING_ENDPOINT` is set: submits job to HuggingFace endpoint
  - Otherwise: POSTs `{"run_id", "dataset_path"}` to local model server `/train`
- Persists `TrainingRun` record with status `RUNNING`

### `poll_training`
- Input: HF job ID
- Polls HF endpoint every 60 s (up to 60 retries = 1 hour)
- On success: downloads artifacts, uploads logs + adapter to S3
- On timeout/failure: marks run `FAILED`

### `run_evaluation`
- Input: artifact directory
- Loads model + new adapter locally
- Runs `Evaluator` against 5 default eval cases
- Uploads eval report to S3
- Transitions session to `DEPLOYING` if pass threshold met

### `deploy_or_rollback`
- Input: eval result + artifact directory
- If eval passed: calls `DeploymentManager.promote()`, runs smoke test
- If smoke test fails or eval failed: calls `DeploymentManager.rollback()`
- Transitions session to `READY` or `FAILED`

## Configuration
| Env Var | Default | Description |
|---------|---------|-------------|
| `MIN_TRAINING_SAMPLES` | `10` | Minimum curated samples to proceed with training |
| `CELERY_BROKER_URL` | — | Redis URL for Celery broker |
| `CELERY_RESULT_BACKEND` | — | Redis URL for Celery results |
| `HF_TRAINING_ENDPOINT` | — | If set, use HuggingFace endpoint; else use local model server |
| `HF_TOKEN` | — | HuggingFace API token |

## Database Tables
- `training_candidates` — extracted + scored turn pairs
- `datasets` — built JSONL with S3 path
- `training_runs` — HF job record with status, logs, artifact paths
- `model_versions` — versioned adapters with eval scores
- `deployment_events` — audit trail: PROMOTE, ROLLBACK, SMOKE_TEST_PASS/FAIL

## Change Log
<!-- Agents: append an entry here after every change -->
| Date | Change | Author |
|------|--------|--------|
| 2026-04-28 | Initial documentation created | opencode |
| 2026-05-05 | Add Slack notifications to extract_knowledge, synthesize_qa, validate_qa (knowledge_extracted, qa_synthesized, training_data_ready) | opencode |
