# /docs — Feature Documentation Index

This folder contains per-feature documentation for the LoRA Chat & Train project.

## Purpose
- **Before modifying a feature:** read the relevant doc to understand current behaviour, file locations, and design decisions.
- **After modifying a feature:** append an entry to the Change Log table at the bottom of the relevant doc.

## Documents

| File | Feature Area |
|------|-------------|
| [session-management.md](./session-management.md) | Session lifecycle, state machine, token budget, DB model |
| [chat-streaming.md](./chat-streaming.md) | SSE streaming, `/sleep` command, token counting, SSE event types |
| [model-server.md](./model-server.md) | Local GPU model server, adapter hot-swap, inference endpoints |
| [training-pipeline.md](./training-pipeline.md) | Full Celery pipeline — Phase 1 and Phase 2 orchestration |
| [curation.md](./curation.md) | Turn pair extraction, PII redaction, quality scoring and filtering |
| [knowledge-pipeline.md](./knowledge-pipeline.md) | Knowledge extraction, Q&A synthesis, validation, corpus merging |
| [lora-training.md](./lora-training.md) | LoRA hyperparameters, HF endpoint, local SFTTrainer |
| [evaluation-deployment.md](./evaluation-deployment.md) | Eval suite, adapter promotion, smoke test, rollback |
| [dataset-writer.md](./dataset-writer.md) | JSONL dataset format for SFTTrainer |
| [storage-notifications.md](./storage-notifications.md) | S3 uploads, local fallback, Slack webhook notifications |
| [database.md](./database.md) | PostgreSQL schema, all 9 tables, async engine setup |
| [frontend-ui.md](./frontend-ui.md) | Next.js chat UI, state management, polling, QA review modal |
| [infrastructure.md](./infrastructure.md) | Docker Compose services, volumes, Makefile, env vars |
| [huggingface-training-hosting.md](./huggingface-training-hosting.md) | HF Inference Endpoint setup, API contract, known bug, GPU tiers |

## Change Log Conventions
Each doc has a Change Log table at the bottom. When making changes:
1. Add a new row with today's date, a concise description of the change, and your identifier (e.g., `opencode` or a PR number).
2. Keep entries in reverse-chronological order (newest first).

```markdown
| Date | Change | Author |
|------|--------|--------|
| YYYY-MM-DD | Description of what changed and why | your-id |
```
