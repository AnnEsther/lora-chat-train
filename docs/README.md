# /docs — Feature Documentation Index

This folder contains per-feature documentation, deployment guides, and a student
textbook for the LoRA Chat & Train project.

## Purpose
- **Before modifying a feature:** read the relevant doc to understand current behaviour, file locations, and design decisions.
- **After modifying a feature:** append an entry to the Change Log table at the bottom of the relevant doc.

## Feature Docs

| File | Feature Area |
|------|-------------|
| [session-management.md](./session-management.md) | Session lifecycle, state machine, token budget, DB model |
| [chat-streaming.md](./chat-streaming.md) | SSE streaming, `/sleep` command, QA synthesis, SSE event types |
| [model-server.md](./model-server.md) | Local GPU model server, adapter hot-swap, inference endpoints, base model unload fix |
| [training-pipeline.md](./training-pipeline.md) | Full Celery pipeline — Phase 1 and Phase 2 orchestration; QA→TrainingCandidate promotion |
| [curation.md](./curation.md) | Turn pair extraction, PII redaction, quality scoring and filtering |
| [knowledge-pipeline.md](./knowledge-pipeline.md) | Knowledge extraction, Q&A synthesis, validation, corpus merging |
| [lora-training.md](./lora-training.md) | LoRA hyperparameters, HF endpoint, local SFTTrainer |
| [evaluation-deployment.md](./evaluation-deployment.md) | Eval suite, adapter promotion, smoke test, rollback |
| [dataset-writer.md](./dataset-writer.md) | JSONL dataset format for SFTTrainer |
| [storage-notifications.md](./storage-notifications.md) | S3 uploads, local fallback, Slack webhook notifications |
| [database.md](./database.md) | PostgreSQL schema, all 9 tables, async engine setup |
| [frontend-ui.md](./frontend-ui.md) | Next.js chat UI, InlineDeck QA review, state management, polling |
| [infrastructure.md](./infrastructure.md) | Docker Compose services, volumes, Makefile, env vars, EC2 partial updates |
| [huggingface-training-hosting.md](./huggingface-training-hosting.md) | HF Inference Endpoint setup, API contract, known bug, GPU tiers |

## Deployment Guides

| File | Purpose |
|------|---------|
| [SetupOnEc2.md](./SetupOnEc2.md) | **Complete EC2 setup guide** — instance launch, storage, NVIDIA drivers, nginx, TLS, Docker, full deployment and smoke test |
| [nginx-ec2-setup.md](./nginx-ec2-setup.md) | nginx config reference, SSL, basic auth, SSE buffering settings |

## Learning

| File | Purpose |
|------|---------|
| [textbook.md](./textbook.md) | **Student textbook** — first-principles explanation of LLMs, LoRA, fine-tuning, system architecture, data flow, and all processes in the system |

## Change Log Conventions
Each doc has a Change Log table at the bottom. When making changes:
1. Add a new row with today's date, a concise description, and your identifier.
2. Keep entries in reverse-chronological order (newest first).

```markdown
| Date | Change | Author |
|------|--------|--------|
| YYYY-MM-DD | Description of what changed and why | your-id |
```
