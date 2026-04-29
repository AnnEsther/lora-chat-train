# AGENTS.md — LoRA Chat & Train

This document provides guidance for agentic coding agents operating in this repository.

---

## Project Overview

A single-user private web application for chat with a small LLM (Llama 3.2:1b) that triggers LoRA fine-tuning when you type `/sleep` or the token budget runs low.

- **Frontend**: Next.js 15 (App Router, React 18, TypeScript, Tailwind CSS)
- **Backend**: FastAPI + async SQLAlchemy + Pydantic v2
- **Worker**: Celery + Redis for async training pipeline
- **Database**: PostgreSQL (async via asyncpg)
- **Storage**: S3 for model artifacts, Slack for notifications

---

## Build / Lint / Test Commands

### Makefile (Python commands)
```bash
make up          # Start all services via docker compose
make down        # Stop all services
make db          # Start only postgres + redis
make init-db     # Initialise the database schema
make backend     # Start FastAPI backend (local, no Docker)
make worker      # Start Celery worker (local, no Docker)
make frontend    # Start Next.js frontend (local)
make test        # Run pytest unit tests
make lint        # Run ruff linter over Python code
make clean       # Remove __pycache__ directories
```

### Python Testing
```bash
cd backend && python -m pytest tests/ -v                    # Run all tests
cd backend && python -m pytest tests/test_core.py -v       # Run specific test file
cd backend && python -m pytest tests/test_core.py::TestSleepCommand -v  # Run specific test class
cd backend && python -m pytest tests/ -k "sleep" -v       # Run tests matching pattern
cd backend && python -m pytest tests/ -x                   # Stop on first failure
```

### Python Linting (Ruff)
```bash
ruff check backend/ worker/ training/ shared/ scripts/      # Lint all Python code
ruff check backend/main.py                                   # Lint single file
ruff check --fix backend/                                    # Auto-fix issues
```

### Frontend Commands
```bash
cd frontend && npm run dev      # Start dev server (localhost:3000)
cd frontend && npm run build    # Production build
cd frontend && npm run lint     # Run Next.js linter
cd frontend && npm run start    # Start production server
```

---

## Code Style Guidelines

### Python

#### Imports
- Always use `from __future__ import annotations` at the top of every module for forward references
- Load `.env` and fix `sys.path` before other project imports (see `backend/main.py:9-15` for pattern)
- Group imports: stdlib → third-party → local (blank line between groups)
- Avoid wildcard imports

```python
from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from models import Session
from schemas import ChatRequest
```

#### Docstrings
- Module docstring: `"""backend/models.py — SQLAlchemy ORM models matching infra/schema.sql."""`
- Class docstring: `"""Session — represents a chat session with token budget tracking."""`
- Function docstring: Concise description of purpose, mention side effects

#### Type Annotations
- Use modern Python 3.11+ syntax: `def foo(x: list[str]) -> dict[str, int]:`
- Use `Optional[X]` for nullable types, not `X | None` (for Pydantic compatibility)
- Explicitly annotate function return types
- Use `uuid.UUID` for all IDs, not strings

#### Naming Conventions
- Classes: `PascalCase` (e.g., `SessionState`, `TrainingRun`)
- Functions/variables: `snake_case` (e.g., `get_db`, `total_tokens`)
- Constants: `SCREAMING_SNAKE_CASE` (e.g., `MAX_SESSION_TOKENS`)
- Private helpers: prefix with `_` (e.g., `_db()`, `_transition()`)
- Enums: `class Foo(str, enum.Enum)` so `.value` returns the string directly

#### Pydantic Schemas (v2)
```python
from pydantic import BaseModel, ConfigDict

class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID
    state: str
    total_tokens: int
    created_at: datetime
    closed_at: Optional[datetime] = None
```

#### SQLAlchemy
- Use async patterns: `AsyncSession`, `async_sessionmaker`, `create_async_engine`
- Celery tasks run in sync context — use sync engine for Celery, async for FastAPI
- Always use `pool_pre_ping=True` on engines for connection reliability
- Use relationship back_populates for bidirectional relations
- JSON columns: use `metadata_ = Column("metadata", JSON, ...)` to avoid Python keyword clash

#### Async/Await
- Use `@asynccontextmanager` for lifespan management
- Use `async for ... in` for async generators
- Wrap streaming responses with `StreamingResponse`
- Always use `await` for all async operations

#### Error Handling
- Use `HTTPException(status_code=404, detail="Session not found.")` for API errors
- Let exceptions propagate in Celery tasks — Celery handles retry/queue
- Use `try/except/finally` for cleanup in database sessions (rollback on error)
- Log structured errors with context: `logger.error("op_failed", extra={"id": str(id), "error": str(exc)})`

#### Logging
- Use `logging.getLogger(__name__)` in every module
- Structured logging: `logger.info("event_name", extra={"key": "value"})`
- Config: `logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")`

---

### TypeScript / React / Next.js

#### Imports
- Use `"use client"` directive for client components
- Use path aliases: `@/*` maps to project root
- Import React hooks explicitly: `import { useState, useEffect, useCallback } from "react";`

#### Type Annotations
- Use `interface` for object shapes, `type` for unions/primitives
- Explicitly annotate component props
- Use TypeScript strict mode (`"strict": true` in tsconfig.json)

```typescript
interface Session {
  id: string;
  state: SessionState;
  total_tokens: number;
  max_tokens: number;
}

type SessionState = "ACTIVE" | "PRE_SLEEP_WARNING" | "INSUFFICIENT_DATA" | "SLEEPING" | "TRAINING" | 
                    "EVALUATING" | "DEPLOYING" | "READY" | "FAILED";
```

#### Naming
- Components: `PascalCase` (e.g., `DiagnosticPanel`, `GaugeBar`)
- Functions/hooks: `camelCase` with `use` prefix for hooks (e.g., `useCallback`)
- Constants: `SCREAMING_SNAKE_CASE` (e.g., `API_URL`, `POLL_INTERVAL_MS`)
- Event handlers: `handle*` (e.g., `handleKeyDown`, `handleSubmit`)

#### Component Patterns
- Keep components small and focused
- Use `useCallback` for functions passed as props to prevent re-renders
- Use `useRef` for DOM references
- Use `AbortSignal.timeout()` for fetch timeouts
- Always handle loading and error states

#### Tailwind CSS
- Use consistent color palette: `bg-gray-*`, `text-gray-*`, `border-gray-*`
- State colors via `Record<SessionState, string>` for consistency
- Use `transition-colors` for interactive elements
- Keep responsive classes inline

---

## Environment Variables

Required in `.env` (see `.env.example`):
- `DATABASE_URL`: PostgreSQL connection (must include `+asyncpg` driver)
- `REDIS_URL` / `CELERY_BROKER_URL`: Redis connection
- `HF_TOKEN`: Hugging Face API token
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `S3_BUCKET`: S3 config
- `SLACK_WEBHOOK_URL`: Slack notifications
- `BASE_MODEL`: e.g., `meta-llama/Llama-3.2-1B-Instruct`
- `MAX_SESSION_TOKENS`: Token budget per session (default 4096)
- `PRE_SLEEP_THRESHOLD`: Warning threshold (default 512)

---

## Directory Structure

```
backend/           FastAPI API + model serving
  main.py          App entry, chat endpoint, session lifecycle
  models.py        SQLAlchemy ORM models
  schemas.py       Pydantic request/response schemas
  database.py      Async SQLAlchemy engine + session factory
  model_client.py  Async model inference wrapper
  routes/          API route modules
  tests/           pytest unit tests

worker/            Celery task definitions (full pipeline orchestration)
training/
  extractor/       Transcript → candidate segments
  curator/         Scoring and filtering
  datasets/        JSONL dataset writer
  trainer/         HF SFTTrainer wrapper
  eval/            Evaluation suite
  deployment/      Promote / rollback logic

shared/            S3 upload, Slack notifications, utilities
frontend/          Next.js chat UI (App Router)
infra/             Docker, Compose, Postgres schema
scripts/           DB init, seed helpers
```

---

## API Contracts

### Sessions
| Method | Path | Description |
|--------|------|-------------|
| POST | `/sessions` | Create a new session |
| GET | `/sessions/{session_id}` | Get session metadata and state |
| GET | `/sessions` | List recent sessions |

### Chat
| Method | Path | Description |
|--------|------|-------------|
| POST | `/sessions/{session_id}/chat` | Send a message, receive SSE stream |

### SSE Events
- `{"type": "start"}`, `{"type": "chunk", "text": "..."}`, `{"type": "end"}`
- `{"type": "status", "remaining_tokens": N, "session_state": "ACTIVE"}`
- `{"type": "sleep_warning", "message": "..."}`, `{"type": "sleeping", "reason": "..."}`

### Health
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Returns `{"status": "ok"}` |

---

## Session State Machine

```
ACTIVE → PRE_SLEEP_WARNING → SLEEPING → TRAINING → EVALUATING → DEPLOYING → READY
                                    ↓                                   ↘ FAILED (rollback)
                              INSUFFICIENT_DATA ← (if < 10 curated samples)
                                    ↓
                              (user continues chatting)
                                    ↓
                              SLEEPING (when /sleep called again)
```

**INSUFFICIENT_DATA**: When curation finds fewer than 10 samples, the session returns to this state. Users can continue chatting to add more data, then trigger `/sleep` again.

---

## Testing Patterns

### Python Tests
- Use `pytest` with `pytest-asyncio` for async tests
- Use `@pytest.mark.asyncio` for async test methods
- Use `unittest.mock.MagicMock`, `AsyncMock`, `patch` for mocking
- Set required env vars before imports: `os.environ.setdefault("DATABASE_URL", "...")`
- Use `tmp_path` fixture for file system operations

### Frontend Tests
- Tests not currently configured; add testing framework as needed

---

## Docker Services

```bash
docker compose up --build       # Build and start all services
docker compose up -d postgres redis   # Start infrastructure only
```

Services: `postgres`, `redis`, `backend`, `worker`, `model_server`, `frontend`

---

## Feature Documentation (`/docs`)

The `docs/` folder contains one Markdown file per feature area. These files are the authoritative reference for each feature's design, file locations, API contracts, and change history.

### Workflow for AI Agents

**Before modifying any feature:**
1. Identify which `docs/*.md` file(s) cover the feature you are touching (see index below).
2. Read the relevant doc(s) in full before writing any code.
3. Pay particular attention to: key files listed, design decisions, configuration, and the Change Log.

**After modifying any feature:**
1. Update the relevant `docs/*.md` file to reflect your changes:
   - Correct any outdated file paths, function names, or behaviour descriptions.
   - Add new sections if you introduced new concepts, endpoints, or configuration.
2. Append a row to the **Change Log** table at the bottom of the doc:
   ```
   | YYYY-MM-DD | Brief description of what changed and why | your-id |
   ```
3. If you created a new feature that has no existing doc, create `docs/<featureName>.md` and add it to the index in `docs/README.md`.

### Documentation Index

| File | Feature Area |
|------|-------------|
| `docs/session-management.md` | Session lifecycle, state machine, token budget, DB model |
| `docs/chat-streaming.md` | SSE streaming, `/sleep` command, token counting, SSE event types |
| `docs/model-server.md` | Local GPU model server, adapter hot-swap, inference endpoints |
| `docs/training-pipeline.md` | Full Celery pipeline — Phase 1 and Phase 2 orchestration |
| `docs/curation.md` | Turn pair extraction, PII redaction, quality scoring and filtering |
| `docs/knowledge-pipeline.md` | Knowledge extraction, Q&A synthesis, validation, corpus merging |
| `docs/lora-training.md` | LoRA hyperparameters, HF endpoint, local SFTTrainer |
| `docs/evaluation-deployment.md` | Eval suite, adapter promotion, smoke test, rollback |
| `docs/dataset-writer.md` | JSONL dataset format for SFTTrainer |
| `docs/storage-notifications.md` | S3 uploads, local fallback, Slack webhook notifications |
| `docs/database.md` | PostgreSQL schema, all 9 tables, async engine setup |
| `docs/frontend-ui.md` | Next.js chat UI, state management, polling, QA review modal |
| `docs/infrastructure.md` | Docker Compose services, volumes, Makefile, env vars |
| `docs/huggingface-training-hosting.md` | HF Inference Endpoint setup, API contract, known bug, GPU tiers |
