# Database

## Overview
The application uses PostgreSQL (16) with two SQLAlchemy engines: an async engine for the FastAPI backend and a sync engine for Celery workers. The schema is managed as a single `infra/schema.sql` DDL file.

## Key Files
- `backend/database.py` — Async engine, session factory, `get_db()` dependency
- `backend/models.py` — All SQLAlchemy ORM models
- `backend/schemas.py` — Pydantic v2 response schemas
- `infra/schema.sql` — Authoritative DDL (applied via Docker initdb or `scripts/init_db.py`)
- `scripts/init_db.py` — Applies schema; safe to run multiple times (`IF NOT EXISTS`)
- `scripts/reset_all.py` — Wipes all data rows + clears adapter files

## Async Engine (`backend/database.py`)
```python
engine = create_async_engine(
    DATABASE_URL,          # must include +asyncpg driver
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,    # validate connections before use
)
AsyncSessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,
    autoflush=False,
)
```

### `get_db()` FastAPI Dependency
```python
async def get_db() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

## Celery Sync Engine
Celery workers use a **separate sync engine** (asyncpg stripped from the URL) because Celery is synchronous. Defined inline in `worker/tasks.py`.

## Tables

### `sessions`
Core session record. Has a `CHECK` constraint enforcing all valid states.
See [Session Management doc](./session-management.md) for column details.

### `turns`
Individual conversation messages.
| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `session_id` | UUID | FK → sessions (CASCADE DELETE) |
| `role` | TEXT | `user`, `assistant`, or `system` |
| `content` | TEXT | Message text |
| `token_count` | INT | Tokens in this turn |
| `created_at` | TIMESTAMPTZ | |

Index: `idx_turns_session(session_id, created_at)` — optimises history loading.

### `training_candidates`
Extracted conversation segments from the extractor/curation pipeline.
| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `session_id` | UUID | FK → sessions |
| `conversation` | JSONB | Full multi-turn segment `[{"role", "content"}, ...]` |
| `quality_score` | FLOAT | Composite quality score (nullable until curated) |
| `included` | BOOL | Whether this segment passes the threshold |
| `rejection_reason` | TEXT | Nullable |
| `created_at` | TIMESTAMPTZ | |

Index: `idx_candidates_session(session_id)`

### `datasets`
Built JSONL datasets.
| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `session_id` | UUID | FK → sessions |
| `s3_path` | TEXT | S3 or `local://` URI |
| `sample_count` | INT | Number of training examples |
| `created_at` | TIMESTAMPTZ | |

### `training_runs`
HuggingFace training job records.
| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `session_id` | UUID | FK → sessions |
| `dataset_id` | UUID | FK → datasets (nullable) |
| `status` | TEXT | CHECK: `PENDING\|RUNNING\|SUCCEEDED\|FAILED\|ROLLED_BACK` |
| `hf_job_id` | TEXT | HF endpoint job identifier (nullable) |
| `config` | JSONB | Full training config dict |
| `logs_s3_path` | TEXT | Nullable |
| `artifact_s3_path` | TEXT | Nullable |
| `eval_s3_path` | TEXT | Nullable |
| `eval_passed` | BOOL | Nullable — set after evaluation |
| `started_at` | TIMESTAMPTZ | Nullable |
| `finished_at` | TIMESTAMPTZ | Nullable |
| `created_at` | TIMESTAMPTZ | |

### `model_versions`
Deployed adapter versions.
| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `run_id` | UUID | FK → training_runs |
| `version_tag` | TEXT | Human-readable version identifier |
| `adapter_s3_path` | TEXT | |
| `is_production` | BOOL | Currently active adapter |
| `eval_score` | FLOAT | Score from eval suite |
| `promoted_at` | TIMESTAMPTZ | |

### `deployment_events`
Audit trail for all deployment actions.
| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `run_id` | UUID | FK → training_runs |
| `event_type` | TEXT | CHECK: `PROMOTE\|ROLLBACK\|SMOKE_TEST_PASS\|SMOKE_TEST_FAIL` |
| `from_version` | TEXT | Previous version tag (nullable) |
| `to_version` | TEXT | New version tag (nullable) |
| `reason` | TEXT | Human-readable reason (nullable) |
| `created_at` | TIMESTAMPTZ | |

### `knowledge_records`
Structured facts extracted from conversations.
| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `session_id` | UUID | FK → sessions (CASCADE DELETE) |
| `topic` | TEXT | Topic domain |
| `facts` | JSONB | `list[dict]` — structured fact records |
| `source_turn_id` | UUID | FK → turns (SET NULL) |
| `created_at` | TIMESTAMPTZ | |

### `synthesized_qa`
Model-generated Q&A pairs.
| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `session_id` | UUID | FK → sessions (CASCADE DELETE) |
| `knowledge_record_id` | UUID | FK → knowledge_records (SET NULL) |
| `question` | TEXT | |
| `answer` | TEXT | |
| `validated` | BOOL | Human or auto-validated flag |
| `edited` | BOOL | User manually edited |
| `retry_count` | INT | Number of validation attempts |
| `validation_notes` | TEXT | Reason for pass/fail |
| `created_at` | TIMESTAMPTZ | |

### `knowledge_corpus`
Cross-session, deduplicated knowledge base.
| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `topic` | TEXT | |
| `facts` | JSONB | `list[dict]` — deduplicated facts |
| `source_session_id` | UUID | FK → sessions |
| `created_at` | TIMESTAMPTZ | |

## Schema Features
- `pgcrypto` extension for `gen_random_uuid()`
- `touch_updated_at()` trigger on `sessions` for automatic `updated_at`
- Cascading deletes: `turns → sessions`, `knowledge_records → sessions`, `synthesized_qa → sessions`
- All IDs are `UUID` via `gen_random_uuid()`

## Configuration
| Env Var | Description |
|---------|-------------|
| `DATABASE_URL` | Must include `+asyncpg` driver for async use (e.g., `postgresql+asyncpg://user:pass@host/db`) |

## Change Log
<!-- Agents: append an entry here after every change -->
| Date | Change | Author |
|------|--------|--------|
| 2026-05-08 | Fix training_candidates: replaced user_turn/assistant_turn/turn_index with conversation JSONB; fix training_runs: replaced eval_result/log_s3_path/completed_at/metadata_ with config/logs_s3_path/artifact_s3_path/eval_s3_path/eval_passed/finished_at; fix deployment_events: replaced version_tag/notes/occurred_at with from_version/to_version/reason/created_at | opencode |
| 2026-04-28 | Initial documentation created | opencode |
| 2026-05-05 | Fix knowledge_records table (add topic, facts, source_turn_id), fix synthesized_qa (add edited, fix retry_count), fix knowledge_corpus (topic, facts, source_session_id) | opencode |
