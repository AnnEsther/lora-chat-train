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
Extracted turn pairs from the extractor/curation pipeline.
| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `session_id` | UUID | FK → sessions |
| `user_turn` | TEXT | |
| `assistant_turn` | TEXT | |
| `turn_index` | INT | Position in original transcript |
| `score` | FLOAT | Composite quality score |
| `included` | BOOL | Whether this pair passes the threshold |
| `rejection_reason` | TEXT | Nullable |
| `created_at` | TIMESTAMPTZ | |

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
| `dataset_id` | UUID | FK → datasets |
| `status` | TEXT | CHECK: `PENDING\|RUNNING\|SUCCEEDED\|FAILED\|ROLLED_BACK` |
| `hf_job_id` | TEXT | HF endpoint job identifier |
| `eval_result` | JSON | Eval report dict |
| `adapter_s3_path` | TEXT | |
| `log_s3_path` | TEXT | |
| `started_at` | TIMESTAMPTZ | |
| `completed_at` | TIMESTAMPTZ | |
| `metadata_` | JSON | Mapped as `metadata` column |

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
| `version_tag` | TEXT | |
| `notes` | TEXT | |
| `occurred_at` | TIMESTAMPTZ | |

### `knowledge_records`
Structured facts extracted from conversations.
| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `session_id` | UUID | FK → sessions (CASCADE DELETE) |
| `topic` | TEXT | Topic domain |
| `fact_type` | TEXT | `fact`, `definition`, `qa_pair`, `code_example`, `step`, `task` |
| `content` | TEXT | |
| `source_turn_index` | INT | |
| `created_at` | TIMESTAMPTZ | |

### `synthesized_qa`
Model-generated Q&A pairs.
| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `session_id` | UUID | FK → sessions (CASCADE DELETE) |
| `knowledge_record_id` | UUID | FK → knowledge_records |
| `question` | TEXT | |
| `answer` | TEXT | |
| `validated` | BOOL | Human or auto-validated flag |
| `validation_score` | FLOAT | Automated quality score |
| `validation_notes` | TEXT | Reason for pass/fail |
| `retry_count` | INT | Number of re-synthesis attempts |
| `created_at` | TIMESTAMPTZ | |

### `knowledge_corpus`
Cross-session, deduplicated knowledge base.
| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `session_id` | UUID | FK → sessions |
| `topic` | TEXT | |
| `fact_type` | TEXT | |
| `content` | TEXT | |
| `dedup_key` | TEXT | `(type, content[:100])` hash |
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
| 2026-04-28 | Initial documentation created | opencode |
