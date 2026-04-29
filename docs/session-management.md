# Session Management

## Overview
Sessions are the core unit of context in this application. Each session tracks a conversation between the user and the model, enforces a token budget, and drives the state machine that triggers LoRA fine-tuning.

## Key Files
- `backend/main.py` — Session CRUD endpoints, state transitions, `_transition()` helper
- `backend/models.py` — `Session` ORM model, `SessionState` enum
- `backend/schemas.py` — `SessionResponse`, `CreateSessionRequest`
- `infra/schema.sql` — `sessions` table DDL with state CHECK constraint
- `frontend/app/page.tsx` — Session switcher, session creation modal, state-reactive UI

## Data Model (`sessions` table)
| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK, `gen_random_uuid()` |
| `state` | TEXT | Enforced by CHECK constraint against full state list |
| `total_tokens` | INT | Cumulative tokens used this session |
| `max_tokens` | INT | Budget ceiling (`MAX_SESSION_TOKENS` env var, default 4096) |
| `system_prompt` | TEXT | Inference system prompt (customisable per session) |
| `training_system_prompt` | TEXT | System prompt injected into the training dataset |
| `created_at` | TIMESTAMPTZ | Set on insert |
| `closed_at` | TIMESTAMPTZ | Set when state enters `SLEEPING` or `FAILED` |
| `updated_at` | TIMESTAMPTZ | Auto-maintained by `touch_updated_at()` trigger |

## State Machine
```
ACTIVE
  ├─ (tokens near threshold) ──────────────→ PRE_SLEEP_WARNING
  ├─ (/sleep or budget exhausted) ─────────→ VALIDATING
  └─ (< MIN_TRAINING_SAMPLES after Phase 1) → INSUFFICIENT_DATA
                                                  │
                                       (chat continues, /sleep again)
                                                  ▼
VALIDATING ──→ [user reviews QA modal] ──→ TRAINING ──→ EVALUATING ──→ DEPLOYING ──→ READY
                                                                                   └──→ FAILED
```

## API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/sessions` | Create session; optional `adapter_id`, `system_prompt`, `training_system_prompt` |
| `GET` | `/sessions/{id}` | Fetch session metadata; also polls model server for training completion |
| `GET` | `/sessions` | List 20 most recent sessions |

## Configuration
| Env Var | Default | Description |
|---------|---------|-------------|
| `MAX_SESSION_TOKENS` | `4096` | Token budget per session |
| `PRE_SLEEP_THRESHOLD` | `512` | Remaining tokens at which warning state is triggered |

## State Transition Logic (`_transition()`)
- Located at `backend/main.py`
- Atomically updates `session.state` via SQLAlchemy
- Sets `closed_at` for terminal states: `SLEEPING`, `FAILED`
- Raises `HTTPException(400)` if the state is already set to the target

## Frontend Behaviour
- On mount: restores last session ID from `localStorage`
- Falls back to most recent non-`READY` session, or creates a new one
- Polls `/sessions/{id}` every 3 seconds while session is not in a terminal state
- Session state drives input disabled/enabled and which modals are shown

## Change Log
<!-- Agents: append an entry here after every change -->
| Date | Change | Author |
|------|--------|--------|
| 2026-04-28 | Initial documentation created | opencode |
