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
| `closed_at` | TIMESTAMPTZ | Set when state enters `SLEEPING` only |
| `failure_reason` | TEXT | Human-readable error written on any `FAILED` transition |
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
| `POST` | `/sessions` | Create session; optional `adapter_id`, `system_prompt`, `training_system_prompt`; loads adapter on model server if `adapter_id` provided |
| `POST` | `/load_adapter` | Load an adapter on the model server without creating a new session |
| `GET` | `/sessions/{id}` | Fetch session metadata; auto-transitions to `READY` if model server reports training completed |
| `GET` | `/sessions` | List 20 most recent sessions |
| `GET` | `/sessions/{id}/qa` | List all synthesized Q&A for the session |
| `PUT` | `/sessions/{id}/qa/{qa_id}` | Edit a Q&A pair (question, answer, validated flag) |
| `DELETE` | `/sessions/{id}/qa/{qa_id}` | Permanently delete a Q&A entry |
| `POST` | `/sessions/{id}/qa/validate-mark` | Bulk-mark all unvalidated Q&A as validated |
| `POST` | `/sessions/{id}/start-training` | Trigger Phase 2 pipeline after user validates QA (requires `VALIDATING` state + at least one validated QA) |
| `POST` | `/sessions/{id}/restart-training` | Re-queue Phase 2 from `launch_training` for a failed run (re-uses existing dataset) |
| `GET` | `/adapters` | Proxy adapter list from the model server (falls back to base-only list on error) |
| `GET` | `/outputs` | List up to 30 most-recent files in `LOCAL_OUTPUT_DIR` for the diagnostic panel |
| `GET` | `/health` | Returns `{"status": "ok"}` |

## Configuration
| Env Var | Default | Description |
|---------|---------|-------------|
| `MAX_SESSION_TOKENS` | `4096` | Token budget per session |
| `PRE_SLEEP_THRESHOLD` | `512` | Remaining tokens at which warning state is triggered |

## State Transition Logic (`_transition()`)
- Located at `backend/main.py`
- Atomically updates `session.state` via SQLAlchemy
- Sets `closed_at` only when transitioning to `SLEEPING` — no other state sets `closed_at`
- `FAILED` and `VALIDATING` sessions do **not** set `closed_at`; the session stays open
- `failure_reason` is written by `_set_failure_reason()` in `worker/tasks.py` at each FAILED site
- Does **not** raise on re-transition to the same state (unlike older versions)

## Frontend Behaviour
- On mount: restores last session ID from `localStorage`
- Falls back to most recent non-`READY` session, or creates a new one
- Polls `/sessions/{id}` every 3 seconds while session is not in a terminal state
- Session state drives input disabled/enabled and which modals are shown
- On transition to `FAILED`: injects a system message in the chat window with the error from `failure_reason`; input remains active so the user can keep chatting or type `/sleep` to retry
- On transition to `INSUFFICIENT_DATA`: injects a system message in the chat window explaining the shortfall and inviting the user to keep chatting

## Change Log
<!-- Agents: append an entry here after every change -->
| Date | Change | Author |
|------|--------|--------|
| 2026-05-08 | Add DELETE /sessions/{id}/qa/{qa_id} endpoint to table | opencode |
| 2026-05-08 | Expanded API endpoints table to include all 11 endpoints; clarified _transition closed_at behaviour (SLEEPING only, not VALIDATING/FAILED); removed erroneous HTTPException(400) note on re-transition | opencode |
| 2026-04-29 | FAILED sessions no longer set closed_at; failure_reason column added; chat input unlocked on FAILED; system messages injected in chat on FAILED and INSUFFICIENT_DATA transitions | opencode |
| 2026-04-28 | Initial documentation created | opencode |
