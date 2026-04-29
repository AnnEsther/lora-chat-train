# Chat & Streaming

## Overview
The chat feature streams model responses to the frontend via Server-Sent Events (SSE). It handles token counting, the `/sleep` command, budget enforcement, and state transitions — all within a single streaming generator.

## Key Files
- `backend/main.py` — `POST /sessions/{id}/chat`, `_stream_reply()`, `_force_sleep()`
- `backend/model_client.py` — `ModelClient.stream()`, SSE parsing from model server
- `backend/token_counter.py` — `count()`, `count_messages()`
- `frontend/app/page.tsx` — `sendMessage()`, SSE event parsing, message rendering

## Streaming Flow

### Backend (`_stream_reply()` in `backend/main.py`)
1. Validate session is in an accepting state; reject with `400` otherwise
2. Detect `/sleep` command — skip model call, jump directly to `_force_sleep()`
3. Load conversation history via `_load_history()` (queries all turns ordered by `created_at`)
4. Call `ModelClient.stream(messages, ...)` — returns an async generator of text chunks
5. Yield `{"type": "start"}`
6. For each chunk: yield `{"type": "chunk", "text": "..."}`
7. After streaming: persist user + assistant turns to DB, update `session.total_tokens`
8. Compute `remaining = session.max_tokens - session.total_tokens`
9. Decide end condition:
   - `INSUFFICIENT_DATA` state → yield status, end
   - Budget exhausted (`remaining ≤ 0`) or deep in `PRE_SLEEP_WARNING` → yield `sleep_warning`, call `_force_sleep()`
   - Otherwise → yield `{"type": "status", "remaining_tokens": N, "session_state": "..."}`, yield `{"type": "end"}`

### `_force_sleep()` generator
- Transitions session to `VALIDATING`
- Enqueues Celery task `enqueue_phase1_pipeline.delay(session_id)`
- Yields `{"type": "validating", "reason": "..."}`

## SSE Event Types
| Event | Payload | Meaning |
|-------|---------|---------|
| `start` | — | Stream about to begin |
| `chunk` | `{"text": "..."}` | A fragment of the assistant reply |
| `end` | — | Stream complete, no sleep triggered |
| `status` | `{"remaining_tokens": N, "session_state": "..."}` | Budget update after response |
| `sleep_warning` | `{"message": "..."}` | Budget nearly exhausted, sleep imminent |
| `sleeping` | `{"reason": "..."}` | Session has transitioned to SLEEPING (legacy) |
| `validating` | `{"reason": "..."}` | Session entering VALIDATING, Phase 1 enqueued |

## Model Client (`backend/model_client.py`)
- `ModelClient.stream(messages, max_new_tokens, temperature)` — async generator
- POSTs to `{MODEL_SERVER_URL}/chat` with `stream: True`
- Parses raw SSE lines: skips blank lines and `data: [DONE]`, extracts `data: <JSON>`
- On `httpx.RequestError` or HTTP error: yields a human-readable error string rather than raising

## Token Counting (`backend/token_counter.py`)
- `count(text)` — uses `tiktoken` (`cl100k_base`) when available; falls back to `len(text) // 4`
- `count_messages(messages)` — sums `count(content) + 4` overhead per message
- Intentionally approximate — accuracy matters less than reliable threshold detection

## Configuration
| Env Var | Default | Description |
|---------|---------|-------------|
| `MODEL_SERVER_URL` | — | Base URL of the model server |
| `MAX_NEW_TOKENS` | — | Max tokens the model generates per response |
| `TEMPERATURE` | — | Sampling temperature |
| `MODEL_REQUEST_TIMEOUT` | — | HTTP timeout for model server requests |

## Frontend (`frontend/app/page.tsx`)
- `sendMessage()` opens a `fetch` SSE stream to `/sessions/{id}/chat`
- Parses each `data:` line, dispatches by `type`
- On `chunk`: appends text to the last streaming message in state
- On `status`: updates `session.total_tokens` and `session.state`
- On `validating` / `sleeping`: inserts a system message, marks session state
- On `end`: clears the `streaming` flag on the last message
- Enter key submits (Shift+Enter inserts newline)
- Textarea auto-resizes up to 160 px

## Change Log
<!-- Agents: append an entry here after every change -->
| Date | Change | Author |
|------|--------|--------|
| 2026-04-28 | Initial documentation created | opencode |
