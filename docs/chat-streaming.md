# Chat & Streaming

## Overview
The chat endpoint now acts as a **Q&A synthesis pipeline** rather than a chat relay. Every message the user sends is treated as a training passage: the backend calls the model server's `/generate` endpoint synchronously to produce Q&A pairs, persists them to `synthesized_qa`, and streams them back as SSE events. The pairs appear inline as editable cards below each message in the chat window.

The `/sleep` command still works and now goes directly to Phase 2 training (skipping Phase 1 extraction) if inline QA pairs already exist.

## Key Files
- `backend/main.py` — `POST /sessions/{id}/chat`, `_synthesize_and_stream()`, `_force_sleep()`
- `training/knowledge/synthesizer.py` — `synthesize_from_passage()` — new single-passage synthesis function
- `frontend/app/page.tsx` — `sendMessage()`, SSE event parsing, inline `QACard` component

## SSE Events (new flow)

| Event type | Payload | Description |
|---|---|---|
| `start` | — | Stream open |
| `qa_pairs` | `{pairs: [{id, question, answer}]}` | Generated Q&A pairs for this passage |
| `qa_count` | `{total, validated, min_required, ready}` | Updated session-wide QA counts |
| `end` | — | Stream closed |
| `sleep_ack` | `{message}` | Emitted only on `/sleep` command |
| `sleeping` | `{reason, message}` | Session entering training (Phase 2 started) |
| `validating` | `{reason, message}` | Legacy: Phase 1 fallback when no inline QA exists |

## Streaming Flow

### Backend (`_synthesize_and_stream()` in `backend/main.py`)
1. Validate session is in an accepting state (`ACTIVE`, `PRE_SLEEP_WARNING`, `INSUFFICIENT_DATA`, `FAILED`)
2. Detect `/sleep` — route to `_handle_sleep_command()` instead
3. Persist user `Turn` to DB
4. Run `synthesize_from_passage(passage, system_prompt)` in a thread pool executor (sync→async bridge)
5. Persist each `SynthesizedQA` row with `source_turn_id = user_turn.id`
6. Yield `{"type": "start"}`
7. Yield `{"type": "qa_pairs", "pairs": [...]}`
8. Query updated QA counts and yield `{"type": "qa_count", ...}`
9. Yield `{"type": "end"}`

### `_force_sleep()` generator (updated)
- If inline QA pairs exist for the session: marks all unvalidated as validated → enqueues `enqueue_phase2_pipeline` → transitions to `TRAINING` → yields `sleeping` event
- If no inline QA exists: falls back to `enqueue_phase1_pipeline` → transitions to `VALIDATING` → yields `validating` event (legacy path)
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

## Chat Request Payload

`POST /sessions/{session_id}/chat` accepts:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `message` | `string` | yes | — | The user's passage text (or `/sleep`) |
| `num_qa` | `integer` | no | `5` | Number of Q&A pairs to generate (1–20) |

The `num_qa` value is passed as `max_parts` to `_split_passage()`, which splits the passage into at most that many segments. One Q&A pair is synthesised per segment, so the final number of pairs equals `min(num_qa, natural_segment_count)`.

## Configuration
| Env Var | Default | Description |
|---------|---------|-------------|
| `MODEL_SERVER_URL` | — | Base URL of the model server |
| `MAX_NEW_TOKENS` | — | Max tokens the model generates per response |
| `TEMPERATURE` | — | Sampling temperature |
| `MODEL_REQUEST_TIMEOUT` | — | HTTP timeout for model server requests |

## Frontend SSE Handling (`frontend/app/page.tsx`)

`sendMessage()` opens a `fetch` SSE stream to `POST /sessions/{id}/chat` and dispatches events:

| Event | Frontend action |
|-------|----------------|
| `start` | Sets `qaPairs: []` and `segmentCount` on the last message; `InlineDeck` appears with skeleton slots |
| `qa_pair` | Appends one `QAPair` to last message's `qaPairs`; skeleton slot replaced by real card |
| `qa_count` | Updates global `qaCount` state; refreshes Start Training button counter |
| `end` | Clears `synthLoading` on last message; all skeletons disappear |
| `sleeping` / `sleep_ack` | Inserts system message; updates session state to `TRAINING` |
| `validating` | Inserts system message; updates session state to `VALIDATING` |
| `error` | Shows error banner; clears `synthLoading` |

QA pairs appear **inline below the user bubble** as an `InlineDeck` component — not in a modal. See [frontend-ui.md](./frontend-ui.md) for full deck behaviour.

## Change Log
<!-- Agents: append an entry here after every change -->
| Date | Change | Author |
|------|--------|--------|
| 2026-07-10 | Add `num_qa` field to `ChatRequest` (default 5, range 1–20); pass through `_synthesize_and_stream`; replace hardcoded `max_parts=5`; add Q&A count number input in frontend footer | opencode |
| 2026-05-20 | Update frontend SSE handling table to reflect InlineDeck (not modal); qa_pair events populate InlineDeck cards progressively; no modal-open trigger on SSE events | opencode |
| 2026-05-18 | Complete redesign of chat flow: every message now synthesises Q&A pairs inline instead of streaming an LLM reply. New SSE events: qa_pairs, qa_count. _force_sleep now routes to Phase 2 directly if inline QA exists. | opencode |
| 2026-04-29 | FAILED sessions no longer freeze chat — input stays active; INSUFFICIENT_DATA transition injects system message | opencode |
| 2026-04-28 | Initial documentation created | opencode |
