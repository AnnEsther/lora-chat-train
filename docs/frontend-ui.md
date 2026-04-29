# Frontend UI

## Overview
A single-page Next.js 15 application (App Router, Client Component) that provides the complete user interface: chat, session management, QA review, adapter selection, and a live diagnostic panel.

## Key Files
- `frontend/app/page.tsx` — Entire UI (1133 lines); all state, components, and logic in one file
- `frontend/app/layout.tsx` — Root HTML shell
- `frontend/next.config.js` — Next.js configuration
- `frontend/tailwind.config.ts` — Tailwind CSS configuration

## State Variables
| State | Type | Description |
|-------|------|-------------|
| `sessions` | `Session[]` | All recent sessions (for switcher dropdown) |
| `session` | `Session \| null` | Currently active session |
| `messages` | `Message[]` | Chat message array with `streaming` flag |
| `input` | `string` | Current textarea value |
| `health` | `ModelHealth` | Model server health snapshot |
| `trainStatus` | `TrainStatus \| null` | Training progress from model server |
| `adapters` | `Adapter[]` | Available adapters |
| `selectedAdapter` | `string \| null` | Adapter to load on next new session |
| `qaItems` | `SynthesizedQA[]` | Q&A pairs for the review modal |
| `systemPrompt` | `string` | Chat system prompt override |
| `trainingSystemPrompt` | `string` | Training dataset system prompt override |

## TypeScript Interfaces
```typescript
interface Session {
  id: string;
  state: SessionState;
  total_tokens: number;
  max_tokens: number;
  created_at: string;
  closed_at: string | null;
  system_prompt: string | null;
  training_system_prompt: string | null;
}

type SessionState =
  | "ACTIVE" | "PRE_SLEEP_WARNING" | "INSUFFICIENT_DATA"
  | "VALIDATING" | "SLEEPING" | "TRAINING"
  | "EVALUATING" | "DEPLOYING" | "READY" | "FAILED";

interface Message {
  role: "user" | "assistant" | "system";
  content: string;
  streaming?: boolean;
}
```

## Session Management
- **On mount:** restores last session ID from `localStorage`; fetches session list; falls back to most recent non-`READY` session or creates a new one
- **New session modal:** user picks adapter + sets both system prompts before calling `POST /sessions`
- **Session switcher dropdown:** in header; switching sets `localStorage` and reloads

## Chat Interaction

### `sendMessage()`
1. Appends user message to `messages` state
2. Opens a `fetch` SSE stream to `POST /sessions/{id}/chat`
3. Parses each `data:` line by `type`:
   - `chunk` → appends text to last streaming message
   - `status` → updates `session.total_tokens` + `session.state`
   - `sleep_warning` → inserts system message
   - `validating` / `sleeping` → inserts system message, marks session state, triggers QA modal load
   - `end` → clears `streaming` flag
4. On stream complete: re-fetches session to get latest state

### Input Behaviour
- **Enter** submits; **Shift+Enter** inserts newline
- Textarea auto-resizes up to 160 px
- Input disabled when session state is not `ACTIVE` or `PRE_SLEEP_WARNING`

## QA Review Modal
- **Trigger:** auto-opens when `session.state === "VALIDATING"`
- **Navigation:** card-by-card with Previous/Next buttons
- **Per-card:** view + inline-edit question and answer; see `validation_notes`; click "Mark Validated"
- **Bulk action:** "Validate All & Start Training" → `POST /qa/validate-mark` then `POST /start-training`
- **API calls:**
  - `GET /sessions/{id}/qa` — load items
  - `PUT /sessions/{id}/qa/{qa_id}` — save edits
  - `POST /sessions/{id}/qa/validate-mark` — bulk validate
  - `POST /sessions/{id}/start-training` — trigger Phase 2

## Polling
Every 5 seconds (background interval):
- Model server `GET /health` → `health` state
- Model server `GET /train/status` → `trainStatus` state
- Backend `GET /outputs` → output file list for diagnostic panel
- Adapter lists from both backend and model server

Every 3 seconds while session is not in a terminal state:
- Backend `GET /sessions/{id}` → refreshes `session` state

## Diagnostic Panel (`DiagnosticPanel` component)
Collapsible panel (toggle in header).

Sections:
- **Model server status** — loaded, adapter path, training flag
- **GPU** — device name, VRAM used/total with color-coded `GaugeBar`
- **Session** — ID, state (human-readable label), token budget `GaugeBar`, started time, current adapter, system prompts
- **Pipeline steps** (`PipelineStep`) — maps session state to done/active/pending icons for 6 stages
- **Training progress** — status, run ID, progress text, elapsed time, VRAM, "Restart Training" button on failure
- **Quick links** — model health, train status, API health, training runs
- **Output files** — lists up to 30 files from `outputs/` directory

## Sub-Components
| Component | Description |
|-----------|-------------|
| `GaugeBar` | Color-coded progress bar for numeric ranges |
| `StatRow` | Key/value display row |
| `SectionHeader` | Section label with optional divider |
| `PipelineStep` | Step icon (done/active/pending) + label |
| `DiagnosticPanel` | Full collapsible diagnostic sidebar |

## Adapter Selection
- Adapter list fetched from `GET /adapters` (backend) and model server `GET /adapters`
- Merged and deduplicated in state
- Selected adapter passed to `POST /sessions` as `adapter_id` on new session creation
- Backend calls `POST /load_adapter` on the model server if `adapter_id` is provided

## Configuration
| Env Var | Description |
|---------|-------------|
| `NEXT_PUBLIC_API_URL` | Backend API base URL (default `http://localhost:8000`) |

## Change Log
<!-- Agents: append an entry here after every change -->
| Date | Change | Author |
|------|--------|--------|
| 2026-04-28 | Initial documentation created | opencode |
