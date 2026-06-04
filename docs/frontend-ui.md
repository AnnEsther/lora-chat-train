# Frontend UI

## Overview
A single-page Next.js 15 application (App Router, Client Component) that provides the complete user interface: chat, session management, inline QA review deck, adapter selection, and a live diagnostic panel.

## Key Files
- `frontend/app/page.tsx` — Entire UI (~1360 lines); all state, sub-components, and logic in one file
- `frontend/app/components/HelpPanel.tsx` — Floating "How it works" help overlay
- `frontend/app/layout.tsx` — Root HTML shell
- `frontend/next.config.js` — Next.js configuration
- `frontend/tailwind.config.ts` — Tailwind CSS configuration

## State Variables
| State | Type | Description |
|-------|------|-------------|
| `sessions` | `Session[]` | All recent sessions (for switcher dropdown) |
| `session` | `Session \| null` | Currently active session |
| `messages` | `Message[]` | Chat message array; user messages carry `qaPairs`, `synthLoading`, `segmentCount` |
| `input` | `string` | Current textarea value |
| `loading` | `boolean` | True while a message stream is in progress |
| `error` | `string \| null` | Global error message |
| `health` | `ModelHealth \| null` | Model server health snapshot |
| `trainStatus` | `TrainStatus \| null` | Training progress from model server |
| `outputFiles` | `OutputFile[]` | Files from `GET /outputs` for the diagnostic panel |
| `adapters` | `Adapter[]` | Available adapters |
| `selectedAdapter` | `string` | Adapter selected for the next new session |
| `trainingSystemPrompt` | `string` | Training dataset system prompt override |
| `systemPrompt` | `string` | Chat system prompt override |
| `lastPoll` | `Date \| null` | Timestamp of last background poll |
| `panelOpen` | `boolean` | Diagnostic panel visibility toggle |
| `startingTraining` | `boolean` | True while Start Training request is in flight |
| `qaCount` | `QACount \| null` | Session-wide validated/total QA counts |

## TypeScript Interfaces & Types
```typescript
interface QAPair {
  id: string;
  question: string;
  answer: string;
  validated: boolean;
  edited: boolean;
}

interface Message {
  role: "user" | "assistant" | "system";
  content: string;
  id?: string;              // turn UUID — used to scope QA mutations
  streaming?: boolean;
  synthLoading?: boolean;   // true while SSE stream is open for this passage
  segmentCount?: number;    // total expected QA pairs (from SSE start event)
  qaPairs?: QAPair[];       // incrementally populated as qa_pair events arrive
}

interface QACount {
  total_count: number;
  validated_count: number;
  min_required: number;
  ready_to_train: boolean;
}

interface Session {
  id: string;
  state: SessionState;
  total_tokens: number;
  max_tokens: number;
  created_at: string;
  system_prompt?: string | null;
  training_system_prompt?: string | null;
  failure_reason?: string | null;
}

type SessionState =
  | "ACTIVE" | "PRE_SLEEP_WARNING" | "INSUFFICIENT_DATA"
  | "VALIDATING" | "SLEEPING" | "TRAINING"
  | "EVALUATING" | "DEPLOYING" | "READY" | "FAILED";

interface TrainStatus {
  status: "idle" | "running" | "completed" | "failed";
  run_id?: string;
  progress: string;
  started_at: string | null;
  finished_at: string | null;
  vram_used_gb: number | null;
  vram_free_gb: number | null;
}

interface ModelHealth {
  status: string;
  model_loaded: boolean;
  adapter: string | null;
  training_active: boolean;
  gpu: { name: string; vram_total_gb: number; vram_used_gb: number } | null;
}

interface OutputFile { name: string; path: string; size: string; }

interface Adapter {
  id: string;
  version: string;
  path: string;
  trained_at: string | null;
  is_current?: boolean;
  is_base?: boolean;
}
```

## Session Management
- **On mount:** restores last session ID from `localStorage`; fetches session list; falls back to most recent non-`READY` session or creates a new one
- **New session modal:** user picks adapter + sets both system prompts before calling `POST /sessions`
- **Session switcher dropdown:** in header; switching sets `localStorage` and reloads

## Chat Interaction

### `sendMessage()`
1. Appends user message to `messages` state with `synthLoading: true`
2. Opens a `fetch` SSE stream to `POST /sessions/{id}/chat`
3. Parses each `data:` line by `type`:
   - `start` → initialises `qaPairs: []` and `segmentCount` on the last message
   - `qa_pair` → appends one `QAPair` to the last message's `qaPairs` array
   - `qa_count` → updates global `qaCount` state (affects Start Training button)
   - `end` → clears `synthLoading` flag on the last message
   - `sleeping` / `sleep_ack` / `validating` → inserts system message, updates session state
   - `error` → shows error banner, clears `synthLoading`
4. `/sleep` command: handled via a separate branch that reads `sleeping`/`validating` events

### Input Behaviour
- **Enter** submits; **Shift+Enter** inserts newline
- Textarea auto-resizes up to 192 px
- Input disabled when session state is not in `{ACTIVE, PRE_SLEEP_WARNING, INSUFFICIENT_DATA, FAILED}`

## Inline QA Deck (`InlineDeck` component)

QA pairs are **not** shown in a modal. They appear inline below each user bubble as a card deck component. There is one `InlineDeck` per message that has QA pairs.

### Layout
```
┌─────────────────────────────────────────────────────────────────┐
│  ● ● ○   1 / 3 generating…          [Validate all]  [←]  [→]  │  ← nav header
├─────────────────────────────────────────────────────────────────┤
│  Q  ┌─ question bubble (red-tinted if unvalidated) ──────────┐  │
│     └───────────────────────────────────────────────────────-┘  │
│  A  ┌─ answer bubble (same tint)  ──────────────────────────┐   │
│     └───────────────────────────────────────────────────────┘   │
│     [Mark validated]  [Edit]                              [✕]   │  ← action row
└─────────────────────────────────────────────────────────────────┘
```

### Navigation
- On-screen `←` / `→` buttons in the nav header
- Keyboard `ArrowLeft` / `ArrowRight` when the deck wrapper div has focus (click to focus)
- Dot indicators: **green** = validated, **red** = not validated, **pulsing grey** = still generating

### Validation behaviour
- **Mark validated**: calls `PUT /sessions/{id}/qa/{qaId}` then **auto-advances** to the next card
- **✓ Validated** (toggle off): stays on current card
- **Validate all**: fires `Promise.all` PUT for every unvalidated pair simultaneously, then returns to card 0; button hidden once all pairs are validated

### Edit mode
- Clicking **Edit** replaces Q and A bubbles with `<textarea>` inputs (blue border)
- **Save** calls `PUT /sessions/{id}/qa/{qaId}` with new question + answer
- Pressing `←`/`→` while editing: auto-saves and navigates
- **Cancel** restores original values

### Delete
- `✕` button always visible (subtle grey); click once shows **Confirm delete** button
- Confirming calls `DELETE /sessions/{id}/qa/{qaId}` and removes the pair from state

### Streaming state
- While `synthLoading` is true: dot indicators for pending cards pulse grey; a shimmer skeleton card fills the current slot if the pair hasn't arrived yet; a ping dot + "X / N generating…" counter shown in the nav header
- Before the `start` SSE event: "Analysing passage…" text shown below the user bubble

### API calls made by `InlineDeck`
| Method | Endpoint | When |
|--------|----------|------|
| `PUT` | `/sessions/{id}/qa/{qaId}` | Edit save, validate toggle, validate-all |
| `DELETE` | `/sessions/{id}/qa/{qaId}` | Delete confirm |

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
| `SectionHeader` | Section label with optional coloured dot |
| `PipelineStep` | Step icon (done/active/pending/failed) + label with spinner on active |
| `DiagnosticPanel` | Full collapsible diagnostic sidebar (288 px fixed width) |
| `InlineDeck` | Inline QA card deck rendered below each user message; see section above |
| `HelpPanel` | Floating `?` button → 6-step workflow guide overlay (`app/components/HelpPanel.tsx`) |

## Adapter Selection
- Adapter list polled every 5 s from model server `GET /adapters` (falls back to backend `GET /adapters`)
- Selected adapter stored in `selectedAdapter` state; passed to `POST /sessions` as `adapter_id` on new session creation
- Backend `POST /sessions` endpoint calls `/reload_adapter` on the model server with the adapter path

## Start Training Button
- Shown in the header when a session is active
- **Enabled** when `qaCount.ready_to_train === true` (i.e. `validated_count >= min_required`)
- Label shows live counter: `Start Training (3/10)`
- Calls `POST /sessions/{id}/start-training`
- Disabled with tooltip explaining how many more validated pairs are needed

## Configuration
| Env Var | Description |
|---------|-------------|
| `NEXT_PUBLIC_API_URL` | Backend API base URL (build arg; default `http://localhost:8000`) |
| `NEXT_PUBLIC_MODEL_SERVER_URL` | Model server base URL (build arg; default `http://localhost:8001`) — used directly by the frontend for health, train status, and adapter listing |

## Change Log
<!-- Agents: append an entry here after every change -->
| Date | Change | Author |
|------|--------|--------|
| 2026-05-20 | Replace inline QACard list with InlineDeck component: one card shown at a time with left/right slide animation (260ms translateX); dot indicators coloured green/red per validation state; validate auto-advances to next card; validate-all button fires Promise.all for all unvalidated pairs; improved button contrast (solid green for Mark validated, always-visible delete button); edit mode auto-saves on arrow-key navigation. Removed QADeck modal overlay and all deck overlay state. | opencode |
| 2026-05-18 | Major redesign: replace QA review modal with inline QACard components rendered below each user message; add Start Training button; add qaCount state and fetchQaCount(); add DiagnosticPanel training data section. | opencode |
| 2026-05-08 | Expand TypeScript interfaces; add NEXT_PUBLIC_MODEL_SERVER_URL to configuration | opencode |
| 2026-05-05 | Add retry logic to fetchQaItems | opencode |
| 2026-04-28 | Initial documentation created | opencode |
