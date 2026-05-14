# Chat API

## Purpose
Client layer (`lib/chat.ts`) that communicates with the Glyph backend. Fetches available model adapters and streams chat responses token-by-token via Server-Sent Events.

## Usage / API

### `fetchAdapters`
```ts
import { fetchAdapters } from '@/lib/chat';

const adapters = await fetchAdapters();
// => Adapter[]
```

### `sendMessage`
```ts
import { sendMessage } from '@/lib/chat';

await sendMessage(
  text,       // user's message string
  adapterId,  // e.g. "current", "base", or a version ID
  history,    // HistoryItem[] — prior conversation turns
  (token) => { /* append token to glyph bubble */ }
);
```

### Types (`lib/types.ts`)
```ts
interface Adapter {
  id: string;
  version: string;
  is_base?: boolean;
  is_current?: boolean;
  trained_at?: string;
}

interface HistoryItem {
  role: 'user' | 'assistant';  // note: "assistant" not "glyph"
  content: string;              // note: "content" not "text"
}
```

## Backend contract

**Base URL:** `NEXT_PUBLIC_API_URL` (= `https://train.anratelier.com/api`)  
**Auth:** `X-Api-Key: NEXT_PUBLIC_API_KEY` header on every request

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/adapters/public` | GET | List available adapters |
| `/chat/direct` | POST | Stream a chat response |

### `/chat/direct` request body
```json
{
  "message": "user text",
  "adapter_id": "current",
  "history": [{ "role": "user", "content": "..." }, { "role": "assistant", "content": "..." }]
}
```

### `/chat/direct` SSE stream events
```
data: { "type": "chunk", "text": "token" }   → append to bubble
data: { "type": "end" }                        → stream complete
data: { "type": "error", "message": "..." }    → throw error
```

## Environment variables

Set in `.env` (gitignored). Embedded at build time — visible in the JS bundle.

```
NEXT_PUBLIC_API_URL=https://train.anratelier.com/api
NEXT_PUBLIC_API_KEY=your-secret-key-here
```

## Files

- `lib/chat.ts` — `fetchAdapters()` + `sendMessage()`
- `lib/types.ts` — `Adapter` + `HistoryItem` types

## Changelog

| Date | Change |
|------|--------|
| 2026-05-13 | Initial stub implementation (mock echo, no real API call) |
| 2026-05-14 | Full implementation: fetchAdapters (GET /adapters/public) + streaming sendMessage (POST /chat/direct, SSE); history mapping glyph→assistant, text→content |
