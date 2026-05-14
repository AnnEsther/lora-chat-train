# Adapter Selector

## Purpose
A persistent floating pill UI that sits top-center on the main page. Fetches the list of available model adapters from the backend on page load and lets the user choose which adapter Glyph uses for chat responses.

## Usage / API

```tsx
<AdapterSelector value={selectedAdapter} onChange={setSelectedAdapter} />
```

| Prop | Type | Description |
|------|------|-------------|
| `value` | `string` | Currently selected adapter ID |
| `onChange` | `(id: string) => void` | Called when user selects a different adapter |

- Fetches `GET /adapters/public` on mount via `fetchAdapters()` from `lib/chat.ts`.
- Defaults to the adapter with `is_current: true`, or the first adapter if none is marked current.
- Shows "Loading…" while fetching and "Unavailable" on error (silently falls back to the `"current"` string ID passed from `page.tsx`).
- Renders as a `position: fixed` pill (`top: 16px`, `left: 50%`) with dark frosted-glass styling and purple border to match the app aesthetic.
- Must be imported with `ssr: false` in `page.tsx` — fetches on mount, needs browser.

## Files

- `components/AdapterSelector.tsx` — component
- `lib/chat.ts` — `fetchAdapters()` used internally
- `lib/types.ts` — `Adapter` type

## Changelog

| Date | Change |
|------|--------|
| 2026-05-14 | Initial implementation |
