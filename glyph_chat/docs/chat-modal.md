# Chat Modal

## Purpose
A MUI Dialog overlay that opens when the user spins the Glyph model. Displays the conversation history and lets the user type and send messages. Glyph responses appear as left-aligned bubbles; user messages as right-aligned.

## Usage / API

```tsx
<ChatModal open={chatOpen} onClose={() => setChatOpen(false)} />
```

| Prop | Type | Description |
|------|------|-------------|
| `open` | `boolean` | Controls dialog visibility |
| `onClose` | `() => void` | Called when the user closes the dialog |

### Message shape

```ts
interface Message {
  role: 'user' | 'glyph';
  text: string;
}
```

- Calls `sendMessage(text)` from `lib/chat.ts` on submit.
- Shows a loading indicator (MUI `CircularProgress`) while awaiting a response.
- Input is disabled while a request is in-flight.
- The message list auto-scrolls to the latest message.
- Dialog paper: `rgba(10,10,10,0.72)` background + `backdropFilter: blur(14px)` frosted-glass effect + subtle border. No title bar.
- Dialog backdrop: transparent — the 3D canvas behind is never dimmed.
- Close button floats at `position: absolute, top: 8, right: 8` inside the paper.
- Message list uses `justifyContent: center` so messages sit vertically centered when the list is short.
- Glyph bubbles: near-transparent background (`rgba(255,255,255,0.04)`) + purple border; text uses CSS `background-clip: text` gradient (`#c084fc → #818cf8 → #38bdf8 → #a78bfa`) for a purple underwater effect.
- User bubbles: solid `rgba(124,106,247,0.85)` purple, white text.

## Files

- `components/ChatModal.tsx` — dialog shell, message list, input bar
- `components/MessageBubble.tsx` — single message bubble (role-aware styling)

## Changelog

| Date | Change |
|------|--------|
| 2026-05-13 | Initial implementation |
| 2026-05-13 | Semi-transparent frosted-glass dialog (rgba + backdropFilter); transparent backdrop so 3D scene stays visible; semi-transparent message bubbles |
| 2026-05-13 | Remove DialogTitle; floating close button; messages vertically centered; glyph text gets purple-to-cyan underwater gradient via background-clip; grey backgrounds removed |
| 2026-05-13 | Mystical bubble redesign: glyph bubble gets dark glass background + animated gradient border (::before mask trick) + layered purple/indigo/cyan box-shadow glow; user bubble gets gradient fill + soft purple glow |
| 2026-05-13 | Both bubbles: gradient background fill + white text; glyph=#4a1a7a→#3730a3→#0e7490→#6d28d9 + 3-layer glow; user=#6d28d9→#7c3aed→#a855f7→#c084fc + purple glow |
| 2026-05-13 | Fade-in + living glow: glyphFadeIn keyframe (opacity+scale+drop-shadow, 600ms) followed by glyphGlowPulse loop (box-shadow breathes, 3s ease-in-out infinite) |
| 2026-05-14 | Bubble fade-in: bubbleFadeIn keyframe (opacity+translateY+scale, glow peaks at 50% then fades to transparent, 0.4s ease-out) on every MessageBubble mount |
