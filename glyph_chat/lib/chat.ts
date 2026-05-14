import type { Adapter, HistoryItem } from './types';

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? '';
const API_KEY  = process.env.NEXT_PUBLIC_API_KEY  ?? '';

// ---------------------------------------------------------------------------
// Fetch available adapters
// ---------------------------------------------------------------------------
export async function fetchAdapters(): Promise<Adapter[]> {
  const resp = await fetch(`${BASE_URL}/adapters/public`, {
    headers: { 'X-Api-Key': API_KEY },
  });
  if (!resp.ok) throw new Error(`fetchAdapters: ${resp.status} ${resp.statusText}`);
  const data = await resp.json();
  return data.adapters as Adapter[];
}

// ---------------------------------------------------------------------------
// Send a message and stream the response token-by-token
// ---------------------------------------------------------------------------
export async function sendMessage(
  text: string,
  adapterId: string,
  history: HistoryItem[],
  onChunk: (token: string) => void,
): Promise<void> {
  const resp = await fetch(`${BASE_URL}/chat/direct`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Api-Key': API_KEY,
    },
    body: JSON.stringify({
      message: text,
      adapter_id: adapterId,
      history,
    }),
  });

  if (!resp.ok) throw new Error(`sendMessage: ${resp.status} ${resp.statusText}`);
  if (!resp.body) throw new Error('sendMessage: response body is null');

  const reader  = resp.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const raw = decoder.decode(value, { stream: true });
    for (const line of raw.split('\n')) {
      if (!line.startsWith('data: ')) continue;
      const payload = line.slice(6).trim();
      if (!payload) continue;

      const event = JSON.parse(payload);
      if (event.type === 'chunk') onChunk(event.text);
      if (event.type === 'end')   return;
      if (event.type === 'error') throw new Error(event.message);
    }
  }
}
