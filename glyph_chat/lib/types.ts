export interface Adapter {
  id: string;
  version: string;
  is_base?: boolean;
  is_current?: boolean;
  trained_at?: string;
}

export interface HistoryItem {
  role: 'user' | 'assistant';
  content: string;
}
