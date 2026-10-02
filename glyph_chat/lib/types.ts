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

// Saved conversations (GET /glyph/api/conversations[/<id>])
export interface ConversationSummary {
  id: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  preview: string;              // first thing the player asked
  last_adapter_id: string | null;
}

export interface SavedMessage {
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
  adapter_id: string | null;
  error: string | null;         // set when the reply failed or was cut short
}

export interface SavedConversation {
  id: string;
  created_at: string;
  updated_at: string;
  messages: SavedMessage[];
}
