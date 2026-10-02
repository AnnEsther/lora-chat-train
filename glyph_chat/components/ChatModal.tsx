'use client';

import { useState, useRef, useEffect, KeyboardEvent } from 'react';
import Dialog from '@mui/material/Dialog';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import TextField from '@mui/material/TextField';
import IconButton from '@mui/material/IconButton';
import Box from '@mui/material/Box';
import SendIcon from '@mui/icons-material/Send';
import CloseIcon from '@mui/icons-material/Close';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import MessageBubble, { Message } from './MessageBubble';
import { sendMessage } from '@/lib/chat';

interface ChatModalProps {
  open: boolean;
  onClose: () => void;
  adapterId: string;
}

const GREETING: Message = { role: 'glyph', text: 'Hello. I am Glyph. What would you like to know?' };

// The conversation survives closing the dialog and reloading the page. The
// server saves every message; this only restores what the player sees.
const STORAGE_KEY = 'glyph_conversation';

interface StoredConversation {
  conversationId: string | null;
  messages: Message[];
}

function loadConversation(): StoredConversation | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as StoredConversation) : null;
  } catch {
    return null;
  }
}

function storeConversation(conversation: StoredConversation | null): void {
  try {
    if (conversation) localStorage.setItem(STORAGE_KEY, JSON.stringify(conversation));
    else localStorage.removeItem(STORAGE_KEY);
  } catch {}
}

export default function ChatModal({ open, onClose, adapterId }: ChatModalProps) {
  const [messages, setMessages] = useState<Message[]>([GREETING]);
  const [input, setInput]     = useState('');
  const [streaming, setStreaming] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Restore the previous conversation once on mount (localStorage is client-only)
  useEffect(() => {
    const saved = loadConversation();
    if (saved?.messages?.length) {
      setMessages(saved.messages);
      setConversationId(saved.conversationId);
    }
  }, []);

  // Persist after each completed exchange (not on every streamed token)
  useEffect(() => {
    if (!streaming && messages.length > 1) storeConversation({ conversationId, messages });
  }, [streaming, messages, conversationId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleNewConversation = () => {
    if (streaming) return;
    storeConversation(null);
    setConversationId(null);
    setMessages([GREETING]);
  };

  const handleSend = async () => {
    const text = input.trim();
    if (!text || streaming) return;

    setInput('');

    // Append user message
    setMessages((prev) => [...prev, { role: 'user', text }]);

    // Append empty glyph bubble immediately — will grow as tokens arrive
    setMessages((prev) => [...prev, { role: 'glyph', text: '' }]);
    setStreaming(true);

    try {
      await sendMessage(
        text,
        adapterId,
        conversationId,
        (token) => {
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            updated[updated.length - 1] = { ...last, text: last.text + token };
            return updated;
          });
        },
        setConversationId,
      );
    } catch (err) {
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: 'glyph',
          text: 'Something went wrong. Please try again.',
        };
        return updated;
      });
    } finally {
      setStreaming(false);
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="sm"
      fullWidth
      slotProps={{
        paper: {
          sx: {
            backgroundColor: 'rgba(10, 10, 10, 0.72)',
            backdropFilter: 'blur(14px)',
            WebkitBackdropFilter: 'blur(14px)',
            border: '1px solid rgba(130,58,136,0.35)',
            color: 'text.primary',
            borderRadius: 3,
            height: '70vh',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            position: 'relative',

            '@keyframes glyphFadeIn': {
              '0%': {
                opacity: 0,
                filter: 'drop-shadow(0 0 0px transparent)',
                transform: 'scale(0.97)',
              },
              '100%': {
                opacity: 1,
                filter: 'drop-shadow(0 0 28px rgba(130,58,136,0.75))',
                transform: 'scale(1)',
              },
            },

            '@keyframes glyphGlowPulse': {
              '0%':   { boxShadow: '0 0 18px rgba(130,58,136,0.45), 0 0 48px rgba(99,58,136,0.2)' },
              '50%':  { boxShadow: '0 0 32px rgba(160,80,180,0.75), 0 0 72px rgba(120,80,200,0.35)' },
              '100%': { boxShadow: '0 0 18px rgba(130,58,136,0.45), 0 0 48px rgba(99,58,136,0.2)' },
            },

            animation: 'glyphFadeIn 0.6s ease-out forwards, glyphGlowPulse 3s ease-in-out 0.6s infinite',
          },
        },
        backdrop: {
          sx: { backgroundColor: 'transparent' },
        },
      }}
    >
      {/* Close button */}
      <IconButton
        onClick={onClose}
        size="small"
        aria-label="close"
        sx={{
          position: 'absolute',
          top: 8,
          right: 8,
          zIndex: 10,
          color: 'rgba(255,255,255,0.5)',
          '&:hover': { color: 'rgba(255,255,255,0.9)' },
        }}
      >
        <CloseIcon fontSize="small" />
      </IconButton>

      {/* New conversation */}
      <IconButton
        onClick={handleNewConversation}
        disabled={streaming || messages.length <= 1}
        size="small"
        aria-label="new conversation"
        title="New conversation"
        sx={{
          position: 'absolute',
          top: 8,
          right: 40,
          zIndex: 10,
          color: 'rgba(255,255,255,0.5)',
          '&:hover': { color: 'rgba(255,255,255,0.9)' },
        }}
      >
        <RestartAltIcon fontSize="small" />
      </IconButton>

      {/* Message list */}
      <DialogContent
        sx={{
          flex: 1,
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          px: 2,
          pt: 4,
          pb: 1,
          backgroundColor: 'transparent',
          '&::-webkit-scrollbar': { width: '4px' },
          '&::-webkit-scrollbar-track': { background: 'transparent' },
          '&::-webkit-scrollbar-thumb': { background: 'rgba(192,132,252,0.2)', borderRadius: '2px' },
        }}
      >
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
          {messages.map((msg, i) => (
            <MessageBubble key={i} message={msg} />
          ))}
          <div ref={bottomRef} />
        </Box>
      </DialogContent>

      <DialogActions sx={{ px: 2, py: 1.5, gap: 1, backgroundColor: 'transparent' }}>
        <TextField
          fullWidth
          size="small"
          placeholder="Type a message…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={streaming}
          multiline
          maxRows={3}
          sx={{
            '& .MuiOutlinedInput-root': { borderRadius: 3 },
          }}
        />
        <IconButton
          onClick={handleSend}
          disabled={streaming || !input.trim()}
          color="primary"
          aria-label="send"
        >
          <SendIcon />
        </IconButton>
      </DialogActions>
    </Dialog>
  );
}
