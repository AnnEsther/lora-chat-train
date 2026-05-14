'use client';

import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';

export interface Message {
  role: 'user' | 'glyph';
  text: string;
}

interface MessageBubbleProps {
  message: Message;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';

  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        mb: 1.5,
      }}
    >
      <Paper
        elevation={0}
        sx={{
          px: 2,
          py: 1.2,
          maxWidth: '75%',
          borderRadius: isUser ? '18px 18px 4px 18px' : '18px 18px 18px 4px',

          // Strip MUI's elevation overlay (--Paper-overlay) that washes over the gradient
          backgroundImage: 'none',

          // Fade in: opacity + slight rise + glow blooms then fades away
          '@keyframes bubbleFadeIn': {
            '0%': {
              opacity: 0,
              transform: 'translateY(6px) scale(0.96)',
              filter: isUser
                ? 'drop-shadow(0 0 0px transparent)'
                : 'drop-shadow(0 0 0px transparent)',
            },
            '50%': {
              filter: isUser
                ? 'drop-shadow(0 0 14px rgba(124,106,247,0.7))'
                : 'drop-shadow(0 0 14px rgba(130,58,136,0.7))',
            },
            '100%': {
              opacity: 1,
              transform: 'translateY(0) scale(1)',
              filter: 'drop-shadow(0 0 0px transparent)',
            },
          },
          animation: 'bubbleFadeIn 0.4s ease-out forwards',

          // Both bubbles — gradient background + mystical glow, white text
          ...(isUser
            ? {
                background: 'linear-gradient(135deg, #6d28d9 0%, #7c3aed 45%, #a855f7 80%, #c084fc 100%)',
                boxShadow: '0 0 12px rgba(139,92,246,0.6), 0 0 28px rgba(139,92,246,0.25)',
                border: '1px solid rgba(196,132,252,0.35)',
              }
            : {
                // Glyph bubble — purple-indigo-cyan gradient + layered mystical glow
                background: 'linear-gradient(135deg, #4a1a7a 0%, #3730a3 50%, #2935dc 100%)',
                boxShadow: [
                  '0 0 10px rgba(192,132,252,0.5)',
                  '0 0 24px rgba(99,102,241,0.3)',
                  '0 0 48px rgba(56,189,248,0.15)',
                ].join(', '),
                border: '1px solid rgba(192,132,252,0.3)',
              }),
        }}
      >
        <Typography variant="body2" sx={{ color: '#fff', fontWeight: 400 }}>
          {message.text}
        </Typography>
      </Paper>
    </Box>
  );
}
