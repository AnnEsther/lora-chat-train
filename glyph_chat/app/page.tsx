'use client';

import { useState, Suspense } from 'react';
import dynamic from 'next/dynamic';
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';

// GlyphScene — SSR disabled (Three.js uses browser APIs)
const GlyphScene = dynamic(() => import('@/components/GlyphScene'), {
  ssr: false,
  loading: () => (
    <Box
      sx={{
        width: '100vw',
        height: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        bgcolor: '#0a0a0a',
      }}
    >
      <CircularProgress />
    </Box>
  ),
});

// ChatModal — SSR disabled (MUI Dialog causes hydration mismatches)
const ChatModal = dynamic(() => import('@/components/ChatModal'), { ssr: false });

// AdapterSelector — SSR disabled (fetches on mount, needs browser)
const AdapterSelector = dynamic(() => import('@/components/AdapterSelector'), { ssr: false });

export default function Home() {
  const [chatOpen, setChatOpen]           = useState(false);
  const [selectedAdapter, setSelectedAdapter] = useState('current');

  return (
    <Box sx={{ width: '100vw', height: '100vh', overflow: 'hidden' }}>
      {/* Persistent adapter selector — always visible */}
      <AdapterSelector value={selectedAdapter} onChange={setSelectedAdapter} />

      <Suspense fallback={null}>
        <GlyphScene onSpun={() => setChatOpen(true)} chatOpen={chatOpen} />
      </Suspense>

      <ChatModal
        open={chatOpen}
        onClose={() => setChatOpen(false)}
        adapterId={selectedAdapter}
      />
    </Box>
  );
}
