'use client';

import { useEffect, useState } from 'react';
import Box from '@mui/material/Box';
import Select, { SelectChangeEvent } from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import Typography from '@mui/material/Typography';
import { fetchAdapters } from '@/lib/chat';
import type { Adapter } from '@/lib/types';

interface AdapterSelectorProps {
  value: string;
  onChange: (id: string) => void;
}

export default function AdapterSelector({ value, onChange }: AdapterSelectorProps) {
  const [adapters, setAdapters] = useState<Adapter[]>([]);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState(false);

  useEffect(() => {
    fetchAdapters()
      .then((list) => {
        setAdapters(list);
        // Default to the "current" adapter if not already set
        const hasCurrent = list.some((a) => a.id === value);
        if (!hasCurrent && list.length > 0) {
          const current = list.find((a) => a.is_current) ?? list[0];
          onChange(current.id);
        }
      })
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleChange = (e: SelectChangeEvent) => onChange(e.target.value);

  const label = (a: Adapter) => {
    if (a.is_current) return `${a.version} ✦ live`;
    if (a.is_base)    return `${a.version} · base`;
    return a.version;
  };

  return (
    <Box
      sx={{
        position: 'fixed',
        top: 16,
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 1300,
        display: 'flex',
        alignItems: 'center',
        gap: 1.5,
        px: 2,
        py: 0.75,
        borderRadius: '999px',
        backgroundColor: 'rgba(10,10,10,0.65)',
        backdropFilter: 'blur(12px)',
        WebkitBackdropFilter: 'blur(12px)',
        border: '1px solid rgba(130,58,136,0.35)',
        boxShadow: '0 0 14px rgba(130,58,136,0.25)',
      }}
    >
      <Typography
        variant="caption"
        sx={{
          color: 'rgba(255,255,255,0.45)',
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          fontSize: '0.65rem',
          userSelect: 'none',
        }}
      >
        Model
      </Typography>

      {loading && (
        <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.35)', fontSize: '0.75rem' }}>
          Loading…
        </Typography>
      )}

      {error && (
        <Typography variant="caption" sx={{ color: 'rgba(255,100,100,0.7)', fontSize: '0.75rem' }}>
          Unavailable
        </Typography>
      )}

      {!loading && !error && (
        <Select
          value={value}
          onChange={handleChange}
          size="small"
          variant="standard"
          disableUnderline
          sx={{
            color: '#c084fc',
            fontSize: '0.78rem',
            fontWeight: 500,
            minWidth: 120,
            '& .MuiSelect-icon': { color: 'rgba(192,132,252,0.6)' },
            '& .MuiSelect-select': { py: 0 },
          }}
          MenuProps={{
            slotProps: {
              paper: {
                sx: {
                  backgroundColor: 'rgba(15,5,30,0.95)',
                  backdropFilter: 'blur(12px)',
                  border: '1px solid rgba(130,58,136,0.4)',
                  boxShadow: '0 0 20px rgba(130,58,136,0.3)',
                  backgroundImage: 'none',
                },
              },
            },
          }}
        >
          {adapters.map((a) => (
            <MenuItem
              key={a.id}
              value={a.id}
              sx={{
                fontSize: '0.78rem',
                color: a.is_current ? '#c084fc' : 'rgba(255,255,255,0.75)',
                '&:hover':    { backgroundColor: 'rgba(130,58,136,0.2)' },
                '&.Mui-selected': { backgroundColor: 'rgba(130,58,136,0.3)' },
              }}
            >
              {label(a)}
            </MenuItem>
          ))}
        </Select>
      )}
    </Box>
  );
}
