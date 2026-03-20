"""backend/token_counter.py

Fast token count approximation used to track session budget in real time.
Uses tiktoken (cl100k_base) when available, falls back to char/4 heuristic.

The approximation does not need to be exact — it just needs to detect
when we're approaching the budget threshold reliably.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        try:
            import tiktoken
            _encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.debug("tiktoken not available — using char/4 heuristic")
            _encoder = False
    return _encoder


def count(text: str) -> int:
    """Return an approximate token count for the given text."""
    enc = _get_encoder()
    if enc is False:
        # Heuristic: ~4 chars per token (works well for English prose)
        return max(1, len(text) // 4)
    try:
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def count_messages(messages: list[dict]) -> int:
    """Count tokens across a list of {"role", "content"} dicts.
    Adds ~4 tokens overhead per message for chat template framing."""
    total = 0
    for msg in messages:
        total += count(msg.get("content", "")) + 4
    return total
