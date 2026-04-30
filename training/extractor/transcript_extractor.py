"""training/extractor/transcript_extractor.py

Extracts overlapping conversation segments (sliding windows) from a raw session
transcript for use as training candidates.

Each candidate is a contiguous slice of N consecutive turns (EXTRACTION_WINDOW_SIZE)
rather than a single user/assistant pair.  This preserves multi-turn context and lets
the model learn from the full flow of a conversation, including exchanges where a single
message contains multiple questions or answers.

Env vars
--------
EXTRACTION_WINDOW_SIZE  Number of turns per segment window (default 4).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

EXTRACTION_WINDOW_SIZE: int = int(os.environ.get("EXTRACTION_WINDOW_SIZE", 4))

# ── Redaction patterns ────────────────────────────────────────────────────────

_REDACT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"sk-[A-Za-z0-9]{20,}", re.I), "[REDACTED_API_KEY]"),
    (
        re.compile(r"(password|passwd|secret|token)\s*[:=]\s*\S+", re.I),
        "[REDACTED_CREDENTIAL]",
    ),
    (re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"), "[REDACTED_CARD]"),
    (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    (
        re.compile(r"\b(?:AWS|GCP|AZURE)_SECRET[_A-Z]*\s*[:=]\s*\S+", re.I),
        "[REDACTED_CLOUD_SECRET]",
    ),
]

# Turns that should never appear in any training segment
_EXCLUDE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*/sleep\s*$", re.I),  # bare /sleep command
    re.compile(r"^system:", re.I),  # system message prefix
]

# Per-turn character limits
MIN_TURN_CHARS = 20
MAX_ASSISTANT_TURN_CHARS = 4000


@dataclass
class Candidate:
    """A sliding-window conversation segment ready for curation."""

    conversation: list[dict]  # list of {"role": str, "content": str}
    turn_index: int  # index of the first turn in the original transcript

    def to_dict(self) -> dict:
        return asdict(self)


class TranscriptExtractor:
    """
    Extract overlapping conversation segments from a raw session transcript.

    Instead of pulling isolated (user, assistant) pairs, this extractor uses a
    sliding window of EXTRACTION_WINDOW_SIZE turns so that multi-turn context —
    including messages with multiple questions or compound answers — is preserved
    as a single training sample.

    The window advances one turn at a time, producing overlapping segments.
    Each segment must:
      - Contain at least one user turn and one assistant turn.
      - Not contain any /sleep commands or system: prefixed turns.
      - Not contain any assistant turn longer than MAX_ASSISTANT_TURN_CHARS chars.
      - Not contain any turn shorter than MIN_TURN_CHARS chars.
    """

    def __init__(self, window_size: Optional[int] = None) -> None:
        self.window_size = (
            window_size if window_size is not None else EXTRACTION_WINDOW_SIZE
        )

    def extract(self, transcript: list[dict]) -> list[Candidate]:
        """
        Parameters
        ----------
        transcript : list of {"role": str, "content": str} dicts, in order.

        Returns
        -------
        list[Candidate] — overlapping window segments, each cleaned and validated.
        """
        # Pre-clean every turn in the transcript first
        cleaned: list[dict] = [
            {"role": t["role"], "content": self._clean(t["content"])}
            for t in transcript
        ]

        candidates: list[Candidate] = []
        n = len(cleaned)

        for i in range(n):
            window = cleaned[i : i + self.window_size]

            # Pad short windows at the end of the transcript — skip them
            if len(window) < 2:
                continue

            if not self._is_valid_segment(window):
                logger.debug("segment_skipped", extra={"turn_index": i})
                continue

            candidates.append(Candidate(conversation=window, turn_index=i))

        logger.info(
            "extraction_complete",
            extra={
                "candidates": len(candidates),
                "window_size": self.window_size,
                "transcript_turns": n,
            },
        )
        return candidates

    # ── Private helpers ───────────────────────────────────────────────────────

    def _clean(self, text: str) -> str:
        """Apply PII redaction and whitespace normalisation to a single turn."""
        for pattern, replacement in _REDACT_PATTERNS:
            text = pattern.sub(replacement, text)
        return text.strip()

    def _is_valid_segment(self, window: list[dict]) -> bool:
        """
        Return False if this segment should be excluded from training.

        Checks (in order):
        1. Must contain at least one user turn and one assistant turn.
        2. No turn may match the global exclude patterns (/sleep, system:).
        3. No turn may be shorter than MIN_TURN_CHARS.
        4. No assistant turn may exceed MAX_ASSISTANT_TURN_CHARS.
        """
        roles = {t["role"] for t in window}
        if "user" not in roles or "assistant" not in roles:
            return False

        for turn in window:
            content = turn["content"]

            # Exclude pattern check
            for pat in _EXCLUDE_PATTERNS:
                if pat.search(content):
                    return False

            # Too short
            if len(content) < MIN_TURN_CHARS:
                return False

            # Assistant turn too long
            if turn["role"] == "assistant" and len(content) > MAX_ASSISTANT_TURN_CHARS:
                return False

        return True
