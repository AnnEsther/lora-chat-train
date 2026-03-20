"""training/extractor/transcript_extractor.py

Refactored from the original EndToEndLoRA syntheticdatageneration.py + preprocessing.py concepts.
Pulls adjacent user/assistant turn pairs from raw transcripts, redacts secrets, and
filters out turns that should not appear in training targets.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)

# ── Redaction patterns ────────────────────────────────────────────────────────

_REDACT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"sk-[A-Za-z0-9]{20,}", re.I), "[REDACTED_API_KEY]"),
    (re.compile(r"(password|passwd|secret|token)\s*[:=]\s*\S+", re.I), "[REDACTED_CREDENTIAL]"),
    (re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"), "[REDACTED_CARD]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:AWS|GCP|AZURE)_SECRET[_A-Z]*\s*[:=]\s*\S+", re.I), "[REDACTED_CLOUD_SECRET]"),
]

# Turns matching these patterns should be excluded from training targets
_EXCLUDE_PATTERNS: list[re.Pattern] = [
    re.compile(r"/sleep", re.I),          # command turns
    re.compile(r"^system:", re.I),        # system messages
]

# Minimum token threshold for a turn to be useful training signal
MIN_TURN_CHARS = 20
MAX_TURN_CHARS = 4000   # avoid overly long completions that pad the context wastefully


@dataclass
class Candidate:
    user_turn: str
    assistant_turn: str
    turn_index: int              # index of the user turn in the transcript

    def to_dict(self) -> dict:
        return asdict(self)


class TranscriptExtractor:
    """
    Extract adjacent (user, assistant) turn pairs from a raw session transcript.

    Original EndToEndLoRA repo used syntheticdatageneration.py to create samples and
    preprocessing.py to clean them. This class consolidates both into one typed module.
    """

    def extract(self, transcript: list[dict]) -> list[Candidate]:
        """
        Parameters
        ----------
        transcript : list of {"role": str, "content": str} dicts

        Returns
        -------
        list[Candidate] — only valid, clean user/assistant pairs
        """
        candidates: list[Candidate] = []

        for i, turn in enumerate(transcript):
            if turn["role"] != "user":
                continue

            # Look for the next assistant turn
            assistant_turn: Optional[dict] = None
            for j in range(i + 1, len(transcript)):
                if transcript[j]["role"] == "assistant":
                    assistant_turn = transcript[j]
                    break
                if transcript[j]["role"] == "user":
                    # Another user turn before any assistant — skip
                    break

            if assistant_turn is None:
                continue

            user_content = self._clean(turn["content"])
            asst_content = self._clean(assistant_turn["content"])

            if not self._is_valid(user_content, asst_content):
                logger.debug("candidate_skipped", extra={"turn_index": i})
                continue

            candidates.append(Candidate(
                user_turn=user_content,
                assistant_turn=asst_content,
                turn_index=i,
            ))

        logger.info("extraction_complete", extra={"candidates": len(candidates)})
        return candidates

    def _clean(self, text: str) -> str:
        """Apply redaction and whitespace normalisation."""
        for pattern, replacement in _REDACT_PATTERNS:
            text = pattern.sub(replacement, text)
        text = text.strip()
        return text

    def _is_valid(self, user_content: str, asst_content: str) -> bool:
        """Return False if this pair should not be included."""
        # Too short
        if len(user_content) < MIN_TURN_CHARS or len(asst_content) < MIN_TURN_CHARS:
            return False
        # Too long
        if len(asst_content) > MAX_TURN_CHARS:
            return False
        # Exclude patterns
        for pattern in _EXCLUDE_PATTERNS:
            if pattern.search(user_content) or pattern.search(asst_content):
                return False
        return True
