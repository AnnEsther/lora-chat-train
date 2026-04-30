"""training/curator/curator.py

Scores and filters conversation segment candidates produced by TranscriptExtractor.

Each candidate is now a multi-turn conversation (list of {"role", "content"} dicts)
rather than an isolated (user_turn, assistant_turn) pair.  Scoring operates across
all turns in the segment.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ── Scoring weights ───────────────────────────────────────────────────────────
_WEIGHTS = {
    "length": 0.25,
    "specificity": 0.25,
    "coherence": 0.25,
    "safety": 0.25,
}

INCLUSION_THRESHOLD = 0.5  # minimum weighted score to include in training

# Patterns that flag low quality or unsafe content in any turn
_UNSAFE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\[REDACTED", re.I),
    re.compile(r"(i can'?t|i am unable|i won'?t) (help|assist|do)", re.I),
    re.compile(r"(error|exception|traceback)", re.I),
]


@dataclass
class ScoredCandidate:
    conversation: list[dict]
    score: float
    included: bool
    rejection_reason: str | None
    _id: Any = None  # DB row id, passed through transparently


class Curator:
    """
    Score and filter training candidates.

    Each candidate carries a `conversation` list (multi-turn segment).
    Scoring aggregates across all turns in the segment so that compound
    exchanges — where a single message contains multiple questions or answers —
    are evaluated holistically rather than penalised for not matching a strict
    one-question-one-answer shape.
    """

    def score_and_filter(self, candidates: list[dict]) -> list[dict]:
        """
        Parameters
        ----------
        candidates : list of dicts with keys ``conversation`` (list[dict]) and ``_id``.

        Returns
        -------
        list of dicts with added keys: score, included, rejection_reason.
        """
        results = []
        for cand in candidates:
            scored = self._score(cand)
            results.append(
                {
                    **cand,
                    "score": scored.score,
                    "included": scored.included,
                    "rejection_reason": scored.rejection_reason,
                }
            )

        included_count = sum(1 for r in results if r["included"])
        logger.info(
            "curation_complete",
            extra={
                "total": len(results),
                "included": included_count,
            },
        )
        return results

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score(self, cand: dict) -> ScoredCandidate:
        conversation: list[dict] = cand.get("conversation") or []

        # Collect all assistant and user text across the segment
        assistant_turns = [
            t["content"] for t in conversation if t.get("role") == "assistant"
        ]
        user_turns = [t["content"] for t in conversation if t.get("role") == "user"]
        all_text = " ".join(t["content"] for t in conversation)

        # Safety hard-check — any unsafe pattern in any turn → immediate exclusion
        for turn in conversation:
            for pattern in _UNSAFE_PATTERNS:
                if pattern.search(turn["content"]):
                    return ScoredCandidate(
                        conversation=conversation,
                        score=0.0,
                        included=False,
                        rejection_reason=f"unsafe_pattern:{pattern.pattern[:40]}",
                        _id=cand.get("_id"),
                    )

        scores = {
            "length": self._score_length(assistant_turns),
            "specificity": self._score_specificity(assistant_turns),
            "coherence": self._score_coherence(user_turns, assistant_turns),
            "safety": 1.0,  # passed hard check above
        }

        final = sum(_WEIGHTS[k] * v for k, v in scores.items())
        included = final >= INCLUSION_THRESHOLD

        return ScoredCandidate(
            conversation=conversation,
            score=round(final, 4),
            included=included,
            rejection_reason=None if included else f"score_below_threshold:{final:.3f}",
            _id=cand.get("_id"),
        )

    def _score_length(self, assistant_turns: list[str]) -> float:
        """Score based on total length of all assistant responses in the segment."""
        total = sum(len(t) for t in assistant_turns)
        if total < 40:
            return 0.1
        if total < 80:
            return 0.5
        if total <= 1200:  # wider upper bound for multi-turn segments
            return 1.0
        if total <= 4000:
            return 0.8
        return 0.5

    def _score_specificity(self, assistant_turns: list[str]) -> float:
        """Reward segments where assistant responses contain numbers, code, or detail."""
        combined = " ".join(assistant_turns)
        specificity = 0.0
        if re.search(r"\d", combined):
            specificity += 0.4
        if re.search(r"```|`[^`]+`", combined):
            specificity += 0.4
        if len(combined.split()) > 15:
            specificity += 0.2
        return min(specificity, 1.0)

    def _score_coherence(
        self, user_turns: list[str], assistant_turns: list[str]
    ) -> float:
        """
        Penalise segments where the assistant responses appear unrelated to the
        user turns.  Uses word-overlap across the full segment rather than
        per-exchange comparison so compound multi-topic exchanges are handled fairly.
        """
        user_words = set(re.findall(r"\b\w{4,}\b", " ".join(user_turns).lower()))
        asst_words = set(re.findall(r"\b\w{4,}\b", " ".join(assistant_turns).lower()))
        if not user_words:
            return 0.5
        overlap = len(user_words & asst_words) / len(user_words)
        return min(0.3 + overlap * 0.7, 1.0)
