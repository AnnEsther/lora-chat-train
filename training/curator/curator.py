"""training/curator/curator.py

Refactored from the original EndToEndLoRA dataquality.py.
Scores each candidate turn pair and filters to a curated training set.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ── Scoring weights ───────────────────────────────────────────────────────────
# Scores are in [0, 1]. Final score is a weighted sum clamped to [0, 1].
_WEIGHTS = {
    "length": 0.25,
    "specificity": 0.25,
    "coherence": 0.25,
    "safety": 0.25,
}

INCLUSION_THRESHOLD = 0.5   # minimum score to include in training

# Patterns that indicate low quality or unsafe content
_UNSAFE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\[REDACTED", re.I),          # redacted fields — don't train on these
    re.compile(r"(i can'?t|i am unable|i won'?t) (help|assist|do)", re.I),  # refusals
    re.compile(r"(error|exception|traceback)", re.I),                         # error messages as answers
]


@dataclass
class ScoredCandidate:
    user_turn: str
    assistant_turn: str
    score: float
    included: bool
    rejection_reason: str | None
    _id: Any = None                  # DB row id, passed through transparently


class Curator:
    """
    Score and filter training candidates.

    Originally dataquality.py ran basic checks; this version adds a multi-factor
    scorer and persists scores back to the DB via the task layer.
    """

    def score_and_filter(self, candidates: list[dict]) -> list[dict]:
        """
        Parameters
        ----------
        candidates : list of dicts with keys user_turn, assistant_turn, _id

        Returns
        -------
        list of dicts with added keys: score, included, rejection_reason
        """
        results = []
        for cand in candidates:
            scored = self._score(cand)
            results.append({
                **cand,
                "score": scored.score,
                "included": scored.included,
                "rejection_reason": scored.rejection_reason,
            })

        included_count = sum(1 for r in results if r["included"])
        logger.info("curation_complete", extra={
            "total": len(results),
            "included": included_count,
        })
        return results

    def _score(self, cand: dict) -> ScoredCandidate:
        user = cand["user_turn"]
        asst = cand["assistant_turn"]

        # Safety check — hard exclude
        for pattern in _UNSAFE_PATTERNS:
            if pattern.search(asst):
                return ScoredCandidate(
                    user_turn=user, assistant_turn=asst,
                    score=0.0, included=False,
                    rejection_reason=f"unsafe_pattern:{pattern.pattern[:30]}",
                    _id=cand.get("_id"),
                )

        scores = {
            "length": self._score_length(asst),
            "specificity": self._score_specificity(user, asst),
            "coherence": self._score_coherence(user, asst),
            "safety": 1.0,          # passed hard check above
        }

        final = sum(_WEIGHTS[k] * v for k, v in scores.items())
        included = final >= INCLUSION_THRESHOLD

        return ScoredCandidate(
            user_turn=user, assistant_turn=asst,
            score=round(final, 4),
            included=included,
            rejection_reason=None if included else f"score_below_threshold:{final:.3f}",
            _id=cand.get("_id"),
        )

    def _score_length(self, asst: str) -> float:
        """Prefer responses in the 80–600 char range."""
        n = len(asst)
        if n < 40:
            return 0.1
        if n < 80:
            return 0.5
        if n <= 600:
            return 1.0
        if n <= 1500:
            return 0.8
        return 0.5

    def _score_specificity(self, user: str, asst: str) -> float:
        """Reward answers that contain numbers, code, or named entities."""
        specificity = 0.0
        if re.search(r"\d", asst):
            specificity += 0.4
        if re.search(r"```|`[^`]+`", asst):          # code blocks
            specificity += 0.4
        if len(asst.split()) > 15:
            specificity += 0.2
        return min(specificity, 1.0)

    def _score_coherence(self, user: str, asst: str) -> float:
        """Penalise answers that appear unrelated to the question."""
        user_words = set(re.findall(r"\b\w{4,}\b", user.lower()))
        asst_words = set(re.findall(r"\b\w{4,}\b", asst.lower()))
        if not user_words:
            return 0.5
        overlap = len(user_words & asst_words) / len(user_words)
        # At least some topical overlap expected
        return min(0.3 + overlap * 0.7, 1.0)
