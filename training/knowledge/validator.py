"""training/knowledge/validator.py — Validate synthesized Q&A pairs.

Automated validation of synthesized Q&A pairs with retry logic.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
VALIDATION_THRESHOLD = 0.6


@dataclass
class ValidationResult:
    valid: bool
    score: float
    notes: str


class QAValidator:
    """
    Validate synthesized Q&A pairs.

    Performs automated validation with retry logic:
    - Auto-retry up to 3 times on failure
    - After 3 failures, mark for manual review
    """

    # Validation criteria weights
    WEIGHTS = {
        "relevance": 0.3,  # Question matches answer topic
        "grammar": 0.2,  # Grammar correctness
        "completeness": 0.25,  # Answer is complete
        "accuracy": 0.25,  # Answer is accurate
    }

    def validate(self, question: str, answer: str) -> ValidationResult:
        """
        Validate a Q&A pair.

        Returns ValidationResult with valid status, score, and notes.
        """
        scores = {
            "relevance": self._score_relevance(question, answer),
            "grammar": self._score_grammar(question, answer),
            "completeness": self._score_completeness(answer),
            "accuracy": self._score_accuracy(answer),
        }

        final_score = sum(self.WEIGHTS[k] * v for k, v in scores.items())
        valid = final_score >= VALIDATION_THRESHOLD

        notes = self._generate_notes(scores)

        logger.debug(
            "validation_complete",
            extra={
                "score": final_score,
                "valid": valid,
                "question": question[:50],
            },
        )

        return ValidationResult(
            valid=valid,
            score=round(final_score, 3),
            notes=notes,
        )

    def _score_relevance(self, question: str, answer: str) -> float:
        """Score how relevant the answer is to the question."""
        q_words = set(re.findall(r"\b\w{4,}\b", question.lower()))
        a_words = set(re.findall(r"\b\w{4,}\b", answer.lower()))

        if not q_words:
            return 0.5

        overlap = len(q_words & a_words) / len(q_words)

        # Penalize if answer is completely unrelated
        if overlap < 0.1:
            return 0.1

        return min(0.3 + overlap * 0.7, 1.0)

    def _score_grammar(self, question: str, answer: str) -> float:
        """Score grammar correctness."""
        # Basic checks
        issues = 0

        # Check for excessive capitalization
        if sum(1 for c in answer if c.isupper()) / len(answer) > 0.3:
            issues += 1

        # Check for obvious grammar issues
        if ".." in answer:
            issues += 1
        if answer.strip() != answer:
            issues += 1

        # Check for incomplete sentences
        sentences = re.split(r"[.!?]", answer)
        for s in sentences:
            s = s.strip()
            if s and len(s.split()) < 2:
                issues += 1

        return max(0.0, 1.0 - (issues * 0.2))

    def _score_completeness(self, answer: str) -> float:
        """Score answer completeness."""
        word_count = len(answer.split())
        char_count = len(answer)

        # Too short
        if word_count < 5:
            return 0.2

        # Good length
        if 20 <= word_count <= 200:
            return 1.0

        # Reasonably detailed
        if 10 <= word_count < 20:
            return 0.7

        # Overly long (might be ranting)
        if word_count > 200:
            return 0.6

        return 0.5

    def _score_accuracy(self, answer: str) -> float:
        """Score answer accuracy (basic heuristics)."""
        # Check for refusal patterns
        refusal_patterns = [
            r"i (can't|cannot|am not able)",
            r"i don't have",
            r"as an ai",
            r"i'm sorry",
        ]

        for pattern in refusal_patterns:
            if re.search(pattern, answer.lower()):
                return 0.2

        # Check for uncertainty markers
        uncertainty = len(
            re.findall(
                r"\b(maybe|perhaps|i think|i believe|not sure)\b", answer.lower()
            )
        )
        if uncertainty > 2:
            return 0.5

        # Check for information density
        facts = len(
            re.findall(r"\b(is|are|was|were|can|will|has|have)\b", answer.lower())
        )
        if facts >= 3:
            return 0.9
        elif facts >= 1:
            return 0.7

        return 0.5

    def _generate_notes(self, scores: dict[str, float]) -> str:
        """Generate human-readable notes from scores."""
        issues = []

        if scores["relevance"] < 0.5:
            issues.append("low relevance")
        if scores["grammar"] < 0.5:
            issues.append("grammar issues")
        if scores["completeness"] < 0.5:
            issues.append("incomplete")
        if scores["accuracy"] < 0.5:
            issues.append("accuracy concerns")

        if not issues:
            return "Passed validation"

        return f"Issues: {', '.join(issues)}"

    def should_retry(
        self, validation_result: ValidationResult, retry_count: int
    ) -> bool:
        """Determine if validation should be retried."""
        if retry_count >= MAX_RETRIES:
            return False

        if not validation_result.valid:
            return True

        return False
