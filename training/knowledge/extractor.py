"""training/knowledge/extractor.py — Extract topics/factors from conversation pairs.

Analyzes user/assistant turn pairs to identify:
- Main topics/subjects discussed
- Tasks being performed
- Key concepts or facts being explained
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ExtractedTopic:
    topic: str
    subtopics: list[str]
    keywords: list[str]
    intent: str  # question, explanation, task, etc.


class KnowledgeExtractor:
    """
    Extract topics and factors from conversation pairs.

    Uses pattern matching and keyword analysis to identify
    what topics/concepts are being discussed.
    """

    # Topic keywords mapping
    TOPIC_PATTERNS = {
        "programming": [
            "code",
            "function",
            "class",
            "debug",
            "algorithm",
            "api",
            "software",
        ],
        "math": ["calculate", "formula", "equation", "number", "solve", "math"],
        "science": [
            "experiment",
            "hypothesis",
            "theory",
            "physics",
            "chemistry",
            "biology",
        ],
        "history": ["year", "war", "century", "ancient", "modern", "timeline"],
        "language": ["grammar", "vocabulary", "translate", "sentence", "word"],
        "cooking": ["recipe", "ingredient", "cook", "bake", "kitchen"],
        "health": ["symptom", "treatment", "medicine", "doctor", "wellness"],
        "technology": ["computer", "device", "software", "internet", "digital"],
        "finance": ["invest", "stock", "money", "bank", "loan", "budget"],
        "general": [],  # fallback
    }

    # Intent patterns
    INTENT_PATTERNS = {
        "question": [
            r"\?",
            r"^what",
            r"^how",
            r"^why",
            r"^when",
            r"^where",
            r"^can (you|i)",
        ],
        "explanation": [
            r"^the ",
            r"^this is",
            r"^it is",
            r"^basically",
            r"^in other words",
        ],
        "task": [r"^write ", r"^create ", r"^build ", r"^make ", r"^implement"],
        "conversation": [],  # fallback
    }

    def extract(self, user_turn: str, assistant_turn: str) -> list[ExtractedTopic]:
        """
        Extract topics from a conversation pair.

        Returns list of topics found in the conversation.
        """
        combined = f"{user_turn} {assistant_turn}".lower()

        topics = []

        # Find main topics
        for topic_name, keywords in self.TOPIC_PATTERNS.items():
            if topic_name == "general":
                continue
            found_keywords = [kw for kw in keywords if kw in combined]
            if found_keywords:
                subtopics = self._extract_subtopics(combined, found_keywords)
                intent = self._extract_intent(user_turn)
                topics.append(
                    ExtractedTopic(
                        topic=topic_name,
                        subtopics=subtopics,
                        keywords=found_keywords,
                        intent=intent,
                    )
                )

        # If no specific topic found, use general
        if not topics:
            keywords = self._extract_general_keywords(combined)
            topics.append(
                ExtractedTopic(
                    topic="general",
                    subtopics=[],
                    keywords=keywords,
                    intent=self._extract_intent(user_turn),
                )
            )

        logger.debug("topics_extracted", extra={"count": len(topics)})
        return topics

    def _extract_subtopics(self, text: str, keywords: list[str]) -> list[str]:
        """Extract subtopics based on surrounding context."""
        subtopics = []
        for kw in keywords:
            # Look for phrases containing the keyword
            pattern = rf"\b\w+\s+\w+\s*{re.escape(kw)}\s*\w+"
            matches = re.findall(pattern, text)
            subtopics.extend([m.strip() for m in matches[:2]])
        return list(set(subtopics))[:3]

    def _extract_intent(self, user_turn: str) -> str:
        """Determine the intent of the user's message."""
        turn_lower = user_turn.lower()
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, turn_lower):
                    return intent
        return "conversation"

    def _extract_general_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords for general topic."""
        words = re.findall(r"\b[a-z]{4,}\b", text)
        # Filter common stopwords
        stopwords = {
            "this",
            "that",
            "with",
            "have",
            "from",
            "they",
            "been",
            "will",
            "would",
            "could",
            "should",
            "what",
            "when",
            "where",
            "which",
            "their",
        }
        keywords = [w for w in words if w not in stopwords]
        return list(set(keywords))[:5]
