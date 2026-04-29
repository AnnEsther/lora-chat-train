"""training/knowledge/corpus.py — Merge validated Q&A into knowledge corpus.

Manages the knowledge corpus - combining knowledge from multiple sessions
into a unified knowledge base.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CorpusEntry:
    topic: str
    facts: list[dict[str, Any]]
    source_session_id: Optional[str] = None


class CorpusManager:
    """
    Manage the knowledge corpus.

    Handles merging knowledge from multiple sessions into a unified corpus.
    """

    def merge(
        self,
        session_id: str,
        knowledge_records: list[dict[str, Any]],
        synthesized_qa: list[dict[str, Any]],
    ) -> list[CorpusEntry]:
        """
        Merge session knowledge into the corpus.

        Args:
            session_id: The session ID
            knowledge_records: Extracted knowledge records
            synthesized_qa: Validated Q&A pairs

        Returns:
            List of corpus entries added
        """
        entries = []

        # Group by topic
        topic_groups: dict[str, list[dict[str, Any]]] = {}

        for record in knowledge_records:
            topic = record.get("topic", "general")
            if topic not in topic_groups:
                topic_groups[topic] = []
            topic_groups[topic].extend(record.get("facts", []))

        # Create corpus entries
        for topic, facts in topic_groups.items():
            # Deduplicate facts
            unique_facts = self._deduplicate_facts(facts)

            entries.append(
                CorpusEntry(
                    topic=topic,
                    facts=unique_facts,
                    source_session_id=session_id,
                )
            )

        # Add Q&A pairs as facts
        for qa in synthesized_qa:
            topic = "general"
            # Try to infer topic from Q&A
            q_words = set(qa.get("question", "").lower().split())

            for known_topic in topic_groups:
                if known_topic in q_words:
                    topic = known_topic
                    break

            # Add Q&A as a fact
            for entry in entries:
                if entry.topic == topic:
                    entry.facts.append(
                        {
                            "type": "qa_pair",
                            "question": qa.get("question", ""),
                            "answer": qa.get("answer", ""),
                        }
                    )

        logger.info(
            "corpus_merged",
            extra={
                "session_id": session_id,
                "entries": len(entries),
            },
        )

        return entries

    def _deduplicate_facts(self, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate facts."""
        seen = set()
        unique = []

        for fact in facts:
            # Create a hashable key
            content = fact.get("content", "")
            fact_type = fact.get("type", "")
            key = (fact_type, content[:100])  # Use first 100 chars for comparison

            if key not in seen:
                seen.add(key)
                unique.append(fact)

        return unique

    def search(self, query: str, corpus: list[CorpusEntry]) -> list[CorpusEntry]:
        """
        Search the corpus for relevant entries.

        Args:
            query: Search query
            corpus: List of corpus entries

        Returns:
            Matching entries
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        results = []

        for entry in corpus:
            # Check topic match
            if query_lower in entry.topic.lower():
                results.append(entry)
                continue

            # Check facts match
            for fact in entry.facts:
                content = fact.get("content", "").lower()
                if any(word in content for word in query_words):
                    results.append(entry)
                    break

        return results

    def get_stats(self, corpus: list[CorpusEntry]) -> dict[str, Any]:
        """Get corpus statistics."""
        topic_counts: dict[str, int] = {}
        total_facts = 0

        for entry in corpus:
            topic_counts[entry.topic] = topic_counts.get(entry.topic, 0) + len(
                entry.facts
            )
            total_facts += len(entry.facts)

        return {
            "total_entries": len(corpus),
            "total_facts": total_facts,
            "topics": topic_counts,
        }
