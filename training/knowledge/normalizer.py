"""training/knowledge/normalizer.py — Convert extracted topics to structured knowledge records.

Takes extracted topics and normalizes them into structured facts
that can be used for Q&A synthesis.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from training.knowledge.extractor import ExtractedTopic

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeRecord:
    topic: str
    facts: list[dict[str, Any]]
    source_type: str  # 'explanation', 'task', 'question', 'conversation'


class KnowledgeNormalizer:
    """
    Normalize extracted topics into structured knowledge records.

    Converts raw conversation content into structured facts
    that can be used for synthesizing Q&A pairs.
    """

    def normalize(
        self,
        user_turn: str,
        assistant_turn: str,
        topics: list[ExtractedTopic],
    ) -> list[KnowledgeRecord]:
        """
        Normalize conversation into knowledge records.

        Args:
            user_turn: The user's message
            assistant_turn: The assistant's response
            topics: Extracted topics from the conversation

        Returns:
            List of knowledge records with structured facts
        """
        records = []

        for topic in topics:
            facts = self._extract_facts(user_turn, assistant_turn, topic)
            if facts:
                records.append(
                    KnowledgeRecord(
                        topic=topic.topic, facts=facts, source_type=topic.intent
                    )
                )

        # If no topics, create a general record
        if not records:
            facts = self._extract_general_facts(user_turn, assistant_turn)
            records.append(
                KnowledgeRecord(
                    topic="general", facts=facts, source_type="conversation"
                )
            )

        logger.debug("knowledge_records_created", extra={"count": len(records)})
        return records

    def _extract_facts( self, user_turn: str, assistant_turn: str, topic: ExtractedTopic, ) -> list[dict[str, Any]]:
        """
        Extract facts always anchored to:
        - user_turn  = the knowledge/answer (teacher explaining)
        - assistant_turn = the question that prompted it (student asking)
        """
        facts = []

        # Every exchange is a qa_pair: student asked, teacher answered
        if user_turn.strip() and assistant_turn.strip():
            facts.append({
                "type": "qa_pair",
                "question": assistant_turn.strip(),  # LLM asked this
                "answer": user_turn.strip(),          # User taught this
                "topic": topic.topic,
            })
            
        # Extract individual knowledge sentences from the USER's response
        # These become additional training examples (model asks about them)
        sentences = re.split(r"[.!?]\s+", user_turn)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 30:
                facts.append({
                    "type": "fact",
                    "content": sentence,          # user's knowledge
                    "student_question": assistant_turn.strip(),  # what the LLM asked
                    "topic": topic.topic,
                    "keywords": topic.keywords,
                })

        # Extract code blocks from the USER's turn (user may paste code to teach)
        code_blocks = re.findall(r"```[\s\S]*?```", user_turn)
        for code in code_blocks[:2]:
            facts.append({
                "type": "code_example",
                "content": code,
                "language": self._detect_language(code),
                "student_question": assistant_turn.strip(),
                "topic": topic.topic,
            })

        return facts[:6]


    def _extract_explanatory_facts( self, assistant_turn: str, topic: ExtractedTopic, ) -> list[dict[str, Any]]:
        """
        The assistant is the STUDENT asking questions.
        The user is the TEACHER providing knowledge.
        Facts come from the USER (teacher) turn — assistant_turn is just the question.
        This method receives assistant_turn for legacy signature compat but we don't use it here.
        """
        # This method is called when intent == "explanation" — meaning the user explained something.
        # The actual content is in user_turn, passed via _extract_facts below.
        return []  # handled in _extract_facts directly

    def _extract_qa_facts(
        self,
        user_turn: str,
        assistant_turn: str,
        topic: ExtractedTopic,
    ) -> list[dict[str, Any]]:
        """Extract facts from Q&A content."""
        facts = []

        # The question and answer themselves are facts
        facts.append(
            {
                "type": "qa_pair",
                "question": user_turn,
                "answer": assistant_turn,
                "topic": topic.topic,
            }
        )

        # Extract any additional facts mentioned in the answer
        sentences = re.split(r"[.!?]\s+", assistant_turn)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 30:
                facts.append(
                    {
                        "type": "fact",
                        "content": sentence,
                        "topic": topic.topic,
                    }
                )

        return facts[:5]

    def _extract_task_facts(
        self,
        user_turn: str,
        assistant_turn: str,
        topic: ExtractedTopic,
    ) -> list[dict[str, Any]]:
        """Extract facts from task/creation content."""
        facts = []

        # Code snippets or implementations
        code_blocks = re.findall(r"```[\s\S]*?```", assistant_turn)
        for i, code in enumerate(code_blocks[:3]):
            facts.append(
                {
                    "type": "code_example",
                    "content": code,
                    "language": self._detect_language(code),
                    "topic": topic.topic,
                }
            )

        # Steps or instructions
        steps = re.split(r"\n(?:\d+\.|\-|\*)\s*", assistant_turn)
        for step in steps[1:4]:  # Skip first if it's the code block
            step = step.strip()
            if len(step) > 20:
                facts.append(
                    {
                        "type": "step",
                        "content": step,
                        "topic": topic.topic,
                    }
                )

        # The task itself
        facts.append(
            {
                "type": "task",
                "request": user_turn,
                "topic": topic.topic,
            }
        )

        return facts[:5]

    def _extract_conversation_facts(
        self,
        user_turn: str,
        assistant_turn: str,
        topic: ExtractedTopic,
    ) -> list[dict[str, Any]]:
        """Extract general facts from conversation."""
        facts = []

        # Key sentences from assistant
        sentences = re.split(r"[.!?]\s+", assistant_turn)
        for sentence in sentences[:3]:
            sentence = sentence.strip()
            if len(sentence) > 30:
                facts.append(
                    {
                        "type": "fact",
                        "content": sentence,
                        "topic": topic.topic,
                        "keywords": topic.keywords,
                    }
                )

        return facts[:3]

    def _extract_general_facts( self, user_turn: str, assistant_turn: str, ) -> list[dict[str, Any]]:
        """Fallback: preserve the full exchange with correct roles."""
        if not user_turn.strip():
            return []
        return [{
            "type": "qa_pair",
            "question": assistant_turn.strip(),  # LLM asked
            "answer": user_turn.strip(),          # User answered/taught
            "topic": "general",
        }]

    def _detect_language(self, code_block: str) -> str:
        """Detect programming language from code block."""
        code = code_block.lower()
        if "def " in code or "import " in code or "class " in code:
            return "python"
        elif "function" in code or "const " in code or "let " in code:
            return "javascript"
        elif "public " in code or "private " in code:
            return "java"
        elif "select " in code or "from " in code:
            return "sql"
        return "unknown"
