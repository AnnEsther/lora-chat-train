"""training/knowledge/synthesizer.py — Synthesize canonical Q&A from knowledge records.

Uses the model server to generate high-quality question-answer pairs
from extracted knowledge records.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://localhost:8001")
MAX_RETRIES = 3


@dataclass
class SynthesizedQA:
    question: str
    answer: str
    source_fact: dict[str, Any]


class QASynthesizer:
    """
    Synthesize canonical Q&A pairs from knowledge records.

    Uses the local model server to generate multiple high-quality
    question-answer pairs from each knowledge record.
    """

    def synthesize(
        self,
        knowledge_records: list[dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> list[SynthesizedQA]:
        """
        Synthesize Q&A pairs from knowledge records.

        Args:
            knowledge_records: List of knowledge records with facts
            system_prompt: Optional system prompt to guide synthesis

        Returns:
            List of synthesized Q&A pairs
        """
        if not knowledge_records:
            logger.warning("No knowledge records to synthesize")
            return []

        all_qa_pairs = []

        for record in knowledge_records:
            facts = record.get("facts", [])
            topic = record.get("topic", "general")

            # Generate multiple Q&A per fact
            for fact in facts:
                qa_pairs = self._synthesize_from_fact(fact, topic, system_prompt)
                all_qa_pairs.extend(qa_pairs)

        logger.info("synthesis_complete", extra={"qa_pairs": len(all_qa_pairs)})
        return all_qa_pairs

    def _synthesize_from_fact(
        self,
        fact: dict[str, Any],
        topic: str,
        system_prompt: Optional[str] = None,
    ) -> list[SynthesizedQA]:
        """Generate Q&A from a single fact."""
        fact_content = fact.get("content", "")
        fact_type = fact.get("type", "fact")

        if not fact_content:
            return []

        # Build prompt for synthesis
        prompt = self._build_synthesis_prompt(fact, topic, system_prompt)

        try:
            response = self._call_model(prompt)
            qa_pairs = self._parse_response(response, fact)

            if not qa_pairs:
                # Fallback: use the fact as-is
                qa_pairs = [
                    SynthesizedQA(
                        question=f"Tell me about {topic}.",
                        answer=fact_content,
                        source_fact=fact,
                    )
                ]

            return qa_pairs
        except Exception as e:
            logger.warning(f"Synthesis failed: {e}")
            # Return fallback
            return [
                SynthesizedQA(
                    question=f"What do you know about {topic}?",
                    answer=fact_content,
                    source_fact=fact,
                )
            ]

    def _build_synthesis_prompt(
        self,
        fact: dict[str, Any],
        topic: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Build the prompt for Q&A synthesis."""
        fact_content = fact.get("content", "")

        base_prompt = f"""Based on the following information about {topic}, generate 2-3 diverse question-answer pairs that could be used for training a language model.

Information: {fact_content}

Generate question-answer pairs in the following JSON format:
[{{"question": "...", "answer": "..."}}, ...]

Make sure the questions are natural, varied, and the answers accurately reflect the information provided.
"""

        if system_prompt:
            base_prompt = f"{system_prompt}\n\n{base_prompt}"

        return base_prompt

    def _call_model(self, prompt: str, max_tokens: int = 500) -> str:
        """Call the model server to generate Q&A."""
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    f"{MODEL_SERVER_URL}/generate",
                    json={
                        "prompt": prompt,
                        "max_new_tokens": max_tokens,
                        "temperature": 0.7,
                    },
                    timeout=60,
                )
                if response.ok:
                    return response.json().get("text", "")
                else:
                    logger.warning(f"Model call failed: {response.status_code}")
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

        raise Exception("Model server unavailable")

    def _parse_response(
        self,
        response: str,
        source_fact: dict[str, Any],
    ) -> list[SynthesizedQA]:
        """Parse the model's response into Q&A pairs."""
        qa_pairs = []

        # Try to extract JSON from response
        try:
            # Find JSON array in response
            start = response.find("[")
            end = response.rfind("]") + 1

            if start != -1 and end > start:
                json_str = response[start:end]
                parsed = json.loads(json_str)

                for item in parsed:
                    if (
                        isinstance(item, dict)
                        and "question" in item
                        and "answer" in item
                    ):
                        qa_pairs.append(
                            SynthesizedQA(
                                question=item["question"],
                                answer=item["answer"],
                                source_fact=source_fact,
                            )
                        )
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: try to extract Q&A from text
        if not qa_pairs:
            lines = response.strip().split("\n")
            current_q = None
            current_a = None

            for line in lines:
                line = line.strip()
                if line.lower().startswith("q:") or line.lower().startswith(
                    "question:"
                ):
                    current_q = line.split(":", 1)[1].strip()
                elif line.lower().startswith("a:") or line.lower().startswith(
                    "answer:"
                ):
                    current_a = line.split(":", 1)[1].strip()
                    if current_q and current_a:
                        qa_pairs.append(
                            SynthesizedQA(
                                question=current_q,
                                answer=current_a,
                                source_fact=source_fact,
                            )
                        )
                        current_q = None
                        current_a = None

        return qa_pairs

    async def synthesize_async(
        self,
        knowledge_records: list[dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> list[SynthesizedQA]:
        """Async version of synthesize."""
        import asyncio

        return await asyncio.to_thread(
            self.synthesize, knowledge_records, system_prompt
        )
