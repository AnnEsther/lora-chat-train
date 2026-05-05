# training/knowledge/synthesizer.py

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://model_server:8001")
BATCH_SIZE = int(os.environ.get("QA_BATCH_SIZE", "20"))  # facts per model call
MAX_RETRIES = 2
CALL_TIMEOUT = int(os.environ.get("QA_SYNTHESIS_TIMEOUT", "60"))


@dataclass
class SynthesizedQA:
    question: str
    answer: str
    source_fact: dict[str, Any]


class QASynthesizer:

    def synthesize(
        self,
        knowledge_records: list[dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> list[SynthesizedQA]:
        if not knowledge_records:
            logger.warning("No knowledge records to synthesize")
            return []

        # Flatten all facts from all records into (fact, topic) pairs
        flat_facts: list[tuple[dict, str]] = []
        for record in knowledge_records:
            topic = record.get("topic", "general")
            for fact in record.get("facts", []):
                if fact.get("content") or fact.get("answer"):
                    flat_facts.append((fact, topic))

        logger.info("synthesis_start", extra={
            "facts": len(flat_facts),
            "batches": -(-len(flat_facts) // BATCH_SIZE),  # ceiling div
        })

        all_qa: list[SynthesizedQA] = []

        # Process in batches — each batch = one model call
        for i in range(0, len(flat_facts), BATCH_SIZE):
            batch = flat_facts[i: i + BATCH_SIZE]
            try:
                pairs = self._synthesize_batch(batch, system_prompt)
                all_qa.extend(pairs)
                logger.info("batch_done", extra={
                    "batch": i // BATCH_SIZE + 1,
                    "qa_pairs": len(pairs),
                })
            except Exception as e:
                logger.warning(f"Batch {i // BATCH_SIZE + 1} failed, using fallback: {e}")
                # Fallback for this batch only — don't lose the whole run
                all_qa.extend(self._fallback_batch(batch))

        logger.info("synthesis_complete", extra={"total_qa": len(all_qa)})
        return all_qa

    def _synthesize_batch( self, batch: list[tuple[dict, str]], system_prompt: Optional[str], ) -> list[SynthesizedQA]:
        """
        The user is the TEACHER. The LLM is the STUDENT.
        We want to generate training examples where:
        - question = what the student (LLM) should ask
        - answer   = what the teacher (user) would say
        """

        fact_lines = []
        for idx, (fact, topic) in enumerate(batch, 1):
            if fact.get("type") == "qa_pair":
                # Already have the real exchange — use it directly
                text = (
                    f"Student asked: {fact.get('question', '')}\n"
                    f"Teacher answered: {fact.get('answer', '')}"
                )
            elif fact.get("student_question"):
                # Fact from teacher's turn, we know what the student asked
                text = (
                    f"Student asked: {fact['student_question']}\n"
                    f"Teacher's knowledge: {fact.get('content', '')}"
                )
            else:
                # Pure knowledge fact from teacher
                text = f"Teacher's knowledge: {fact.get('content', '')}"

            fact_lines.append(f"{idx}. [{topic}]\n{text}")

        facts_block = "\n\n".join(fact_lines)

        prompt = f"""{system_prompt + chr(10) if system_prompt else ""}You are generating fine-tuning training data. 

Context: A student (AI model) is learning from a teacher (human user). The student asks questions and the teacher provides knowledge.

Your job: For each exchange below, generate ONE training example with:
- "question": a natural question the student would ask to draw out this knowledge
- "answer": the teacher's knowledge response (use the teacher's actual words where possible)

The goal is to train the model to ask good questions that help the teacher share their knowledge.

{facts_block}

Respond ONLY with a JSON array of exactly {len(batch)} objects:
[
  {{"question": "question student asks", "answer": "teacher knowledge response"}},
  ...
]

Rules:
- Questions must be specific to the actual content, not generic
- Answers must come from the teacher's knowledge, not be invented
- Questions should sound like a curious student asking to learn more
- No preamble, just the JSON array"""

        response_text = self._call_model(prompt, max_tokens=min(150 * len(batch), 3000))
        return self._parse_batch_response(response_text, batch)

    def _call_model(self, prompt: str, max_tokens: int = 1000) -> str:
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(
                    f"{MODEL_SERVER_URL}/generate",
                    json={
                        "prompt": prompt,
                        "max_new_tokens": max_tokens,
                        "temperature": 0.4,  # lower = more consistent JSON
                    },
                    timeout=CALL_TIMEOUT,
                )
                if resp.ok:
                    return resp.json().get("response", resp.json().get("text", ""))
                logger.warning(f"Model returned {resp.status_code}, attempt {attempt + 1}")
            except requests.RequestException as e:
                logger.warning(f"Model call attempt {attempt + 1} failed: {e}")

        raise Exception(f"Model server failed after {MAX_RETRIES} attempts")

    def _parse_batch_response(
        self,
        response: str,
        batch: list[tuple[dict, str]],
    ) -> list[SynthesizedQA]:
        """Parse the JSON array response, fall back per-item if needed."""
        # Strip markdown fences if model wrapped output
        clean = re.sub(r"```(?:json)?|```", "", response).strip()

        # Find the JSON array (model sometimes adds preamble text)
        array_match = re.search(r"\[.*\]", clean, re.DOTALL)
        if not array_match:
            logger.warning("No JSON array found in response, using fallback")
            return self._fallback_batch(batch)

        try:
            items = json.loads(array_match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}, using fallback")
            return self._fallback_batch(batch)

        qa_pairs = []
        for i, (fact, topic) in enumerate(batch):
            if i < len(items):
                item = items[i]
                q = str(item.get("question", "")).strip()
                a = str(item.get("answer", "")).strip()
                if q and a and len(q) > 5 and len(a) > 5:
                    qa_pairs.append(SynthesizedQA(question=q, answer=a, source_fact=fact))
                    continue
            # Item missing or malformed — use fallback for just this one
            qa_pairs.extend(self._fallback_single(fact, topic))

        return qa_pairs

    def _fallback_batch(self, batch: list[tuple[dict, str]]) -> list[SynthesizedQA]:
        """Fast no-model fallback for an entire batch."""
        pairs = []
        for fact, topic in batch:
            pairs.extend(self._fallback_single(fact, topic))
        return pairs

    def _fallback_single(self, fact: dict, topic: str) -> list[SynthesizedQA]:
        """Fallback without model — correct roles: LLM asks, user answers."""
        if fact.get("type") == "qa_pair":
            q = fact.get("question", "").strip()  # LLM's question
            a = fact.get("answer", "").strip()     # user's knowledge
            if q and a:
                return [SynthesizedQA(question=q, answer=a, source_fact=fact)]

        # For plain facts — generate a student question about the teacher's knowledge
        content = fact.get("content", "").strip()
        student_q = fact.get("student_question", "").strip()
        if not content:
            return []

        # Prefer the actual student question if we have it
        if student_q:
            return [SynthesizedQA(question=student_q, answer=content, source_fact=fact)]

        # Derive a curiosity-style question from content
        q = self._derive_student_question(content, topic)
        return [SynthesizedQA(question=q, answer=content, source_fact=fact)]

    def _derive_student_question(self, content: str, topic: str) -> str:
        """Derive a student-style question from teacher knowledge content."""
        import re

        # "X is/was Y" → "Can you explain what X is?"
        m = re.match(r"^(.+?)\s+(is|was|are|were)\s+(.+)", content, re.I)
        if m and len(m.group(1)) < 60:
            return f"Can you explain what {m.group(1).strip()} is?"

        # "In 1789 / In the X century" → "What can you tell me about what happened in X?"
        m = re.match(r"^In (\d{4}|the \w+ century),?\s+(.+)", content, re.I)
        if m:
            return f"What happened in {m.group(1)}?"

        # "X caused/led to Y" → "Why did X cause that?"
        m = re.match(r"^(.+?)\s+(caused|led to|resulted in)\s+(.+)", content, re.I)
        if m and len(m.group(1)) < 60:
            return f"Why did {m.group(1).strip()} {m.group(2)}?"

        # Generic curiosity question
        words = content.split()
        subject = " ".join(words[:5]).rstrip(".,;:")
        return f"Could you tell me more about {subject}?"