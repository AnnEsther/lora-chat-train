"""training/datasets/dataset_writer.py — Write curated candidates as SFTTrainer-compatible JSONL.

Each training sample is now a full multi-turn conversation segment (list of
{"role", "content"} dicts) rather than an isolated (user_turn, assistant_turn) pair.
The system prompt is prepended as the first message; the segment turns follow as-is.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# System prompt prepended to every training example.
DEFAULT_SYSTEM_PROMPT = (
    "You are an eager student sitting in a classroom, listening attentively to your teacher. "
    "When your teacher explains something, pay close attention and ask thoughtful questions to deepen your understanding. "
    "Seek clarification on anything you find confusing, and ask follow-up questions to explore topics more deeply. "
    "Show genuine curiosity and enthusiasm for learning new concepts."
)


class DatasetWriter:
    """
    Writes curated multi-turn conversation segments into JSONL format consumable
    by transformers + trl SFTTrainer using the chat template format.

    Each line:
        {"messages": [{"role": "system", ...}, <turn 1>, <turn 2>, ...]}

    The system prompt is always the first message.  All turns from the conversation
    segment follow in their original order, preserving multi-turn context including
    segments that contain multiple questions or compound answers within a single
    exchange.
    """

    def __init__(self, system_prompt: Optional[str] = None) -> None:
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    def write_jsonl(self, samples: list[dict]) -> str:
        """
        Parameters
        ----------
        samples : list of {"conversation": list[{"role": str, "content": str}]}

        Returns
        -------
        JSONL string — one JSON object per line.
        """
        lines = []
        for sample in samples:
            conversation: list[dict] = sample.get("conversation") or []

            # Skip empty or degenerate segments
            if not conversation:
                continue

            record = {
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    *conversation,
                ]
            }
            lines.append(json.dumps(record, ensure_ascii=False))

        jsonl = "\n".join(lines)
        logger.info(
            "dataset_written", extra={"samples": len(lines), "chars": len(jsonl)}
        )
        return jsonl

    def write_to_file(self, samples: list[dict], path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        jsonl = self.write_jsonl(samples)
        path.write_text(jsonl, encoding="utf-8")
        logger.info(
            "dataset_file_written", extra={"path": str(path), "samples": len(samples)}
        )
        return path
