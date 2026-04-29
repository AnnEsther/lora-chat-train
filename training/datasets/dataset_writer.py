"""training/datasets/dataset_writer.py — Write curated candidates as SFTTrainer-compatible JSONL."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# System prompt prepended to every training example.
# Swap this out to tune personality / domain.
DEFAULT_SYSTEM_PROMPT = (
    "You are an eager student sitting in a classroom, listening attentively to your teacher. "
    "When your teacher explains something, pay close attention and ask thoughtful questions to deepen your understanding. "
    "Seek clarification on anything you find confusing, and ask follow-up questions to explore topics more deeply. "
    "Show genuine curiosity and enthusiasm for learning new concepts."
)


class DatasetWriter:
    """
    Writes curated (user, assistant) pairs into JSONL format consumable by
    transformers + trl SFTTrainer using the chat template format.

    Each line: {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

    This matches the format expected when SFTTrainer is initialised with
    `dataset_text_field` not set and `formatting_func` uses apply_chat_template.
    """

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    def write_jsonl(self, samples: list[dict]) -> str:
        """
        Parameters
        ----------
        samples : list of {"user_turn": str, "assistant_turn": str}

        Returns
        -------
        JSONL string — one JSON object per line
        """
        lines = []
        for sample in samples:
            record = {
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": sample["user_turn"]},
                    {"role": "assistant", "content": sample["assistant_turn"]},
                ]
            }
            lines.append(json.dumps(record, ensure_ascii=False))

        jsonl = "\n".join(lines)
        logger.info(
            "dataset_written", extra={"samples": len(samples), "chars": len(jsonl)}
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
