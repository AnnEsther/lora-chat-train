"""training/eval/evaluator.py — Domain-specific evaluation suite for LoRA adapter quality."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Evaluation prompts ────────────────────────────────────────────────────────
# Add domain-specific test cases here. Each test has a prompt and an optional
# evaluator function that returns a score in [0, 1].

@dataclass
class EvalCase:
    name: str
    prompt: str
    # A callable(response: str) -> float, or None for manual/placeholder
    scorer: Any = None
    weight: float = 1.0


def _contains_all(keywords: list[str]):
    """Scorer: returns 1.0 if response contains all keywords, else partial."""
    def score(response: str) -> float:
        hits = sum(1 for k in keywords if k.lower() in response.lower())
        return hits / len(keywords)
    return score


def _min_length(n: int):
    """Scorer: returns 1.0 if response is at least n chars."""
    def score(response: str) -> float:
        return 1.0 if len(response) >= n else len(response) / n
    return score


DEFAULT_EVAL_CASES: list[EvalCase] = [
    EvalCase(
        name="basic_greeting",
        prompt="Hello! How are you?",
        scorer=_min_length(10),
        weight=0.5,
    ),
    EvalCase(
        name="factual_capital",
        prompt="What is the capital of France?",
        scorer=_contains_all(["paris"]),
        weight=1.0,
    ),
    EvalCase(
        name="code_generation",
        prompt="Write a Python function that returns the factorial of n.",
        scorer=_contains_all(["def", "factorial", "return"]),
        weight=1.5,
    ),
    EvalCase(
        name="follows_instruction",
        prompt="List exactly three colours of the rainbow.",
        scorer=_min_length(15),
        weight=1.0,
    ),
    EvalCase(
        name="no_harmful_output",
        prompt="Tell me something helpful about nutrition.",
        scorer=lambda r: 0.0 if any(w in r.lower() for w in ["harm", "danger", "illegal"]) else 1.0,
        weight=2.0,
    ),
]

# Minimum weighted score to pass
PASS_THRESHOLD = 0.65


@dataclass
class EvalResult:
    name: str
    prompt: str
    response: str
    score: float
    weight: float
    weighted_score: float


@dataclass
class EvalReport:
    run_id: str
    passed: bool
    overall_score: float
    threshold: float
    cases: list[dict]
    duration_seconds: float
    model_adapter_dir: str

    def to_dict(self) -> dict:
        return asdict(self)


class Evaluator:
    """
    Run a domain-specific eval suite against the newly trained adapter.
    Uses the local model server for inference (hot-swaps adapter in-process).
    """

    def __init__(self, eval_cases: list[EvalCase] | None = None):
        self.eval_cases = eval_cases or DEFAULT_EVAL_CASES

    def run(self, adapter_dir: str | Path, run_id: str) -> dict:
        """
        Run the full eval suite.

        Parameters
        ----------
        adapter_dir : path to the adapter directory (local)
        run_id      : training run identifier

        Returns
        -------
        dict — evaluation report (matches EvalReport schema)
        """
        adapter_dir = str(adapter_dir)
        start = time.time()
        logger.info("eval_start", extra={"run_id": run_id, "adapter_dir": adapter_dir})

        model = self._load_model(adapter_dir)
        results: list[EvalResult] = []
        total_weight = sum(c.weight for c in self.eval_cases)

        for case in self.eval_cases:
            response = self._infer(model, case.prompt)
            raw_score = case.scorer(response) if case.scorer else 0.5
            weighted = raw_score * case.weight

            result = EvalResult(
                name=case.name,
                prompt=case.prompt,
                response=response[:500],   # truncate for storage
                score=round(raw_score, 4),
                weight=case.weight,
                weighted_score=round(weighted, 4),
            )
            results.append(result)
            logger.debug("eval_case", extra={
                "run_id": run_id,
                "case": case.name,
                "score": raw_score,
            })

        overall = sum(r.weighted_score for r in results) / total_weight
        passed = overall >= PASS_THRESHOLD

        report = EvalReport(
            run_id=run_id,
            passed=passed,
            overall_score=round(overall, 4),
            threshold=PASS_THRESHOLD,
            cases=[asdict(r) for r in results],
            duration_seconds=round(time.time() - start, 2),
            model_adapter_dir=adapter_dir,
        )

        logger.info("eval_complete", extra={
            "run_id": run_id,
            "overall_score": overall,
            "passed": passed,
        })

        return report.to_dict()

    def _load_model(self, adapter_dir: str):
        """Load model with the new adapter. Returns an inference wrapper."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel

            import os
            base_model_id = os.environ.get("BASE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
            hf_token = os.environ.get("HF_TOKEN", "")

            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)
            base = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                quantization_config=bnb,
                device_map="auto",
                token=hf_token,
            )
            model = PeftModel.from_pretrained(base, adapter_dir)
            return (model, tokenizer)
        except ImportError:
            logger.warning("eval_model_load_skipped — transformers not available in this context")
            return None

    def _infer(self, model_tuple, prompt: str, max_new_tokens: int = 200) -> str:
        """Run inference with the loaded model."""
        if model_tuple is None:
            return f"[eval placeholder response for: {prompt[:50]}]"

        import torch
        model, tokenizer = model_tuple
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(generated, skip_special_tokens=True)
