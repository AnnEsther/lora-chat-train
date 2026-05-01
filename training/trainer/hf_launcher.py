"""training/trainer/hf_launcher.py

Refactored from the original EndToEndLoRA train.py.
Launches and polls LoRA fine-tuning jobs on HuggingFace (A100 endpoint).
Also contains the local SFTTrainer wrapper used when running training directly
(e.g. on RunPod or inside the container itself).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Literal

import requests

logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
BASE_MODEL = os.environ.get("BASE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
HF_TRAINING_ENDPOINT = os.environ.get("HF_TRAINING_ENDPOINT", "")

# LoRA / training hyperparameters — override via env
LORA_R = int(os.environ.get("LORA_R", 16))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", 32))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", 0.05))
LORA_TARGET_MODULES = os.environ.get("LORA_TARGET_MODULES", "q_proj,v_proj").split(",")

TRAIN_EPOCHS = int(os.environ.get("TRAIN_EPOCHS", 3))
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 4))
TRAIN_GRAD_ACCUM = int(os.environ.get("TRAIN_GRAD_ACCUM", 4))
TRAIN_LR = float(os.environ.get("TRAIN_LR", 2e-4))
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", 512))


class HFTrainingLauncher:
    """Launch LoRA SFT jobs on HuggingFace Inference Endpoints / AutoTrain."""

    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        }

    def build_config(
        self,
        run_id: str,
        session_id: str,
        dataset_s3_path: str,
        dataset_local_path: str = "",
    ) -> dict:
        return {
            "run_id": run_id,
            "session_id": session_id,
            "base_model": BASE_MODEL,
            "dataset_s3_path": dataset_s3_path,
            "dataset_local_path": dataset_local_path,
            "lora": {
                "r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "target_modules": LORA_TARGET_MODULES,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
            "training": {
                "num_train_epochs": TRAIN_EPOCHS,
                "per_device_train_batch_size": TRAIN_BATCH_SIZE,
                "gradient_accumulation_steps": TRAIN_GRAD_ACCUM,
                "learning_rate": TRAIN_LR,
                "max_seq_length": MAX_SEQ_LENGTH,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.05,
                "fp16": True,
                "optim": "paged_adamw_8bit",
                "logging_steps": 10,
                "save_steps": 50,
                "output_dir": f"/tmp/lora_run_{run_id}",
            },
        }

    def launch(self, config: dict) -> str:
        """Submit training job to HF endpoint. Returns the HF job ID."""
        if not HF_TRAINING_ENDPOINT:
            # Fall back to local GPU training via model server
            import os, requests as req

            model_server = os.environ.get("MODEL_SERVER_URL", "http://localhost:8001")
            dataset_path = config.get("dataset_local_path", "")
            run_id = config.get("run_id", "unknown")
            resp = req.post(
                f"{model_server}/train",
                json={"run_id": run_id, "dataset_path": dataset_path},
                timeout=30,
            )
            resp.raise_for_status()
            return f"local_{run_id}"
        else:
            # Submit to HuggingFace training endpoint
            run_id = config.get("run_id", "unknown")
            payload = {
                "run_id": run_id,
                "dataset_s3_path": config.get("dataset_s3_path", ""),
                "base_model": config.get("base_model", os.environ.get("BASE_MODEL", "")),
                "lora_r": config.get("lora_r", 16),
                "lora_alpha": config.get("lora_alpha", 32),
                "lora_dropout": config.get("lora_dropout", 0.05),
                "epochs": config.get("epochs", 3),
                "batch_size": config.get("batch_size", 4),
                "learning_rate": config.get("learning_rate", 2e-4),
            }
            try:
                resp = requests.post(
                    HF_TRAINING_ENDPOINT,
                    headers=self.headers,
                    json=payload,
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                job_id = data.get("id") or data.get("job_id") or run_id
                logger.info("hf_job_submitted", extra={"job_id": job_id, "run_id": run_id})
                return job_id
            except requests.RequestException as exc:
                logger.error("hf_submit_error", extra={"error": str(exc)})
                # Fall back to treating run_id as job_id for polling
                return f"local_{run_id}"            

    def poll(self, hf_job_id: str) -> Literal["running", "succeeded", "failed"]:
        """Returns one of: running | succeeded | failed."""
        url = f"{HF_TRAINING_ENDPOINT}/{hf_job_id}"
        try:
            resp = requests.get(url, headers=self.headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            raw_status = data.get("status", "unknown").lower()
        except requests.RequestException as exc:
            logger.warning(
                "hf_poll_error", extra={"job_id": hf_job_id, "error": str(exc)}
            )
            return "running"  # assume still running on transient error

        if raw_status in ("completed", "succeeded", "done"):
            return "succeeded"
        if raw_status in ("failed", "error", "cancelled"):
            return "failed"
        return "running"

    def get_error(self, hf_job_id: str) -> str:
        url = f"{HF_TRAINING_ENDPOINT}/{hf_job_id}"
        try:
            resp = requests.get(url, headers=self.headers, timeout=15)
            data = resp.json()
            return data.get("error", "unknown error")
        except Exception:
            return "could not retrieve error details"

    def get_logs(self, hf_job_id: str) -> str:
        url = f"{HF_TRAINING_ENDPOINT}/{hf_job_id}/logs"
        try:
            resp = requests.get(url, headers=self.headers, timeout=30)
            return resp.text[:50_000]  # cap log size
        except Exception as exc:
            return f"[log retrieval failed: {exc}]"

    def download_artifacts(self, hf_job_id: str, run_id: str) -> Path:
        """Download trained adapter to local temp dir. Returns path."""
        import tempfile
        import zipfile
        import io

        output_dir = Path(tempfile.mkdtemp()) / f"adapter_{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        artifact_url = f"{HF_TRAINING_ENDPOINT}/{hf_job_id}/artifacts"
        resp = requests.get(
            artifact_url, headers=self.headers, timeout=120, stream=True
        )

        if resp.status_code == 200 and "zip" in resp.headers.get("Content-Type", ""):
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                zf.extractall(output_dir)
        else:
            # Fallback: write raw bytes as a single artifact file
            (output_dir / "adapter.bin").write_bytes(resp.content)

        logger.info(
            "artifacts_downloaded", extra={"run_id": run_id, "path": str(output_dir)}
        )
        return output_dir


# ── Local SFTTrainer (run inside container or RunPod) ─────────────────────────


def train_local(config: dict, dataset_path: Path) -> Path:
    """
    Run LoRA SFT training locally using transformers + peft + trl.
    This is the direct refactor of train.py from the original EndToEndLoRA repo.

    Returns path to the saved adapter.
    """
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig,
    )
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

    run_id = config["run_id"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "local_training_start",
        extra={"run_id": run_id, "base_model": config["base_model"]},
    )

    # 4-bit quantisation for memory efficiency on smaller GPUs
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"],
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        fp16=train_cfg.get("fp16", True),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 50),
        report_to="none",  # avoid wandb dependency
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=train_cfg.get("max_seq_length", 512),
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logger.info("local_training_done", extra={"output_dir": str(output_dir)})
    return output_dir
