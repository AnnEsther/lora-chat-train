"""
backend/model_server/local_gpu_serve.py

Model server + local LoRA trainer for RTX 4060 (8GB VRAM).
Uses Qwen2.5 / Llama with bf16 throughout to avoid bfloat16/fp16 mismatch.

Run with:
    python backend/model_server/local_gpu_serve.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

from dotenv import load_dotenv
_root = Path(__file__).parent.parent.parent
load_dotenv(_root / ".env")
sys.path.insert(0, str(_root))

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer
from datasets import Dataset as HFDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL          = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
HF_TOKEN            = os.environ.get("HF_TOKEN", "")
ADAPTER_DIR         = Path(os.environ.get("ADAPTER_DIR",         _root / "adapters" / "current"))
HISTORY_DIR         = Path(os.environ.get("ADAPTER_HISTORY_DIR", _root / "adapters" / "history"))
OUTPUT_DIR          = Path(os.environ.get("LOCAL_OUTPUT_DIR",    _root / "outputs"))
LORA_R              = int(os.environ.get("LORA_R", 16))
LORA_ALPHA          = int(os.environ.get("LORA_ALPHA", 32))
LORA_DROPOUT        = float(os.environ.get("LORA_DROPOUT", 0.05))
LORA_TARGET_MODULES = os.environ.get("LORA_TARGET_MODULES", "q_proj,v_proj").split(",")
TRAIN_EPOCHS        = int(os.environ.get("TRAIN_EPOCHS", 3))
TRAIN_BATCH_SIZE    = int(os.environ.get("TRAIN_BATCH_SIZE", 1))
TRAIN_GRAD_ACCUM    = int(os.environ.get("TRAIN_GRAD_ACCUM", 8))
TRAIN_LR            = float(os.environ.get("TRAIN_LR", 2e-4))
MAX_SEQ_LENGTH      = int(os.environ.get("MAX_SEQ_LENGTH", 256))

# ── Global state ──────────────────────────────────────────────────────────────
_model_lock      = threading.Lock()
_model           = None
_tokenizer       = None
_adapter_path: Optional[str] = None
_training_active = False
_training_status: dict = {"status": "idle", "progress": "", "started_at": None, "finished_at": None}


# ── GPU check ─────────────────────────────────────────────────────────────────

def _check_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found. Install torch with: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"GPU detected: {name} ({vram:.1f} GB VRAM)")


# ── BnB config — bf16 throughout to match Qwen's native dtype ────────────────

def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # must match torch_dtype below
    )


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_base_model() -> None:
    global _model, _tokenizer
    logger.info(f"Loading base model: {BASE_MODEL}")

    _tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, token=HF_TOKEN or None, trust_remote_code=True,
    )
    _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.padding_side = "right"

    _model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=_bnb_config(),
        device_map="auto",
        token=HF_TOKEN or None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,             # load everything in bf16
    )
    _model.config.use_cache = False
    _model.eval()
    logger.info("Base model loaded")


def _load_adapter(adapter_dir: Path) -> None:
    global _model, _adapter_path
    if not adapter_dir.exists():
        logger.warning(f"Adapter dir not found: {adapter_dir} — using base model")
        return
    with _model_lock:
        if isinstance(_model, PeftModel):
            logger.info("Unloading existing adapter")
            _model = _model.merge_and_unload()
            _model.eval()
        logger.info(f"Loading adapter: {adapter_dir}")
        _model = PeftModel.from_pretrained(_model, str(adapter_dir), is_trainable=False)
        _model.eval()
        _adapter_path = str(adapter_dir)
        logger.info("Adapter loaded")


# ── Inference ─────────────────────────────────────────────────────────────────

def _build_prompt(messages: list[dict]) -> str:
    return _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _stream_tokens(prompt: str, max_new_tokens: int, temperature: float):
    inputs = _tokenizer(prompt, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature if temperature > 0 else None,
        do_sample=temperature > 0,
        pad_token_id=_tokenizer.eos_token_id,
    )
    def _generate():
        with _model_lock:
            with torch.no_grad():
                _model.generate(**gen_kwargs)
    t = threading.Thread(target=_generate, daemon=True)
    t.start()
    return streamer, t


# ── Training ──────────────────────────────────────────────────────────────────

def _run_training(run_id: str, dataset_jsonl_path: Path) -> None:
    global _training_active, _training_status, _model

    _training_active = True
    _training_status = {
        "status": "running",
        "run_id": run_id,
        "progress": "Starting…",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
    }

    output_dir = OUTPUT_DIR / "training_runs" / run_id / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── Load dataset ──────────────────────────────────────────────────────
        _training_status["progress"] = "Loading dataset…"
        records = [
            json.loads(line)
            for line in dataset_jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records:
            raise ValueError("Dataset is empty")

        def _fmt(ex):
            return {"text": _tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False)}

        hf_dataset = HFDataset.from_list(records).map(_fmt)
        logger.info(f"Dataset: {len(records)} samples")
        _training_status["progress"] = f"Training on {len(records)} samples…"

        # ── Unload inference adapter before training ───────────────────────────
        with _model_lock:
            if isinstance(_model, PeftModel):
                _model = _model.merge_and_unload()
                _model.eval()

        # ── Load fresh trainable model ─────────────────────────────────────────
        train_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=_bnb_config(),
            device_map="auto",
            token=HF_TOKEN or None,
            torch_dtype=torch.bfloat16,         # bf16 throughout — no fp16 mixing
        )
        train_model.config.use_cache = False

        lora_config = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES, bias="none", task_type=TaskType.CAUSAL_LM,
        )
        train_model = get_peft_model(train_model, lora_config)
        train_model.print_trainable_parameters()

        # ── SFTConfig — bf16=True, fp16=False ─────────────────────────────────
        sft_config = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=TRAIN_EPOCHS,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=TRAIN_GRAD_ACCUM,
            learning_rate=TRAIN_LR,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            fp16=False,                         # must be False for Qwen/bf16 models
            bf16=True,                          # use bf16 — matches model's native dtype
            optim="paged_adamw_8bit",
            logging_steps=5,
            save_strategy="no",
            report_to="none",
            dataloader_pin_memory=False,        # avoids Windows CUDA issues
            max_length=MAX_SEQ_LENGTH,          # correct arg name in trl >= 0.13
            dataset_text_field="text",
        )

        trainer = SFTTrainer(
            model=train_model,
            args=sft_config,
            train_dataset=hf_dataset,
            processing_class=_tokenizer,        # replaces tokenizer= in trl >= 0.9
        )

        trainer.train()

        # ── Save adapter ───────────────────────────────────────────────────────
        train_model.save_pretrained(str(output_dir))
        _tokenizer.save_pretrained(str(output_dir))

        del train_model
        torch.cuda.empty_cache()

        logger.info(f"Training complete: {output_dir}")
        _training_status["progress"] = "Complete — loading new adapter…"

        # ── Hot-swap adapter ───────────────────────────────────────────────────
        _archive_current_adapter(run_id)
        _load_adapter(output_dir)

        manifest = {
            "version": run_id[:8],
            "run_id": run_id,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "samples": len(records),
            "adapter_dir": str(output_dir),
        }
        ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
        (ADAPTER_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

        _training_status.update({
            "status": "completed",
            "progress": f"Adapter {run_id[:8]} is live",
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.info("Adapter hot-swapped — model ready")

    except Exception as exc:
        logger.error(f"Training failed: {exc}", exc_info=True)
        _training_status.update({
            "status": "failed",
            "progress": f"Failed: {exc}",
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
    finally:
        _training_active = False


def _archive_current_adapter(run_id: str) -> None:
    import shutil
    if not ADAPTER_DIR.exists():
        return
    version = "unknown"
    manifest_path = ADAPTER_DIR / "manifest.json"
    if manifest_path.exists():
        try:
            version = json.loads(manifest_path.read_text()).get("version", "unknown")
        except Exception:
            pass
    archive = HISTORY_DIR / version
    archive.mkdir(parents=True, exist_ok=True)
    for item in ADAPTER_DIR.iterdir():
        shutil.copy2(item, archive / item.name)
    logger.info(f"Archived adapter: {version}")


# ── FastAPI ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _check_gpu()
    _load_base_model()
    if ADAPTER_DIR.exists() and any(ADAPTER_DIR.iterdir()):
        _load_adapter(ADAPTER_DIR)
    else:
        logger.info("No existing adapter — using base model")
    yield


app = FastAPI(title="LoRA Local GPU Server", version="1.0.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    messages: list[dict]
    max_new_tokens: int = 512
    temperature: float = 0.7
    stream: bool = True


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.0


class TrainRequest(BaseModel):
    run_id: str
    dataset_path: str


class ReloadRequest(BaseModel):
    adapter_dir: str


@app.post("/chat")
async def chat(req: ChatRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    if _training_active:
        raise HTTPException(503, "Training in progress — please wait")
    prompt = _build_prompt(req.messages)

    async def sse():
        streamer, thread = _stream_tokens(prompt, req.max_new_tokens, req.temperature)
        loop = asyncio.get_event_loop()
        while True:
            token = await loop.run_in_executor(None, lambda: next(streamer, None))
            if token is None:
                break
            yield f"data: {json.dumps({'text': token})}\n\n"
        yield "data: [DONE]\n\n"
        thread.join(timeout=5)

    if req.stream:
        return StreamingResponse(sse(), media_type="text/event-stream")
    streamer, thread = _stream_tokens(prompt, req.max_new_tokens, req.temperature)
    full = "".join(streamer)
    thread.join(timeout=5)
    return {"response": full}


@app.post("/generate")
async def generate(req: GenerateRequest) -> dict:
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    prompt = _build_prompt([{"role": "user", "content": req.prompt}])
    inputs = _tokenizer(prompt, return_tensors="pt").to("cuda")
    with _model_lock, torch.no_grad():
        outputs = _model.generate(
            **inputs, max_new_tokens=req.max_new_tokens,
            do_sample=False, pad_token_id=_tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    return {"response": _tokenizer.decode(generated, skip_special_tokens=True)}


@app.post("/train")
async def train(req: TrainRequest) -> dict:
    global _training_active
    if _training_active:
        raise HTTPException(409, "Training already in progress")

    dataset_path = Path(req.dataset_path)
    if not dataset_path.exists():
        # Try relative to project root
        dataset_path = _root / req.dataset_path
    if not dataset_path.exists():
        raise HTTPException(404, f"Dataset not found: {req.dataset_path}")

    # Real thread — BackgroundTasks is unreliable for long jobs on Windows
    t = threading.Thread(
        target=_run_training,
        args=(req.run_id, dataset_path),
        daemon=True,
        name=f"training-{req.run_id[:8]}",
    )
    t.start()
    logger.info(f"Training thread started: {t.name}")
    return {"status": "started", "run_id": req.run_id}


@app.get("/train/status")
async def train_status() -> dict:
    vram_used = vram_free = None
    if torch.cuda.is_available():
        vram_used = round(torch.cuda.memory_allocated(0) / 1024**3, 2)
        vram_free = round(
            (torch.cuda.get_device_properties(0).total_memory
             - torch.cuda.memory_allocated(0)) / 1024**3, 2
        )
    return {**_training_status, "vram_used_gb": vram_used, "vram_free_gb": vram_free}


@app.post("/reload_adapter")
async def reload_adapter(req: ReloadRequest) -> dict:
    adapter_path = Path(req.adapter_dir)
    if not adapter_path.exists():
        raise HTTPException(404, f"Adapter dir not found: {req.adapter_dir}")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_adapter, adapter_path)
    return {"status": "ok", "adapter_dir": str(adapter_path)}


@app.get("/health")
async def health() -> dict:
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
            "vram_used_gb":  round(torch.cuda.memory_allocated(0) / 1024**3, 2),
        }
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "adapter": _adapter_path,
        "training_active": _training_active,
        "gpu": gpu_info,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")