"""backend/model_server/serve.py

FastAPI model server — loads Llama 3.2:1b with the current production LoRA adapter
and exposes streaming inference + a hot-reload endpoint for adapter swapping.

Run with:
    python -m model_server.serve
"""

from __future__ import annotations

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

BASE_MODEL = os.environ.get("BASE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ADAPTER_DIR = Path(os.environ.get("ADAPTER_DIR", "/adapters/current"))

# ── Global model state ────────────────────────────────────────────────────────
_model_lock = threading.Lock()
_model = None
_tokenizer = None
_adapter_loaded: Optional[str] = None


def _load_base_model():
    global _model, _tokenizer

    logger.info("loading_base_model", extra={"model": BASE_MODEL})
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    _tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, token=HF_TOKEN, trust_remote_code=True,
    )
    _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    _model.eval()
    logger.info("base_model_loaded")


def _load_adapter(adapter_dir: Path) -> None:
    """Hot-swap the LoRA adapter. Thread-safe."""
    global _model, _adapter_loaded

    if not adapter_dir.exists():
        logger.warning("adapter_dir_not_found", extra={"dir": str(adapter_dir)})
        return

    from peft import PeftModel

    with _model_lock:
        # If adapter already loaded via PEFT, unload first
        if hasattr(_model, "disable_adapter"):
            try:
                _model = _model.merge_and_unload()
            except Exception:
                pass  # base model, nothing to unload

        logger.info("loading_adapter", extra={"dir": str(adapter_dir)})
        _model = PeftModel.from_pretrained(
            _model, str(adapter_dir), is_trainable=False,
        )
        _model.eval()
        _adapter_loaded = str(adapter_dir)
        logger.info("adapter_loaded", extra={"dir": str(adapter_dir)})


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _load_base_model()
    if ADAPTER_DIR.exists():
        _load_adapter(ADAPTER_DIR)
    yield


app = FastAPI(title="LoRA Model Server", version="1.0.0", lifespan=lifespan)


# ── Request / response schemas ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    messages: list[dict]
    max_new_tokens: int = 512
    temperature: float = 0.7
    stream: bool = True


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.0
    stream: bool = False


class ReloadRequest(BaseModel):
    adapter_dir: str


# ── Inference helpers ─────────────────────────────────────────────────────────

def _build_prompt(messages: list[dict]) -> str:
    """Apply the Llama 3 chat template."""
    return _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def _stream_tokens(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> AsyncIterator[str]:
    """Returns a sync generator of decoded token strings via TextIteratorStreamer."""
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    streamer = TextIteratorStreamer(
        _tokenizer, skip_prompt=True, skip_special_tokens=True,
    )
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

    thread = threading.Thread(target=_generate, daemon=True)
    thread.start()
    return streamer, thread


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = _build_prompt(req.messages)

    async def sse_generator():
        streamer, thread = _stream_tokens(prompt, req.max_new_tokens, req.temperature)
        loop = asyncio.get_event_loop()

        def _next_token():
            try:
                return next(streamer)
            except StopIteration:
                return None

        while True:
            token = await loop.run_in_executor(None, _next_token)
            if token is None:
                break
            payload = json.dumps({"text": token})
            yield f"data: {payload}\n\n"

        yield "data: [DONE]\n\n"
        thread.join(timeout=5)

    if req.stream:
        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    # Non-streaming fallback
    streamer, thread = _stream_tokens(prompt, req.max_new_tokens, req.temperature)
    full = "".join(streamer)
    thread.join(timeout=5)
    return {"response": full}


@app.post("/generate")
async def generate(req: GenerateRequest) -> dict:
    """Simple non-streaming single prompt inference for smoke tests / eval."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    messages = [{"role": "user", "content": req.prompt}]
    prompt = _build_prompt(messages)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    with _model_lock, torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    text = _tokenizer.decode(generated, skip_special_tokens=True)
    return {"response": text}


@app.post("/reload_adapter")
async def reload_adapter(req: ReloadRequest) -> dict:
    """Hot-swap the LoRA adapter. Called by DeploymentManager after promotion."""
    adapter_path = Path(req.adapter_dir)
    if not adapter_path.exists():
        raise HTTPException(status_code=404, detail=f"Adapter dir not found: {req.adapter_dir}")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_adapter, adapter_path)
    return {"status": "ok", "adapter_dir": str(adapter_path)}


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "adapter": _adapter_loaded,
    }
    
@app.get("/adapters")
async def list_adapters() -> dict:
    import json
    adapters = [{"id": "base", "version": "Base model", "path": "", "is_base": True}]
    history_dir = ADAPTER_DIR.parent / "history"
    try:
        if history_dir.exists():
            for hist_dir in sorted(history_dir.iterdir()):
                if hist_dir.is_dir():
                    manifest_path = hist_dir / "manifest.json"
                    version = hist_dir.name
                    trained_at = None
                    if manifest_path.exists():
                        try:
                            manifest = json.loads(manifest_path.read_text())
                            version = manifest.get("version", hist_dir.name)
                            trained_at = manifest.get("trained_at")
                        except Exception:
                            pass
                    adapters.append({"id": hist_dir.name, "version": version, "path": str(hist_dir), "trained_at": trained_at})
        if ADAPTER_DIR.exists() and any(ADAPTER_DIR.iterdir()):
            manifest_path = ADAPTER_DIR / "manifest.json"
            version = "v1 (live)"
            trained_at = None
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text())
                    version = manifest.get("version", "v1") + " (live)"
                    trained_at = manifest.get("trained_at")
                except Exception:
                    pass
            adapters.append({"id": "current", "version": version, "path": str(ADAPTER_DIR), "trained_at": trained_at, "is_current": True})
    except Exception as exc:
        logger.warning(f"adapters_list_error: {exc}")
    return {"adapters": adapters}

@app.get("/train/status")
async def train_status() -> dict:
    return {"status": "idle", "progress": "", "started_at": None, "finished_at": None, "vram_used_gb": None, "vram_free_gb": None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
