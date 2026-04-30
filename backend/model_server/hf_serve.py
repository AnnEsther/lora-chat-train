"""backend/model_server/hf_serve.py

Thin proxy model server that forwards inference to a **private** HuggingFace
Dedicated Inference Endpoint (vLLM) instead of loading weights onto a local GPU.

Drop-in replacement for local_gpu_serve.py — exposes the identical FastAPI
surface so ModelClient, the backend, Celery workers, and the frontend diagnostic
panel all work without modification.

Inference flow
--------------
  client → ModelClient → /chat (this server) → HF vLLM /v1/chat/completions
                                               (your private endpoint, HF_TOKEN auth)

The vLLM endpoint returns OpenAI-format SSE:
  data: {"choices": [{"delta": {"content": "..."}, "finish_reason": null}]}

This server translates that to the internal SSE format consumed by ModelClient:
  data: {"text": "..."}

Training
--------
/train is intentionally a no-op stub.  Training still runs locally via
local_gpu_serve.py or via HF_TRAINING_ENDPOINT / hf_launcher.py.

Adapter hot-swap
----------------
/reload_adapter logs a warning and returns success without doing anything.
The base model on the vLLM endpoint is always used for inference.

Required env vars
-----------------
HF_ENDPOINT_URL   Full URL of your private vLLM endpoint, e.g.
                  https://your-name.us-east-1.aws.endpoints.huggingface.cloud
HF_TOKEN          Your HuggingFace access token (write scope)
HF_ENDPOINT_MODEL HuggingFace model ID sent in the request body — must match
                  the model deployed on the endpoint, e.g.
                  meta-llama/Llama-3.2-1B-Instruct

Optional env vars
-----------------
MAX_NEW_TOKENS    Default max tokens per response (default: 512)
TEMPERATURE       Default sampling temperature (default: 0.7)

Run with:
    python backend/model_server/hf_serve.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from dotenv import load_dotenv

_root = Path(__file__).parent.parent.parent
load_dotenv(_root / ".env")
sys.path.insert(0, str(_root))

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

HF_ENDPOINT_URL: str = os.environ.get("HF_ENDPOINT_URL", "").rstrip("/")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
# vLLM requires the real model ID in the request body, not a generic string.
# Defaults to BASE_MODEL so you only need to set one env var.
HF_ENDPOINT_MODEL: str = os.environ.get(
    "HF_ENDPOINT_MODEL",
    os.environ.get("BASE_MODEL", "meta-llama/Llama-3.2-1B-Instruct"),
)
DEFAULT_MAX_NEW_TOKENS: int = int(os.environ.get("MAX_NEW_TOKENS", 512))
DEFAULT_TEMPERATURE: float = float(os.environ.get("TEMPERATURE", 0.7))

# Local adapter directory — still read for the /adapters listing endpoint
ADAPTER_DIR = Path(os.environ.get("ADAPTER_DIR", _root / "adapters" / "current"))
HISTORY_DIR = Path(
    os.environ.get("ADAPTER_HISTORY_DIR", _root / "adapters" / "history")
)

# ── Validation ────────────────────────────────────────────────────────────────


def _validate_config() -> None:
    if not HF_ENDPOINT_URL:
        raise RuntimeError(
            "HF_ENDPOINT_URL is not set.\n"
            "Set it in .env to your private HF Inference Endpoint URL, e.g.:\n"
            "  HF_ENDPOINT_URL=https://your-endpoint.us-east-1.aws.endpoints.huggingface.cloud"
        )
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set. Add your HuggingFace token to .env.")
    logger.info("hf_endpoint_configured", extra={"url": HF_ENDPOINT_URL})


# ── HTTP client ───────────────────────────────────────────────────────────────

_http: Optional[httpx.AsyncClient] = None


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }


# ── FastAPI lifespan ──────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _http
    _validate_config()
    _http = httpx.AsyncClient(
        base_url=HF_ENDPOINT_URL,
        headers=_headers(),
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0),
    )
    logger.info(
        "hf_endpoint_configured",
        extra={"url": HF_ENDPOINT_URL, "model": HF_ENDPOINT_MODEL},
    )
    # Verify the private endpoint is reachable — vLLM exposes /health
    try:
        resp = await _http.get("/health")
        if resp.is_success:
            logger.info("hf_endpoint_reachable", extra={"url": HF_ENDPOINT_URL})
        else:
            logger.warning(
                "hf_endpoint_health_check_failed",
                extra={"status": resp.status_code, "body": resp.text[:200]},
            )
    except Exception as exc:
        logger.warning("hf_endpoint_unreachable", extra={"error": str(exc)})

    yield

    await _http.aclose()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="LoRA HF vLLM Proxy Server", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request schemas ───────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    messages: list[dict]
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
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


# ── SSE translation helpers ───────────────────────────────────────────────────


def _translate_vllm_chunk(raw: str) -> Optional[str]:
    """
    Convert one vLLM/OpenAI SSE data line into the internal {"text": "..."} format.

    vLLM returns lines like:
      data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}
    or:
      data: [DONE]

    Returns the translated JSON string to yield, or None if the line should be
    skipped (empty delta, [DONE], non-data lines, parse errors).
    """
    raw = raw.strip()
    if not raw.startswith("data: "):
        return None
    payload = raw[6:].strip()
    if payload == "[DONE]":
        return None
    try:
        obj = json.loads(payload)
        choices = obj.get("choices") or []
        if not choices:
            return None
        delta = choices[0].get("delta") or {}
        text = delta.get("content", "")
        if not text:
            return None
        return json.dumps({"text": text})
    except (json.JSONDecodeError, KeyError, IndexError):
        return None


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Stream inference from the private vLLM endpoint."""
    assert _http is not None

    payload = {
        "model": HF_ENDPOINT_MODEL,
        "messages": req.messages,
        "max_tokens": req.max_new_tokens,
        "temperature": req.temperature if req.temperature > 0 else 1e-7,
        "stream": True,
    }

    async def sse_generator():
        try:
            async with _http.stream(
                "POST", "/v1/chat/completions", json=payload
            ) as resp:
                if not resp.is_success:
                    body = await resp.aread()
                    error_msg = body.decode(errors="replace")[:300]
                    logger.error(
                        "vllm_chat_error",
                        extra={"status": resp.status_code, "body": error_msg},
                    )
                    yield f"data: {json.dumps({'text': f'[Endpoint error {resp.status_code}: {error_msg}]'})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                async for line in resp.aiter_lines():
                    translated = _translate_vllm_chunk(line)
                    if translated is not None:
                        yield f"data: {translated}\n\n"

        except httpx.TimeoutException:
            logger.error("vllm_chat_timeout")
            yield f"data: {json.dumps({'text': '[Request timed out — the HF endpoint may be cold-starting. Try again in a moment.]'})}\n\n"
        except httpx.RequestError as exc:
            logger.error("vllm_chat_request_error", extra={"error": str(exc)})
            yield f"data: {json.dumps({'text': f'[Could not reach HF endpoint: {exc}]'})}\n\n"

        yield "data: [DONE]\n\n"

    if req.stream:
        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    # Non-streaming fallback: collect all chunks
    chunks: list[str] = []
    async for line in sse_generator():
        if line.startswith("data: ") and not line.strip().endswith("[DONE]"):
            try:
                obj = json.loads(line[6:])
                chunks.append(obj.get("text", ""))
            except json.JSONDecodeError:
                pass
    return {"response": "".join(chunks)}


@app.post("/generate")
async def generate(req: GenerateRequest) -> dict:
    """Non-streaming single-turn generation — used by eval and smoke tests."""
    assert _http is not None

    payload = {
        "model": HF_ENDPOINT_MODEL,
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": req.max_new_tokens,
        "temperature": req.temperature if req.temperature > 0 else 1e-7,
        "stream": False,
    }

    try:
        resp = await _http.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"response": text}
    except httpx.HTTPStatusError as exc:
        logger.error(
            "vllm_generate_error",
            extra={"status": exc.response.status_code, "body": exc.response.text[:200]},
        )
        raise HTTPException(502, f"HF endpoint error: {exc.response.status_code}")
    except httpx.RequestError as exc:
        logger.error("vllm_generate_request_error", extra={"error": str(exc)})
        raise HTTPException(503, f"HF endpoint unreachable: {exc}")


@app.post("/train")
async def train(req: TrainRequest) -> dict:
    """
    Training is not supported through the HF vLLM inference endpoint.
    Use local_gpu_serve.py or set HF_TRAINING_ENDPOINT for Celery-based training.
    """
    logger.warning(
        "train_not_supported",
        extra={
            "run_id": req.run_id,
            "hint": "Set HF_TRAINING_ENDPOINT or run local_gpu_serve.py to train",
        },
    )
    raise HTTPException(
        501,
        detail=(
            "Training is not supported by this server (hf_serve.py). "
            "To run training: either set HF_TRAINING_ENDPOINT in .env for remote training, "
            "or switch to local_gpu_serve.py."
        ),
    )


@app.get("/train/status")
async def train_status() -> dict:
    """Always idle — training does not run through this server."""
    return {
        "status": "idle",
        "progress": "Training not supported via HF vLLM inference endpoint — use local_gpu_serve.py or HF_TRAINING_ENDPOINT.",
        "started_at": None,
        "finished_at": None,
        "vram_used_gb": None,
        "vram_free_gb": None,
    }


@app.post("/reload_adapter")
async def reload_adapter(req: ReloadRequest) -> dict:
    """
    Adapter hot-swap is not supported for remote vLLM endpoints via this proxy.
    The vLLM endpoint always serves the base model it was deployed with.
    To use a fine-tuned adapter, redeploy the endpoint with the new model weights.
    """
    logger.warning(
        "reload_adapter_not_supported",
        extra={
            "requested_dir": req.adapter_dir,
            "hint": "Adapter hot-swap requires local_gpu_serve.py",
        },
    )
    # Return success shape so callers (DeploymentManager) do not crash
    return {
        "status": "ok",
        "adapter_dir": req.adapter_dir,
        "warning": "Adapter hot-swap is not supported for the HF vLLM inference endpoint — base model is always used.",
    }


@app.get("/adapters")
async def list_adapters() -> dict:
    """List locally saved adapters (history only — not loaded into the vLLM endpoint)."""
    adapters = [
        {
            "id": "base",
            "version": "Base model (remote vLLM)",
            "path": "",
            "is_base": True,
        }
    ]
    try:
        if HISTORY_DIR.exists():
            for hist_dir in sorted(HISTORY_DIR.iterdir()):
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
                    adapters.append(
                        {
                            "id": str(hist_dir.name),
                            "version": version,
                            "path": str(hist_dir),
                            "trained_at": trained_at,
                            "note": "saved locally — not active in vLLM endpoint",
                        }
                    )
        if ADAPTER_DIR.exists() and any(ADAPTER_DIR.iterdir()):
            current_manifest = ADAPTER_DIR / "manifest.json"
            current_version = "v1 (local only)"
            if current_manifest.exists():
                try:
                    current_version = (
                        json.loads(current_manifest.read_text()).get("version", "v1")
                        + " (local only)"
                    )
                except Exception:
                    pass
            adapters.append(
                {
                    "id": "current",
                    "version": current_version,
                    "path": str(ADAPTER_DIR),
                    "trained_at": None,
                    "is_current": True,
                    "note": "saved locally — not active in vLLM endpoint",
                }
            )
    except Exception as exc:
        logger.warning("adapters_list_error", extra={"error": str(exc)})
    return {"adapters": adapters}


@app.get("/health")
async def health() -> dict:
    """
    Check health of the remote vLLM endpoint and return in the standard shape
    that the frontend diagnostic panel expects.
    """
    assert _http is not None
    vllm_ok = False
    try:
        resp = await _http.get("/health", timeout=5.0)
        vllm_ok = resp.is_success
    except Exception as exc:
        logger.warning("vllm_health_error", extra={"error": str(exc)})

    return {
        "status": "ok" if vllm_ok else "degraded",
        "model_loaded": vllm_ok,
        "adapter": None,
        "training_active": False,
        "gpu": None,
        "remote_endpoint": HF_ENDPOINT_URL,
        "remote_healthy": vllm_ok,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
