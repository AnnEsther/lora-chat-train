# Model Server

## Overview
The model server is a standalone FastAPI process that the backend talks to exclusively over HTTP. Three implementations exist — pick one at startup:

| File | Use case | GPU required |
|------|----------|-------------|
| `local_gpu_serve.py` | Local dev with RTX 4060; includes inline LoRA training | Yes |
| `hf_serve.py` | **Private HF Dedicated Inference Endpoint (vLLM)** — no local GPU needed | No |
| `serve.py` | Docker container deployment; inference + adapter hot-swap only | Yes |

All three expose **identical API endpoints** so `ModelClient`, the backend, Celery workers, and the frontend diagnostic panel work without any code changes regardless of which server is running.

## Key Files
- `backend/model_server/local_gpu_serve.py` — Primary server for local RTX 4060 (8 GB VRAM); includes inline training
- `backend/model_server/hf_serve.py` — Thin proxy to a **private** HF Dedicated Inference Endpoint (vLLM); no local GPU needed
- `backend/model_server/serve.py` — Container-oriented server; inference + adapter hot-swap only, no training
- `backend/model_client.py` — Async HTTP client used by the FastAPI backend to call the model server

## `hf_serve.py` — Private HF Endpoint Proxy (vLLM)

### How it works
```
ModelClient → POST /chat (hf_serve.py:8001) → POST /v1/chat/completions (HF vLLM endpoint)
```
vLLM returns OpenAI-format SSE (`choices[].delta.content`). `hf_serve.py` translates this to the internal format (`{"text": "..."}`) before yielding, so `ModelClient` needs no changes.

> **Note:** HF has deprecated TGI for new endpoints. Use **vLLM** (or SGLang) when creating new endpoints.

### Setup
1. Go to [huggingface.co/inference-endpoints](https://huggingface.co/inference-endpoints) and create a **New Endpoint**:
   - Model: your `BASE_MODEL` (e.g. `meta-llama/Llama-3.2-1B-Instruct`)
   - Framework: **vLLM**
   - Hardware: Nvidia L4 (24 GB) or larger
   - Visibility: **Private** (only your `HF_TOKEN` can access it)
2. Copy the endpoint URL into `.env`:
   ```ini
   HF_ENDPOINT_URL=https://your-endpoint-name.us-east-1.aws.endpoints.huggingface.cloud
   HF_ENDPOINT_MODEL=meta-llama/Llama-3.2-1B-Instruct  # must match BASE_MODEL
   ```
   > **Important:** vLLM requires the real HuggingFace model ID in the request body (not `"tgi"`). `HF_ENDPOINT_MODEL` defaults to `BASE_MODEL` so you only need to set it if they differ.
3. Start with:
   ```
   python backend/model_server/hf_serve.py
   ```
   instead of `python backend/model_server/local_gpu_serve.py`.

### Limitations
- **Training (`/train`)** — returns HTTP 501. Use `local_gpu_serve.py` or set `HF_TRAINING_ENDPOINT` for Celery-based training.
- **Adapter hot-swap (`/reload_adapter`)** — no-op stub; always logs a warning and returns `status: ok`. The vLLM endpoint always serves the model it was deployed with.
- **`/train/status`** — always returns `status: idle`.
- **`/health`** — proxies to vLLM's `/health` endpoint; returns `remote_healthy: true/false`.

### Configuration
| Env Var | Default | Description |
|---------|---------|-------------|
| `HF_ENDPOINT_URL` | — | Full URL of your private vLLM endpoint |
| `HF_TOKEN` | — | HF access token (write scope) |
| `HF_ENDPOINT_MODEL` | value of `BASE_MODEL` | Model ID in vLLM request body — must match deployed model |
| `MAX_NEW_TOKENS` | `512` | Default max tokens per response |
| `TEMPERATURE` | `0.7` | Default sampling temperature |

## Design Decisions
- **bf16 throughout** (`local_gpu_serve.py`) — matches Qwen2.5's native dtype; avoids bfloat16/fp16 mismatch errors
- **4-bit NF4 quantization** via `BitsAndBytesConfig` — fits 7B+ models within 8 GB VRAM
- **`_model_lock` (threading.Lock)** — protects all model state; inference returns HTTP 503 during active training
- **Training on daemon thread** — `threading.Thread(daemon=True)` so training does not block the HTTP server

## Global State (`local_gpu_serve.py`)
| Variable | Type | Description |
|----------|------|-------------|
| `_model` | `PreTrainedModel` | Loaded base model (with NF4 quant) |
| `_tokenizer` | `PreTrainedTokenizer` | Tokenizer for the base model |
| `_adapter_path` | `str \| None` | Path of the currently loaded PEFT adapter |
| `_training_active` | `bool` | Blocks inference and concurrent training when `True` |
| `_training_status` | `dict` | Live status: `status`, `progress`, timestamps, run_id |

## Endpoints (`local_gpu_serve.py`)
| Method | Path | Notes |
|--------|------|-------|
| `POST` | `/chat` | Streaming SSE inference; returns 503 during training |
| `POST` | `/generate` | Non-streaming single response; for eval and smoke tests |
| `POST` | `/train` | Launches `_run_training()` on a background thread |
| `GET` | `/train/status` | Training progress + VRAM stats |
| `POST` | `/reload_adapter` | Hot-swaps PEFT adapter; `"base"` unloads adapter |
| `GET` | `/adapters` | Lists adapters from `adapters/history/` + `adapters/current/` |
| `GET` | `/health` | Model loaded, adapter path, training flag, GPU info |

## Endpoints (`serve.py` — container server)
| Method | Path | Notes |
|--------|------|-------|
| `POST` | `/chat` | Streaming SSE inference |
| `POST` | `/generate` | Non-streaming single response |
| `POST` | `/reload_adapter` | Hot-swap PEFT adapter (thread-safe via `_model_lock`) |
| `GET` | `/health` | `{status, model_loaded, adapter}` |
| `GET` | `/adapters` | Lists base + `adapters/history/` + `adapters/current/`; reads `manifest.json` |
| `GET` | `/train/status` | Always returns `{status: "idle", ...}` (no in-process training in this variant) |

## Key Functions (`local_gpu_serve.py`)

### `_load_base_model()`
- Loads the base model from `BASE_MODEL` env var
- Applies `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=bfloat16, bnb_4bit_quant_type="nf4")`
- Sets `device_map="auto"` for multi-GPU spread

### `_load_adapter(adapter_dir)`
- If an adapter is already loaded: calls `model.merge_and_unload()` first to clear it
- Wraps model with `PeftModel.from_pretrained(model, adapter_dir)`
- Updates `_adapter_path`

### `_stream_tokens(prompt, max_new_tokens, temperature)`
- Creates a `TextIteratorStreamer` and launches tokenization + `model.generate()` in a background thread
- Returns `(streamer, thread)` — caller iterates the streamer for token strings

### `_run_training(run_id, dataset_path)`
1. Sets `_training_active = True`
2. Runs full LoRA SFT training via `SFTTrainer`
3. Saves adapter to `outputs/<run_id>/`
4. Archives current adapter via `_archive_current_adapter(run_id)`
5. Copies new adapter to `adapters/current/`, writes `manifest.json`
6. Hot-swaps to the new adapter
7. Sets `_training_active = False`

### `_archive_current_adapter(run_id)`
- Copies `adapters/current/` → `adapters/history/<version>/`
- Reads version from `manifest.json`; defaults to `run_id` if not found

## Adapter Storage Layout
```
adapters/
├── current/          ← active production adapter + manifest.json
└── history/
    └── <version>/    ← archived previous adapters
```

## Configuration
| Env Var | Default | Description |
|---------|---------|-------------|
| `BASE_MODEL` | `meta-llama/Llama-3.2-1B-Instruct` (serve.py) / `Qwen/Qwen2.5-1.5B-Instruct` (local_gpu_serve.py) | HuggingFace model ID |
| `MODEL_SERVER_URL` | — | URL backend uses to reach this server |
| `MAX_NEW_TOKENS` | `512` | Max tokens generated per response |
| `TEMPERATURE` | `0.7` | Sampling temperature |
| `ADAPTER_DIR` | `/adapters/current` | Production adapter path |
| `ADAPTER_HISTORY_DIR` | `/adapters/history` | Archive path for previous adapters |
| `HF_TOKEN` | — | Required by `serve.py` to load gated HF models |

## Change Log
<!-- Agents: append an entry here after every change -->
| Date | Change | Author |
|------|--------|--------|
| 2026-05-08 | Expand serve.py endpoint table to include /adapters and /train/status; expand configuration table with defaults, ADAPTER_HISTORY_DIR, HF_TOKEN | opencode |
| 2026-04-29 | Switched hf_serve.py from TGI to vLLM (TGI deprecated for new endpoints); HF_ENDPOINT_MODEL now defaults to BASE_MODEL | opencode |
| 2026-04-29 | Added hf_serve.py — thin proxy to private HF Dedicated Inference Endpoint; no local GPU needed for inference | opencode |
| 2026-04-28 | Initial documentation created | opencode |
