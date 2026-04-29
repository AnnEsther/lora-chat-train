# Model Server

## Overview
The model server is a standalone FastAPI process that owns the GPU. It loads the base LLM (and optionally a LoRA adapter), serves streaming inference, and runs LoRA training on a background thread. The backend talks to it exclusively over HTTP.

## Key Files
- `backend/model_server/local_gpu_serve.py` — Primary server for local RTX 4060 (8 GB VRAM); includes inline training
- `backend/model_server/serve.py` — Simpler container-oriented server; inference + adapter hot-swap only, no training
- `backend/model_client.py` — Async HTTP client used by the FastAPI backend to call the model server

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
| `POST` | `/chat` | Streaming SSE |
| `POST` | `/generate` | Non-streaming |
| `POST` | `/reload_adapter` | Hot-swap adapter |
| `GET` | `/health` | Model state |

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
| `BASE_MODEL` | — | HuggingFace model ID (e.g., `meta-llama/Llama-3.2-1B-Instruct`) |
| `MODEL_SERVER_URL` | — | URL backend uses to reach this server |
| `MAX_NEW_TOKENS` | — | Max tokens generated per response |
| `TEMPERATURE` | — | Sampling temperature |
| `ADAPTER_DIR` | `/adapters/current` | Production adapter path |

## Change Log
<!-- Agents: append an entry here after every change -->
| Date | Change | Author |
|------|--------|--------|
| 2026-04-28 | Initial documentation created | opencode |
