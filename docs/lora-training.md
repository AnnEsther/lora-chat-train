# LoRA Training

## Overview
This module handles the actual fine-tuning of the base LLM using Low-Rank Adaptation (LoRA). It supports two backends: a HuggingFace Inference Endpoint (for cloud training) and a local SFTTrainer (for GPU-equipped development machines).

## Key Files
- `training/trainer/hf_launcher.py` — `HFTrainingLauncher` class + `train_local()` function
- `backend/model_server/local_gpu_serve.py` — `_run_training()` for in-process training on local model server
- `worker/tasks.py` — `launch_training`, `poll_training` Celery tasks

## `HFTrainingLauncher` Class

### `build_config(run_id, session_id, ...)`
Constructs the full training config dict. All hyperparameters are overridable via env vars.

### `launch(config)`
1. If `HF_TRAINING_ENDPOINT` is set: submits job to HuggingFace Inference Endpoints, returns HF job ID
2. Otherwise: POSTs `{"run_id": ..., "dataset_path": ...}` to the local model server's `/train` endpoint

### `poll(hf_job_id)`
- Polls HF endpoint status
- Returns `"running" | "succeeded" | "failed"`

### `download_artifacts(hf_job_id, run_id)`
- Downloads ZIP from HF endpoint, extracts to temp directory
- Falls back to writing raw bytes as `adapter.bin` if extraction fails

## `train_local(config, dataset_path)` (standalone function)
Full local SFTTrainer run:
1. Load base model with 4-bit NF4 quantization
2. Apply `LoraConfig`
3. Load JSONL dataset via HuggingFace `datasets`
4. Run `SFTTrainer.train()`
5. Save adapter to output directory

Returns the output directory path.

## LoRA Hyperparameters
All env-overridable with these defaults:

| Env Var | Default | Description |
|---------|---------|-------------|
| `LORA_R` | `16` | LoRA rank |
| `LORA_ALPHA` | `32` | LoRA scaling factor |
| `LORA_DROPOUT` | `0.05` | LoRA dropout rate |
| `LORA_TARGET_MODULES` | `q_proj,v_proj` | Transformer modules to apply LoRA to |
| `TRAIN_EPOCHS` | `3` | Number of training epochs |
| `TRAIN_BATCH_SIZE` | `4` | Per-device training batch size |
| `TRAIN_GRAD_ACCUM` | `4` | Gradient accumulation steps (effective batch = 16) |
| `TRAIN_LR` | `2e-4` | Learning rate |
| `MAX_SEQ_LENGTH` | `512` | Maximum token sequence length |

## In-Process Training (`local_gpu_serve.py`)
The local GPU model server runs training in a daemon `threading.Thread`:
- `_model_lock` prevents inference during training (HTTP 503 returned)
- `_training_active` flag guards against concurrent training runs
- After training: automatically archives current adapter and hot-swaps the new one
- Progress tracked in `_training_status` dict (exposed via `GET /train/status`)

## Dataset Format
Training data is JSONL, each line containing:
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```
See [Dataset Writer doc](./dataset-writer.md) for details.

## Celery Tasks

### `launch_training`
- Creates `TrainingRun` DB record with status `PENDING` → `RUNNING`
- Calls `HFTrainingLauncher.launch(config)`
- Passes HF job ID to next task in chain

### `poll_training`
- Polls HF endpoint every 60 seconds
- Timeout: 60 retries = 1 hour maximum
- On success: downloads artifacts, uploads logs + adapter to S3
- On failure: marks run `FAILED`

## Configuration
| Env Var | Default | Description |
|---------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace API token |
| `HF_TRAINING_ENDPOINT` | — | HF endpoint URL; if absent, uses local model server |
| `BASE_MODEL` | — | Base model ID (e.g., `meta-llama/Llama-3.2-1B-Instruct`) |

## Change Log
<!-- Agents: append an entry here after every change -->
| Date | Change | Author |
|------|--------|--------|
| 2026-04-28 | Initial documentation created | opencode |
