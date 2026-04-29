# HuggingFace Training Hosting

## Overview
LoRA fine-tuning can be offloaded to a HuggingFace Inference Endpoint instead of running on the local GPU. When `HF_TRAINING_ENDPOINT` is set, the Celery `launch_training` task submits the training job to HF and polls it until completion. When the variable is absent, the system falls back to posting the job to the local model server's `/train` endpoint.

## Key Files
- `training/trainer/hf_launcher.py` — `HFTrainingLauncher` class; all HF HTTP calls
- `worker/tasks.py` — `launch_training` and `poll_training` Celery tasks
- `.env.example` — Reference for all required environment variables

---

## Prerequisites

### 1. HuggingFace Account & API Token
1. Create an account at [huggingface.co](https://huggingface.co)
2. Go to **Settings → Access Tokens**
3. Create a token with **write** scope (needed to access private models and endpoints)
4. Copy the token — it starts with `hf_`

### 2. Base Model Access
If your `BASE_MODEL` is a gated model (e.g., `meta-llama/Llama-3.2-1B-Instruct`):
1. Go to the model page on HuggingFace
2. Accept the license agreement to request access
3. Wait for approval (usually instant for Llama 3.2)

### 3. HuggingFace Inference Endpoint
The system expects an **Inference Endpoint** that can accept a custom training payload. This is most straightforwardly set up as an **AutoTrain** endpoint or a custom endpoint container that exposes the expected API (see [API Contract](#api-contract) below).

#### Creating an Endpoint (HF Console)
1. Go to [huggingface.co/inference-endpoints](https://huggingface.co/inference-endpoints)
2. Click **New Endpoint**
3. Select your base model (e.g., `meta-llama/Llama-3.2-1B-Instruct`)
4. Choose a GPU instance — recommended minimum: **A10G (24 GB VRAM)** for a 1B model with fp16; **A100 (80 GB)** for 7B+
5. Set **Endpoint type** to **Private**
6. Note the endpoint URL — it will look like:
   ```
   https://api.endpoints.huggingface.co/v2/endpoint/your-namespace/your-endpoint
   ```
7. Copy this full URL into `HF_TRAINING_ENDPOINT`

---

## Environment Variables

Set these in your `.env` file (see `.env.example`):

```ini
# Required
HF_TOKEN=hf_your_token_here
HF_TRAINING_ENDPOINT=https://api.endpoints.huggingface.co/v2/endpoint/your-namespace/your-endpoint
BASE_MODEL=meta-llama/Llama-3.2-1B-Instruct

# Optional — LoRA hyperparameters (all have defaults)
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES=q_proj,v_proj

# Optional — SFT training arguments (all have defaults)
TRAIN_EPOCHS=3
TRAIN_BATCH_SIZE=4
TRAIN_GRAD_ACCUM=4
TRAIN_LR=2e-4
MAX_SEQ_LENGTH=512
```

When `HF_TRAINING_ENDPOINT` is **absent or empty**, the system automatically falls back to posting the training job to the local model server (`MODEL_SERVER_URL/train`). No code change is required to switch modes.

---

## How It Works

### Step 1 — Dataset is Built (Phase 2 starts)
The `build_dataset` Celery task writes the curated conversation pairs as a JSONL file to S3 (or local fallback). The S3 path and local path are passed to `launch_training`.

### Step 2 — Training Config is Built (`build_config`)
`HFTrainingLauncher.build_config()` assembles a config dict containing:
- `run_id` and `session_id` — for traceability
- `base_model` — HF model ID from `BASE_MODEL` env var
- `dataset_s3_path` and `dataset_local_path` — where the JSONL lives
- `lora` block — all LoRA hyperparameters
- `training` block — all SFT training arguments

Full shape:
```json
{
  "run_id": "...",
  "session_id": "...",
  "base_model": "meta-llama/Llama-3.2-1B-Instruct",
  "dataset_s3_path": "s3://bucket/sessions/.../dataset.jsonl",
  "dataset_local_path": "/path/to/dataset.jsonl",
  "lora": {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM"
  },
  "training": {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 0.0002,
    "max_seq_length": 512,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "fp16": true,
    "optim": "paged_adamw_8bit",
    "logging_steps": 10,
    "save_steps": 50,
    "output_dir": "/tmp/lora_run_<run_id>"
  }
}
```

### Step 3 — Job Submitted (`launch`)
`HFTrainingLauncher.launch(config)` is called. With `HF_TRAINING_ENDPOINT` set it should POST the config to the endpoint. See [Known Bug](#known-bug) below — the HF POST call is currently missing from the code; only the local fallback path is implemented.

Auth header sent with every HF request:
```
Authorization: Bearer <HF_TOKEN>
Content-Type: application/json
```

The HF job ID returned by the endpoint is stored in the `training_runs` DB table as `hf_job_id`.

### Step 4 — Polling (`poll_training` Celery task)
The Celery task retries every **60 seconds**, up to **60 retries** (maximum 1 hour total).

Each poll makes:
```
GET {HF_TRAINING_ENDPOINT}/{hf_job_id}
Authorization: Bearer {HF_TOKEN}
```

Expected response JSON: `{"status": "running" | "completed" | "succeeded" | "done" | "failed" | "error" | "cancelled"}`

Status mapping:
| HF status | Internal result |
|-----------|----------------|
| `completed`, `succeeded`, `done` | `"succeeded"` |
| `failed`, `error`, `cancelled` | `"failed"` |
| anything else | `"running"` (retry) |
| network error | `"running"` (retry — transient assumed) |

### Step 5 — Artifact Download (`download_artifacts`)
When polling returns `"succeeded"`:
```
GET {HF_TRAINING_ENDPOINT}/{hf_job_id}/artifacts
Authorization: Bearer {HF_TOKEN}
```
Timeout: 120 seconds, streamed.

- If response is `200` and `Content-Type` contains `"zip"`: extracted as a ZIP archive
- Otherwise: raw bytes written to `adapter.bin` in a temp directory

The downloaded adapter directory is then uploaded to S3 and passed to the evaluation task.

### Step 6 — Logs Retrieved (`get_logs`)
```
GET {HF_TRAINING_ENDPOINT}/{hf_job_id}/logs
```
Response capped at 50,000 characters. Uploaded to S3 as `training_runs/{run_id}/logs/training.log`.

---

## API Contract

Your HF endpoint must implement these routes:

| Method | Path | Request | Response |
|--------|------|---------|----------|
| `POST` | `/` (root) | Training config JSON (see shape above) | `{"job_id": "<id>"}` |
| `GET` | `/{job_id}` | — | `{"status": "running\|succeeded\|failed\|..."}` |
| `GET` | `/{job_id}/artifacts` | — | ZIP file or raw adapter bytes |
| `GET` | `/{job_id}/logs` | — | Plain text training logs |
| `GET` | `/{job_id}` | — | `{"error": "..."}` field when status is failed |

All requests authenticated with `Authorization: Bearer {HF_TOKEN}`.

---

## Known Bug

**`HFTrainingLauncher.launch()` is incomplete for the HF path.**

In `training/trainer/hf_launcher.py`, the `launch()` method only implements the local fallback (`if not HF_TRAINING_ENDPOINT:`). When `HF_TRAINING_ENDPOINT` *is* set, the function falls through the `if` block and returns `None` implicitly. This means:

- The training config is **never POSTed** to the HF endpoint
- `hf_job_id` becomes `None` in the DB
- `poll_training` will attempt `GET {HF_TRAINING_ENDPOINT}/None` and loop until timeout

**Fix required** — add the HF submission branch to `launch()`:

```python
def launch(self, config: dict) -> str:
    if not HF_TRAINING_ENDPOINT:
        # local fallback
        model_server = os.environ.get("MODEL_SERVER_URL", "http://localhost:8001")
        resp = requests.post(
            f"{model_server}/train",
            json={"run_id": config["run_id"], "dataset_path": config.get("dataset_local_path", "")},
            timeout=30,
        )
        resp.raise_for_status()
        return f"local_{config['run_id']}"

    # HF endpoint submission  ← THIS BRANCH IS MISSING AND NEEDS TO BE ADDED
    resp = requests.post(
        HF_TRAINING_ENDPOINT,
        headers=self.headers,
        json=config,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["job_id"]
```

---

## Hyperparameter Defaults Reference

Both `hf_launcher.py` and `local_gpu_serve.py` read from the same env vars, but use **different defaults** tuned for their respective hardware:

| Env Var | HF Launcher default | Local GPU default | Notes |
|---------|--------------------|--------------------|-------|
| `BASE_MODEL` | `meta-llama/Llama-3.2-1B-Instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | Set explicitly to ensure consistency |
| `LORA_R` | `16` | `16` | Higher values = more parameters = better fit, slower training |
| `LORA_ALPHA` | `32` | `32` | Typically `2 × LORA_R` |
| `LORA_DROPOUT` | `0.05` | `0.05` | |
| `LORA_TARGET_MODULES` | `q_proj,v_proj` | `q_proj,v_proj` | Comma-separated; model-architecture dependent |
| `TRAIN_EPOCHS` | `3` | `3` | |
| `TRAIN_BATCH_SIZE` | `4` | `1` | Local GPU reduced for 8 GB VRAM |
| `TRAIN_GRAD_ACCUM` | `4` | `8` | Local GPU increases accum to compensate for smaller batch |
| `TRAIN_LR` | `2e-4` | `2e-4` | |
| `MAX_SEQ_LENGTH` | `512` | `256` | Local GPU reduced for memory |

> The local model server uses `bf16=True, fp16=False`. The HF launcher config sends `fp16=True`. If your HF endpoint runs the same `_run_training` code as the local server, this will be overridden by the server's own config. If your HF endpoint is a separate service, ensure dtype consistency with the `base_model`'s native dtype.

---

## Recommended GPU Tiers for HF Endpoints

| Model size | Recommended HF instance | VRAM | Est. training time (3 epochs, 50 samples) |
|-----------|------------------------|------|------------------------------------------|
| 1B | NVIDIA A10G (24 GB) | 24 GB | ~5–10 min |
| 3B | NVIDIA A10G (24 GB) | 24 GB | ~15–25 min |
| 7B | NVIDIA A100 (40 GB) | 40 GB | ~30–60 min |
| 13B | NVIDIA A100 (80 GB) | 80 GB | ~60–90 min |

All sizes assume 4-bit NF4 quantization and LoRA (not full fine-tuning).

---

## Switching Between Local and HF Training

The only change needed is setting or clearing `HF_TRAINING_ENDPOINT` in `.env`:

```ini
# Use HuggingFace endpoint
HF_TRAINING_ENDPOINT=https://api.endpoints.huggingface.co/v2/endpoint/...

# Use local model server (comment out or leave empty)
# HF_TRAINING_ENDPOINT=
```

No code changes are required. The `launch()` function checks the variable at call time.

---

## Troubleshooting

### Training never starts / `hf_job_id` is `None`
The HF submission branch in `launch()` is not implemented (see [Known Bug](#known-bug)). Fix the method or use local training.

### `401 Unauthorized` from HF endpoint
- Check `HF_TOKEN` is set and has **write** scope
- Confirm the endpoint is **Private** and the token has access

### `404` on artifact download
- The endpoint may store artifacts at a different path
- Check HF endpoint documentation for your specific endpoint type
- As a workaround, download artifacts manually and place them in `adapters/current/`

### Poll timeout (60 retries exceeded)
Training took longer than 1 hour. Options:
- Increase `max_retries` on the `poll_training` task in `worker/tasks.py`
- Reduce dataset size or number of epochs
- Upgrade to a larger GPU instance

### `INSUFFICIENT_DATA` — training never triggered
Curation filtered out too many samples. See [Curation doc](./curation.md). Check `MIN_TRAINING_SAMPLES` env var (default `10`).

---

## Change Log
<!-- Agents: append an entry here after every change -->
| Date | Change | Author |
|------|--------|--------|
| 2026-04-28 | Initial documentation created; includes known bug in launch() HF path | opencode |
