#!/usr/bin/env bash
# =============================================================================
# entrypoint.sh — model_server container startup
#
# Downloads the HuggingFace base model to the hf_cache volume on first run.
# Subsequent starts skip the download because the cache already exists.
# Then execs the model server process (replacing this shell with the server).
# =============================================================================
set -euo pipefail

HF_CACHE_DIR="${HF_HOME:-/root/.cache/huggingface}"
# BASE_MODEL is set at runtime via env_file: .env in docker-compose.yml.
# The fallback here is only used when running the container directly (outside Compose).
# To change the model, edit .env or use ./start.sh which presents a model selector.
BASE_MODEL="${BASE_MODEL:-cognitivecomputations/dolphin-2.9-mistral-7b-v2}"
HF_TOKEN="${HF_TOKEN:-}"

# Derive the expected cache subdirectory from the model repo id.
# HuggingFace stores snapshots at:
#   ~/.cache/huggingface/hub/models--<org>--<name>/snapshots/
# We check for the existence of that directory to decide whether to download.
SAFE_MODEL_ID="$(echo "$BASE_MODEL" | tr '/' '--')"
SNAPSHOT_DIR="${HF_CACHE_DIR}/hub/models--${SAFE_MODEL_ID}/snapshots"

if [ -d "$SNAPSHOT_DIR" ] && [ "$(ls -A "$SNAPSHOT_DIR" 2>/dev/null)" ]; then
    echo "[entrypoint] Model cache found at ${SNAPSHOT_DIR} — skipping download."
else
    echo "[entrypoint] Model cache not found. Downloading ${BASE_MODEL} ..."
    echo "[entrypoint] This is a one-time download (~2.4 GB). Subsequent starts will be instant."

    DOWNLOAD_ARGS="--repo-id ${BASE_MODEL} --local-dir-use-symlinks False"

    # Pass token only if provided
    if [ -n "$HF_TOKEN" ]; then
        DOWNLOAD_ARGS="${DOWNLOAD_ARGS} --token ${HF_TOKEN}"
    fi

    # Ignore non-essential weight formats to save ~500 MB
    python3 -c "
import os, sys
from huggingface_hub import snapshot_download

token = os.environ.get('HF_TOKEN') or None
model = os.environ.get('BASE_MODEL', 'cognitivecomputations/dolphin-2.9-mistral-7b-v2')
cache = os.environ.get('HF_HOME', '/root/.cache/huggingface')

print(f'[entrypoint] Downloading {model} to {cache} ...', flush=True)
path = snapshot_download(
    repo_id=model,
    cache_dir=cache + '/hub',
    token=token,
    ignore_patterns=[
        '*.msgpack', '*.h5',
        'flax_model*', 'tf_model*', 'rust_model*',
        'onnx/*',
    ],
)
print(f'[entrypoint] Download complete: {path}', flush=True)
"
    echo "[entrypoint] Download finished."
fi

echo "[entrypoint] Starting model server: python -m model_server.local_gpu_serve"
exec python -m model_server.local_gpu_serve
