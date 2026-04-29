# scripts/resume_training.py
import sys, os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")
sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("backend").resolve()))

# Paste your session_id and run_id from the Celery logs
SESSION_ID = "ab0911a1-3768-47d9-acfd-521ca9446ce1"   # e.g. "b1dcc350-a003-4efc-99c5-98f8f9408754"
RUN_ID     = "0c3f8085-c671-4225-9d47-f0345630e654"       # e.g. "3c15a7cf-ce48-4feb-9c52-e5ee25b7992f"

# Find the dataset that was already built
output_dir = Path(os.environ.get("LOCAL_OUTPUT_DIR", "outputs"))
dataset_path = output_dir / "sessions" / SESSION_ID / "dataset" / "dataset.jsonl"

if not dataset_path.exists():
    print(f"ERROR: dataset not found at {dataset_path}")
    sys.exit(1)

print(f"Found dataset: {dataset_path} ({dataset_path.stat().st_size} bytes)")

from worker.tasks import launch_training, poll_training, run_evaluation, deploy_or_rollback
from celery import chain

prev = {
    "session_id": SESSION_ID,
    "run_id":     RUN_ID,
    "s3_path":    f"local://{dataset_path}",
    "local_path": str(dataset_path),
}

print("Requeueing pipeline from launch_training...")
(
    launch_training.s(prev, SESSION_ID, RUN_ID)
    | poll_training.s(SESSION_ID, RUN_ID)
    | run_evaluation.s(SESSION_ID, RUN_ID)
    | deploy_or_rollback.s(SESSION_ID, RUN_ID)
).apply_async()

print("Done — watch your Celery terminal for progress.")
# Fill in your IDs — you can find them in the Celery log lines:
# Task tasks.launch_training[3c15a7cf-ce48-4feb-9c52-e5ee25b7992f]

# check
# curl -X POST http://localhost:8001/train -H "Content-Type: application/json" -d "{\"run_id\": \"3c15a7cf-ce48-4feb-9c52-e5ee25b7992f\", \"dataset_path\": \"outputs/sessions/391d3009-a6e9-4b9a-804c-ce3c090f6534/dataset/dataset.jsonl\"}"