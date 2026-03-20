"""backend/tests/conftest.py — Set env vars and sys.path before any test module is imported."""

import os
import sys
from pathlib import Path

# ── Env vars must be set before project modules are imported ──────────────────
# Several modules (database.py, s3_uploader.py) read os.environ at module level.
# conftest.py is executed by pytest before test files are collected.
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://lora:lora@localhost:5432/lora")
os.environ.setdefault("S3_BUCKET", "test-bucket")
os.environ.setdefault("SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
os.environ.setdefault("MAX_SESSION_TOKENS", "4096")
os.environ.setdefault("PRE_SLEEP_THRESHOLD", "512")
os.environ.setdefault("MODEL_SERVER_URL", "http://localhost:8001")
os.environ.setdefault("BASE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

# ── sys.path — make all packages importable from the project root ─────────────
root = Path(__file__).parent.parent.parent   # lora-chat-train/
for p in [root, root / "backend", root / "shared", root / "training", root / "worker"]:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
