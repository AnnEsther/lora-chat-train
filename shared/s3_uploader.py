"""shared/s3_uploader.py — S3 uploads with local filesystem fallback."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BUCKET: str = os.environ.get("S3_BUCKET", "")
_client = None


def _s3_available() -> bool:
    return bool(BUCKET) and bool(os.environ.get("AWS_ACCESS_KEY_ID"))


def _get_client():
    global _client
    if _client is None:
        import boto3
        _client = boto3.client(
            "s3",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )
    return _client


# ── Core upload (with local fallback) ────────────────────────────────────────

def upload_bytes(
    data: bytes,
    key: str,
    content_type: str = "application/octet-stream",
    retries: int = 3,
    backoff: float = 1.5,
) -> str:
    if not _s3_available():
        from shared.local_storage import save_bytes
        uri = save_bytes(data, key)
        logger.info("local_save_ok", extra={"key": key, "bytes": len(data)})
        return uri

    from botocore.exceptions import BotoCoreError, ClientError
    s3 = _get_client()
    for attempt in range(1, retries + 1):
        try:
            s3.put_object(Bucket=BUCKET, Key=key, Body=data, ContentType=content_type)
            uri = f"s3://{BUCKET}/{key}"
            logger.info("s3_upload_ok", extra={"key": key, "bytes": len(data)})
            return uri
        except (BotoCoreError, ClientError) as exc:
            logger.warning("s3_upload_retry",
                           extra={"key": key, "attempt": attempt, "error": str(exc)})
            if attempt == retries:
                raise
            time.sleep(backoff ** attempt)
    raise RuntimeError("unreachable")


def upload_text(text: str, key: str, content_type: str = "text/plain") -> str:
    return upload_bytes(text.encode(), key, content_type)


def upload_json(obj: Any, key: str) -> str:
    return upload_bytes(
        json.dumps(obj, indent=2, default=str).encode(), key, "application/json"
    )


def upload_file(local_path: Path | str, key: str, content_type: str | None = None) -> str:
    local_path = Path(local_path)
    if not _s3_available():
        from shared.local_storage import save_file
        return save_file(local_path, key)
    suffix = local_path.suffix.lower()
    content_type = content_type or {
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".txt": "text/plain",
        ".log": "text/plain",
    }.get(suffix, "application/octet-stream")
    return upload_bytes(local_path.read_bytes(), key, content_type)


def upload_directory(local_dir: Path | str, prefix: str) -> list[str]:
    local_dir = Path(local_dir)
    if not _s3_available():
        from shared.local_storage import save_directory
        return save_directory(local_dir, prefix)
    uris = []
    for p in sorted(local_dir.rglob("*")):
        if p.is_file():
            key = f"{prefix.rstrip('/')}/{p.relative_to(local_dir)}"
            uris.append(upload_file(p, key))
    return uris


# ── S3 prefix helpers ─────────────────────────────────────────────────────────

def session_prefix(session_id: str, stage: str) -> str:
    return f"sessions/{session_id}/{stage}/"


def run_prefix(run_id: str, stage: str) -> str:
    return f"training_runs/{run_id}/{stage}/"


def production_prefix(sub: str = "current") -> str:
    return f"production/{sub}/"


# ── Named stage uploads ───────────────────────────────────────────────────────

def upload_raw_transcript(session_id: str, transcript: list[dict]) -> str:
    key = session_prefix(session_id, "raw") + "transcript.json"
    return upload_json({"session_id": session_id, "turns": transcript}, key)


def upload_candidates(session_id: str, candidates: list[dict]) -> str:
    key = session_prefix(session_id, "candidates") + "candidates.json"
    return upload_json({"session_id": session_id, "candidates": candidates}, key)


def upload_curated(session_id: str, curated: list[dict]) -> str:
    key = session_prefix(session_id, "curated") + "curated.json"
    return upload_json({"session_id": session_id, "curated": curated}, key)


def upload_dataset_jsonl(session_id: str, jsonl_text: str) -> str:
    key = session_prefix(session_id, "dataset") + "dataset.jsonl"
    return upload_text(jsonl_text, key, "application/x-ndjson")


def upload_training_config(run_id: str, config: dict) -> str:
    key = run_prefix(run_id, "config") + "config.json"
    return upload_json(config, key)


def upload_training_logs(run_id: str, logs: str) -> str:
    key = run_prefix(run_id, "logs") + "training.log"
    return upload_text(logs, key)


def upload_adapter(run_id: str, adapter_dir: Path) -> str:
    prefix = run_prefix(run_id, "artifacts")
    upload_directory(adapter_dir, prefix)
    if _s3_available():
        return f"s3://{BUCKET}/{prefix}"
    from shared.local_storage import OUTPUT_DIR
    return f"local://{OUTPUT_DIR / prefix}"


def upload_eval_report(run_id: str, report: dict) -> str:
    key = run_prefix(run_id, "eval") + "eval_report.json"
    return upload_json(report, key)


def upload_deployment_manifest(run_id: str, manifest: dict) -> str:
    key = production_prefix("current") + "manifest.json"
    return upload_json(manifest, key)


def upload_rollback_manifest(run_id: str, manifest: dict) -> str:
    key = production_prefix(f"history/{run_id}") + "rollback_manifest.json"
    return upload_json(manifest, key)


def sync_adapter_to_production(run_id: str, adapter_dir: Path) -> str:
    prefix = production_prefix("current")
    upload_directory(adapter_dir, prefix)
    upload_directory(adapter_dir, production_prefix(f"history/{run_id}"))
    if _s3_available():
        return f"s3://{BUCKET}/{prefix}"
    from shared.local_storage import OUTPUT_DIR
    return f"local://{OUTPUT_DIR / prefix}"