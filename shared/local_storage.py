"""shared/local_storage.py — Local filesystem fallback when S3 is not configured."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

# Default output directory — override via env var
OUTPUT_DIR = Path(os.environ.get("LOCAL_OUTPUT_DIR", Path(__file__).parent.parent / "outputs"))


def _ensure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_bytes(data: bytes, key: str) -> str:
    path = _ensure(OUTPUT_DIR / key)
    path.write_bytes(data)
    return f"local://{path}"


def save_text(text: str, key: str) -> str:
    path = _ensure(OUTPUT_DIR / key)
    path.write_text(text, encoding="utf-8")
    return f"local://{path}"


def save_json(obj: Any, key: str) -> str:
    return save_text(json.dumps(obj, indent=2, default=str), key)


def save_file(local_path: Path, key: str) -> str:
    dest = _ensure(OUTPUT_DIR / key)
    shutil.copy2(local_path, dest)
    return f"local://{dest}"


def save_directory(local_dir: Path, prefix: str) -> list[str]:
    uris = []
    for p in sorted(local_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(local_dir)
            uris.append(save_file(p, f"{prefix.rstrip('/')}/{rel}"))
    return uris