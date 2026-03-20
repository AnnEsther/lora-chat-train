"""training/deployment/deploy.py — Promote LoRA adapter to production, smoke test, and rollback."""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import requests

from shared.s3_uploader import (
    sync_adapter_to_production,
    upload_deployment_manifest,
    upload_rollback_manifest,
)

logger = logging.getLogger(__name__)

MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://model_server:8001")
PRODUCTION_ADAPTER_DIR = Path(os.environ.get("ADAPTER_DIR", "/adapters/current"))
HISTORY_DIR = Path(os.environ.get("ADAPTER_HISTORY_DIR", "/adapters/history"))

# Smoke test prompts that the new adapter must respond to non-emptily
SMOKE_TEST_PROMPTS = [
    "Hello, are you working?",
    "What is 2 + 2?",
]


class DeploymentManager:
    """
    Handles the full promote → smoke-test → rollback lifecycle.
    Manages the production adapter pointer on both local disk and S3.
    """

    def get_current_production_version(self) -> str | None:
        """Return the version tag of the current production adapter, or None."""
        manifest_path = PRODUCTION_ADAPTER_DIR / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            return manifest.get("version")
        return None

    def promote(self, run_id: str, adapter_dir: str | Path) -> None:
        """
        Copy the new adapter into the production slot and update S3.
        Also archives the previous adapter for rollback.
        """
        adapter_dir = Path(adapter_dir)
        version = run_id[:8]

        # Archive current production adapter (if any)
        if PRODUCTION_ADAPTER_DIR.exists():
            prev_version = self.get_current_production_version()
            if prev_version:
                archive_path = HISTORY_DIR / prev_version
                archive_path.mkdir(parents=True, exist_ok=True)
                for item in PRODUCTION_ADAPTER_DIR.iterdir():
                    dest = archive_path / item.name
                    if item.is_file():
                        shutil.copy2(item, dest)
                logger.info("adapter_archived", extra={"version": prev_version})

        # Overwrite production slot
        PRODUCTION_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
        for item in adapter_dir.iterdir():
            dest = PRODUCTION_ADAPTER_DIR / item.name
            if item.is_file():
                shutil.copy2(item, dest)
            elif item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)

        # Write manifest
        manifest = {
            "version": version,
            "run_id": run_id,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }
        (PRODUCTION_ADAPTER_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # Sync to S3
        s3_path = sync_adapter_to_production(run_id, PRODUCTION_ADAPTER_DIR)
        upload_deployment_manifest(run_id, manifest)

        # Hot-reload in model server
        self._reload_model_server()

        logger.info("adapter_promoted", extra={"version": version, "s3_path": s3_path})

    def rollback(self, to_version: str | None) -> None:
        """Restore a previous adapter from the history directory."""
        if to_version is None:
            logger.warning("rollback_no_previous_version — nothing to roll back to")
            return

        archive_path = HISTORY_DIR / to_version
        if not archive_path.exists():
            logger.error("rollback_archive_missing", extra={"to_version": to_version})
            raise FileNotFoundError(f"No archived adapter for version {to_version}")

        # Overwrite production slot with archived adapter
        PRODUCTION_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
        for item in archive_path.iterdir():
            dest = PRODUCTION_ADAPTER_DIR / item.name
            if item.is_file():
                shutil.copy2(item, dest)

        # Update manifest
        manifest = {
            "version": to_version,
            "rolled_back_at": datetime.now(timezone.utc).isoformat(),
        }
        (PRODUCTION_ADAPTER_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # Sync rollback state to S3
        upload_rollback_manifest("rollback", manifest)
        sync_adapter_to_production("rollback", PRODUCTION_ADAPTER_DIR)

        # Hot-reload in model server
        self._reload_model_server()

        logger.info("rollback_complete", extra={"to_version": to_version})

    def smoke_test(self) -> bool:
        """
        Send a few simple prompts to the model server and verify non-empty responses.
        Returns True if all prompts return coherent output.
        """
        logger.info("smoke_test_start")
        try:
            for prompt in SMOKE_TEST_PROMPTS:
                resp = requests.post(
                    f"{MODEL_SERVER_URL}/generate",
                    json={"prompt": prompt, "max_new_tokens": 50},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                response_text = data.get("response", "")
                if len(response_text.strip()) < 3:
                    logger.warning("smoke_test_empty_response", extra={"prompt": prompt})
                    return False

            logger.info("smoke_test_passed")
            return True

        except requests.RequestException as exc:
            logger.error("smoke_test_error", extra={"error": str(exc)})
            return False

    def _reload_model_server(self) -> None:
        """Signal the model server to hot-reload the adapter from disk."""
        try:
            resp = requests.post(
                f"{MODEL_SERVER_URL}/reload_adapter",
                json={"adapter_dir": str(PRODUCTION_ADAPTER_DIR)},
                timeout=60,
            )
            resp.raise_for_status()
            logger.info("model_server_reload_ok")
        except requests.RequestException as exc:
            logger.error("model_server_reload_failed", extra={"error": str(exc)})
            raise
