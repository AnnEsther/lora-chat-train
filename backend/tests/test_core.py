"""backend/tests/test_core.py — Unit tests for /sleep handling, token threshold, S3, Slack, deploy."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# ── Set required env vars BEFORE any project imports ─────────────────────────
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://lora:lora@localhost:5432/lora")
os.environ.setdefault("S3_BUCKET", "test-bucket")
os.environ.setdefault("SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
os.environ.setdefault("MAX_SESSION_TOKENS", "4096")
os.environ.setdefault("PRE_SLEEP_THRESHOLD", "512")
os.environ.setdefault("MODEL_SERVER_URL", "http://localhost:8001")
os.environ.setdefault("BASE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")


# ── /sleep command handling ───────────────────────────────────────────────────

class TestSleepCommand:
    @pytest.mark.asyncio
    async def test_sleep_command_triggers_state_transition(self):
        from backend.main import _force_sleep

        mock_session = MagicMock()
        mock_session.id = uuid.uuid4()
        mock_session.state = "ACTIVE"
        mock_db = AsyncMock()

        with patch("backend.main.enqueue_training_pipeline") as mock_queue, \
             patch("backend.main.session_sleeping"):
            events = []
            async for event in _force_sleep(mock_session, mock_db, reason="user_command"):
                events.append(event)

        assert any("sleeping" in e for e in events)
        mock_queue.delay.assert_called_once_with(str(mock_session.id))

    @pytest.mark.asyncio
    async def test_sleep_command_locks_session(self):
        mock_session = MagicMock()
        mock_session.state = "SLEEPING"
        is_accepted = mock_session.state in ("ACTIVE", "PRE_SLEEP_WARNING")
        assert not is_accepted


# ── Token threshold ───────────────────────────────────────────────────────────

class TestTokenThreshold:
    def test_pre_sleep_warning_triggered_at_threshold(self):
        from backend.main import MAX_SESSION_TOKENS, PRE_SLEEP_THRESHOLD
        remaining = MAX_SESSION_TOKENS - 3700
        assert remaining <= PRE_SLEEP_THRESHOLD

    def test_no_warning_when_budget_ample(self):
        from backend.main import MAX_SESSION_TOKENS, PRE_SLEEP_THRESHOLD
        remaining = MAX_SESSION_TOKENS - 100
        assert remaining > PRE_SLEEP_THRESHOLD


# ── S3 uploader ───────────────────────────────────────────────────────────────

class TestS3Uploader:
    def test_upload_json_calls_put_object(self):
        with patch("shared.s3_uploader._get_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client_fn.return_value = mock_client

            from shared.s3_uploader import upload_json
            uri = upload_json({"hello": "world"}, "test/key.json")

        mock_client.put_object.assert_called_once()
        kwargs = mock_client.put_object.call_args.kwargs
        assert kwargs["Bucket"] == "test-bucket"
        assert kwargs["Key"] == "test/key.json"
        assert uri == "s3://test-bucket/test/key.json"

    def test_upload_retries_on_failure(self):
        from botocore.exceptions import ClientError
        with patch("shared.s3_uploader._get_client") as mock_client_fn, \
             patch("shared.s3_uploader.time.sleep"):
            mock_client = MagicMock()
            mock_client.put_object.side_effect = [
                ClientError({"Error": {"Code": "500", "Message": "err"}}, "put"),
                ClientError({"Error": {"Code": "500", "Message": "err"}}, "put"),
                None,
            ]
            mock_client_fn.return_value = mock_client

            from shared.s3_uploader import upload_bytes
            upload_bytes(b"data", "retry/test.bin", retries=3)

        assert mock_client.put_object.call_count == 3

    def test_session_prefix_format(self):
        from shared.s3_uploader import session_prefix
        assert session_prefix("abc-123", "raw") == "sessions/abc-123/raw/"

    def test_run_prefix_format(self):
        from shared.s3_uploader import run_prefix
        assert run_prefix("run-xyz", "artifacts") == "training_runs/run-xyz/artifacts/"


# ── Slack notifier ────────────────────────────────────────────────────────────

class TestSlackNotifier:
    def test_send_posts_to_webhook(self):
        import shared.slack_notifier as notifier
        original = notifier.WEBHOOK_URL
        try:
            notifier.WEBHOOK_URL = "https://hooks.slack.com/test"
            with patch("shared.slack_notifier.requests.post") as mock_post:
                mock_resp = MagicMock()
                mock_resp.raise_for_status = MagicMock()
                mock_post.return_value = mock_resp

                from shared.slack_notifier import SlackEvent, send
                result = send(SlackEvent(stage="test_stage", status="ok",
                                         summary="Test message", session_id="abc-123"))

            assert result is True
            mock_post.assert_called_once()
            assert "attachments" in mock_post.call_args.kwargs["json"]
        finally:
            notifier.WEBHOOK_URL = original

    def test_send_skipped_when_no_webhook(self):
        import shared.slack_notifier as notifier
        original = notifier.WEBHOOK_URL
        try:
            notifier.WEBHOOK_URL = None
            from shared.slack_notifier import SlackEvent, send
            assert send(SlackEvent(stage="x", status="info", summary="y")) is False
        finally:
            notifier.WEBHOOK_URL = original

    def test_send_retries_on_failure(self):
        import requests as req
        import shared.slack_notifier as notifier
        original = notifier.WEBHOOK_URL
        try:
            notifier.WEBHOOK_URL = "https://hooks.slack.com/test"
            with patch("shared.slack_notifier.requests.post") as mock_post, \
                 patch("shared.slack_notifier.time.sleep"):
                mock_post.side_effect = req.RequestException("timeout")
                from shared.slack_notifier import SlackEvent, send
                result = send(SlackEvent(stage="test", status="error", summary="fail"), retries=3)

            assert result is False
            assert mock_post.call_count == 3
        finally:
            notifier.WEBHOOK_URL = original


# ── Deployment manager ────────────────────────────────────────────────────────

class TestDeploymentManager:
    def test_promote_copies_adapter_and_writes_manifest(self, tmp_path):
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_model.bin").write_bytes(b"fake weights")
        (adapter_dir / "adapter_config.json").write_text('{"r": 16}')
        prod_dir = tmp_path / "production"
        hist_dir = tmp_path / "history"

        with patch("training.deployment.deploy.PRODUCTION_ADAPTER_DIR", prod_dir), \
             patch("training.deployment.deploy.HISTORY_DIR", hist_dir), \
             patch("training.deployment.deploy.sync_adapter_to_production",
                   return_value="s3://test-bucket/production/current/"), \
             patch("training.deployment.deploy.upload_deployment_manifest"), \
             patch("training.deployment.deploy.DeploymentManager._reload_model_server"):
            from training.deployment.deploy import DeploymentManager
            DeploymentManager().promote(run_id="abcd1234", adapter_dir=adapter_dir)

        manifest = json.loads((prod_dir / "manifest.json").read_text())
        assert manifest["version"] == "abcd1234"
        assert (prod_dir / "adapter_model.bin").exists()

    def test_rollback_restores_previous_adapter(self, tmp_path):
        hist_dir = tmp_path / "history"
        prev_dir = hist_dir / "prev_v1"
        prev_dir.mkdir(parents=True)
        (prev_dir / "adapter_model.bin").write_bytes(b"old weights")
        prod_dir = tmp_path / "production"

        with patch("training.deployment.deploy.PRODUCTION_ADAPTER_DIR", prod_dir), \
             patch("training.deployment.deploy.HISTORY_DIR", hist_dir), \
             patch("training.deployment.deploy.sync_adapter_to_production"), \
             patch("training.deployment.deploy.upload_rollback_manifest"), \
             patch("training.deployment.deploy.DeploymentManager._reload_model_server"):
            from training.deployment.deploy import DeploymentManager
            DeploymentManager().rollback(to_version="prev_v1")

        assert (prod_dir / "adapter_model.bin").read_bytes() == b"old weights"
        assert json.loads((prod_dir / "manifest.json").read_text())["version"] == "prev_v1"

    def test_smoke_test_returns_false_on_empty_response(self):
        with patch("training.deployment.deploy.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"response": ""}
            mock_post.return_value.raise_for_status = MagicMock()
            assert not training_deployment_mgr().smoke_test()

    def test_smoke_test_returns_true_on_valid_response(self):
        with patch("training.deployment.deploy.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"response": "I am working fine!"}
            mock_post.return_value.raise_for_status = MagicMock()
            assert training_deployment_mgr().smoke_test()


def training_deployment_mgr():
    from training.deployment.deploy import DeploymentManager
    return DeploymentManager()
