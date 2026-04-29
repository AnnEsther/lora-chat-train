"""shared/slack_notifier.py — Send structured Slack webhook notifications at every pipeline stage."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import requests

logger = logging.getLogger(__name__)

WEBHOOK_URL: str | None = os.environ.get("SLACK_WEBHOOK_URL")


@dataclass
class SlackEvent:
    stage: str
    status: str  # ok | warn | error | info
    summary: str
    session_id: str | None = None
    run_id: str | None = None
    model_version: str | None = None
    adapter_version: str | None = None
    s3_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


_STATUS_EMOJI = {
    "ok": ":white_check_mark:",
    "warn": ":warning:",
    "error": ":x:",
    "info": ":information_source:",
}

_STAGE_COLORS = {
    "ok": "#36a64f",
    "warn": "#ffcc00",
    "error": "#cc0000",
    "info": "#439fe0",
}


def _build_payload(event: SlackEvent) -> dict:
    emoji = _STATUS_EMOJI.get(event.status, ":robot_face:")
    color = _STAGE_COLORS.get(event.status, "#999999")
    fields = []

    if event.session_id:
        fields.append(
            {"title": "Session", "value": event.session_id[:8] + "…", "short": True}
        )
    if event.run_id:
        fields.append({"title": "Run", "value": event.run_id[:8] + "…", "short": True})
    if event.model_version:
        fields.append({"title": "Model", "value": event.model_version, "short": True})
    if event.adapter_version:
        fields.append(
            {"title": "Adapter", "value": event.adapter_version, "short": True}
        )
    if event.s3_path:
        fields.append({"title": "S3 artifact", "value": event.s3_path, "short": False})
    for k, v in event.extra.items():
        fields.append({"title": k, "value": str(v), "short": True})

    return {
        "attachments": [
            {
                "color": color,
                "fallback": f"{emoji} [{event.stage}] {event.summary}",
                "title": f"{emoji} {event.stage.replace('_', ' ').title()}",
                "text": event.summary,
                "fields": fields,
                "footer": "LoRA Chat & Train",
                "ts": int(time.time()),
            }
        ]
    }


def send(event: SlackEvent, retries: int = 3, backoff: float = 1.5) -> bool:
    """Send a Slack notification. Returns True on success."""
    if not WEBHOOK_URL:
        logger.debug("slack_skip_no_webhook", extra=asdict(event))
        return False

    payload = _build_payload(event)

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                WEBHOOK_URL,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            logger.info(
                "slack_sent", extra={"stage": event.stage, "status": event.status}
            )
            return True
        except requests.RequestException as exc:
            logger.warning(
                "slack_retry",
                extra={"attempt": attempt, "stage": event.stage, "error": str(exc)},
            )
            if attempt == retries:
                logger.error(
                    "slack_failed", extra={"stage": event.stage, "error": str(exc)}
                )
                return False
            time.sleep(backoff**attempt)
    return False


# ── Convenience wrappers for each pipeline stage ──────────────────────────────


def session_started(session_id: str) -> None:
    send(
        SlackEvent(
            stage="session_started",
            status="info",
            summary="New chat session is now active.",
            session_id=session_id,
        )
    )


def pre_sleep_warning(session_id: str, tokens_remaining: int) -> None:
    send(
        SlackEvent(
            stage="pre_sleep_warning",
            status="warn",
            summary=f"Session nearing token limit — {tokens_remaining} tokens remaining.",
            session_id=session_id,
            extra={"tokens_remaining": tokens_remaining},
        )
    )


def session_sleeping(session_id: str) -> None:
    send(
        SlackEvent(
            stage="session_sleeping",
            status="info",
            summary="Session locked and queued for fine-tuning.",
            session_id=session_id,
        )
    )


def extraction_started(session_id: str) -> None:
    send(
        SlackEvent(
            stage="extraction_started",
            status="info",
            summary="Extracting training candidates from transcript.",
            session_id=session_id,
        )
    )


def extraction_completed(session_id: str, count: int, s3_path: str) -> None:
    send(
        SlackEvent(
            stage="extraction_completed",
            status="ok",
            summary=f"Extracted {count} candidate turn pairs.",
            session_id=session_id,
            s3_path=s3_path,
            extra={"count": count},
        )
    )


def curation_started(session_id: str) -> None:
    send(
        SlackEvent(
            stage="curation_started",
            status="info",
            summary="Scoring and filtering candidates.",
            session_id=session_id,
        )
    )


def curation_completed(session_id: str, kept: int, total: int, s3_path: str) -> None:
    send(
        SlackEvent(
            stage="curation_completed",
            status="ok",
            summary=f"Kept {kept}/{total} candidates after scoring.",
            session_id=session_id,
            s3_path=s3_path,
            extra={"kept": kept, "total": total},
        )
    )


def dataset_built(session_id: str, sample_count: int, s3_path: str) -> None:
    send(
        SlackEvent(
            stage="dataset_built",
            status="ok",
            summary=f"SFT dataset built with {sample_count} samples.",
            session_id=session_id,
            s3_path=s3_path,
            extra={"samples": sample_count},
        )
    )


def artifact_uploaded(run_id: str, artifact_type: str, s3_path: str) -> None:
    send(
        SlackEvent(
            stage="artifact_uploaded",
            status="info",
            summary=f"Artifact uploaded: {artifact_type}",
            run_id=run_id,
            s3_path=s3_path,
        )
    )


def training_started(run_id: str, session_id: str, hf_job_id: str) -> None:
    send(
        SlackEvent(
            stage="training_started",
            status="info",
            summary="HuggingFace training job launched.",
            run_id=run_id,
            session_id=session_id,
            extra={"hf_job_id": hf_job_id},
        )
    )


def training_succeeded(run_id: str, s3_path: str) -> None:
    send(
        SlackEvent(
            stage="training_succeeded",
            status="ok",
            summary="Training job completed successfully.",
            run_id=run_id,
            s3_path=s3_path,
        )
    )


def training_failed(run_id: str, error: str) -> None:
    send(
        SlackEvent(
            stage="training_failed",
            status="error",
            summary=f"Training job failed: {error}",
            run_id=run_id,
            extra={"error": error},
        )
    )


def evaluation_started(run_id: str) -> None:
    send(
        SlackEvent(
            stage="evaluation_started",
            status="info",
            summary="Running evaluation suite.",
            run_id=run_id,
        )
    )


def evaluation_completed(run_id: str, passed: bool, score: float, s3_path: str) -> None:
    send(
        SlackEvent(
            stage="evaluation_completed",
            status="ok" if passed else "warn",
            summary=f"Evaluation {'passed' if passed else 'failed'} — score {score:.3f}",
            run_id=run_id,
            s3_path=s3_path,
            extra={"passed": passed, "score": score},
        )
    )


def deployment_approved(run_id: str, version: str) -> None:
    send(
        SlackEvent(
            stage="deployment_approved",
            status="ok",
            summary=f"Adapter {version} approved for production.",
            run_id=run_id,
            adapter_version=version,
        )
    )


def deployment_rejected(run_id: str, reason: str) -> None:
    send(
        SlackEvent(
            stage="deployment_rejected",
            status="warn",
            summary=f"Deployment rejected: {reason}",
            run_id=run_id,
            extra={"reason": reason},
        )
    )


def adapter_switch_succeeded(run_id: str, version: str) -> None:
    send(
        SlackEvent(
            stage="adapter_switch_succeeded",
            status="ok",
            summary=f"Production adapter switched to {version}.",
            run_id=run_id,
            adapter_version=version,
        )
    )


def adapter_switch_failed(run_id: str, error: str) -> None:
    send(
        SlackEvent(
            stage="adapter_switch_failed",
            status="error",
            summary=f"Adapter switch failed: {error}",
            run_id=run_id,
            extra={"error": error},
        )
    )


def rollback_triggered(run_id: str, to_version: str) -> None:
    send(
        SlackEvent(
            stage="rollback_triggered",
            status="warn",
            summary=f"Smoke test failed — rolling back to {to_version}.",
            run_id=run_id,
            adapter_version=to_version,
        )
    )


def rollback_completed(run_id: str, to_version: str) -> None:
    send(
        SlackEvent(
            stage="rollback_completed",
            status="ok",
            summary=f"Rollback to {to_version} completed.",
            run_id=run_id,
            adapter_version=to_version,
        )
    )


def insufficient_data_warning(session_id: str, kept: int, required: int) -> None:
    send(
        SlackEvent(
            stage="insufficient_data_warning",
            status="warn",
            summary=f"Not enough training data: {kept}/{required} samples. Session can continue chatting.",
            session_id=session_id,
            extra={"kept": kept, "required": required},
        )
    )
