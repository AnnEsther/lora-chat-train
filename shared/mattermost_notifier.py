"""shared/mattermost_notifier.py — Professional Mattermost update notifier.

Posts one message per training run to a dedicated Mattermost channel and edits
it in-place as each pipeline phase completes, giving a live checklist view.

Requires three env vars:
  MATTERMOST_BOT_TOKEN   — bot personal access token (post:create + post:write)
  MATTERMOST_API_URL     — root URL of your Mattermost server (no trailing slash)
  MATTERMOST_CHANNEL_ID  — channel ID of the professional updates channel

All public functions are no-ops (with a debug log) when any of the three vars
are absent, so the pipeline works normally without Mattermost configured.

The post_id returned by pipeline_started() must be persisted by the caller
(stored in training_runs.config["mm_post_id"]) so subsequent tasks can retrieve
it without passing it through the Celery chain.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

BOT_TOKEN: str | None = os.environ.get("MATTERMOST_BOT_TOKEN")
API_URL: str | None = os.environ.get("MATTERMOST_API_URL", "").rstrip("/") or None
CHANNEL_ID: str | None = os.environ.get("MATTERMOST_CHANNEL_ID")

# ── Stage definitions ─────────────────────────────────────────────────────────
# Each stage is a key used in the _STAGE_LABELS map.
# The message table is rebuilt from a list of completed/active/pending stage keys.

_STAGE_LABELS = {
    "phase1": "Phase 1 — Extraction, curation & QA synthesis",
    "qa_review": "QA review",
    "training": "Model training",
    "evaluation": "Evaluation",
    "deployment": "Deployment",
}

_ICON_DONE = ":white_check_mark:"
_ICON_ACTIVE = ":hourglass_spinning:"
_ICON_PENDING = ":grey_question:"
_ICON_FAILED = ":x:"
_ICON_ROLLED = ":arrows_counterclockwise:"

_RETRIES = 3
_BACKOFF = 1.5


# ── Internal HTTP helpers ─────────────────────────────────────────────────────


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {BOT_TOKEN}",
        "Content-Type": "application/json",
    }


def _available() -> bool:
    if not (BOT_TOKEN and API_URL and CHANNEL_ID):
        logger.debug("mattermost_skip_not_configured")
        return False
    return True


def _post_message(text: str, card: str) -> str | None:
    """Create a new post in CHANNEL_ID. Returns the post_id on success."""
    payload = {
        "channel_id": CHANNEL_ID,
        "message": text,
        "props": {"card": card} if card else {},
    }
    for attempt in range(1, _RETRIES + 1):
        try:
            resp = requests.post(
                f"{API_URL}/api/v4/posts",
                json=payload,
                headers=_headers(),
                timeout=10,
            )
            resp.raise_for_status()
            post_id: str = resp.json()["id"]
            logger.info("mattermost_post_created", extra={"post_id": post_id})
            return post_id
        except requests.RequestException as exc:
            logger.warning(
                "mattermost_post_retry",
                extra={"attempt": attempt, "error": str(exc)},
            )
            if attempt == _RETRIES:
                logger.error("mattermost_post_failed", extra={"error": str(exc)})
                return None
            time.sleep(_BACKOFF**attempt)
    return None


def _edit_message(post_id: str, text: str, card: str) -> None:
    """Edit an existing post in-place."""
    payload = {
        "id": post_id,
        "message": text,
        "props": {"card": card} if card else {},
    }
    for attempt in range(1, _RETRIES + 1):
        try:
            resp = requests.put(
                f"{API_URL}/api/v4/posts/{post_id}",
                json=payload,
                headers=_headers(),
                timeout=10,
            )
            resp.raise_for_status()
            logger.info("mattermost_post_edited", extra={"post_id": post_id})
            return
        except requests.RequestException as exc:
            logger.warning(
                "mattermost_edit_retry",
                extra={"attempt": attempt, "post_id": post_id, "error": str(exc)},
            )
            if attempt == _RETRIES:
                logger.error(
                    "mattermost_edit_failed",
                    extra={"post_id": post_id, "error": str(exc)},
                )
                return
            time.sleep(_BACKOFF**attempt)


# ── Message builder ───────────────────────────────────────────────────────────


def _build_message(
    session_id: str,
    run_id: str,
    stages: list[tuple[str, str, str]],  # (stage_key, icon, detail_text)
    card_extra: dict | None = None,
) -> tuple[str, str]:
    """Return (message_text, card_text) for the current pipeline state.

    stages is a list of (stage_key, icon, detail) tuples — one per stage,
    ordered as they appear in _STAGE_LABELS.  Pending stages should be passed
    as (key, _ICON_PENDING, "Pending").
    """
    sid = session_id[:8] + "…"
    rid = run_id[:8] + "…"

    # Determine overall header icon from the last stage's icon
    last_icon = stages[-1][1] if stages else _ICON_ACTIVE
    if last_icon == _ICON_FAILED:
        header_icon = ":x:"
        header_suffix = "— **failed**"
    elif last_icon == _ICON_ROLLED:
        header_icon = ":arrows_counterclockwise:"
        header_suffix = "— rolled back"
    elif all(icon == _ICON_DONE for _, icon, _ in stages):
        header_icon = ":tada:"
        header_suffix = "— **complete**"
    else:
        header_icon = ":robot_face:"
        header_suffix = "— in progress…"

    lines = [
        f"#### {header_icon} LoRA Training Run {header_suffix}",
        f"Session `{sid}` · Run `{rid}`",
        "",
        "| Stage | Status |",
        "|---|---|",
    ]
    for stage_key, icon, detail in stages:
        label = _STAGE_LABELS.get(stage_key, stage_key.replace("_", " ").title())
        lines.append(f"| {icon} {label} | {detail} |")

    text = "\n".join(lines)

    # ── Side-panel card (detail view) ─────────────────────────────────────────
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    card_lines = [
        "### Pipeline Details",
        f"**Run ID:** `{run_id}`",
        f"**Session:** `{session_id}`",
        f"**Updated:** {now}",
    ]
    if card_extra:
        for k, v in card_extra.items():
            if v is not None:
                card_lines.append(f"**{k}:** {v}")

    card_text = "\n".join(card_lines)
    return text, card_text


# ── Stage list helpers ────────────────────────────────────────────────────────


def _pending_stages(from_stage: str) -> list[tuple[str, str, str]]:
    """Return all stages from from_stage onwards as pending."""
    keys = list(_STAGE_LABELS.keys())
    idx = keys.index(from_stage) if from_stage in keys else 0
    return [(k, _ICON_PENDING, "Pending") for k in keys[idx:]]


# ── Public API ────────────────────────────────────────────────────────────────


def pipeline_started(session_id: str, run_id: str) -> Optional[str]:
    """Post the initial pipeline message. Returns post_id (store in run.config).

    Called at the start of enqueue_phase1_pipeline.
    """
    if not _available():
        return None

    stages = [
        ("phase1", _ICON_ACTIVE, "Running — extracting and curating turns…"),
        ("qa_review", _ICON_PENDING, "Pending"),
        ("training", _ICON_PENDING, "Pending"),
        ("evaluation", _ICON_PENDING, "Pending"),
        ("deployment", _ICON_PENDING, "Pending"),
    ]
    text, card = _build_message(session_id, run_id, stages)
    return _post_message(text, card)


def qa_ready(
    session_id: str,
    run_id: str,
    post_id: str,
    qa_count: int,
    validated_count: int,
) -> None:
    """Phase 1 complete — QA pairs are ready for human review.

    Called at the end of validate_qa task.
    """
    if not _available() or not post_id:
        return

    stages = [
        (
            "phase1",
            _ICON_DONE,
            f"Done — {qa_count} Q&A pairs synthesised, {validated_count} auto-validated",
        ),
        ("qa_review", _ICON_ACTIVE, "Awaiting human review…"),
        ("training", _ICON_PENDING, "Pending"),
        ("evaluation", _ICON_PENDING, "Pending"),
        ("deployment", _ICON_PENDING, "Pending"),
    ]
    text, card = _build_message(
        session_id,
        run_id,
        stages,
        card_extra={"Q&A pairs": qa_count, "Auto-validated": validated_count},
    )
    _edit_message(post_id, text, card)


def insufficient_data(
    session_id: str,
    run_id: str,
    post_id: str,
    kept: int,
    required: int,
) -> None:
    """Phase 1 stopped early — not enough curated samples.

    Called when curate_candidates finds kept < MIN_TRAINING_SAMPLES.
    """
    if not _available() or not post_id:
        return

    stages = [
        (
            "phase1",
            _ICON_FAILED,
            f"Insufficient data — {kept}/{required} samples (need more chat)",
        ),
        ("qa_review", _ICON_PENDING, "Skipped"),
        ("training", _ICON_PENDING, "Skipped"),
        ("evaluation", _ICON_PENDING, "Skipped"),
        ("deployment", _ICON_PENDING, "Skipped"),
    ]
    text, card = _build_message(
        session_id,
        run_id,
        stages,
        card_extra={"Kept samples": kept, "Required": required},
    )
    _edit_message(post_id, text, card)


def training_launched(
    session_id: str,
    run_id: str,
    post_id: str,
    job_id: str,
    sample_count: int,
) -> None:
    """Phase 2 started — dataset built, training job submitted.

    Called from enqueue_phase2_pipeline after the chain is queued.
    """
    if not _available() or not post_id:
        return

    stages = [
        ("phase1", _ICON_DONE, "Done"),
        ("qa_review", _ICON_DONE, "Review complete — training approved"),
        ("training", _ICON_ACTIVE, f"Running — job `{job_id[:12]}…`"),
        ("evaluation", _ICON_PENDING, "Pending"),
        ("deployment", _ICON_PENDING, "Pending"),
    ]
    text, card = _build_message(
        session_id,
        run_id,
        stages,
        card_extra={"Job ID": job_id, "Dataset samples": sample_count},
    )
    _edit_message(post_id, text, card)


def training_done(
    session_id: str,
    run_id: str,
    post_id: str,
    artifact_s3: str,
) -> None:
    """Training job completed successfully — moving to evaluation.

    Called from launch_training (local) or poll_training (HF).
    """
    if not _available() or not post_id:
        return

    stages = [
        ("phase1", _ICON_DONE, "Done"),
        ("qa_review", _ICON_DONE, "Done"),
        ("training", _ICON_DONE, "Completed — adapter artifact uploaded"),
        ("evaluation", _ICON_ACTIVE, "Running eval suite…"),
        ("deployment", _ICON_PENDING, "Pending"),
    ]
    text, card = _build_message(
        session_id,
        run_id,
        stages,
        card_extra={"Adapter S3": artifact_s3},
    )
    _edit_message(post_id, text, card)


def training_failed(
    session_id: str,
    run_id: str,
    post_id: str,
    error: str,
) -> None:
    """Training job failed.

    Called from launch_training (local failure) or poll_training (HF failure).
    """
    if not _available() or not post_id:
        return

    stages = [
        ("phase1", _ICON_DONE, "Done"),
        ("qa_review", _ICON_DONE, "Done"),
        ("training", _ICON_FAILED, f"Failed — {error[:120]}"),
        ("evaluation", _ICON_PENDING, "Skipped"),
        ("deployment", _ICON_PENDING, "Skipped"),
    ]
    text, card = _build_message(
        session_id,
        run_id,
        stages,
        card_extra={"Error": error},
    )
    _edit_message(post_id, text, card)


def eval_result(
    session_id: str,
    run_id: str,
    post_id: str,
    passed: bool,
    score: float,
    eval_s3: str,
) -> None:
    """Evaluation suite finished — show pass/fail with score.

    Called from run_evaluation task.
    """
    if not _available() or not post_id:
        return

    if passed:
        eval_icon = _ICON_DONE
        eval_detail = f"Passed — score **{score:.3f}**"
        deploy_icon = _ICON_ACTIVE
        deploy_detail = "Promoting adapter…"
    else:
        eval_icon = _ICON_FAILED
        eval_detail = f"Failed — score **{score:.3f}** (threshold 0.65)"
        deploy_icon = _ICON_PENDING
        deploy_detail = "Skipped — eval did not pass"

    stages = [
        ("phase1", _ICON_DONE, "Done"),
        ("qa_review", _ICON_DONE, "Done"),
        ("training", _ICON_DONE, "Done"),
        ("evaluation", eval_icon, eval_detail),
        ("deployment", deploy_icon, deploy_detail),
    ]
    text, card = _build_message(
        session_id,
        run_id,
        stages,
        card_extra={"Eval score": f"{score:.4f}", "Eval report": eval_s3},
    )
    _edit_message(post_id, text, card)


def pipeline_finished(
    session_id: str,
    run_id: str,
    post_id: str,
    status: str,  # "deployed" | "rolled_back" | "failed"
    version: str | None = None,
    reason: str | None = None,
) -> None:
    """Final pipeline state — deployed, rolled back, or failed.

    Called from deploy_or_rollback task.
    """
    if not _available() or not post_id:
        return

    if status == "deployed":
        deploy_icon = _ICON_DONE
        deploy_detail = f"Adapter `{version}` promoted to production :tada:"
    elif status == "rolled_back":
        deploy_icon = _ICON_ROLLED
        deploy_detail = f"Smoke test failed — rolled back to `{version or 'base'}`"
    else:  # failed
        deploy_icon = _ICON_FAILED
        deploy_detail = reason or "Deployment failed"

    stages = [
        ("phase1", _ICON_DONE, "Done"),
        ("qa_review", _ICON_DONE, "Done"),
        ("training", _ICON_DONE, "Done"),
        ("evaluation", _ICON_DONE, "Done"),
        ("deployment", deploy_icon, deploy_detail),
    ]
    card_extra: dict = {}
    if version:
        card_extra["Version"] = version
    if reason:
        card_extra["Reason"] = reason

    text, card = _build_message(session_id, run_id, stages, card_extra=card_extra)
    _edit_message(post_id, text, card)
