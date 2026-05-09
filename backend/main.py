"""backend/main.py — FastAPI application: chat streaming, session lifecycle, /sleep handling."""

from __future__ import annotations

import sys
from pathlib import Path
from dotenv import load_dotenv

# Fix sys.path and load .env before all other imports
_root = Path(__file__).parent.parent
_backend = Path(__file__).parent
for _p in [str(_root), str(_backend)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
load_dotenv(_root / ".env")

import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select

from database import get_db, AsyncSession
from models import Session as ChatSession, Turn, SessionState
from schemas import (
    ChatRequest,
    CreateSessionRequest,
    SessionResponse,
    SessionListResponse,
)
from model_client import ModelClient
import token_counter
from worker.tasks import enqueue_training_pipeline
from shared.slack_notifier import (
    session_started,
    pre_sleep_warning,
    session_sleeping,
    insufficient_data_warning,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

MAX_SESSION_TOKENS: int = int(os.environ.get("MAX_SESSION_TOKENS", 4096))
PRE_SLEEP_THRESHOLD: int = int(os.environ.get("PRE_SLEEP_THRESHOLD", 512))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.model_client = ModelClient()
    await app.state.model_client.load()
    logger.info("model_ready")
    yield
    await app.state.model_client.unload()


app = FastAPI(title="LoRA Chat & Train", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Session endpoints ─────────────────────────────────────────────────────────


@app.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_session(
    request: CreateSessionRequest | None = None,
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    session = ChatSession(
        id=uuid.uuid4(),
        state=SessionState.ACTIVE,
        max_tokens=MAX_SESSION_TOKENS,
        system_prompt=request.system_prompt if request else None,
        training_system_prompt=request.training_system_prompt if request else None,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    if request and request.adapter_id and request.adapter_id != "base":
        try:
            import requests

            model_url = os.environ.get("MODEL_SERVER_URL", "http://localhost:8001")
            adapters_resp = requests.get(f"{model_url}/adapters", timeout=5)
            if adapters_resp.ok:
                adapters = adapters_resp.json().get("adapters", [])
                selected = next(
                    (a for a in adapters if a["id"] == request.adapter_id), None
                )
                if selected and selected.get("path"):
                    reload_resp = requests.post(
                        f"{model_url}/reload_adapter",
                        json={"adapter_dir": selected["path"]},
                        timeout=30,
                    )
                    if not reload_resp.ok:
                        logger.warning(
                            "failed_to_load_adapter",
                            extra={"adapter_id": request.adapter_id},
                        )
        except Exception as exc:
            logger.warning(
                "adapter_load_error",
                extra={"adapter_id": request.adapter_id, "error": str(exc)},
            )

    return SessionResponse.from_orm(session)


@app.post("/load_adapter")
async def load_adapter(request: CreateSessionRequest) -> dict:
    """Load an adapter without creating a session."""
    import requests as req

    model_url = os.environ.get("MODEL_SERVER_URL", "http://localhost:8001")
    adapters_resp = req.get(f"{model_url}/adapters", timeout=5)
    if not adapters_resp.ok:
        return {
            "status": "error",
            "message": "Could not fetch adapters from model server",
        }

    adapters = adapters_resp.json().get("adapters", [])
    selected = next((a for a in adapters if a["id"] == request.adapter_id), None)

    if not selected:
        return {"status": "error", "message": f"Adapter {request.adapter_id} not found"}

    if request.adapter_id == "base":
        reload_resp = req.post(
            f"{model_url}/reload_adapter",
            json={"adapter_dir": "base"},
            timeout=30,
        )
    elif selected.get("path"):
        reload_resp = req.post(
            f"{model_url}/reload_adapter",
            json={"adapter_dir": selected["path"]},
            timeout=30,
        )
    else:
        return {"status": "error", "message": "No path for adapter"}

    if reload_resp.ok:
        return {"status": "ok", "adapter_id": request.adapter_id}
    return {"status": "error", "message": reload_resp.text}


@app.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> SessionResponse:
    session = await _get_active_session(session_id, db)
    if session.state in (
        SessionState.TRAINING,
        SessionState.EVALUATING,
        SessionState.DEPLOYING,
    ):
        try:
            import requests

            model_url = os.environ.get("MODEL_SERVER_URL", "http://localhost:8001")
            resp = requests.get(f"{model_url}/train/status", timeout=5)
            if resp.ok:
                train_status = resp.json()
                if train_status.get("status") == "completed":
                    await _transition(session, SessionState.READY, db)
        except Exception:
            pass
    return SessionResponse.from_orm(session)


@app.get("/sessions", response_model=SessionListResponse)
async def list_sessions(db: AsyncSession = Depends(get_db)) -> SessionListResponse:
    from sqlalchemy import select

    result = await db.execute(
        select(ChatSession).order_by(ChatSession.created_at.desc()).limit(20)
    )
    sessions = result.scalars().all()
    return SessionListResponse(sessions=[SessionResponse.from_orm(s) for s in sessions])


# ── Chat endpoint ─────────────────────────────────────────────────────────────


@app.post("/sessions/{session_id}/chat")
async def chat(
    session_id: uuid.UUID,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    session = await _get_active_session(session_id, db)

    if session.state not in (
        SessionState.ACTIVE,
        SessionState.PRE_SLEEP_WARNING,
        SessionState.INSUFFICIENT_DATA,
        SessionState.FAILED,
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Session is {session.state}, not accepting messages.",
        )

    # /sleep command — immediate clean shutdown
    if request.message.strip() == "/sleep":
        return StreamingResponse(
            _handle_sleep_command(session, db),
            media_type="text/event-stream",
        )

    # Persist user turn
    user_tokens = token_counter.count(request.message)
    user_turn = Turn(
        id=uuid.uuid4(),
        session_id=session.id,
        role="user",
        content=request.message,
        token_count=user_tokens,
    )
    db.add(user_turn)
    session.total_tokens += user_tokens
    await db.commit()

    # Check if we are approaching limit before generating reply
    remaining_before = MAX_SESSION_TOKENS - session.total_tokens

    if remaining_before <= 0:
        # Force sleep immediately — no reply
        return StreamingResponse(
            _force_sleep(session, db, reason="budget_exhausted"),
            media_type="text/event-stream",
        )

    if remaining_before <= PRE_SLEEP_THRESHOLD and session.state == SessionState.ACTIVE:
        await _transition(session, SessionState.PRE_SLEEP_WARNING, db)
        pre_sleep_warning(str(session.id), remaining_before)

    # Build conversation history for model
    history = await _load_history(session_id, db)

    model_client: ModelClient = app.state.model_client

    return StreamingResponse(
        _stream_reply(session, history, model_client, db),
        media_type="text/event-stream",
    )


async def _stream_reply(
    session: ChatSession,
    history: list[dict],
    model_client: ModelClient,
    db: AsyncSession,
):
    """Stream the assistant reply, persist it, and trigger sleep if budget hit."""
    assistant_text = ""
    assistant_tokens = 0

    yield 'data: {"type":"start"}\n\n'

    async for chunk in model_client.stream(history):
        assistant_text += chunk
        assistant_tokens += token_counter.count(chunk)
        import json

        yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"

    # Persist assistant turn
    assistant_turn = Turn(
        id=uuid.uuid4(),
        session_id=session.id,
        role="assistant",
        content=assistant_text,
        token_count=assistant_tokens,
    )
    db.add(assistant_turn)
    session.total_tokens += assistant_tokens
    await db.commit()

    remaining = MAX_SESSION_TOKENS - session.total_tokens
    import json

    if session.state == SessionState.INSUFFICIENT_DATA:
        status = {"remaining_tokens": remaining, "session_state": session.state}
        yield f"data: {json.dumps({'type': 'status', **status})}\n\n"
        yield 'data: {"type":"end"}\n\n'
    elif (
        remaining <= 0
        or session.state == SessionState.PRE_SLEEP_WARNING
        and remaining <= PRE_SLEEP_THRESHOLD // 2
    ):
        yield f"data: {json.dumps({'type': 'sleep_warning', 'message': 'Session closing — starting fine-tuning…'})}\n\n"
        yield 'data: {"type":"end"}\n\n'
        async for event in _force_sleep(session, db, reason="token_threshold"):
            yield event
    else:
        status = {"remaining_tokens": remaining, "session_state": session.state}
        yield f"data: {json.dumps({'type': 'status', **status})}\n\n"
        yield 'data: {"type":"end"}\n\n'


async def _handle_sleep_command(session: ChatSession, db: AsyncSession):
    import json

    yield f"data: {json.dumps({'type': 'sleep_ack', 'message': 'Going to sleep — see you after fine-tuning!'})}\n\n"
    yield 'data: {"type":"end"}\n\n'
    async for event in _force_sleep(session, db, reason="user_command"):
        yield event


async def _force_sleep(session: ChatSession, db: AsyncSession, reason: str):
    import json
    from worker.tasks import enqueue_phase1_pipeline

    await _transition(session, SessionState.VALIDATING, db)
    logger.info(
        "session_validating", extra={"session_id": str(session.id), "reason": reason}
    )
    enqueue_phase1_pipeline.delay(str(session.id))
    yield f"data: {json.dumps({'type': 'validating', 'reason': reason})}\n\n"


# ── Helper utilities ──────────────────────────────────────────────────────────


async def _get_active_session(session_id: uuid.UUID, db: AsyncSession) -> ChatSession:
    from sqlalchemy import select

    result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


async def _transition(
    session: ChatSession, new_state: SessionState, db: AsyncSession
) -> None:
    old_state = session.state
    session.state = new_state
    if new_state == SessionState.SLEEPING:
        from datetime import datetime, timezone

        session.closed_at = datetime.now(timezone.utc)
    await db.commit()
    logger.info(
        "state_transition",
        extra={
            "session_id": str(session.id),
            "from": old_state,
            "to": new_state,
        },
    )


async def _load_history(session_id: uuid.UUID, db: AsyncSession) -> list[dict]:
    from sqlalchemy import select

    result = await db.execute(
        select(Turn).where(Turn.session_id == session_id).order_by(Turn.created_at)
    )
    turns = result.scalars().all()
    return [{"role": t.role, "content": t.content} for t in turns]


# ── QA Review endpoints ───────────────────────────────────────────────────────


class QAUpdateRequest(BaseModel):
    question: Optional[str] = None
    answer: Optional[str] = None
    validated: Optional[bool] = None


@app.get("/sessions/{session_id}/qa")
async def get_session_qa(
    session_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> list[dict]:
    """Get all synthesized Q&A for a session."""
    from models import SynthesizedQA

    result = await db.execute(
        select(SynthesizedQA).where(SynthesizedQA.session_id == session_id)
    )
    qa_items = result.scalars().all()

    return [
        {
            "id": str(qa.id),
            "question": qa.question,
            "answer": qa.answer,
            "validated": qa.validated,
            "edited": qa.edited,
            "retry_count": qa.retry_count,
            "validation_notes": qa.validation_notes,
        }
        for qa in qa_items
    ]


@app.put("/sessions/{session_id}/qa/{qa_id}")
async def update_qa(
    session_id: uuid.UUID,
    qa_id: uuid.UUID,
    request: QAUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Update a Q&A pair (edit)."""
    from models import SynthesizedQA

    result = await db.execute(
        select(SynthesizedQA).where(
            SynthesizedQA.id == qa_id, SynthesizedQA.session_id == session_id
        )
    )
    qa = result.scalar_one_or_none()

    if not qa:
        raise HTTPException(404, "Q&A not found")

    if request.question is not None:
        qa.question = request.question
        qa.edited = True
    if request.answer is not None:
        qa.answer = request.answer
        qa.edited = True
    if request.validated is not None:
        qa.validated = request.validated

    await db.commit()

    return {"status": "ok", "id": str(qa.id)}


@app.post("/sessions/{session_id}/qa/validate-mark")
async def mark_qa_validated(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Mark all Q&A as validated (user clicked Start Training)."""
    from models import SynthesizedQA

    result = await db.execute(
        select(SynthesizedQA).where(
            SynthesizedQA.session_id == session_id,
            SynthesizedQA.validated == False,
        )
    )
    qa_items = result.scalars().all()

    count = 0
    for qa in qa_items:
        qa.validated = True
        count += 1

    await db.commit()

    return {"status": "ok", "validated_count": count}


@app.delete("/sessions/{session_id}/qa/{qa_id}")
async def delete_qa(
    session_id: uuid.UUID,
    qa_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Delete a synthesized Q&A entry."""
    from models import SynthesizedQA

    result = await db.execute(
        select(SynthesizedQA).where(
            SynthesizedQA.id == qa_id,
            SynthesizedQA.session_id == session_id,
        )
    )
    qa = result.scalar_one_or_none()
    if not qa:
        raise HTTPException(404, "Q&A not found")
    await db.delete(qa)
    await db.commit()
    return {"status": "ok", "id": str(qa_id)}


@app.post("/sessions/{session_id}/restart-training")
async def restart_training(
    session_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> dict:
    """Restart training from the training phase (for failed training)."""
    from sqlalchemy import select
    from worker.tasks import (
        launch_training,
        poll_training,
        run_evaluation,
        deploy_or_rollback,
    )
    import uuid as uuid_module

    session = await _get_active_session(session_id, db)

    # Get the most recent training run for this session
    from models import TrainingRun

    result = await db.execute(
        select(TrainingRun)
        .where(TrainingRun.session_id == session_id)
        .order_by(TrainingRun.created_at.desc())
    )
    training_run = result.scalars().first()

    if not training_run:
        raise HTTPException(404, "No training run found for this session")

    run_id = str(training_run.id)

    # Get dataset path
    from models import Dataset

    result = await db.execute(
        select(Dataset)
        .where(Dataset.session_id == session_id)
        .order_by(Dataset.created_at.desc())
    )
    dataset = result.scalars().first()

    if not dataset:
        raise HTTPException(404, "No dataset found for this session")

    # Restart from training
    prev = {
        "session_id": str(session_id),
        "run_id": run_id,
        "s3_path": dataset.s3_path,
    }

    # Re-queue the pipeline from training
    from celery import chain

    (
        launch_training.s(prev, str(session_id), run_id)
        | poll_training.s(str(session_id), run_id)
        | run_evaluation.s(str(session_id), run_id)
        | deploy_or_rollback.s(str(session_id), run_id)
    ).apply_async()

    await _transition(session, SessionState.TRAINING, db)

    logger.info(
        "training_restarted", extra={"session_id": str(session_id), "run_id": run_id}
    )
    return {"status": "ok", "run_id": run_id}


@app.post("/sessions/{session_id}/start-training")
async def start_training(
    session_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> dict:
    """Start Phase 2 of training after user validates QA."""
    from sqlalchemy import select
    from worker.tasks import enqueue_phase2_pipeline

    session = await _get_active_session(session_id, db)

    if session.state != "VALIDATING":
        raise HTTPException(
            400, "Session must be in VALIDATING state to start training"
        )

    # Check if there's at least one validated QA
    from models import SynthesizedQA

    result = await db.execute(
        select(SynthesizedQA).where(
            SynthesizedQA.session_id == session_id,
            SynthesizedQA.validated == True,
        )
    )
    validated_qa = result.scalars().all()

    if len(validated_qa) == 0:
        raise HTTPException(
            400, "No validated QA found. Please validate at least one QA pair."
        )

    # Transition to TRAINING and trigger Phase 2
    await _transition(session, SessionState.TRAINING, db)
    enqueue_phase2_pipeline.delay(str(session_id))

    logger.info("training_started", extra={"session_id": str(session_id)})
    return {"status": "ok"}


# ── Health check ──────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/adapters")
async def list_adapters() -> dict:
    """Proxy adapter list from model server which has the adapter_store volume."""
    import requests as req

    model_url = os.environ.get("MODEL_SERVER_URL", "http://model_server:8001")
    try:
        resp = req.get(f"{model_url}/adapters", timeout=5)
        if resp.ok:
            return resp.json()
    except Exception as exc:
        logger.warning("adapters_proxy_error", extra={"error": str(exc)})
    # Fallback — base model only
    return {
        "adapters": [
            {"id": "base", "version": "Base model", "path": "", "is_base": True}
        ]
    }


# ── Outputs listing (used by diagnostic panel) ────────────────────────────────


@app.get("/outputs")
async def list_outputs() -> list[dict]:
    """Return a flat list of files in the local outputs directory for the frontend panel."""
    import os
    from pathlib import Path

    output_dir = Path(
        os.environ.get("LOCAL_OUTPUT_DIR", Path(__file__).parent.parent / "outputs")
    )

    if not output_dir.exists():
        return []

    files = []
    for p in sorted(output_dir.rglob("*")):
        if not p.is_file():
            continue
        size_bytes = p.stat().st_size
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024**2:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes / 1024**2:.1f} MB"

        files.append(
            {
                "name": p.name,
                "path": str(p.relative_to(output_dir)),
                "size": size_str,
            }
        )

    # Return most recent 30 files to keep the panel compact
    return files[-30:]

@app.get("/sessions/{session_id}/turns")
async def get_session_turns(session_id: uuid.UUID, db: AsyncSession = Depends(get_db),) -> list[dict]:
    history = await _load_history(session_id, db)
    return history