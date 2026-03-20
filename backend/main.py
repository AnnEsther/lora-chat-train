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
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from database import get_db, AsyncSession
from models import Session as ChatSession, Turn, SessionState
from schemas import (
    ChatRequest, SessionResponse, SessionListResponse
)
from model_client import ModelClient
import token_counter
from worker.tasks import enqueue_training_pipeline
from shared.slack_notifier import (
    session_started, pre_sleep_warning, session_sleeping
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
async def create_session(db: AsyncSession = Depends(get_db)) -> SessionResponse:
    session = ChatSession(
        id=uuid.uuid4(),
        state=SessionState.ACTIVE,
        max_tokens=MAX_SESSION_TOKENS,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    session_started(str(session.id))
    logger.info("session_created", extra={"session_id": str(session.id)})
    return SessionResponse.from_orm(session)


@app.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> SessionResponse:
    session = await _get_active_session(session_id, db)
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

    if session.state not in (SessionState.ACTIVE, SessionState.PRE_SLEEP_WARNING):
        raise HTTPException(status_code=409, detail=f"Session is {session.state}, not accepting messages.")

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

    yield "data: {\"type\":\"start\"}\n\n"

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

    if remaining <= 0 or session.state == SessionState.PRE_SLEEP_WARNING and remaining <= PRE_SLEEP_THRESHOLD // 2:
        yield f"data: {json.dumps({'type': 'sleep_warning', 'message': 'Session closing — starting fine-tuning…'})}\n\n"
        yield "data: {\"type\":\"end\"}\n\n"
        async for event in _force_sleep(session, db, reason="token_threshold"):
            yield event
    else:
        status = {"remaining_tokens": remaining, "session_state": session.state}
        yield f"data: {json.dumps({'type': 'status', **status})}\n\n"
        yield "data: {\"type\":\"end\"}\n\n"


async def _handle_sleep_command(session: ChatSession, db: AsyncSession):
    import json
    yield f"data: {json.dumps({'type': 'sleep_ack', 'message': 'Going to sleep — see you after fine-tuning!'})}\n\n"
    yield "data: {\"type\":\"end\"}\n\n"
    async for event in _force_sleep(session, db, reason="user_command"):
        yield event


async def _force_sleep(session: ChatSession, db: AsyncSession, reason: str):
    import json
    await _transition(session, SessionState.SLEEPING, db)
    session_sleeping(str(session.id))
    logger.info("session_sleeping", extra={"session_id": str(session.id), "reason": reason})
    enqueue_training_pipeline.delay(str(session.id))
    yield f"data: {json.dumps({'type': 'sleeping', 'reason': reason})}\n\n"


# ── Helper utilities ──────────────────────────────────────────────────────────

async def _get_active_session(session_id: uuid.UUID, db: AsyncSession) -> ChatSession:
    from sqlalchemy import select
    result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


async def _transition(session: ChatSession, new_state: SessionState, db: AsyncSession) -> None:
    old_state = session.state
    session.state = new_state
    if new_state in (SessionState.SLEEPING, SessionState.FAILED):
        from datetime import datetime, timezone
        session.closed_at = datetime.now(timezone.utc)
    await db.commit()
    logger.info("state_transition", extra={
        "session_id": str(session.id),
        "from": old_state,
        "to": new_state,
    })


async def _load_history(session_id: uuid.UUID, db: AsyncSession) -> list[dict]:
    from sqlalchemy import select
    result = await db.execute(
        select(Turn)
        .where(Turn.session_id == session_id)
        .order_by(Turn.created_at)
    )
    turns = result.scalars().all()
    return [{"role": t.role, "content": t.content} for t in turns]


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# ── Outputs listing (used by diagnostic panel) ────────────────────────────────

@app.get("/outputs")
async def list_outputs() -> list[dict]:
    """Return a flat list of files in the local outputs directory for the frontend panel."""
    import os
    from pathlib import Path

    output_dir = Path(os.environ.get("LOCAL_OUTPUT_DIR",
                      Path(__file__).parent.parent / "outputs"))

    if not output_dir.exists():
        return []

    files = []
    for p in sorted(output_dir.rglob("*")):
        if not p.is_file():
            continue
        size_bytes = p.stat().st_size
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 ** 2:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes / 1024**2:.1f} MB"

        files.append({
            "name": p.name,
            "path": str(p.relative_to(output_dir)),
            "size": size_str,
        })

    # Return most recent 30 files to keep the panel compact
    return files[-30:]
