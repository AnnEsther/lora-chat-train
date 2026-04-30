"""backend/schemas.py — Pydantic v2 request/response schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


# ── Session schemas ───────────────────────────────────────────────────────────


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    state: str
    total_tokens: int
    max_tokens: int
    system_prompt: Optional[str] = None
    training_system_prompt: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    closed_at: Optional[datetime] = None
    failure_reason: Optional[str] = None


class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]


class CreateSessionRequest(BaseModel):
    adapter_id: Optional[str] = None
    system_prompt: Optional[str] = None
    training_system_prompt: Optional[str] = None


# ── Chat schemas ──────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str


# ── Training run schemas ──────────────────────────────────────────────────────


class TrainingRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    session_id: uuid.UUID
    status: str
    hf_job_id: Optional[str] = None
    eval_passed: Optional[bool] = None
    artifact_s3_path: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


# ── Model version schemas ─────────────────────────────────────────────────────


class ModelVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    version_tag: str
    adapter_s3_path: str
    is_production: bool
    eval_score: Optional[float] = None
    promoted_at: Optional[datetime] = None
    created_at: datetime
