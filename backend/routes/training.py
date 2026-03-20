"""backend/routes/training.py — Training run and model version status endpoints."""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select

from database import AsyncSession, get_db
from models import TrainingRun, ModelVersion, DeploymentEvent
from schemas import TrainingRunResponse, ModelVersionResponse

router = APIRouter(prefix="/training", tags=["training"])


@router.get("/runs/{run_id}", response_model=TrainingRunResponse)
async def get_run(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> TrainingRunResponse:
    result = await db.execute(select(TrainingRun).where(TrainingRun.id == run_id))
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Training run not found")
    return TrainingRunResponse.from_orm(run)


@router.get("/runs", response_model=list[TrainingRunResponse])
async def list_runs(
    session_id: Optional[uuid.UUID] = None,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
) -> list[TrainingRunResponse]:
    q = select(TrainingRun).order_by(TrainingRun.created_at.desc()).limit(limit)
    if session_id:
        q = q.where(TrainingRun.session_id == session_id)
    result = await db.execute(q)
    return [TrainingRunResponse.from_orm(r) for r in result.scalars().all()]


@router.get("/models/current", response_model=Optional[ModelVersionResponse])
async def get_current_model(db: AsyncSession = Depends(get_db)) -> Optional[ModelVersionResponse]:
    result = await db.execute(
        select(ModelVersion)
        .where(ModelVersion.is_production == True)
        .order_by(ModelVersion.promoted_at.desc())
        .limit(1)
    )
    mv = result.scalar_one_or_none()
    if mv is None:
        return None
    return ModelVersionResponse.from_orm(mv)


@router.get("/models", response_model=list[ModelVersionResponse])
async def list_models(db: AsyncSession = Depends(get_db)) -> list[ModelVersionResponse]:
    result = await db.execute(
        select(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(20)
    )
    return [ModelVersionResponse.from_orm(mv) for mv in result.scalars().all()]


@router.get("/deployments", response_model=list[dict])
async def list_deployments(db: AsyncSession = Depends(get_db)) -> list[dict]:
    result = await db.execute(
        select(DeploymentEvent).order_by(DeploymentEvent.created_at.desc()).limit(50)
    )
    events = result.scalars().all()
    return [
        {
            "id": str(e.id),
            "run_id": str(e.run_id),
            "event_type": e.event_type,
            "from_version": e.from_version,
            "to_version": e.to_version,
            "reason": e.reason,
            "created_at": e.created_at.isoformat(),
        }
        for e in events
    ]
