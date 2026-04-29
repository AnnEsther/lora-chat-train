"""backend/models.py — SQLAlchemy ORM models matching infra/schema.sql."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SessionState(str, enum.Enum):
    ACTIVE = "ACTIVE"
    PRE_SLEEP_WARNING = "PRE_SLEEP_WARNING"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    VALIDATING = "VALIDATING"
    SLEEPING = "SLEEPING"
    TRAINING = "TRAINING"
    EVALUATING = "EVALUATING"
    DEPLOYING = "DEPLOYING"
    READY = "READY"
    FAILED = "FAILED"


class Session(Base):
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    state = Column(String, nullable=False, default=SessionState.ACTIVE)
    total_tokens = Column(Integer, nullable=False, default=0)
    max_tokens = Column(Integer, nullable=False, default=4096)
    system_prompt = Column(Text, nullable=True)
    training_system_prompt = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=_now, onupdate=_now
    )
    closed_at = Column(DateTime(timezone=True), nullable=True)
    metadata_ = Column("metadata", JSON, nullable=False, default=dict)

    turns = relationship("Turn", back_populates="session", cascade="all, delete-orphan")
    candidates = relationship("TrainingCandidate", back_populates="session")
    datasets = relationship("Dataset", back_populates="session")
    training_runs = relationship("TrainingRun", back_populates="session")


class Turn(Base):
    __tablename__ = "turns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role = Column(String, nullable=False)  # user | assistant | system
    content = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    metadata_ = Column("metadata", JSON, nullable=False, default=dict)

    session = relationship("Session", back_populates="turns")


class TrainingCandidate(Base):
    __tablename__ = "training_candidates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    user_turn = Column(Text, nullable=False)
    assistant_turn = Column(Text, nullable=False)
    quality_score = Column(Float, nullable=True)
    included = Column(Boolean, nullable=False, default=False)
    rejection_reason = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)

    session = relationship("Session", back_populates="candidates")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    s3_path = Column(String, nullable=False)
    sample_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)

    session = relationship("Session", back_populates="datasets")


class TrainingRunStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=True)
    status = Column(String, nullable=False, default=TrainingRunStatus.PENDING)
    hf_job_id = Column(String, nullable=True)
    config = Column(JSON, nullable=False, default=dict)
    logs_s3_path = Column(String, nullable=True)
    artifact_s3_path = Column(String, nullable=True)
    eval_s3_path = Column(String, nullable=True)
    eval_passed = Column(Boolean, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)

    session = relationship("Session", back_populates="training_runs")
    dataset = relationship("Dataset")
    model_versions = relationship("ModelVersion", back_populates="run")
    deployment_events = relationship("DeploymentEvent", back_populates="run")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("training_runs.id"), nullable=False)
    version_tag = Column(String, nullable=False)  # e.g. "v0.4"
    adapter_s3_path = Column(String, nullable=False)
    is_production = Column(Boolean, nullable=False, default=False)
    eval_score = Column(Float, nullable=True)
    promoted_at = Column(DateTime(timezone=True), nullable=True)
    retired_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)

    run = relationship("TrainingRun", back_populates="model_versions")


class DeploymentEvent(Base):
    __tablename__ = "deployment_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("training_runs.id"), nullable=False)
    event_type = Column(
        String, nullable=False
    )  # PROMOTE | ROLLBACK | SMOKE_TEST_PASS | SMOKE_TEST_FAIL
    from_version = Column(String, nullable=True)
    to_version = Column(String, nullable=True)
    reason = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)

    run = relationship("TrainingRun", back_populates="deployment_events")


class KnowledgeRecord(Base):
    __tablename__ = "knowledge_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    topic = Column(String, nullable=False)
    facts = Column(JSON, nullable=False, default=list)
    source_turn_id = Column(UUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)

    session = relationship("Session")


class SynthesizedQA(Base):
    __tablename__ = "synthesized_qa"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    knowledge_record_id = Column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_records.id", ondelete="SET NULL"),
        nullable=True,
    )
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    validated = Column(Boolean, nullable=False, default=False)
    edited = Column(Boolean, nullable=False, default=False)
    retry_count = Column(Integer, nullable=False, default=0)
    validation_notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)

    session = relationship("Session")
    knowledge_record = relationship("KnowledgeRecord")


class KnowledgeCorpus(Base):
    __tablename__ = "knowledge_corpus"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic = Column(String, nullable=False)
    facts = Column(JSON, nullable=False, default=list)
    source_session_id = Column(UUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
