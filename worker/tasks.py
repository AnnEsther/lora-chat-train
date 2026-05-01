"""worker/tasks.py — Celery task definitions orchestrating the full training pipeline."""

from __future__ import annotations

import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ── Load .env and fix sys.path before any other project imports ───────────────
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))  # project root
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))  # for models

from celery import Celery
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session as DBSession

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")

# Build sync DB URL from the async one (strip +asyncpg driver prefix)
_raw_db_url = os.environ.get("DATABASE_URL", "")
DATABASE_URL_SYNC = _raw_db_url.replace("+asyncpg", "") if _raw_db_url else ""

if not DATABASE_URL_SYNC:
    raise RuntimeError("DATABASE_URL is not set — check your .env file")

MIN_TRAINING_SAMPLES = int(os.environ.get("MIN_TRAINING_SAMPLES", 10))

app = Celery(
    "lora_worker",
    broker=REDIS_URL,
    backend=os.environ.get("CELERY_RESULT_BACKEND", REDIS_URL),
)
app.conf.task_serializer = "json"
app.conf.result_serializer = "json"

# Sync engine for Celery tasks (Celery runs in a sync context)
engine = create_engine(DATABASE_URL_SYNC, pool_pre_ping=True)


def _db() -> DBSession:
    return DBSession(engine)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _update_session_state(session_id: str, state: str, db: DBSession) -> None:
    from backend.models import Session as ChatSession

    session = db.get(ChatSession, uuid.UUID(session_id))
    if session:
        session.state = state
        db.commit()


def _update_run_status(run_id: str, status: str, db: DBSession, **kwargs) -> None:
    from backend.models import TrainingRun

    run = db.get(TrainingRun, uuid.UUID(run_id))
    if run:
        run.status = status
        for k, v in kwargs.items():
            setattr(run, k, v)
        db.commit()


def _set_failure_reason(session_id: str, reason: str, db: DBSession) -> None:
    """Persist a human-readable failure reason on the session row."""
    from backend.models import Session as ChatSession

    session = db.get(ChatSession, uuid.UUID(session_id))
    if session:
        session.failure_reason = reason
        db.commit()


# ── Pipeline tasks ────────────────────────────────────────────────────────────


@app.task(name="tasks.enqueue_training_pipeline", bind=True, max_retries=1)
def enqueue_training_pipeline(self, session_id: str) -> None:
    """Legacy: Full pipeline (for backward compatibility)."""
    from backend.models import TrainingRun
    from shared.slack_notifier import extraction_started

    logger.info("pipeline_start", extra={"session_id": session_id})
    extraction_started(session_id)

    with _db() as db:
        run = TrainingRun(
            id=uuid.uuid4(),
            session_id=uuid.UUID(session_id),
            status="PENDING",
            created_at=datetime.now(timezone.utc),
        )
        db.add(run)
        db.commit()
        run_id = str(run.id)
        _update_session_state(session_id, "TRAINING", db)

    # Full pipeline stages
    (
        extract_candidates.s(None, session_id, run_id)
        | curate_candidates.s(session_id, run_id)
        | build_dataset.s(session_id, run_id)
        | launch_training.s(session_id, run_id)
        | poll_training.s(session_id, run_id)
        | run_evaluation.s(session_id, run_id)
        | deploy_or_rollback.s(session_id, run_id)
    ).apply_async()


@app.task(name="tasks.enqueue_phase1_pipeline", bind=True, max_retries=1)
def enqueue_phase1_pipeline(self, session_id: str) -> None:
    """Phase 1: Extract candidates, curate, extract knowledge, synthesize QA, validate."""
    from backend.models import TrainingRun
    from shared.slack_notifier import extraction_started

    logger.info("phase1_start", extra={"session_id": session_id})
    extraction_started(session_id)

    with _db() as db:
        run = TrainingRun(
            id=uuid.uuid4(),
            session_id=uuid.UUID(session_id),
            status="PENDING",
            created_at=datetime.now(timezone.utc),
        )
        db.add(run)
        db.commit()
        run_id = str(run.id)

    # Phase 1: Extract → Curate → Knowledge → Synthesize → Validate
    (
        extract_candidates.s(None, session_id, run_id)
        | curate_candidates.s(session_id, run_id)
        | extract_knowledge.s(session_id, run_id)
        | synthesize_qa.s(session_id, run_id)
        | validate_qa.s(session_id, run_id)
    ).apply_async()

    logger.info("phase1_enqueued", extra={"session_id": session_id, "run_id": run_id})


@app.task(name="tasks.enqueue_phase2_pipeline", bind=True, max_retries=1)
def enqueue_phase2_pipeline(self, session_id: str) -> None:
    """Phase 2: Build dataset, train, evaluate, deploy. Called after user validates QA."""
    from backend.models import TrainingRun
    from shared.slack_notifier import training_started

    logger.info("phase2_start", extra={"session_id": session_id})

    with _db() as db:
        run = TrainingRun(
            id=uuid.uuid4(),
            session_id=uuid.UUID(session_id),
            status="PENDING",
            created_at=datetime.now(timezone.utc),
        )
        db.add(run)
        db.commit()
        run_id = str(run.id)
        _update_session_state(session_id, "TRAINING", db)

    training_started(session_id)

    # Phase 2: Build dataset → Train → Evaluate → Deploy
    (
        build_dataset.s(session_id, run_id)
        | launch_training.s(session_id, run_id)
        | poll_training.s(session_id, run_id)
        | run_evaluation.s(session_id, run_id)
        | deploy_or_rollback.s(session_id, run_id)
    ).apply_async()

    logger.info("phase2_enqueued", extra={"session_id": session_id, "run_id": run_id})


@app.task(name="tasks.extract_candidates", bind=True)
def extract_candidates(self, _prev, session_id: str, run_id: str) -> dict:
    """Extract user/assistant turn pairs from the raw transcript."""
    from training.extractor.transcript_extractor import TranscriptExtractor
    from shared.s3_uploader import upload_raw_transcript, upload_candidates
    from shared.slack_notifier import extraction_completed

    with _db() as db:
        from backend.models import Turn

        turns = (
            db.execute(
                select(Turn)
                .where(Turn.session_id == uuid.UUID(session_id))
                .order_by(Turn.created_at)
            )
            .scalars()
            .all()
        )
        raw_transcript = [{"role": t.role, "content": t.content} for t in turns]

    raw_s3 = upload_raw_transcript(session_id, raw_transcript)

    extractor = TranscriptExtractor()
    candidates = extractor.extract(raw_transcript)

    cand_s3 = upload_candidates(session_id, [c.to_dict() for c in candidates])

    with _db() as db:
        from backend.models import TrainingCandidate

        for c in candidates:
            db.add(
                TrainingCandidate(
                    id=uuid.uuid4(),
                    session_id=uuid.UUID(session_id),
                    conversation=c.conversation,
                )
            )
        db.commit()

    extraction_completed(session_id, len(candidates), cand_s3)
    logger.info(
        "extraction_done", extra={"session_id": session_id, "count": len(candidates)}
    )
    return {
        "session_id": session_id,
        "run_id": run_id,
        "candidate_count": len(candidates),
    }


@app.task(name="tasks.curate_candidates", bind=True)
def curate_candidates(self, prev: dict, session_id: str, run_id: str) -> dict:
    """Score and filter candidates. Persist scores to DB."""
    from training.curator.curator import Curator
    from shared.s3_uploader import upload_curated
    from shared.slack_notifier import curation_started, curation_completed

    curation_started(session_id)

    with _db() as db:
        from backend.models import TrainingCandidate

        rows = (
            db.execute(
                select(TrainingCandidate).where(
                    TrainingCandidate.session_id == uuid.UUID(session_id)
                )
            )
            .scalars()
            .all()
        )
        candidates = [{"conversation": r.conversation, "_id": r.id} for r in rows]

    curator = Curator()
    curated = curator.score_and_filter(candidates)

    with _db() as db:
        from backend.models import TrainingCandidate

        for item in curated:
            row = db.get(TrainingCandidate, item["_id"])
            if row:
                row.quality_score = item["score"]
                row.included = item["included"]
                row.rejection_reason = item.get("rejection_reason")
        db.commit()

    kept = [c for c in curated if c["included"]]
    curation_s3 = upload_curated(session_id, kept)
    curation_completed(session_id, len(kept), len(curated), curation_s3)

    logger.info(
        "curation_done",
        extra={
            "session_id": session_id,
            "kept": len(kept),
            "total": len(curated),
        },
    )

    if len(kept) < MIN_TRAINING_SAMPLES:
        from shared.slack_notifier import insufficient_data_warning

        insufficient_data_warning(session_id, len(kept), MIN_TRAINING_SAMPLES)
        _update_run_status(
            run_id, "FAILED", _db(), finished_at=datetime.now(timezone.utc)
        )
        _update_session_state(session_id, "INSUFFICIENT_DATA", _db())
        logger.info(
            "insufficient_data",
            extra={
                "session_id": session_id,
                "kept": len(kept),
                "required": MIN_TRAINING_SAMPLES,
            },
        )
        return {
            "session_id": session_id,
            "run_id": run_id,
            "kept": len(kept),
            "sufficient": False,
            "required": MIN_TRAINING_SAMPLES,
        }

    return {
        "session_id": session_id,
        "run_id": run_id,
        "kept": len(kept),
        "sufficient": True,
    }


@app.task(name="tasks.build_dataset", bind=True)
def build_dataset(self, prev: dict, session_id: str, run_id: str) -> dict:
    """Write curated candidates to JSONL and upload dataset."""
    if not prev.get("sufficient", True):
        logger.info(
            "skipping_dataset_build_insufficient_data", extra={"session_id": session_id}
        )
        return prev

    from training.datasets.dataset_writer import DatasetWriter
    from shared.s3_uploader import upload_dataset_jsonl
    from shared.slack_notifier import dataset_built

    training_system_prompt = None
    with _db() as db:
        from backend.models import TrainingCandidate, Session as ChatSession

        session = db.get(ChatSession, uuid.UUID(session_id))
        if session:
            training_system_prompt = session.training_system_prompt

        rows = (
            db.execute(
                select(TrainingCandidate).where(
                    TrainingCandidate.session_id == uuid.UUID(session_id),
                    TrainingCandidate.included == True,
                )
            )
            .scalars()
            .all()
        )
        samples = [{"conversation": r.conversation} for r in rows]

    writer = DatasetWriter(system_prompt=training_system_prompt)
    jsonl_text = writer.write_jsonl(samples)

    dataset_s3 = upload_dataset_jsonl(session_id, jsonl_text)

    with _db() as db:
        from backend.models import Dataset

        dataset = Dataset(
            id=uuid.uuid4(),
            session_id=uuid.UUID(session_id),
            s3_path=dataset_s3,
            sample_count=len(samples),
        )
        db.add(dataset)
        db.commit()
        dataset_id = str(dataset.id)

    dataset_built(session_id, len(samples), dataset_s3)
    return {
        "session_id": session_id,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "s3_path": dataset_s3,
    }


@app.task(name="tasks.extract_knowledge", bind=True)
def extract_knowledge(self, prev: dict, session_id: str, run_id: str) -> dict:
    """Extract knowledge records from curated candidates."""
    from training.knowledge.extractor import KnowledgeExtractor
    from training.knowledge.normalizer import KnowledgeNormalizer

    logger.info("extract_knowledge_start", extra={"session_id": session_id})

    with _db() as db:
        from backend.models import TrainingCandidate

        rows = (
            db.execute(
                select(TrainingCandidate).where(
                    TrainingCandidate.session_id == uuid.UUID(session_id),
                    TrainingCandidate.included == True,
                )
            )
            .scalars()
            .all()
        )

    logger.info(
        "extract_knowledge", extra={"session_id": session_id, "candidates": len(rows)}
    )

    extractor = KnowledgeExtractor()
    normalizer = KnowledgeNormalizer()
    knowledge_records = []

    with _db() as db:
        from backend.models import KnowledgeRecord

        for row in rows:
            # Flatten conversation into combined user/assistant text for knowledge extraction
            user_text = " ".join(
                t["content"]
                for t in (row.conversation or [])
                if t.get("role") == "user"
            )
            assistant_text = " ".join(
                t["content"]
                for t in (row.conversation or [])
                if t.get("role") == "assistant"
            )
            topics = extractor.extract(user_text, assistant_text)
            records = normalizer.normalize(user_text, assistant_text, topics)

            for record in records:
                kr = KnowledgeRecord(
                    id=uuid.uuid4(),
                    session_id=uuid.UUID(session_id),
                    topic=record.topic,
                    facts=record.facts,
                    source_turn_id=None,
                )
                db.add(kr)
                knowledge_records.append(
                    {
                        "id": str(kr.id),
                        "topic": record.topic,
                        "facts": record.facts,
                    }
                )

        db.commit()

    logger.info(
        "knowledge_extracted",
        extra={
            "session_id": session_id,
            "records": len(knowledge_records),
        },
    )

    return {
        "session_id": session_id,
        "run_id": run_id,
        "knowledge_records": len(knowledge_records),
    }


@app.task(name="tasks.synthesize_qa", bind=True)
def synthesize_qa(self, prev: dict, session_id: str, run_id: str) -> dict:
    """Synthesize Q&A pairs from knowledge records."""
    from training.knowledge.synthesizer import QASynthesizer

    system_prompt = None
    with _db() as db:
        from backend.models import Session as ChatSession, KnowledgeRecord

        session = db.get(ChatSession, uuid.UUID(session_id))
        if session:
            system_prompt = session.training_system_prompt

        records = (
            db.execute(
                select(KnowledgeRecord).where(
                    KnowledgeRecord.session_id == uuid.UUID(session_id)
                )
            )
            .scalars()
            .all()
        )

    knowledge_records = [{"topic": r.topic, "facts": r.facts} for r in records]
    logger.info(
        "synthesize_qa",
        extra={"session_id": session_id, "records": len(knowledge_records)},
    )

    # If no knowledge records, fall back to using curated candidates directly
    if not knowledge_records:
        logger.warning("No knowledge records found, using curated candidates")
        with _db() as db:
            from backend.models import TrainingCandidate, SynthesizedQA

            rows = (
                db.execute(
                    select(TrainingCandidate).where(
                        TrainingCandidate.session_id == uuid.UUID(session_id),
                        TrainingCandidate.included == True,
                    )
                )
                .scalars()
                .all()
            )
            for r in rows:
                user_text = " ".join(
                    t["content"]
                    for t in (r.conversation or [])
                    if t.get("role") == "user"
                )
                assistant_text = " ".join(
                    t["content"]
                    for t in (r.conversation or [])
                    if t.get("role") == "assistant"
                )
                sq = SynthesizedQA(
                    id=uuid.uuid4(),
                    session_id=uuid.UUID(session_id),
                    question=user_text,
                    answer=assistant_text,
                )
                db.add(sq)
            db.commit()
        return {"session_id": session_id, "run_id": run_id, "qa_pairs": len(rows)}

    try:
        synthesizer = QASynthesizer()
        qa_pairs = synthesizer.synthesize(knowledge_records, system_prompt)
    except Exception as e:
        logger.warning(f"Synthesis failed, using fallback: {e}")
        # Fallback: use curated candidates directly
        with _db() as db:
            from backend.models import TrainingCandidate, SynthesizedQA

            rows = (
                db.execute(
                    select(TrainingCandidate).where(
                        TrainingCandidate.session_id == uuid.UUID(session_id),
                        TrainingCandidate.included == True,
                    )
                )
                .scalars()
                .all()
            )
            qa_pairs = []
            for r in rows:
                user_text = " ".join(
                    t["content"]
                    for t in (r.conversation or [])
                    if t.get("role") == "user"
                )
                assistant_text = " ".join(
                    t["content"]
                    for t in (r.conversation or [])
                    if t.get("role") == "assistant"
                )
                sq = SynthesizedQA(
                    id=uuid.uuid4(),
                    session_id=uuid.UUID(session_id),
                    question=user_text,
                    answer=assistant_text,
                )
                db.add(sq)
                qa_pairs.append(
                    type(
                        "obj",
                        (object,),
                        {"question": user_text, "answer": assistant_text},
                    )()
                )
            db.commit()

    with _db() as db:
        from backend.models import SynthesizedQA

        for qa in qa_pairs:
            sq = SynthesizedQA(
                id=uuid.uuid4(),
                session_id=uuid.UUID(session_id),
                question=qa.question,
                answer=qa.answer,
            )
            db.add(sq)

        db.commit()

    logger.info(
        "qa_synthesized",
        extra={
            "session_id": session_id,
            "qa_pairs": len(qa_pairs),
        },
    )

    return {
        "session_id": session_id,
        "run_id": run_id,
        "qa_pairs": len(qa_pairs),
    }


@app.task(name="tasks.validate_qa", bind=True)
def validate_qa(self, prev: dict, session_id: str, run_id: str) -> dict:
    """Validate synthesized Q&A pairs."""
    from training.knowledge.validator import QAValidator

    with _db() as db:
        from backend.models import SynthesizedQA

        qa_items = (
            db.execute(
                select(SynthesizedQA).where(
                    SynthesizedQA.session_id == uuid.UUID(session_id)
                )
            )
            .scalars()
            .all()
        )

    validator = QAValidator()
    validated_count = 0
    needs_review_count = 0

    for qa in qa_items:
        result = validator.validate(qa.question, qa.answer)

        if result.valid:
            qa.validated = True
            qa.validation_notes = result.notes
            validated_count += 1
        else:
            qa.retry_count += 1
            qa.validation_notes = result.notes

            if qa.retry_count >= 3:
                needs_review_count += 1

    db.commit()

    logger.info(
        "qa_validated",
        extra={
            "session_id": session_id,
            "validated": validated_count,
            "needs_review": needs_review_count,
        },
    )

    return {
        "session_id": session_id,
        "run_id": run_id,
        "validated": validated_count,
        "needs_review": needs_review_count,
    }


@app.task(name="tasks.merge_corpus", bind=True)
def merge_corpus(self, prev: dict, session_id: str, run_id: str) -> dict:
    """Merge validated Q&A into knowledge corpus."""
    from training.knowledge.corpus import CorpusManager

    with _db() as db:
        from backend.models import KnowledgeRecord, SynthesizedQA, KnowledgeCorpus

        records = (
            db.execute(
                select(KnowledgeRecord).where(
                    KnowledgeRecord.session_id == uuid.UUID(session_id)
                )
            )
            .scalars()
            .all()
        )

        qa_items = (
            db.execute(
                select(SynthesizedQA).where(
                    SynthesizedQA.session_id == uuid.UUID(session_id),
                    SynthesizedQA.validated == True,
                )
            )
            .scalars()
            .all()
        )

    knowledge_records = [{"topic": r.topic, "facts": r.facts} for r in records]

    synthesized_qa = [{"question": q.question, "answer": q.answer} for q in qa_items]

    corpus_manager = CorpusManager()
    entries = corpus_manager.merge(session_id, knowledge_records, synthesized_qa)

    with _db() as db:
        for entry in entries:
            ec = KnowledgeCorpus(
                id=uuid.uuid4(),
                topic=entry.topic,
                facts=entry.facts,
                source_session_id=uuid.UUID(session_id),
            )
            db.add(ec)

        db.commit()

    logger.info(
        "corpus_merged",
        extra={
            "session_id": session_id,
            "entries": len(entries),
        },
    )

    return {
        "session_id": session_id,
        "run_id": run_id,
        "corpus_entries": len(entries),
    }


@app.task(name="tasks.launch_training", bind=True)
def launch_training(self, prev: dict, session_id: str, run_id: str) -> dict:
    """Launch HuggingFace training job."""
    if not prev.get("sufficient", True):
        logger.info(
            "skipping_training_insufficient_data", extra={"session_id": session_id}
        )
        return prev

    from training.trainer.hf_launcher import HFTrainingLauncher
    from shared.s3_uploader import upload_training_config
    from shared.slack_notifier import training_started, artifact_uploaded

    dataset_s3 = prev["s3_path"]
    dataset_local = prev.get("local_path", "")

    launcher = HFTrainingLauncher()
    config = launcher.build_config(
        run_id=run_id,
        session_id=session_id,
        dataset_s3_path=dataset_s3,
        dataset_local_path=dataset_local,
    )

    config_s3 = upload_training_config(run_id, config)
    artifact_uploaded(run_id, "training_config", config_s3)

    hf_job_id = launcher.launch(config)

    with _db() as db:
        _update_run_status(
            run_id,
            "RUNNING",
            db,
            hf_job_id=hf_job_id,
            config=config,
            started_at=datetime.now(timezone.utc),
        )

    training_started(run_id, session_id, hf_job_id)
    logger.info("training_launched", extra={"run_id": run_id, "hf_job_id": hf_job_id})
    return {"session_id": session_id, "run_id": run_id, "hf_job_id": hf_job_id}


@app.task(name="tasks.poll_training", bind=True, max_retries=60, default_retry_delay=60)
def poll_training(self, prev: dict, session_id: str, run_id: str) -> dict:
    """Poll HuggingFace until training job completes. Retries every 60s for up to 1h."""
    if not prev.get("sufficient", True):
        logger.info(
            "skipping_poll_training_insufficient_data", extra={"session_id": session_id}
        )
        return prev

    from training.trainer.hf_launcher import HFTrainingLauncher
    from shared.s3_uploader import upload_training_logs, upload_adapter
    from shared.slack_notifier import (
        training_succeeded,
        training_failed,
        artifact_uploaded,
    )

    hf_job_id = prev["hf_job_id"]
    launcher = HFTrainingLauncher()
    status = launcher.poll(hf_job_id)

    if status == "running":
        raise self.retry()

    if status == "failed":
        error = launcher.get_error(hf_job_id)
        with _db() as db:
            _update_run_status(
                run_id, "FAILED", db, finished_at=datetime.now(timezone.utc)
            )
            _update_session_state(session_id, "FAILED", db)
            _set_failure_reason(
                session_id, str(error) or "Training job failed on remote.", db
            )
        training_failed(run_id, error)
        raise RuntimeError(f"Training failed: {error}")

    # succeeded — download artifacts
    adapter_dir = launcher.download_artifacts(hf_job_id, run_id)
    logs = launcher.get_logs(hf_job_id)

    logs_s3 = upload_training_logs(run_id, logs)
    artifact_s3 = upload_adapter(run_id, adapter_dir)

    artifact_uploaded(run_id, "adapter", artifact_s3)
    artifact_uploaded(run_id, "training_logs", logs_s3)

    with _db() as db:
        _update_run_status(
            run_id,
            "SUCCEEDED",
            db,
            logs_s3_path=logs_s3,
            artifact_s3_path=artifact_s3,
            finished_at=datetime.now(timezone.utc),
        )
        _update_session_state(session_id, "EVALUATING", db)

    training_succeeded(run_id, artifact_s3)
    return {
        "session_id": session_id,
        "run_id": run_id,
        "artifact_dir": str(adapter_dir),
    }


@app.task(name="tasks.run_evaluation", bind=True)
def run_evaluation(self, prev: dict, session_id: str, run_id: str) -> dict:
    """Run domain eval suite on new adapter."""
    if not prev.get("sufficient", True):
        logger.info(
            "skipping_evaluation_insufficient_data", extra={"session_id": session_id}
        )
        return prev

    from training.eval.evaluator import Evaluator
    from shared.s3_uploader import upload_eval_report
    from shared.slack_notifier import (
        evaluation_started,
        evaluation_completed,
        artifact_uploaded,
    )

    evaluation_started(run_id)

    artifact_dir = prev["artifact_dir"]
    evaluator = Evaluator()
    report = evaluator.run(adapter_dir=artifact_dir, run_id=run_id)

    eval_s3 = upload_eval_report(run_id, report)
    artifact_uploaded(run_id, "eval_report", eval_s3)

    passed = report["passed"]
    score = report["overall_score"]

    with _db() as db:
        _update_run_status(
            run_id,
            "SUCCEEDED",
            db,
            eval_s3_path=eval_s3,
            eval_passed=passed,
        )
        _update_session_state(session_id, "DEPLOYING", db)

    evaluation_completed(run_id, passed, score, eval_s3)
    return {
        "session_id": session_id,
        "run_id": run_id,
        "eval_passed": passed,
        "artifact_dir": artifact_dir,
    }


@app.task(name="tasks.deploy_or_rollback", bind=True)
def deploy_or_rollback(self, prev: dict, session_id: str, run_id: str) -> dict:
    """Promote new adapter or rollback if smoke test fails."""
    if not prev.get("sufficient", True):
        logger.info(
            "skipping_deployment_insufficient_data", extra={"session_id": session_id}
        )
        return prev

    from training.deployment.deploy import DeploymentManager
    from shared.slack_notifier import (
        deployment_approved,
        deployment_rejected,
        adapter_switch_succeeded,
        adapter_switch_failed,
        rollback_triggered,
        rollback_completed,
    )

    eval_passed = prev["eval_passed"]
    artifact_dir = prev["artifact_dir"]

    manager = DeploymentManager()

    if not eval_passed:
        deployment_rejected(run_id, reason="eval_failed")
        with _db() as db:
            _update_run_status(run_id, "FAILED", db)
            _update_session_state(session_id, "FAILED", db)
            _set_failure_reason(
                session_id,
                "Evaluation failed — the new adapter did not pass quality checks.",
                db,
            )
        return {"status": "rejected"}

    deployment_approved(run_id, version=run_id[:8])

    try:
        previous_version = manager.get_current_production_version()
        manager.promote(run_id=run_id, adapter_dir=artifact_dir)
        adapter_switch_succeeded(run_id, version=run_id[:8])

        # Smoke test
        if not manager.smoke_test():
            rollback_triggered(run_id, to_version=previous_version or "base")
            manager.rollback(to_version=previous_version)
            rollback_completed(run_id, to_version=previous_version or "base")
            with _db() as db:
                _update_run_status(run_id, "ROLLED_BACK", db)
                _update_session_state(session_id, "READY", db)
            return {"status": "rolled_back"}

        with _db() as db:
            _update_run_status(run_id, "SUCCEEDED", db)
            _update_session_state(session_id, "READY", db)
        return {"status": "deployed", "version": run_id[:8]}

    except Exception as exc:
        adapter_switch_failed(run_id, str(exc))
        with _db() as db:
            _update_run_status(run_id, "FAILED", db)
            _update_session_state(session_id, "FAILED", db)
            _set_failure_reason(session_id, f"Deployment error: {exc}", db)
        raise
