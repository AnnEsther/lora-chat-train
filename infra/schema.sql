-- LoRA Chat & Train — Postgres schema
-- Run via: psql $DATABASE_URL -f infra/schema.sql

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── Sessions ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    state           TEXT NOT NULL DEFAULT 'ACTIVE'
                    CHECK (state IN (
                        'ACTIVE','PRE_SLEEP_WARNING','SLEEPING',
                        'TRAINING','EVALUATING','DEPLOYING',
                        'READY','FAILED'
                    )),
    total_tokens    INTEGER NOT NULL DEFAULT 0,
    max_tokens      INTEGER NOT NULL DEFAULT 4096,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at       TIMESTAMPTZ,
    metadata        JSONB NOT NULL DEFAULT '{}'
);

-- ── Turns ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS turns (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
    content         TEXT NOT NULL,
    token_count     INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata        JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, created_at);

-- ── Training candidates ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS training_candidates (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES sessions(id),
    user_turn       TEXT NOT NULL,
    assistant_turn  TEXT NOT NULL,
    quality_score   FLOAT,
    included        BOOLEAN NOT NULL DEFAULT FALSE,
    rejection_reason TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_candidates_session ON training_candidates(session_id);

-- ── Datasets ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS datasets (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES sessions(id),
    s3_path         TEXT NOT NULL,
    sample_count    INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Training runs ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS training_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES sessions(id),
    dataset_id      UUID REFERENCES datasets(id),
    status          TEXT NOT NULL DEFAULT 'PENDING'
                    CHECK (status IN (
                        'PENDING','RUNNING','SUCCEEDED','FAILED','ROLLED_BACK'
                    )),
    hf_job_id       TEXT,
    config          JSONB NOT NULL DEFAULT '{}',
    logs_s3_path    TEXT,
    artifact_s3_path TEXT,
    eval_s3_path    TEXT,
    eval_passed     BOOLEAN,
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Model versions ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_versions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id          UUID NOT NULL REFERENCES training_runs(id),
    version_tag     TEXT NOT NULL,               -- e.g. "v0.4"
    adapter_s3_path TEXT NOT NULL,
    is_production   BOOLEAN NOT NULL DEFAULT FALSE,
    eval_score      FLOAT,
    promoted_at     TIMESTAMPTZ,
    retired_at      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_model_versions_prod ON model_versions(is_production);

-- ── Deployment events ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS deployment_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id          UUID NOT NULL REFERENCES training_runs(id),
    event_type      TEXT NOT NULL
                    CHECK (event_type IN (
                        'PROMOTE','ROLLBACK','SMOKE_TEST_PASS','SMOKE_TEST_FAIL'
                    )),
    from_version    TEXT,
    to_version      TEXT,
    reason          TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Auto-update updated_at ───────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION touch_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$;

CREATE OR REPLACE TRIGGER sessions_updated_at
BEFORE UPDATE ON sessions
FOR EACH ROW EXECUTE FUNCTION touch_updated_at();
