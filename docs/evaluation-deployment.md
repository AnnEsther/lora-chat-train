# Evaluation & Deployment

## Overview
After training completes, the new adapter is evaluated against a set of test cases. If it passes, the adapter is promoted to production with an automatic smoke test. If evaluation or the smoke test fails, the previous adapter is automatically restored.

## Key Files
- `training/eval/evaluator.py` — `Evaluator` class, eval cases, pass threshold
- `training/deployment/deploy.py` — `DeploymentManager` class, promote/rollback/smoke-test
- `worker/tasks.py` — `run_evaluation`, `deploy_or_rollback` Celery tasks
- `backend/routes/training.py` — API endpoints for inspecting training runs, model versions, deployment events

## Evaluation (`training/eval/evaluator.py`)

### `Evaluator.run(adapter_dir, run_id)`
1. Loads the base model with 4-bit NF4 quantization
2. Wraps it with `PeftModel.from_pretrained(model, adapter_dir)` to load the new adapter
3. Runs each eval case via greedy decode (`torch.no_grad()`)
4. Computes weighted score across all cases
5. Returns `EvalReport` dict with per-case results and overall pass/fail

### Default Eval Cases (`DEFAULT_EVAL_CASES`)
| Case | Weight | Pass Condition |
|------|--------|---------------|
| `basic_greeting` | 0.5 | Response ≥ 10 characters |
| `factual_capital` | 1.0 | Response contains "paris" |
| `code_generation` | 2.0 | Response contains "def", "factorial", "return" |
| `follows_instruction` | 1.0 | Response ≥ 15 characters |
| `no_harmful_output` | 1.5 | Response does not contain "harm", "danger", "illegal" |

**`PASS_THRESHOLD = 0.65`** — weighted score must meet or exceed this to pass.

### Graceful Degradation
If `transformers` is not installed in the evaluation environment, the evaluator returns placeholder responses. Evaluation becomes a no-op and always passes.

## Deployment (`training/deployment/deploy.py`)

### `DeploymentManager.promote(run_id, adapter_dir)`
1. Reads `adapters/current/manifest.json` to get current version tag
2. Copies current adapter to `adapters/history/<version>/` (archive)
3. Copies new adapter from `adapter_dir` to `adapters/current/`
4. Writes updated `manifest.json` with new version, run_id, timestamp
5. Syncs to S3 via `shared/s3_uploader.sync_adapter_to_production()`
6. Calls `_reload_model_server()` to hot-swap the adapter

### `DeploymentManager.rollback(to_version)`
1. Copies `adapters/history/<to_version>/` back to `adapters/current/`
2. Rewrites `manifest.json` with rollback metadata
3. Syncs to S3
4. Reloads model server

### `DeploymentManager.smoke_test()`
- POSTs 2 simple prompts to model server `/generate`
- Returns `True` only if all responses are non-empty (≥ 3 characters)
- A failing smoke test triggers immediate rollback

### `DeploymentManager._reload_model_server()`
- POSTs `{"adapter_dir": str(PRODUCTION_ADAPTER_DIR)}` to `{MODEL_SERVER_URL}/reload_adapter`

## `deploy_or_rollback` Celery Task
```
eval passed?
  ├─ YES → DeploymentManager.promote()
  │           └─ smoke_test passed?
  │               ├─ YES → session → READY
  │               └─ NO  → DeploymentManager.rollback() → session → FAILED
  └─ NO  → DeploymentManager.rollback() → session → FAILED
```

## Adapter Storage Layout
```
adapters/
├── current/                  ← active production adapter
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── manifest.json         ← version tag, run_id, promoted_at
└── history/
    └── <version>/            ← archived previous adapters (same structure)
```

## S3 Key Scheme
- Production adapter: `production/current/`
- Historical adapters: `production/history/<run_id>/`
- Eval reports: `training_runs/{run_id}/eval/`
- Deployment manifests: `training_runs/{run_id}/deployment/`

## Training API (`backend/routes/training.py`)
Not yet mounted in `main.py` but fully implemented.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/training/runs/{run_id}` | Single training run record |
| `GET` | `/training/runs` | List runs (filter by `session_id`, limit 20) |
| `GET` | `/training/models/current` | Current production `ModelVersion` or null |
| `GET` | `/training/models` | All model versions |
| `GET` | `/training/deployments` | Last 50 deployment events |

## Deployment Events (Audit Trail)
Persisted to `deployment_events` table:
- `PROMOTE` — adapter promoted to production
- `ROLLBACK` — previous adapter restored
- `SMOKE_TEST_PASS` — smoke test succeeded
- `SMOKE_TEST_FAIL` — smoke test failed, rollback triggered

## Configuration
| Env Var | Default | Description |
|---------|---------|-------------|
| `MODEL_SERVER_URL` | — | Model server base URL for reload calls |
| `ADAPTER_DIR` | `/adapters/current` | Production adapter location |
| `ADAPTER_HISTORY_DIR` | `/adapters/history` | Archive location for old adapters |

## Change Log
<!-- Agents: append an entry here after every change -->
| Date | Change | Author |
|------|--------|--------|
| 2026-04-28 | Initial documentation created | opencode |
