# Curation

## Overview
The curation module scores every extracted turn pair on four quality dimensions and filters out those below a threshold. It determines what data actually enters the training dataset.

## Key Files
- `training/curator/curator.py` — `Curator` class, scoring logic, `INCLUSION_THRESHOLD`
- `training/extractor/transcript_extractor.py` — Upstream step that produces candidates for curation
- `worker/tasks.py` — `curate_candidates` Celery task that invokes `Curator`

## Scoring Dimensions

Each dimension produces a score in `[0.0, 1.0]`. The final score is a weighted sum.

| Dimension | Weight | Logic |
|-----------|--------|-------|
| `length` | 0.25 | Optimal range: 80–600 chars assistant response. Scores below/above are penalised linearly. |
| `specificity` | 0.25 | Numbers present → +0.4. Code block present → +0.4. Word count > 15 → +0.2. Capped at 1.0. |
| `coherence` | 0.25 | Jaccard-like word overlap between user turn and assistant turn. |
| `safety` | 0.25 | Hard-exclude: `[REDACTED` prefix, refusal phrases, error/traceback mentions. Any trigger → 0.0. |

**`INCLUSION_THRESHOLD = 0.5`** — candidates with `score < 0.5` are excluded.

## `Curator` Class Interface

```python
class Curator:
    def score_and_filter(
        self, candidates: list[Candidate]
    ) -> list[ScoredCandidate]:
        ...
```

Returns the same list with added fields: `score`, `included`, `rejection_reason`.

## `ScoredCandidate` Fields
| Field | Type | Description |
|-------|------|-------------|
| `user_turn` | `str` | User message |
| `assistant_turn` | `str` | Assistant message |
| `turn_index` | `int` | Position in original transcript |
| `score` | `float` | Weighted composite score `[0.0, 1.0]` |
| `included` | `bool` | `True` if `score >= INCLUSION_THRESHOLD` |
| `rejection_reason` | `str \| None` | Human-readable reason for exclusion |

## Upstream: `TranscriptExtractor`
Before curation, `extract_candidates` runs extraction which:
- Iterates turns to find adjacent `user → assistant` pairs
- Applies 5 PII redaction patterns (API keys, passwords, credit cards, emails, cloud secrets)
- Rejects: too short (< 20 chars either side), too long assistant (> 4000 chars), `/sleep` commands, `system:` prefixed turns

Constants: `MIN_TURN_CHARS = 20`, `MAX_TURN_CHARS = 4000`

## Pipeline Integration
```
extract_candidates (TranscriptExtractor)
    ↓
curate_candidates (Curator)
    ├─ if included_count < MIN_TRAINING_SAMPLES → session → INSUFFICIENT_DATA (stop)
    └─ else → extract_knowledge → ...
```

## `MIN_TRAINING_SAMPLES`
Controlled by `MIN_TRAINING_SAMPLES` env var (default `10`). If curation produces fewer included samples than this, the entire Phase 1 pipeline halts and the session enters `INSUFFICIENT_DATA`. The user can continue chatting and re-trigger `/sleep` to try again.

## Change Log
<!-- Agents: append an entry here after every change -->
| Date | Change | Author |
|------|--------|--------|
| 2026-04-28 | Initial documentation created | opencode |
