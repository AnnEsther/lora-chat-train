# Curation

## Overview
The curation module scores every extracted **conversation segment** on four quality dimensions and filters out those below a threshold. It determines what data actually enters the training dataset.

Segments are multi-turn sliding windows of `EXTRACTION_WINDOW_SIZE` consecutive turns (default 4), not isolated single-exchange pairs. This means a segment can contain multiple questions and answers from a single natural exchange.

## Key Files
- `training/curator/curator.py` — `Curator` class, scoring logic, `INCLUSION_THRESHOLD`
- `training/extractor/transcript_extractor.py` — Upstream step that produces candidates for curation
- `worker/tasks.py` — `curate_candidates` Celery task that invokes `Curator`

## Scoring Dimensions

Each dimension produces a score in `[0.0, 1.0]`. The final score is a weighted sum.

| Dimension | Weight | Logic |
|-----------|--------|-------|
| `length` | 0.25 | Total length of all assistant turns in the segment. Optimal range: 80–1200 chars. |
| `specificity` | 0.25 | Numbers present → +0.4. Code block present → +0.4. Word count > 15 → +0.2. Capped at 1.0. |
| `coherence` | 0.25 | Jaccard-like word overlap between all user content and all assistant content in the segment. |
| `safety` | 0.25 | Hard-exclude: `[REDACTED` prefix, refusal phrases, error/traceback mentions in any turn. Any trigger → 0.0. |

**`INCLUSION_THRESHOLD = 0.5`** — candidates with `score < 0.5` are excluded.

## `Curator` Class Interface

```python
class Curator:
    def score_and_filter(
        self, candidates: list[dict]  # each has "conversation": list[dict]
    ) -> list[dict]:
        ...
```

Returns the same list with added fields: `score`, `included`, `rejection_reason`.

## `ScoredCandidate` Fields
| Field | Type | Description |
|-------|------|-------------|
| `conversation` | `list[dict]` | Full multi-turn segment `[{"role", "content"}, ...]` |
| `score` | `float` | Weighted composite score `[0.0, 1.0]` |
| `included` | `bool` | `True` if `score >= INCLUSION_THRESHOLD` |
| `rejection_reason` | `str \| None` | Human-readable reason for exclusion |

## Upstream: `TranscriptExtractor`
Before curation, `extract_candidates` runs extraction which:
- Uses a **sliding window** of `EXTRACTION_WINDOW_SIZE` consecutive turns (configurable via env var, default 4)
- Advances the window one turn at a time, producing overlapping segments that preserve multi-turn context
- Applies 5 PII redaction patterns to every turn (API keys, passwords, credit cards, emails, cloud secrets)
- Rejects segments that: lack both a user and an assistant turn; contain a `/sleep` command or `system:` prefix; have any turn shorter than 20 chars; have any assistant turn longer than 4000 chars

Constants: `MIN_TURN_CHARS = 20`, `MAX_ASSISTANT_TURN_CHARS = 4000`, `EXTRACTION_WINDOW_SIZE` (env, default 4)

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
| 2026-04-29 | Switched from single-pair extraction to sliding-window conversation segments; Curator updated to score across full segment; EXTRACTION_WINDOW_SIZE env var added | opencode |
| 2026-04-28 | Initial documentation created | opencode |
