# Dataset Writer

## Overview
The dataset writer converts curated `(user, assistant)` turn pairs into a JSONL file formatted for HuggingFace SFTTrainer. It is the final data transformation step before training begins.

## Key Files
- `training/datasets/dataset_writer.py` — `DatasetWriter` class
- `worker/tasks.py` — `build_dataset` Celery task

## `DatasetWriter` Class

### `__init__(system_prompt: str)`
- `system_prompt` defaults to the "eager student" persona if not provided
- The same system prompt is injected into every training example

### `write_jsonl(samples: list[dict]) -> str`
- Accepts a list of `{"user": ..., "assistant": ...}` dicts (curated candidates)
- Returns a JSONL string; each line is a JSON object

### `write_to_file(samples: list[dict], path: str) -> None`
- Calls `write_jsonl()` and writes the result to disk at `path`

## Output Format
Each line in the JSONL file:
```json
{
  "messages": [
    {"role": "system", "content": "<system_prompt>"},
    {"role": "user",   "content": "<user_turn>"},
    {"role": "assistant", "content": "<assistant_turn>"}
  ]
}
```

This format is directly compatible with HuggingFace `SFTTrainer` when using `dataset_text_field="text"` after applying `tokenizer.apply_chat_template()`.

## `build_dataset` Celery Task
- Queries included `TrainingCandidate` rows from DB for the session
- Instantiates `DatasetWriter` with the session's `training_system_prompt` (falls back to default if null)
- Calls `write_to_file()` to write the JSONL locally
- Uploads JSONL to S3 via `shared/s3_uploader.upload_dataset_jsonl()`
- Persists a `Dataset` DB record with S3 path + sample count

## S3 Key
`sessions/{session_id}/dataset/dataset.jsonl`

## System Prompt
The training system prompt is configurable per session via the session creation modal in the frontend. It determines the persona and context injected into every training example. If not set, the default "eager student" persona is used.

To customise: set `training_system_prompt` in `POST /sessions` request body or via the frontend new-session modal.

## Change Log
<!-- Agents: append an entry here after every change -->
| Date | Change | Author |
|------|--------|--------|
| 2026-04-28 | Initial documentation created | opencode |
