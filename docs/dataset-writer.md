# Dataset Writer

## Overview
The dataset writer converts curated multi-turn conversation segments into a JSONL file formatted for HuggingFace SFTTrainer. It is the final data transformation step before training begins.

## Key Files
- `training/datasets/dataset_writer.py` — `DatasetWriter` class
- `worker/tasks.py` — `build_dataset` Celery task

## `DatasetWriter` Class

### `__init__(system_prompt: str)`
- `system_prompt` defaults to the "eager student" persona if not provided
- The same system prompt is injected into every training example

### `write_jsonl(samples: list[dict]) -> str`
- Accepts a list of dicts, each with a `"conversation"` key containing a `list[dict]` of `{"role", "content"}` turns (the format produced by `TrainingCandidate.conversation`)
- Returns a JSONL string; each line is a JSON object

### `write_to_file(samples: list[dict], path: str) -> Path`
- Calls `write_jsonl()` and writes the result to disk at `path`; returns the `Path` of the written file

## Output Format
Each line in the JSONL file:
```json
{
  "messages": [
    {"role": "system", "content": "<system_prompt>"},
    {"role": "user",   "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

The system prompt is prepended once; the remaining turns come directly from the `conversation` list (which may contain multiple alternating user/assistant turns). This format is directly compatible with HuggingFace `SFTTrainer` when using `tokenizer.apply_chat_template()`.

## `build_dataset` Celery Task
- Queries included `TrainingCandidate` rows from DB for the session (where `included = True`)
- Instantiates `DatasetWriter` with the session's `training_system_prompt` (falls back to default if null)
- Passes each candidate's `conversation` field as a `{"conversation": [...]}` dict
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
| 2026-05-08 | Updated to reflect multi-turn conversation segments: write_jsonl now accepts conversation list[dict] not user/assistant pair dicts; write_to_file returns Path; output format note updated | opencode |
| 2026-04-28 | Initial documentation created | opencode |
