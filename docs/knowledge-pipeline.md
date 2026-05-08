# Knowledge Pipeline

## Overview
The knowledge pipeline sits between curation and dataset building (Phase 1). It transforms raw conversation turn pairs into structured facts, synthesizes Q&A pairs from those facts, validates the Q&A quality, and merges everything into a cross-session knowledge corpus.

## Key Files
- `training/knowledge/extractor.py` — `KnowledgeExtractor` — topic + intent classification
- `training/knowledge/normalizer.py` — `KnowledgeNormalizer` — structured fact extraction
- `training/knowledge/synthesizer.py` — `QASynthesizer` — model-powered Q&A generation
- `training/knowledge/validator.py` — `QAValidator` — automated quality scoring
- `training/knowledge/corpus.py` — `CorpusManager` — cross-session deduplication + merging
- `worker/tasks.py` — `extract_knowledge`, `synthesize_qa`, `validate_qa`, `merge_corpus` Celery tasks

## Pipeline Steps

### 1. `KnowledgeExtractor` — Topic & Intent Classification
**Input:** `(user_turn, assistant_turn)` pair
**Output:** `list[ExtractedTopic]`

- Classifies into one of 10 topic domains using keyword matching against `TOPIC_PATTERNS`:
  `programming`, `math`, `science`, `history`, `language`, `cooking`, `health`, `technology`, `finance`, `general`
- Classifies user intent: `question | explanation | task | conversation` via regex on the user turn
- Each `ExtractedTopic` carries: `topic`, `subtopics: list[str]`, `keywords: list[str]`, `intent: str`
- There is **no** `confidence` field on `ExtractedTopic`

### 2. `KnowledgeNormalizer` — Structured Fact Extraction
**Input:** `user_turn`, `assistant_turn`, `list[ExtractedTopic]`
**Output:** `list[KnowledgeRecord]`

- Converts raw text into typed fact records
- Handles 5 fact types: `qa_pair`, `fact`, `code_example`, `step`, `task`
- Each `KnowledgeRecord` carries: `topic: str`, `facts: list[dict]`, `source_type: str`
- The primary extraction always creates a `qa_pair` (LLM-asks/user-answers framing), then extracts factual sentences and code blocks from the assistant turn

### 3. `QASynthesizer` — Q&A Generation
**Input:** `list[KnowledgeRecord]`, `system_prompt`
**Output:** `list[SynthesizedQA]`

- Calls the model server's `/generate` endpoint per fact
- Prompt asks model to return JSON: `[{"question": "...", "answer": "..."}]`
- Generates 2–3 Q&A pairs per knowledge fact
- **Fallback 1:** If JSON parsing fails, attempts plaintext `Q:` / `A:` parsing
- **Fallback 2:** If model call fails entirely, uses the raw fact as both question and answer

### 4. `QAValidator` — Quality Scoring
**Input:** `question: str`, `answer: str`
**Output:** `ValidationResult(valid, score, notes)`

Scores on 4 dimensions:
| Dimension | Weight | Logic |
|-----------|--------|-------|
| `relevance` | 0.30 | Keyword overlap between question and answer |
| `grammar` | 0.20 | Penalty for very short sentences, unusual punctuation patterns |
| `completeness` | 0.25 | Minimum answer length, penalise truncated/incomplete answers |
| `accuracy` | 0.25 | Penalise refusal phrases ("as an AI", "I can't"), excessive uncertainty markers |

**`VALIDATION_THRESHOLD = 0.6`** — items below this threshold are flagged for retry.
**`should_retry(result, retry_count)`** — returns `True` if failed and `retry_count < 3`. After 3 failures, auto-marked for human review.

### 5. `CorpusManager` — Cross-Session Knowledge Base
**Input:** `session_id`, `list[KnowledgeRecord]`, `list[SynthesizedQA]`
**Output:** `list[CorpusEntry]`

- Groups facts by topic
- Deduplicates using `(type, content[:100])` as a key — prevents near-duplicate facts accumulating across sessions
- `search(query, corpus)` — keyword search across topic labels and fact content
- `get_stats(corpus)` — returns `{total_entries, total_facts, topics: {topic: count}}`

## Database Tables
- `knowledge_records` — structured facts with topic, facts (JSONB), source_turn_id
- `synthesized_qa` — Q&A pairs with `validated` flag, `edited`, `retry_count`, `validation_notes`
- `knowledge_corpus` — cross-session merged, deduplicated knowledge base

## QA Review Modal (Frontend)
- Auto-opens when session enters `VALIDATING` state
- Header shows total pair count and a progress bar indicating how many have been validated
- Card-by-card navigation (Previous / Next) with an unsaved-changes guard: if edits are pending, the user is prompted to Save & Continue or Discard & Continue before moving
- Inline editing of question and answer text; an "Unsaved changes" indicator appears on the card when local edits exist
- "Save changes" button appears on the card only when there are unsaved edits
- Per-item "Mark as Validated" / "Unmark validation" toggle button
- Validated cards have a green left border; needs-review cards have an amber border
- "Delete entry" button in the navigation row; triggers an inline confirmation before deleting (calls `DELETE /sessions/{id}/qa/{qa_id}`)
- "Validate All & Start Training" — calls `POST /sessions/{id}/qa/validate-mark` then `POST /sessions/{id}/start-training`
- Shows `validation_notes` labelled as "Automated validator notes" on each card

## API Endpoints (QA Review)
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/sessions/{id}/qa` | List all synthesized Q&A for the session |
| `PUT` | `/sessions/{id}/qa/{qa_id}` | Edit a Q&A pair (question, answer, validated flag) |
| `DELETE` | `/sessions/{id}/qa/{qa_id}` | Permanently delete a Q&A entry |
| `POST` | `/sessions/{id}/qa/validate-mark` | Bulk-mark all Q&A as validated |
| `POST` | `/sessions/{id}/start-training` | Trigger Phase 2 after user validates QA |

## Change Log
<!-- Agents: append an entry here after every change -->
| Date | Change | Author |
|------|--------|--------|
| 2026-05-08 | Add DELETE /sessions/{id}/qa/{qa_id} endpoint; overhaul QA Review Modal: progress bar, card status badges, inline delete confirmation, unsaved-changes guard, Save button, unmark validation | opencode |
| 2026-05-08 | Fix ExtractedTopic fields (subtopics + keywords + intent, no confidence); fix topic domains (business→finance); fix KnowledgeRecord fact types (5 types, not 6; no definition type); add source_type to KnowledgeRecord | opencode |
| 2026-04-28 | Initial documentation created | opencode |
| 2026-05-05 | Update KnowledgeRecord output to match corrected model (topic, facts list[dict]) | opencode |
