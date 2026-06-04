# LoRA Chat & Train — Student Textbook

A complete technical reference for students learning how this system works, why each
design decision was made, and how all the pieces fit together. Covers everything from
first principles through to production deployment.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [Large Language Models — A Primer](#2-large-language-models--a-primer)
3. [Fine-Tuning and LoRA](#3-fine-tuning-and-lora)
4. [System Architecture](#4-system-architecture)
5. [The Database](#5-the-database)
6. [The Backend API](#6-the-backend-api)
7. [The Chat Flow](#7-the-chat-flow)
8. [The Model Server](#8-the-model-server)
9. [The Training Pipeline](#9-the-training-pipeline)
10. [The Frontend UI](#10-the-frontend-ui)
11. [Infrastructure and Deployment](#11-infrastructure-and-deployment)
12. [Data Flow — End to End](#12-data-flow--end-to-end)
13. [Known Bugs Fixed and Why](#13-known-bugs-fixed-and-why)
14. [Glossary](#14-glossary)

---

## 1. What This System Does

This application teaches a small language model new things by having a conversation
with it. The workflow is:

1. **You chat** — paste a passage of text into the chat window.
2. **The system generates Q&A pairs** — the model reads your passage and produces
   question/answer pairs that capture the knowledge in it.
3. **You review** — a card-based UI lets you validate, edit, or delete each pair.
4. **Training runs** — once you have enough validated pairs, the system fine-tunes the
   model on them using LoRA, a technique that updates a tiny fraction of the model's
   weights without touching the rest.
5. **The new adapter goes live** — the updated model is hot-swapped into the serving
   layer and is immediately available for chat.

The result is a model that is better at answering questions about the specific topics
you taught it, while still behaving normally on everything else.

---

## 2. Large Language Models — A Primer

### What a language model is

A language model is a function that takes a sequence of tokens (words, subwords, or
characters) and predicts the probability distribution of the next token. By sampling
from this distribution repeatedly, you get generated text.

Modern large language models (LLMs) like Llama 3 or Qwen2.5 are transformer neural
networks with billions of parameters. They are pre-trained on hundreds of billions of
tokens of internet text, which gives them broad general knowledge and language
fluency.

### Tokens

Text is split into tokens before being fed to the model. A tokenizer maps text to
integer IDs from a fixed vocabulary. "Token" is roughly a word or syllable — "hello"
might be one token, "transformers" might be two.

Token counts matter because:
- The model has a **context window** — a maximum number of tokens it can process at
  once (e.g. 4096 tokens for Llama 3.2 1B).
- Inference cost and latency scale with token count.
- Training cost scales with sequence length.

### Chat templates

Instruction-tuned models (like the ones used here) are trained to follow a structured
format with role markers. For Llama 3.2:

```
<|system|>
You are a helpful assistant.
<|user|>
What is photosynthesis?
<|assistant|>
Photosynthesis is the process by which plants...
```

The tokenizer's `apply_chat_template()` method adds these markers automatically.
Training data must use the same format the model was fine-tuned on, otherwise the
model won't know when to start and stop generating.

### Quantization

A 1B parameter model in 32-bit float precision uses 4 GB of memory (1B × 4 bytes).
In 16-bit precision: 2 GB. With 4-bit NF4 quantization (used here): ~700 MB.

**NF4 (Normal Float 4)** stores each weight as one of 16 values chosen to minimize
quantization error for normally-distributed weights. The `bitsandbytes` library handles
this transparently. Compute still happens in `bfloat16` — only storage is 4-bit.

---

## 3. Fine-Tuning and LoRA

### Why fine-tune?

Pre-trained models have broad knowledge but may not behave exactly how you want:
- They may not know about your specific domain.
- They may use a style or format you don't want.
- They may lack knowledge of events after their training cutoff.

Fine-tuning trains the model on new examples so it learns to behave differently.
Full fine-tuning updates all parameters — expensive and risks overwriting general
knowledge (catastrophic forgetting). **Parameter-efficient fine-tuning (PEFT)**
methods update only a small number of new parameters.

### LoRA — Low-Rank Adaptation

LoRA (Hu et al., 2021) is the most widely used PEFT method. The key insight:
model weight matrices don't need to change in all directions to learn a new task.
They can change along a **low-rank subspace**.

For a weight matrix `W` of shape `(d, k)`, LoRA adds a bypass:

```
output = W·x + (B·A)·x · (α/r)
```

Where:
- `A` is a `(r, k)` matrix — initialised with random Gaussian values
- `B` is a `(d, r)` matrix — initialised to zero (so the adapter starts as identity)
- `r` is the **rank** — a small number like 8 or 16
- `α` is the **scaling factor** — usually set equal to `r`

During training, `W` is frozen. Only `A` and `B` are trained. The number of trainable
parameters is `2 × r × (d + k)` instead of `d × k`. For a typical transformer
attention projection (d=k=2048, r=16), this is ~65K instead of ~4M — a 60× reduction.

After training, `B·A` can be merged into `W` permanently (`merge_and_unload()`), or
kept as a separate **adapter** that is loaded on top of the base model at inference time.
This system uses the latter: the base model stays unchanged; adapters are hot-swapped.

### LoRA hyperparameters

| Parameter | Typical value | Effect |
|-----------|---------------|--------|
| `r` (rank) | 8–64 | Higher = more expressive adapter, more parameters, slower training |
| `alpha` | = r (or 2×r) | Scaling factor; effective learning rate scales as `alpha/r` |
| `dropout` | 0.05–0.1 | Regularization; set to 0 for small datasets |
| `target_modules` | `q_proj,v_proj` | Which weight matrices to apply LoRA to; query+value is standard |

### The SFT training format

**Supervised Fine-Tuning (SFT)** trains the model to produce specific outputs given
specific inputs. In this system, each training sample is a 2-turn conversation:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user",   "content": "What is photosynthesis?"},
    {"role": "assistant", "content": "Photosynthesis is the process..."}
  ]
}
```

The HuggingFace `SFTTrainer` (from the `trl` library) handles the chat template
application and loss masking automatically — it only computes loss on the assistant
tokens, not on the user prompt.

### Why this approach teaches the model

The model sees many examples of (question, expected answer) pairs. Gradient descent
adjusts the LoRA parameters so the model becomes more likely to produce the correct
answer. Over multiple epochs, the loss drops and the model "memorises" the knowledge
encoded in the training pairs — but because the base model is frozen, it doesn't
forget its other capabilities.

---

## 4. System Architecture

```
                          ┌──────────────────┐
                          │   Browser (user) │
                          └────────┬─────────┘
                                   │ HTTPS
                          ┌────────▼─────────┐
                          │      nginx       │  ← TLS termination, reverse proxy
                          └────────┬─────────┘
           ┌─────────────┬─────────┴──────────┬──────────────┐
           │ /           │ /api/*              │ /model/*     │
    ┌──────▼──────┐ ┌────▼────────────┐ ┌─────▼────────────┐ │
    │  frontend   │ │    backend      │ │   model_server   │ │
    │  Next.js    │ │   FastAPI       │ │  local_gpu_serve │ │
    │  :3000      │ │   :8000         │ │  :8001           │ │
    └─────────────┘ └────────┬────────┘ └──────────────────┘ │
                             │                      ▲          │
                      ┌──────┴──────┐               │          │
                      │  postgres   │         adapter_store     │
                      │  :5432      │         (named volume)    │
                      └─────────────┘                          │
                             │                                 │
                      ┌──────┴──────┐  ┌─────────────────────┐│
                      │    redis    │  │      worker          ││
                      │   :6379     │──│   Celery + training  ││
                      └─────────────┘  └─────────────────────┘│
```

### Service responsibilities

| Service | Technology | Responsibility |
|---------|-----------|----------------|
| `frontend` | Next.js 15, React 18, Tailwind | Chat UI, QA deck, session management, diagnostic panel |
| `backend` | FastAPI, SQLAlchemy (async), Pydantic v2 | REST API, SSE streaming, session state machine, QA synthesis orchestration |
| `model_server` | FastAPI, PyTorch, PEFT, bitsandbytes | Load/serve/train the LLM; adapter hot-swap |
| `worker` | Celery, SQLAlchemy (sync) | Async training pipeline orchestration |
| `postgres` | PostgreSQL 16 | Persistent storage for all application data |
| `redis` | Redis 7 | Celery message broker and result backend |
| `nginx` | nginx + Certbot | TLS termination, routing, basic auth |

### Communication patterns

| From → To | Protocol | Notes |
|-----------|----------|-------|
| Browser → nginx | HTTPS | All traffic; nginx proxies to services |
| frontend → backend | HTTP/1.1 + SSE | SSE for streaming; REST for everything else |
| backend → model_server | HTTP (sync via httpx) | For chat generation and adapter management |
| backend → redis | TCP | Celery task dispatch |
| worker → postgres | TCP (sync SQLAlchemy) | Celery runs synchronously |
| worker → model_server | HTTP | To trigger local training |
| model_server → disk | File I/O | Read/write adapter files on `adapter_store` volume |

---

## 5. The Database

PostgreSQL is the single source of truth for all application data. The schema has 9 tables.

### Sessions (`sessions`)

A session represents one "learning session" — the unit of training. The user chats
within a session, accumulates Q&A pairs, and then triggers training. After training,
a new session should be created to use the updated adapter.

| Column | Type | Purpose |
|--------|------|---------|
| `id` | UUID PK | Session identifier |
| `state` | VARCHAR | State machine state (see below) |
| `total_tokens` | INT | Tokens consumed so far |
| `max_tokens` | INT | Budget (default 4096) |
| `system_prompt` | TEXT | Chat system prompt override |
| `training_system_prompt` | TEXT | System prompt injected into training samples |
| `failure_reason` | TEXT | Error message when state = FAILED |

#### Session state machine

```
ACTIVE → PRE_SLEEP_WARNING (token budget low)
       → SLEEPING → TRAINING → EVALUATING → DEPLOYING → READY
                                                       ↘ FAILED
       → INSUFFICIENT_DATA ← (curation found < 10 samples; user continues chatting)
ACTIVE → VALIDATING (Phase 1 legacy path)
```

The state machine is enforced in `backend/main.py` via the `_transition()` helper
which validates allowed transitions and logs every state change.

### Turns (`turns`)

Each chat message (user or assistant) is a turn. In the new flow, only user turns
are created — assistant turns are not stored because the "reply" is Q&A pairs, not
a natural language response.

| Column | Type | Purpose |
|--------|------|---------|
| `session_id` | UUID FK | Parent session |
| `role` | VARCHAR | `user`, `assistant`, or `system` |
| `content` | TEXT | Message text |
| `token_count` | INT | Approximate tokens |

### Synthesized Q&A (`synthesized_qa`)

The central table for the new inline chat flow. Each row is one Q&A pair generated
from a user's passage.

| Column | Type | Purpose |
|--------|------|---------|
| `session_id` | UUID FK | Parent session |
| `source_turn_id` | UUID FK | The user turn that generated this pair |
| `question` | TEXT | Generated question |
| `answer` | TEXT | Generated answer |
| `validated` | BOOLEAN | True after user validates |
| `edited` | BOOLEAN | True if user changed question or answer |

### Training Candidates (`training_candidates`)

Used by the Phase 1 curation pipeline AND by the inline flow promotion step
(`_promote_qa_to_candidates()`). Each row is a multi-turn conversation segment
ready for the dataset writer.

| Column | Type | Purpose |
|--------|------|---------|
| `session_id` | UUID FK | Parent session |
| `conversation` | JSONB | `[{"role": "user", "content": "..."}, {"role": "assistant", ...}]` |
| `quality_score` | FLOAT | Curation score (0–1); 1.0 for user-validated QA |
| `included` | BOOLEAN | True = will be included in training dataset |
| `rejection_reason` | TEXT | Why excluded (or `"qa:<uuid>"` idempotency key for promoted pairs) |

### Other tables

| Table | Purpose |
|-------|---------|
| `datasets` | JSONL datasets built by `build_dataset`; stores S3 path and sample count |
| `training_runs` | One row per training attempt; tracks status, HF job ID, S3 paths for logs/artifacts |
| `model_versions` | Versioned adapters with eval scores and promotion timestamps |
| `deployment_events` | Audit log: PROMOTE, ROLLBACK, SMOKE_TEST_PASS, SMOKE_TEST_FAIL |
| `knowledge_records` | Structured facts extracted from candidates (Phase 1 legacy path) |

---

## 6. The Backend API

The backend is a FastAPI application in `backend/main.py`. All endpoints are async
using `asyncpg` as the database driver.

### Key design patterns

**Dependency injection for DB sessions:**
```python
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session

@app.get("/sessions/{id}")
async def get_session(id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    ...
```

Each request gets its own database session, automatically cleaned up on response.

**State machine enforcement:**
```python
async def _transition(session, new_state, db):
    session.state = new_state
    await db.commit()
    logger.info("state_transition", extra={"from": old_state, "to": new_state})
```

**SSE streaming:**
```python
async def _synthesize_and_stream(session, message, db):
    yield f"data: {json.dumps({'type': 'start', 'segment_count': n})}\n\n"
    for pair in qa_pairs:
        yield f"data: {json.dumps({'type': 'qa_pair', 'pair': pair})}\n\n"
    yield f"data: {json.dumps({'type': 'end'})}\n\n"

return StreamingResponse(generator(), media_type="text/event-stream")
```

The double `\n\n` after each `data:` line is required by the SSE protocol — the
browser's `EventSource` API uses it to delimit events.

### Session lifecycle endpoints

| Method | Path | What it does |
|--------|------|-------------|
| `POST` | `/sessions` | Create session; optionally load adapter first |
| `GET` | `/sessions` | List recent sessions (last 20) |
| `GET` | `/sessions/{id}` | Get session state, token count, prompts |
| `POST` | `/sessions/{id}/chat` | Send message; returns SSE stream of QA pairs |
| `GET` | `/sessions/{id}/turns` | Load all turns with attached QA pairs (for page reload) |
| `GET` | `/sessions/{id}/qa/count` | Validated/total QA counts for Start Training button |
| `PUT` | `/sessions/{id}/qa/{qa_id}` | Edit question, answer, or validated flag |
| `DELETE` | `/sessions/{id}/qa/{qa_id}` | Remove a QA pair |
| `POST` | `/sessions/{id}/start-training` | Promote QA → TrainingCandidate, enqueue Phase 2 |

---

## 7. The Chat Flow

### What happens when you send a message

This is the most important flow to understand. When the user submits text:

```
User types text → clicks Send
   ↓
frontend: POST /sessions/{id}/chat  {"message": "..."}
   ↓
backend: _synthesize_and_stream()
   │
   ├── 1. Validate session state (must be ACTIVE/PRE_SLEEP_WARNING/etc.)
   ├── 2. Save user Turn to DB
   ├── 3. Call synthesize_from_passage() in thread pool
   │        │
   │        └── POST /generate to model_server
   │             → model reads passage, returns Q&A pairs as JSON
   │
   ├── 4. Save SynthesizedQA rows to DB (one per pair)
   ├── 5. yield SSE: {"type": "start", "segment_count": N}
   ├── 6. yield SSE: {"type": "qa_pair", "pair": {...}} × N
   ├── 7. yield SSE: {"type": "qa_count", "total": X, "validated": Y, ...}
   └── 8. yield SSE: {"type": "end"}
   ↓
frontend: parseSSE()
   ├── on "start": set qaPairs=[], segmentCount=N on last message
   ├── on "qa_pair": append pair to qaPairs → InlineDeck card appears
   ├── on "qa_count": update Start Training button counter
   └── on "end": clear synthLoading flag → skeletons disappear
```

### The `/sleep` command

```
User types /sleep
   ↓
backend: _force_sleep()
   │
   ├── Count SynthesizedQA rows for this session
   │
   ├── If QA exists:
   │    ├── Mark all unvalidated pairs as validated
   │    ├── Call _promote_qa_to_candidates()    ← converts QA → TrainingCandidate
   │    ├── Transition session → TRAINING
   │    ├── enqueue_phase2_pipeline.delay()      ← Celery task chain
   │    └── yield SSE: {"type": "sleeping", ...}
   │
   └── If no QA (legacy path):
        ├── Transition session → VALIDATING
        ├── enqueue_phase1_pipeline.delay()      ← Phase 1 extraction
        └── yield SSE: {"type": "validating", ...}
```

### Token budget

Each session has a maximum token budget (`MAX_SESSION_TOKENS`, default 4096). Every
user message's tokens are added to `total_tokens`. When:
- `remaining < PRE_SLEEP_THRESHOLD` (default 512): session moves to `PRE_SLEEP_WARNING`
- `remaining <= 0`: session is forced to sleep automatically

This prevents infinite chatting and ensures the model doesn't run out of context.

---

## 8. The Model Server

The model server is a separate FastAPI process running on port 8001. It is the only
service that directly interacts with the GPU. The backend calls it over HTTP.

### Why a separate process?

1. **Isolation**: a crash in the model server doesn't crash the API.
2. **Hot-swap without downtime**: the model server can swap adapters while the API
   keeps serving other requests.
3. **Training on a background thread**: training runs on a daemon thread inside the
   model server — the HTTP server keeps responding to health checks during training.

### Loading the model

On startup, `local_gpu_serve.py` calls `_load_base_model()`:

```python
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    ),
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
```

`device_map="auto"` spreads layers across available GPUs (or GPU+CPU if VRAM is tight).
For a 1B model on a T4, the entire model fits in VRAM.

### Streaming inference

```python
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
thread = Thread(target=model.generate, kwargs={
    "inputs": input_ids,
    "streamer": streamer,
    "max_new_tokens": max_new_tokens,
    "do_sample": True,
    "temperature": temperature,
})
thread.start()
for token in streamer:
    yield token   # sent back to backend as SSE
```

`TextIteratorStreamer` is a HuggingFace utility that puts generated tokens into a queue
as the model produces them. This lets the HTTP response stream tokens one by one without
waiting for the full generation to complete.

### Adapter hot-swap

```python
@app.post("/reload_adapter")
async def reload_adapter(req: ReloadRequest):
    with _model_lock:    # threading.Lock — prevents concurrent access
        if req.adapter_dir == "base":
            # Detach LoRA adapter, return to base model
            if isinstance(_model, PeftModel):
                _model = _model.merge_and_unload()
            _adapter_path = None
        else:
            # Load new LoRA adapter on top of base model
            if isinstance(_model, PeftModel):
                _model = _model.merge_and_unload()   # unload previous first
            _model = PeftModel.from_pretrained(_model, req.adapter_dir)
            _adapter_path = req.adapter_dir
```

`merge_and_unload()` mathematically merges the LoRA matrices into the base model
weights and returns a plain `PreTrainedModel` — after this, the model is identical to
the base model. The LoRA parameters are discarded from memory.

---

## 9. The Training Pipeline

The pipeline is a Celery task chain. Each step is an independent async function that
reads its inputs from the database, does work, and returns a dict that is passed to
the next step.

### Why Celery?

Training takes 5–15 minutes. You don't want the HTTP request to stay open that long.
Celery lets the backend return immediately ("training started") and continue the work
in a background worker process. If the worker crashes, Celery can retry tasks.

### Phase 2 pipeline (the one triggered by the chat UI)

```
_promote_qa_to_candidates()     ← in backend, before enqueuing
    ↓
build_dataset(session_id)
    ├── Query TrainingCandidate WHERE session_id = X AND included = True
    ├── For each row: convert conversation → JSONL line
    │       {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    ├── Upload JSONL to S3 (or local filesystem)
    └── Save Dataset record to DB
    ↓
launch_training(session_id, run_id)
    ├── Build LoRA config (r, alpha, dropout, target_modules)
    ├── POST /train to model_server with {run_id, dataset_path}
    └── Return {local: True}
    ↓
poll_training(prev)
    └── If prev["local"] == True: SKIP (model server handles it internally)
    ↓
run_evaluation(session_id)
    ├── Load model + new adapter
    ├── Run 5 eval prompts
    ├── Score responses (relevance, completeness, grammar)
    └── Pass if avg score >= 0.65
    ↓
deploy_or_rollback(session_id)
    ├── If eval passed: copy adapter to adapters/current/, write manifest.json
    │                   POST /reload_adapter to model server
    │                   Run smoke test (one inference call)
    └── If eval failed or smoke test fails: keep previous adapter
        Transition session → READY or FAILED
```

### The dataset format

Each training sample is a JSON object with a `"messages"` key:

```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant..."},
  {"role": "user",   "content": "What is the capital of France?"},
  {"role": "assistant", "content": "The capital of France is Paris."}
]}
```

One sample per line (JSONL). The system prompt is always the first message, prepended
by `DatasetWriter`. The Q and A from the validated pair become the user and assistant
turns.

### Why Q&A pairs as training data?

The model learns from examples of questions followed by correct answers. During
inference, when the model sees a similar question, it's more likely to generate an
accurate answer because it has been reinforced for that pattern.

The quality of training data directly determines how well the model learns:
- **Too short**: not enough information for the model to learn from
- **Incorrect**: model learns wrong information
- **Poorly formatted**: model learns a bad style
- **Too many from one passage**: model overfits to one topic

This is why the validation step matters — garbage in, garbage out.

### Local training with SFTTrainer

```python
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir=output_dir,
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        bf16=True,
    ),
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model(output_dir)
```

`SFTTrainer` from the `trl` library handles:
- Chat template application (converting `messages` dicts to tokenized sequences)
- Loss masking (computing loss only on assistant tokens, not on the prompt)
- Gradient accumulation (simulating larger batch sizes)
- Mixed precision training (`bf16=True`)

---

## 10. The Frontend UI

### Architecture

The entire UI lives in a single file: `frontend/app/page.tsx`. This is a React client
component ("use client") — it runs in the browser and manages all state.

There is no routing, no Redux, no Zustand. State is plain React `useState` and
`useRef`. Callbacks flow down via props.

### The message model

Each message in the `messages[]` array is:

```typescript
interface Message {
  role: "user" | "assistant" | "system";
  content: string;
  id?: string;          // turn UUID for scoping QA mutations
  synthLoading?: boolean;  // true while SSE stream is still open
  segmentCount?: number;   // how many pairs the backend said it would produce
  qaPairs?: QAPair[];      // grows as qa_pair SSE events arrive
}
```

When a user sends a passage, a Message is appended with `synthLoading: true` and
`qaPairs: undefined`. As SSE events arrive:
1. `start` event: sets `qaPairs: []` and `segmentCount`
2. `qa_pair` events: each appends one `QAPair` to `qaPairs`
3. `end` event: clears `synthLoading`

The `InlineDeck` component reads `qaPairs` and renders one card at a time.

### The InlineDeck component

This is the core QA review UI. It is defined inline in `page.tsx` and rendered below
each user message bubble that has QA pairs.

Key state inside `InlineDeck`:
- `cardIdx`: which pair is currently shown (0-indexed)
- `slideDir`: `"left"` | `"right"` | `null` — direction of the current animation
- `animating`: true during the 260ms CSS transition

Navigation triggers a `translateX` CSS transition:
- Going forward (→): current card exits left (`-translate-x-full`), next card enters
  from right. React state updates (`cardIdx += 1`) happen after the transition completes
  via `setTimeout(260ms)`.
- Going back (←): mirror of the above.

### Polling

Three polling intervals run simultaneously in the background:

```
Every 5s:
  GET /health        → model server status, GPU stats
  GET /train/status  → training progress
  GET /outputs       → file listing for diagnostic panel
  GET /adapters      → adapter list

Every 10s (while session is active):
  GET /sessions/{id}/qa/count  → Start Training button counter

Every 3s (while session is in a non-terminal training state):
  GET /sessions/{id}  → session state transitions (TRAINING → EVALUATING → DEPLOYING → READY)
```

These are all managed with `useEffect` + `setInterval` + cleanup functions to prevent
memory leaks when the component unmounts.

---

## 11. Infrastructure and Deployment

### Docker Compose

All services are defined in `docker-compose.yml` with explicit health checks and
dependency ordering:

```yaml
backend:
  depends_on:
    postgres:
      condition: service_healthy   # wait for pg_isready
    redis:
      condition: service_healthy   # wait for redis-cli ping
```

### Named volumes

Two named volumes persist data across container restarts:

- **`adapter_store`**: mounted at `/adapters` in both `model_server` and `worker`.
  The worker writes new adapters here after training; the model server reads from it
  for hot-swap. Without this shared volume, training would produce an adapter that
  the model server could never find.

- **`hf_cache`**: mounted at `~/.cache/huggingface` in both GPU containers.
  The base model (~2.4 GB of weights) is downloaded once and cached here. Without
  this, every container restart would re-download the model.

### nginx as reverse proxy

nginx sits on the EC2 host (not containerised) and routes:
- `/` → frontend on :3000
- `/api/*` → backend on :8000 (with `/api` prefix stripped)
- `/model/*` → model server on :8001 (with `/model` prefix stripped)

Two critical settings for SSE to work:

```nginx
proxy_buffering off;     # disable response buffering — required for SSE
proxy_read_timeout 3600s; # allow long-lived connections
```

Without `proxy_buffering off`, nginx would buffer the entire SSE stream and deliver
it all at once at the end, defeating the purpose of streaming.

### Build-time vs runtime environment variables

Next.js has two kinds of environment variables:
- **Build-time** (`NEXT_PUBLIC_*`): baked into the JavaScript bundle at `npm run build`.
  If they change, the frontend image must be rebuilt.
- **Runtime** (server-side only): read from `process.env` at request time.
  Never accessible in the browser.

`NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_MODEL_SERVER_URL` are build-time. This is why
you set them in `.env` before running `docker compose build frontend`, not after.

---

## 12. Data Flow — End to End

Here is the complete journey of one piece of knowledge through the system:

```
1. User pastes a paragraph about photosynthesis into the chat.

2. Frontend POSTs to /sessions/{id}/chat.

3. Backend calls model_server /generate with a synthesis prompt:
   "Generate 3 Q&A pairs from this passage: [paragraph]"

4. Model server runs inference, returns JSON list of Q&A pairs.

5. Backend saves each pair as a SynthesizedQA row:
   {session_id, source_turn_id, question: "What is...", answer: "...", validated: false}

6. Backend streams qa_pair events to the browser over SSE.

7. Frontend appends each pair to the InlineDeck below the user bubble.

8. User clicks "Mark validated" on a pair:
   → Frontend PUTs /sessions/{id}/qa/{id} {validated: true}
   → Backend sets synthesized_qa.validated = True
   → Frontend auto-advances to next card

9. User clicks "Start Training" (enabled once 10+ pairs are validated):
   → Frontend POSTs /sessions/{id}/start-training
   → Backend calls _promote_qa_to_candidates():
       INSERT INTO training_candidates (session_id, conversation=[
         {role: "user", content: "What is photosynthesis?"},
         {role: "assistant", content: "Photosynthesis is..."}
       ], quality_score=1.0, included=true)
   → Backend transitions session → TRAINING
   → Backend calls enqueue_phase2_pipeline.delay(session_id)

10. Celery worker runs build_dataset:
    → Reads training_candidates WHERE included=True
    → Writes JSONL: {"messages": [system, user, assistant]}
    → Uploads to S3 (or local filesystem)

11. Celery worker runs launch_training:
    → POSTs /train to model_server with dataset_path
    → Model server runs SFTTrainer on a background thread

12. Training completes:
    → New adapter saved to adapters/current/
    → Model server hot-swaps to new adapter
    → Session transitions → DEPLOYING → READY

13. User starts a new session with the new adapter.
    → They ask "What is photosynthesis?"
    → The model now gives a better answer because it was trained on this Q&A pair.
```

---

## 13. Known Bugs Fixed and Why

### Bug 1: Training failed with "An error occurred while generating the dataset"

**Root cause:** A data source mismatch. The inline chat flow stores knowledge in the
`synthesized_qa` table. The `build_dataset` Celery task reads from `training_candidates`.
When Phase 2 was triggered directly (no Phase 1 curation), `training_candidates` was
empty. HuggingFace's `load_dataset()` called on an empty JSONL file throws this error.

**Fix:** Added `_promote_qa_to_candidates()` in `backend/main.py`. Called before
`enqueue_phase2_pipeline` from both `start_training` and `_force_sleep`. Converts each
validated `SynthesizedQA` into a `TrainingCandidate` row with the correct format.
`build_dataset` is unchanged.

**Lesson:** When two flows write to different tables but the same downstream task reads
from only one table, the gap must be bridged at the transition point. The fix is
idempotent (uses `rejection_reason = "qa:<uuid>"` as a deduplication key) so retries
are safe.

### Bug 2: Switching to "Base Model" in Glyph Chat had no effect

**Root cause:** In `backend/main.py`'s `direct_chat` endpoint, the adapter-loading
logic was:

```python
if request.adapter_id and request.adapter_id != "base":
    # call /reload_adapter to load adapter
    ...
# else: do nothing
```

The `do nothing` case meant the model server kept whatever adapter it had previously
loaded. The model server's `/reload_adapter` endpoint correctly handles `"base"` by
calling `merge_and_unload()`, but it was never being called.

**Fix:** Added an `else` branch that explicitly POSTs `{"adapter_dir": "base"}` to the
model server whenever `adapter_id == "base"`, ensuring LoRA weights are always detached
before inference.

**Lesson:** Null cases in conditional logic are often bugs in disguise. "Do nothing
for base model" should have been "call the correct unload path for base model".

---

## 14. Glossary

| Term | Definition |
|------|-----------|
| **Adapter** | A set of small weight matrices (LoRA) that modify a model's behaviour without changing the base weights |
| **Base model** | The pre-trained LLM before any fine-tuning; used as the starting point |
| **bfloat16** | 16-bit floating point format with the same exponent range as float32; preferred for training |
| **Celery** | Python distributed task queue; tasks are sent to Redis (broker) and executed by worker processes |
| **Chat template** | The structured format (role markers, special tokens) that instruction-tuned models expect |
| **Context window** | Maximum number of tokens a model can process in a single forward pass |
| **CUDA** | NVIDIA's GPU computing platform; required for GPU-accelerated training and inference |
| **DatasetWriter** | Class in `training/datasets/dataset_writer.py` that converts Q&A pairs to SFTTrainer-compatible JSONL |
| **Docker Compose** | Tool for defining and running multi-container Docker applications from a YAML file |
| **Fine-tuning** | Training a pre-trained model on new data to specialise its behaviour |
| **JSONL** | JSON Lines format — one JSON object per line; used for training datasets |
| **LoRA** | Low-Rank Adaptation; PEFT technique that adds small trainable matrices to frozen model weights |
| **merge_and_unload()** | PeftModel method that integrates LoRA matrices into base weights and returns a plain model |
| **nginx** | High-performance web server used here as a reverse proxy and TLS terminator |
| **NF4** | Normal Float 4; 4-bit quantization format from bitsandbytes; stores weights in 1/8th the memory |
| **PEFT** | Parameter-Efficient Fine-Tuning; techniques that train a small fraction of parameters |
| **Quantization** | Storing model weights in reduced precision (e.g. 4-bit instead of 32-bit) to save memory |
| **rank (r)** | The LoRA hyperparameter controlling the dimension of the low-rank decomposition |
| **Redis** | In-memory data store; used here as the Celery message broker |
| **Session** | One user learning session; the unit of training data collection and pipeline execution |
| **SFT** | Supervised Fine-Tuning; training the model to produce specific outputs for specific inputs |
| **SFTTrainer** | HuggingFace `trl` library class that implements supervised fine-tuning with chat template support |
| **SSE** | Server-Sent Events; HTTP-based unidirectional streaming from server to browser |
| **SynthesizedQA** | Database table storing Q&A pairs generated from user passages |
| **Token** | Basic unit of text after tokenization; roughly a word or syllable |
| **TrainingCandidate** | Database table storing conversation segments ready for dataset building |
| **VRAM** | Video RAM; GPU memory where model weights, activations, and gradients are stored during inference/training |
