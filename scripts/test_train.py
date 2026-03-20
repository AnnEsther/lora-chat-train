# scripts/test_train.py
import sys, os, json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")
sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("backend").resolve()))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset as HFDataset
from trl import SFTConfig, SFTTrainer

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
HF_TOKEN   = os.environ.get("HF_TOKEN", "")

# ── Find dataset ───────────────────────────────────────────────────────────────
output_dir   = Path(os.environ.get("LOCAL_OUTPUT_DIR", "outputs"))
dataset_path = sorted(output_dir.rglob("dataset.jsonl"))[0]
print(f"Dataset: {dataset_path}")

records = [json.loads(l) for l in dataset_path.read_text().splitlines() if l.strip()]
print(f"Samples: {len(records)}")

# ── Tokenizer ──────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN or None)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── Format dataset ─────────────────────────────────────────────────────────────
def fmt(ex):
    return {"text": tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False)}

hf_dataset = HFDataset.from_list(records).map(fmt)

# ── Model — force float16 explicitly to avoid bfloat16 issues ─────────────────
# — use bf16 throughout, don't mix with fp16 ──────────────────────────
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # match the model's native dtype
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map="auto",
    token=HF_TOKEN or None,
    torch_dtype=torch.bfloat16,             # load everything in bf16
)
model.config.use_cache = False

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none", task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ── Train ──────────────────────────────────────────────────────────────────────
adapter_out = Path("outputs/test_adapter")
adapter_out.mkdir(parents=True, exist_ok=True)

cfg = SFTConfig(
    output_dir=str(adapter_out),
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=False,                             # disable fp16
    bf16=True,                              # use bf16 — matches Qwen's native dtype
    optim="paged_adamw_8bit",
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    dataloader_pin_memory=False,
    max_length=256,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=hf_dataset,
    processing_class=tokenizer,
)

print("Training...")
trainer.train()
model.save_pretrained(str(adapter_out))
tokenizer.save_pretrained(str(adapter_out))
print(f"Done — adapter saved to {adapter_out}")