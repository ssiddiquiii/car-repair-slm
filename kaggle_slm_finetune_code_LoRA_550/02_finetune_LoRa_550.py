# ============================================================
# CELL 1 — Environment verification
# ============================================================

import torch, sys

print("=" * 60)
print("KAGGLE ENVIRONMENT CHECK")
print("=" * 60)
print(f"\nPython:  {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.cuda.is_available()}")

if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
    print(f"\nGPU count: {n_gpu}")
    for i in range(n_gpu):
        name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {name} ({vram:.1f} GB)")
    
    if "T4" in torch.cuda.get_device_name(0):
        print("\n✓ T4 confirmed")
else:
    print("\n✗ NO GPU — enable from sidebar")

try:
    from kaggle_secrets import UserSecretsClient
    us = UserSecretsClient()
    _ = us.get_secret("HF_TOKEN")
    _ = us.get_secret("GROQ_API_KEY")
    print("✓ Secrets loaded")
except Exception as e:
    print(f"✗ Secrets issue: {e}")

# ============================================================
# CELL 2 — Install Unsloth for optimized Gemma-4 QLoRA
# ============================================================

# Unsloth has custom path for Gemma-4 E2B-it (handles shared KV + PLE)
# !pip install -q -U unsloth  #uncomment when run!!
# !pip install -q -U "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" #uncomment when run!!

# Supporting libraries
# !pip install -q -U trl peft accelerate bitsandbytes datasets #uncomment when run!!
# !pip install -q -U huggingface_hub sentencepiece protobuf #uncomment when run!!

import transformers, peft, trl, accelerate, bitsandbytes, torch
try:
    import unsloth
    unsloth_ver = getattr(unsloth, '__version__', 'installed')
except ImportError:
    unsloth_ver = "NOT INSTALLED"

print("=" * 60)
print("VERSION CHECK")
print("=" * 60)
print(f"  unsloth:       {unsloth_ver}")
print(f"  transformers:  {transformers.__version__}")
print(f"  peft:          {peft.__version__}")
print(f"  trl:           {trl.__version__}")
print(f"  bitsandbytes:  {bitsandbytes.__version__}")
print(f"  torch:         {torch.__version__}")

# ============================================================
# CELL 3 — Authenticate HuggingFace + Groq
# ============================================================

import os
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
GROQ_API_KEY = user_secrets.get_secret("GROQ_API_KEY")

os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['HUGGINGFACE_TOKEN'] = HF_TOKEN
os.environ['GROQ_API_KEY'] = GROQ_API_KEY
os.environ['LITELLM_SUPPRESS_DEBUG_INFO'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from huggingface_hub import login
login(token=HF_TOKEN, add_to_git_credential=False)

print("✓ HuggingFace authenticated")
print("✓ Groq API key loaded")
print("✓ Environment variables set")

# ============================================================
# CELL 4 — Config (Gemma-4 E2B-it + Pure LoRA)
# ============================================================

from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    model_id: str = "unsloth/gemma-4-E2B-it"
    
    # --- DATA GATEKEEPER PATCH ---
    baseline_dir: str = "/kaggle/input/datasets/sameedsiddiqui0347/car-repairs-datasets-updated"
    dataset_repo: str = "ssiddiquii/car-repair-hq-550" # CRITICAL: Updated to the new 550-sample repo
    train_file: str = "train.json"
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # --- PURE LORA HARDWARE HACKS ---
    epochs: int = 5 
    per_device_batch_size: int = 1        # Reduced to 1 to prevent OOM
    grad_accumulation: int = 16           # Increased to 16 to maintain Effective Batch Size = 16
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    warmup_ratio: float = 0.03
    
    val_ratio: float = 0.10
    work_dir: str = "/kaggle/working"

CONFIG = Config()

WORK = Path(CONFIG.work_dir)
CHECKPOINTS = WORK / "checkpoints_pure_lora"
ADAPTERS = WORK / "lora_adapters_pure_lora"
LOGS = WORK / "logs"

for d in [CHECKPOINTS, ADAPTERS, LOGS]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PURE LORA CONFIG LOCKED")
print("=" * 60)
print(f"  Model:         {CONFIG.model_id}")
print(f"  Dataset:       {CONFIG.dataset_repo}")
print(f"  LoRA rank:     {CONFIG.lora_r}")
print(f"  Epochs:        {CONFIG.epochs}")
print(f"  Device Batch:  {CONFIG.per_device_batch_size}")
print(f"  Accumulation:  {CONFIG.grad_accumulation}")
print(f"  Effective BS:  {CONFIG.per_device_batch_size * CONFIG.grad_accumulation}")
print("=" * 60)

# ============================================================
# CELL 5 — Download train.json from HuggingFace
# ============================================================

import json
from huggingface_hub import hf_hub_download

print(f"Downloading {CONFIG.train_file} from {CONFIG.dataset_repo}...")

train_path = hf_hub_download(
    repo_id=CONFIG.dataset_repo,
    filename=CONFIG.train_file,
    repo_type="dataset",
    token=HF_TOKEN,
)

with open(train_path) as f:
    train_data = json.load(f)

print(f"\n✓ Loaded {len(train_data)} training samples")
print(f"First sample keys: {list(train_data[0].keys())}")
print(f"\nPreview:")
print(json.dumps(train_data[0], indent=2, ensure_ascii=False)[:400])

# ============================================================
# CELL 6 — Auto-detect question/answer fields
# ============================================================

Q_CANDIDATES = ['question', 'input', 'query', 'prompt', 'instruction']
A_CANDIDATES = ['answer', 'expected_answer', 'output', 'response', 'completion']

def find_key(item, candidates):
    for k in candidates:
        if k in item and item[k]:
            return k
    return None

sample = train_data[0]
Q_KEY = find_key(sample, Q_CANDIDATES)
A_KEY = find_key(sample, A_CANDIDATES)

assert Q_KEY, f"No question field in: {list(sample.keys())}"
assert A_KEY, f"No answer field in: {list(sample.keys())}"

print(f"Question field: '{Q_KEY}'")
print(f"Answer field:   '{A_KEY}'")
print(f"\nSample Q: {sample[Q_KEY][:150]}")
print(f"Sample A: {sample[A_KEY][:200]}")

# ============================================================
# CELL 7 — Clean + format as Gemma-4 chat messages
# ============================================================

SYSTEM_PROMPT = (
    "You are an expert car repair assistant. Answer the user's question concisely "
    "and accurately. Be technically precise about parts, diagnostics, and procedures."
)

def to_messages(item):
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": str(item[Q_KEY]).strip()},
            {"role": "assistant", "content": str(item[A_KEY]).strip()},
        ]
    }

clean = [
    it for it in train_data
    if it.get(Q_KEY) and it.get(A_KEY)
    and len(str(it[Q_KEY]).strip()) > 0
    and len(str(it[A_KEY]).strip()) > 0
]

print(f"Cleaning:")
print(f"  Original: {len(train_data)}")
print(f"  Cleaned:  {len(clean)}")
print(f"  Dropped:  {len(train_data) - len(clean)}")

formatted = [to_messages(it) for it in clean]
print(f"\n✓ Formatted {len(formatted)} samples in Gemma-4 chat structure")


# ============================================================
# CELL 8 — Split + pre-render text (stable Unsloth path)
# ============================================================

import random
from datasets import Dataset

random.seed(42)
shuffled = formatted.copy()
random.shuffle(shuffled)

split_idx = int(len(shuffled) * (1 - CONFIG.val_ratio))
train_rows = shuffled[:split_idx]
val_rows   = shuffled[split_idx:]

train_ds = Dataset.from_list(train_rows)
val_ds   = Dataset.from_list(val_rows)

print(f"Split:")
print(f"  Train: {len(train_ds)} samples (90%)")
print(f"  Val:   {len(val_ds)} samples (10%)")
print("\n✓ Raw splits ready — will pre-render after tokenizer loads in Cell 9")

# ============================================================
# CELL 9 — Load Gemma-4 E2B-it + Pure LoRA + Pre-render
# ============================================================

import torch, gc
gc.collect()
torch.cuda.empty_cache()

from unsloth import FastModel

print("Loading Gemma-4 E2B-it via Unsloth in PURE 16-BIT...")
print("WARNING: VRAM will spike to ~7GB during load.\n")

# --- Step 1: Load model + tokenizer (PURE LORA PATH) ---
model, tokenizer = FastModel.from_pretrained(
    model_name=CONFIG.model_id,
    max_seq_length=CONFIG.max_seq_length,
    load_in_4bit=False,                   # CRITICAL PATCH: Disabled quantization
    torch_dtype=torch.float16,            # CRITICAL PATCH: Force native 16-bit for T4
    full_finetuning=False,
    token=HF_TOKEN,
)

print("\n✓ 16-Bit Base Model loaded")

# Memory after load
for i in range(torch.cuda.device_count()):
    alloc = torch.cuda.memory_allocated(i) / 1024**3
    print(f"  GPU {i} after load: {alloc:.2f} GB")

# --- Step 2: Attach LoRA ---
model = FastModel.get_peft_model(
    model,
    r=CONFIG.lora_r,
    lora_alpha=CONFIG.lora_alpha,
    lora_dropout=CONFIG.lora_dropout,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
)

print("\n" + "=" * 60)
print("PURE LORA ATTACHED")
print("=" * 60)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
pct = 100 * trainable / total

print(f"  Trainable params: {trainable:,}")
print(f"  Total params:     {total:,}")
print(f"  Trainable %:      {pct:.2f}%")

for i in range(torch.cuda.device_count()):
    alloc = torch.cuda.memory_allocated(i) / 1024**3
    total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    free = total_mem - alloc
    print(f"  GPU {i}: {alloc:.2f} / {total_mem:.1f} GB  ({free:.2f} GB free)")

# --- Step 3: Pre-render chat template into 'text' column ---
print("\n\nPre-rendering chat template into 'text' column...")

def render_row(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }

train_ds = train_ds.map(render_row, remove_columns=["messages"], desc="Rendering train")
val_ds = val_ds.map(render_row, remove_columns=["messages"], desc="Rendering val")

print(f"\n✓ Datasets ready for SFTTrainer")

# ============================================================
# CELL 10 — Training (Pure LoRA + 8-bit AdamW Optimization)
# ============================================================

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

assert "text" in train_ds.column_names, "Run Cell 9 pre-render block first!"

sample_text = train_ds[0]["text"]
if "<|turn>user" in sample_text:
    INSTRUCTION_PART = "<|turn>user\n"
    RESPONSE_PART = "<|turn>model\n"
elif "<start_of_turn>user" in sample_text:
    INSTRUCTION_PART = "<start_of_turn>user\n"
    RESPONSE_PART = "<start_of_turn>model\n"
else:
    raise RuntimeError("Unknown template")

import torch
gpu_cap = torch.cuda.get_device_capability(0)
USE_BF16 = gpu_cap[0] >= 8
USE_FP16 = not USE_BF16

# --- SFTConfig ---
sft_config = SFTConfig(
    output_dir=str(CHECKPOINTS),
    num_train_epochs=CONFIG.epochs,
    per_device_train_batch_size=CONFIG.per_device_batch_size, # Now 1
    per_device_eval_batch_size=CONFIG.per_device_batch_size,
    gradient_accumulation_steps=CONFIG.grad_accumulation,     # Now 16
    learning_rate=CONFIG.learning_rate,
    lr_scheduler_type="cosine",
    warmup_ratio=CONFIG.warmup_ratio,
    max_grad_norm=0.3,
    optim="adamw_8bit",          # CRITICAL: Do not change. Required for Pure LoRA on T4
    weight_decay=0.0,
    bf16=USE_BF16,
    fp16=USE_FP16,
    max_seq_length=CONFIG.max_seq_length,
    packing=False,
    dataset_text_field="text", 
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=200,
    report_to="none",
    seed=42,
    dataloader_num_workers=0,
    remove_unused_columns=False,
)

trainer_kwargs = {
    "model": model,
    "args": sft_config,
    "train_dataset": train_ds,
    "eval_dataset": val_ds,
}

try:
    trainer = SFTTrainer(**trainer_kwargs, processing_class=tokenizer)
except TypeError:
    trainer = SFTTrainer(**trainer_kwargs, tokenizer=tokenizer)

trainer = train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART)

import math
total_steps = math.ceil(len(train_ds) / (CONFIG.per_device_batch_size * CONFIG.grad_accumulation)) * CONFIG.epochs

print(f"\n{'=' * 60}")
print("PURE LORA TRAINING PLAN")
print(f"{'=' * 60}")
print(f"  Effective batch:  {CONFIG.per_device_batch_size * CONFIG.grad_accumulation}")
print(f"  Total steps:      {total_steps}")
print(f"{'=' * 60}")

print("\nStarting training. Monitor VRAM closely! ☕\n")

train_result = trainer.train()

print(f"\n{'=' * 60}")
print("TRAINING COMPLETE")
print(f"{'=' * 60}")
print(f"  Final train loss:    {train_result.training_loss:.4f}")

print(f"\n  Peak VRAM:")
for i in range(torch.cuda.device_count()):
    peak = torch.cuda.max_memory_allocated(i) / 1024**3
    print(f"    GPU {i}: {peak:.2f} GB")
print(f"{'=' * 60}")

# ============================================================
# CELL 11 — Save LoRA adapters (URGENT, before session expires)
# ============================================================

import gc, json

print(f"Saving adapters to {ADAPTERS}...")
model.save_pretrained(str(ADAPTERS))
tokenizer.save_pretrained(str(ADAPTERS))

# Metadata
metadata = {
    "base_model": CONFIG.model_id,
    "framework": "unsloth",
    "training": {
        "lora_r": CONFIG.lora_r,
        "lora_alpha": CONFIG.lora_alpha,
        "lora_dropout": CONFIG.lora_dropout,
        "epochs": CONFIG.epochs,
        "effective_batch_size": 16,
        "learning_rate": CONFIG.learning_rate,
        "max_seq_length": CONFIG.max_seq_length,
        "precision": "fp16",
    },
    "data": {
        "source": CONFIG.dataset_repo,
    },
    "notes": "Fine-tuning complete. Transitioning to Post-Eval.",
}

with open(ADAPTERS / "training_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nFiles saved:")
total_mb = 0
for f in sorted(ADAPTERS.iterdir()):
    size_mb = f.stat().st_size / 1024 / 1024
    total_mb += size_mb
    print(f"  {f.name:45s}  {size_mb:>8.2f} MB")
print(f"  {'TOTAL':45s}  {total_mb:>8.2f} MB")

print(f"\n✓ Adapters saved to {ADAPTERS}")

# ============================================================
# ==================== END ===================================
# ============================================================