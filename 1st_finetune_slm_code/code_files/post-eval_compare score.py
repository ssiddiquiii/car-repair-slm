# ============================================================
# POST-EVAL CELL 1 — Verify environment + dataset paths
# ============================================================

import torch, sys, subprocess, os
from pathlib import Path

print("=" * 60)
print("POST-EVAL ENVIRONMENT CHECK")
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
    
    gpu_cap = torch.cuda.get_device_capability(0)
    print(f"\nGPU compute capability: sm_{gpu_cap[0]}{gpu_cap[1]}")

# Discover ALL mounted datasets and files
print(f"\n{'=' * 60}")
print("MOUNTED DATASETS")
print(f"{'=' * 60}")

result = subprocess.run(['find', '/kaggle/input', '-type', 'f'],
                       capture_output=True, text=True)
all_files = result.stdout.strip().split('\n') if result.stdout.strip() else []

# Organize by parent directory
from collections import defaultdict
by_dir = defaultdict(list)
for f in all_files:
    parent = str(Path(f).parent)
    by_dir[parent].append(Path(f).name)

print(f"\nTotal files found: {len(all_files)}")
for d, files in by_dir.items():
    print(f"\n  📁 {d}")
    for fname in sorted(files)[:10]:
        print(f"     {fname}")

# Auto-detect critical paths
BASELINE_DIR = None
ADAPTER_DIR = None

for d in by_dir.keys():
    dname = d.lower()
    if 'baseline' in dname or 'car-repairs-datasets' in dname:
        # Check it has baseline_answers.json
        if 'baseline_answers.json' in by_dir[d]:
            BASELINE_DIR = Path(d)
    if 'lora' in dname or 'adapter' in dname:
        if any('adapter' in f for f in by_dir[d]):
            ADAPTER_DIR = Path(d)

print(f"\n{'=' * 60}")
print("CRITICAL PATHS")
print(f"{'=' * 60}")
print(f"  BASELINE_DIR:  {BASELINE_DIR}")
print(f"  ADAPTER_DIR:   {ADAPTER_DIR}")

assert BASELINE_DIR is not None, "Baseline dataset not attached or path mismatch"
assert ADAPTER_DIR is not None, "Adapter dataset not attached or path mismatch"

# Check critical files
print(f"\n  Checking critical files...")
assert (BASELINE_DIR / "baseline_answers.json").exists(), "baseline_answers.json missing"
assert (BASELINE_DIR / "baseline_scores.json").exists(), "baseline_scores.json missing"
assert (ADAPTER_DIR / "adapter_model.safetensors").exists(), "adapter_model.safetensors missing"
assert (ADAPTER_DIR / "adapter_config.json").exists(), "adapter_config.json missing"
print("  ✓ All critical files present")

# Secrets
print(f"\n{'=' * 60}")
print("SECRETS")
print(f"{'=' * 60}")
try:
    from kaggle_secrets import UserSecretsClient
    us = UserSecretsClient()
    _ = us.get_secret("HF_TOKEN")
    _ = us.get_secret("GROQ_API_KEY")
    print("  ✓ HF_TOKEN loaded")
    print("  ✓ GROQ_API_KEY loaded")
except Exception as e:
    print(f"  ✗ Secrets issue: {e}")

print(f"\n{'=' * 60}")
print("✓ ENVIRONMENT READY FOR POST-EVAL")
print(f"{'=' * 60}")

# ============================================================
# POST-EVAL CELL 2 — Install dependencies
# ============================================================

# Unsloth — same framework as training, needed to load LoRA correctly
!pip install -q -U unsloth
!pip install -q -U "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Supporting
!pip install -q -U trl peft accelerate bitsandbytes datasets
!pip install -q -U huggingface_hub sentencepiece protobuf

# Evaluation — same as baseline
!pip install -q -U deepeval litellm

# Version check
import transformers, peft, trl, bitsandbytes, torch, deepeval
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
print(f"  deepeval:      {deepeval.__version__}")
print(f"  torch:         {torch.__version__}")

# ============================================================
# POST-EVAL CELL 3 — Config + load baseline artifacts
# ============================================================

import os
import json
from pathlib import Path
from dataclasses import dataclass
from kaggle_secrets import UserSecretsClient

# --- Secrets ---
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

# --- Config (must match baseline exactly for fair comparison) ---
@dataclass
class PostEvalConfig:
    base_model: str = "unsloth/gemma-4-E2B-it"
    adapter_dir: str = str(ADAPTER_DIR)          # from Cell 1
    baseline_dir: str = str(BASELINE_DIR)         # from Cell 1
    
    # Evaluation — must match baseline judge
    eval_model: str = "groq/llama-3.1-8b-instant"
    eval_threshold: float = 0.7
    
    # Generation — deterministic for reproducibility
    max_new_tokens: int = 256
    max_input_length: int = 1024
    
    # Output
    work_dir: str = "/kaggle/working"

CFG = PostEvalConfig()
WORK = Path(CFG.work_dir)
RESULTS = WORK / "post_eval_results"
RESULTS.mkdir(parents=True, exist_ok=True)

FINETUNED_ANSWERS_PATH = RESULTS / "finetuned_answers.json"
POSTEVAL_SCORES_PATH = RESULTS / "post_eval_scores.json"
DELTA_REPORT_PATH = RESULTS / "delta_report.json"

# --- Load baseline artifacts ---
print("=" * 60)
print("LOADING BASELINE ARTIFACTS")
print("=" * 60)

baseline_answers_path = Path(CFG.baseline_dir) / "baseline_answers.json"
baseline_scores_path = Path(CFG.baseline_dir) / "baseline_scores.json"

with open(baseline_answers_path) as f:
    baseline_answers = json.load(f)

with open(baseline_scores_path) as f:
    baseline_scores = json.load(f)

print(f"\n✓ Loaded baseline_answers: {len(baseline_answers)} entries")
print(f"✓ Loaded baseline_scores")

print(f"\nBASELINE SCORES (Gemma-4 E2B, pre-fine-tune):")
print(f"  Model:   {baseline_scores.get('model')}")
print(f"  Judge:   {baseline_scores.get('judge')}")
print(f"  Samples: {baseline_scores.get('n_samples')}")
print(f"  Metrics:")
for metric_name, metric_data in baseline_scores.get('metrics', {}).items():
    avg = metric_data.get('avg_score', 0)
    pass_rate = metric_data.get('pass_rate', 0)
    print(f"    {metric_name:32s}  avg={avg:.3f}  pass={pass_rate:.1%}")

# Sample baseline question for sanity
print(f"\nSample question (will re-run on fine-tuned model):")
print(f"  Q: {baseline_answers[0]['question'][:120]}...")
print(f"  Expected: {baseline_answers[0]['expected_answer'][:120]}...")
print(f"  Baseline A: {baseline_answers[0]['generated_answer'][:120]}...")

print(f"\n{'=' * 60}")
print("CONFIG LOCKED")
print(f"{'=' * 60}")
print(f"  Base model:    {CFG.base_model}")
print(f"  Adapter dir:   {CFG.adapter_dir}")
print(f"  Judge:         {CFG.eval_model}")
print(f"  Threshold:     {CFG.eval_threshold}")
print(f"  Output dir:    {RESULTS}")

# Verify adapter files exist
print(f"\nAdapter files:")
for f in sorted(Path(CFG.adapter_dir).iterdir()):
    size_mb = f.stat().st_size / 1024 / 1024
    marker = "⭐" if 'adapter' in f.name.lower() else "  "
    print(f"  {marker} {f.name:35s}  {size_mb:>7.2f} MB")

print(f"\n✓ Ready to load fine-tuned model")

# ============================================================
# POST-EVAL CELL 4 — Load model + adapters (FINAL FIX)
# Multimodal tokenizer requires `text=` keyword argument
# ============================================================

import torch
import gc
gc.collect()
torch.cuda.empty_cache()

from unsloth import FastModel

print("=" * 60)
print("LOADING FINE-TUNED MODEL")
print("=" * 60)

print(f"\nLoading base + LoRA adapters in one call...")
print(f"  Adapter dir: {CFG.adapter_dir}")
print(f"  (Unsloth auto-detects base from adapter_config.json)\n")

# FastModel loads adapter + base together, handles custom Gemma-4 layers
model, tokenizer = FastModel.from_pretrained(
    model_name=CFG.adapter_dir,
    max_seq_length=CFG.max_input_length,
    load_in_4bit=True,
    full_finetuning=False,
    token=HF_TOKEN,
)

print(f"\n✓ Model + adapters loaded")

for i in range(torch.cuda.device_count()):
    alloc = torch.cuda.memory_allocated(i) / 1024**3
    total = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"  GPU {i}: {alloc:.2f} / {total:.1f} GB")

# Verify adapter attached
lora_count = sum(1 for name, _ in model.named_modules() if 'lora' in name.lower())
print(f"\n  LoRA modules detected: {lora_count}")
assert lora_count > 0, "No LoRA modules — adapter not attached"
print(f"  ✓ Adapter integrated")

# Enable Unsloth inference optimizations (2x faster gen)
try:
    FastModel.for_inference(model)
    print(f"  ✓ Inference mode enabled (Unsloth optimized)")
except AttributeError:
    # Older Unsloth versions don't have this
    pass

model.eval()

# --- Quick test (with FIXED tokenizer call) ---
print(f"\n{'=' * 60}")
print("QUICK INFERENCE TEST")
print(f"{'=' * 60}")

test_question = "What should I check if my car won't start on a cold morning?"

messages = [
    {"role": "system", "content": "You are an expert car repair assistant. Answer the user's question concisely and accurately. Be technically precise about parts, diagnostics, and procedures."},
    {"role": "user", "content": test_question},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# FIX: multimodal tokenizer requires `text=` keyword
inputs = tokenizer(
    text=prompt,
    return_tensors="pt",
    truncation=True,
    max_length=CFG.max_input_length,
).to(model.device)

print(f"Input shape: {inputs['input_ids'].shape}")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        do_sample=False,   # deterministic
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

generated = tokenizer.decode(
    outputs[0][inputs['input_ids'].shape[1]:],
    skip_special_tokens=True,
).strip()

print(f"\nQ: {test_question}")
print(f"\nA (fine-tuned):\n{generated}")
print(f"\n{'=' * 60}")
print("✓ Fine-tuned model generating successfully")
print(f"{'=' * 60}")

# ============================================================
# POST-EVAL CELL 5 — Generate fine-tuned answers on all 30 baseline questions
# ============================================================

import json
from tqdm.auto import tqdm
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an expert car repair assistant. Answer the user's question concisely "
    "and accurately. Be technically precise about parts, diagnostics, and procedures."
)

@torch.no_grad()
def generate_answer(question: str) -> str:
    """Generate fine-tuned answer for a single question (deterministic)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Use text= keyword (multimodal tokenizer requirement)
    inputs = tokenizer(
        text=prompt,
        return_tensors="pt",
        truncation=True,
        max_length=CFG.max_input_length,
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=CFG.max_new_tokens,
        do_sample=False,   # deterministic / reproducible
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Decode only new tokens (exclude input prompt)
    generated = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    
    return generated


# --- Generate for all 30 questions ---
print("=" * 60)
print(f"GENERATING FINE-TUNED ANSWERS  ({len(baseline_answers)} questions)")
print("=" * 60)
print("  Deterministic mode (do_sample=False)")
print("  Expected time: ~3-5 min on T4\n")

finetuned_results = []

for i, item in enumerate(tqdm(baseline_answers, desc="Generating")):
    question = item["question"]
    expected_answer = item["expected_answer"]
    baseline_answer = item["generated_answer"]
    context = item.get("context", "")
    
    # Generate with fine-tuned model
    finetuned_answer = generate_answer(question)
    
    finetuned_results.append({
        "idx": i,
        "question": question,
        "expected_answer": expected_answer,
        "baseline_answer": baseline_answer,       # for comparison
        "finetuned_answer": finetuned_answer,
        "context": context,
    })

# Save
with open(FINETUNED_ANSWERS_PATH, 'w') as f:
    json.dump(finetuned_results, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved {len(finetuned_results)} fine-tuned answers")
print(f"  → {FINETUNED_ANSWERS_PATH}")

# --- Quick visual diff on first 3 samples ---
print(f"\n{'=' * 60}")
print("SAMPLE COMPARISON — Base vs Fine-Tuned (first 3 questions)")
print(f"{'=' * 60}")

for i in range(min(3, len(finetuned_results))):
    r = finetuned_results[i]
    print(f"\n[Q{i+1}] {r['question'][:130]}...")
    print(f"\n  BASELINE   (Gemma-4 pure):")
    print(f"  {r['baseline_answer'][:250]}...")
    print(f"\n  FINE-TUNED (Gemma-4 + LoRA):")
    print(f"  {r['finetuned_answer'][:250]}...")
    print(f"\n  EXPECTED:")
    print(f"  {r['expected_answer'][:250]}...")
    print("-" * 60)

print(f"\n✓ Ready for DeepEval scoring (Cell 6)")

# ============================================================
# POST-EVAL CELL 6 — Configure DeepEval with Groq judge
# Same setup as baseline — apples-to-apples comparison
# ============================================================

import os, time, threading, re
import litellm
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

litellm.suppress_debug_info = True


class LiteLLMJudge(DeepEvalBaseLLM):
    """
    Groq judge via LiteLLM.
    Same proven class used in baseline eval.
    TPM-aware, no logprobs, JSON retry built-in.
    """
    _lock = threading.Lock()
    _call_log = []

    def __init__(self, model_name, api_key, rpm_limit=30, tpm_limit=6000,
                 safety=0.70, max_tokens=1024):
        self.model_name = model_name
        self.api_key = api_key
        self.rpm_limit = rpm_limit
        self.tpm_budget = int(tpm_limit * safety)
        self.max_tokens = max_tokens

    @classmethod
    def _prune(cls):
        now = time.time()
        cls._call_log = [(t, n) for t, n in cls._call_log if now - t < 60]

    def _throttle(self, estimated_tokens):
        while True:
            with LiteLLMJudge._lock:
                self._prune()
                calls_in_window = len(LiteLLMJudge._call_log)
                tokens_in_window = sum(n for _, n in LiteLLMJudge._call_log)
                if (calls_in_window < self.rpm_limit and
                    tokens_in_window + estimated_tokens <= self.tpm_budget):
                    return
                oldest = LiteLLMJudge._call_log[0][0] if LiteLLMJudge._call_log else time.time()
                wait = (oldest + 61) - time.time()
            wait = max(1.0, min(wait, 30.0))
            time.sleep(wait)

    @staticmethod
    def _parse_retry(msg):
        m = re.search(r"try again in\s*(\d+(?:\.\d+)?)\s*(?:s|ms)", msg, re.I)
        if m:
            return float(m.group(1)) + 1.0
        m = re.search(r"retry[_\s-]?after[^\d]*(\d+(?:\.\d+)?)", msg, re.I)
        return float(m.group(1)) + 1.0 if m else None

    def _call(self, prompt, schema=None, retries=5):
        estimated = len(prompt) // 4 + self.max_tokens
        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "api_key": self.api_key,
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        last_err = None
        for attempt in range(retries):
            self._throttle(estimated)
            try:
                resp = litellm.completion(**kwargs)
                usage = getattr(resp, 'usage', None)
                actual_tokens = usage.total_tokens if usage else estimated
                with LiteLLMJudge._lock:
                    LiteLLMJudge._call_log.append((time.time(), actual_tokens))

                text = resp.choices[0].message.content.strip()
                if text.startswith("```"):
                    parts = text.split("```")
                    if len(parts) >= 2:
                        text = parts[1]
                        if text.startswith("json"):
                            text = text[4:]
                        text = text.strip()

                if schema is not None:
                    try:
                        return schema.model_validate_json(text)
                    except Exception as pe:
                        last_err = pe
                        if attempt < retries - 1:
                            kwargs["max_tokens"] = min(kwargs["max_tokens"] * 2, 4096)
                            continue
                        raise
                return text

            except Exception as e:
                last_err = e
                msg = str(e)
                ml = msg.lower()

                if "503" in msg or "unavailable" in ml or "overloaded" in ml:
                    wait = min(15 * (2 ** attempt), 60)
                    time.sleep(wait)
                    continue

                retry_after = self._parse_retry(msg)
                if retry_after:
                    time.sleep(retry_after)
                    with LiteLLMJudge._lock:
                        LiteLLMJudge._call_log.append((time.time(), estimated))
                    continue
                if "rate_limit" in ml or "429" in msg:
                    time.sleep(20)
                    continue

                if "validation" in ml or "json" in ml:
                    if attempt < retries - 1:
                        kwargs["max_tokens"] = min(kwargs["max_tokens"] * 2, 4096)
                        continue
                raise
        raise RuntimeError(f"Judge failed after {retries} retries: {last_err}")

    def load_model(self):
        return self.model_name

    def generate(self, prompt, schema=None):
        return self._call(prompt, schema)

    async def a_generate(self, prompt, schema=None):
        return self._call(prompt, schema)

    def generate_raw_response(self, prompt, top_logprobs=None, schema=None):
        text = self._call(prompt, schema)
        class _FakeChoice:
            def __init__(self, t): self.message = type("M", (), {"content": t})
        class _FakeResp:
            def __init__(self, t):
                self.choices = [_FakeChoice(t)]
                self.usage = type("U", (), {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0})
        return _FakeResp(text if isinstance(text, str) else str(text)), 0.0

    async def a_generate_raw_response(self, prompt, top_logprobs=None, schema=None):
        return self.generate_raw_response(prompt, top_logprobs, schema)

    def get_model_name(self):
        return self.model_name


# Reset global state
LiteLLMJudge._call_log = []

# Build judge (same as baseline)
judge = LiteLLMJudge(
    model_name=CFG.eval_model,
    api_key=GROQ_API_KEY,
    rpm_limit=30,
    tpm_limit=6000,
    safety=0.70,
    max_tokens=1024,
)

# Smoke test
print("Smoke-testing Groq judge...")
probe = judge.generate("Reply with exactly one word: OK")
print(f"  Judge reply: {str(probe)[:80]}")
print("  ✓ Judge working\n")

# Build metrics — SAME as baseline
metrics = [
    AnswerRelevancyMetric(threshold=CFG.eval_threshold, model=judge, async_mode=False),
    GEval(
        name="Correctness",
        criteria=(
            "Evaluate whether the actual_output is a factually correct, technically accurate "
            "answer to the input question, using expected_output as ground truth. For car repair, "
            "check: correct parts/components, correct diagnostic reasoning, correct procedures, "
            "safe advice. Heavily penalize vague, generic, or incorrect technical claims."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=CFG.eval_threshold,
        model=judge,
        async_mode=False,
    ),
]

print(f"Judge:   {CFG.eval_model} (TPM-aware, ~4200 usable TPM)")
print(f"Metrics: {len(metrics)} (sequential mode)")
for m in metrics:
    tag = f" ({m.name})" if isinstance(m, GEval) else ""
    print(f"  - {m.__class__.__name__}{tag}")

print(f"\n✓ DeepEval configured. Run Cell 7 to score all 30 fine-tuned answers.")

# ============================================================
# POST-EVAL CELL 7 — Score fine-tuned answers with DeepEval
# ============================================================

from collections import defaultdict
from tqdm.auto import tqdm
import json

# --- Load generated answers from Cell 5 ---
with open(FINETUNED_ANSWERS_PATH) as f:
    finetuned_results = json.load(f)

print(f"Loaded {len(finetuned_results)} fine-tuned answers for scoring")

# --- Build test cases ---
test_cases = []
for r in finetuned_results:
    ctx = r.get("context", "")
    ctx_list = [ctx] if ctx else None
    test_cases.append(LLMTestCase(
        input=r["question"],
        actual_output=r["finetuned_answer"],     # SCORING fine-tuned answer
        expected_output=r["expected_answer"],
        retrieval_context=ctx_list,
    ))

total_calls = len(test_cases) * len(metrics)
est_min = (total_calls * 6.0) / 60

print(f"\nSequential eval: {len(test_cases)} cases × {len(metrics)} metrics = {total_calls} judge calls")
print(f"Rate-limited (~6s/call) → ~{est_min:.0f} min")
print(f"Incremental save after each case to prevent data loss\n")

# --- Incremental scoring with checkpoint ---
POSTEVAL_CHECKPOINT = RESULTS / "post_eval_checkpoint.jsonl"
scores_by_metric = defaultdict(list)
per_case = []
failed_count = 0

# Resume support: load existing checkpoint if any
completed_indices = set()
if POSTEVAL_CHECKPOINT.exists():
    with open(POSTEVAL_CHECKPOINT) as f:
        for line in f:
            try:
                entry = json.loads(line)
                completed_indices.add(entry["idx"])
                per_case.append(entry)
                for k, v in entry.get("scores", {}).items():
                    if isinstance(v, (int, float)):
                        scores_by_metric[k].append(v)
            except Exception:
                pass
    if completed_indices:
        print(f"Resuming from checkpoint: {len(completed_indices)} cases already scored\n")

# --- Score each case ---
with open(POSTEVAL_CHECKPOINT, 'a') as ckpt_file:
    for i, tc in enumerate(tqdm(test_cases, desc="Scoring")):
        if i in completed_indices:
            continue
        
        case_scores = {}
        case_failed = False
        
        for metric in metrics:
            mname = getattr(metric, 'name', None) or metric.__class__.__name__
            try:
                metric.measure(tc)
                score = getattr(metric, 'score', None)
                if score is not None:
                    scores_by_metric[mname].append(score)
                    case_scores[mname] = round(score, 3)
            except Exception as e:
                print(f"\n  [case {i}] {mname} failed: {str(e)[:200]}")
                case_failed = True
        
        if case_failed and not case_scores:
            failed_count += 1
        
        entry = {
            "idx": i,
            "question": finetuned_results[i]["question"][:150],
            "scores": case_scores,
        }
        per_case.append(entry)
        
        # Durable write
        ckpt_file.write(json.dumps(entry) + "\n")
        ckpt_file.flush()

# --- Aggregate + save summary ---
summary = {
    "model": "google/gemma-4-E2B-it + LoRA (fine-tuned)",
    "adapter_source": CFG.adapter_dir,
    "judge": CFG.eval_model,
    "n_samples": len(test_cases),
    "n_failed": failed_count,
    "threshold": CFG.eval_threshold,
    "metrics": {},
    "per_case": sorted(per_case, key=lambda x: x["idx"]),
}

print(f"\n{'=' * 70}")
print(f"POST-FINETUNE SCORES — Gemma-4 E2B-it + LoRA v1")
print(f"{'=' * 70}")
for name, vals in scores_by_metric.items():
    if not vals:
        continue
    avg = sum(vals) / len(vals)
    pass_rate = sum(1 for v in vals if v >= CFG.eval_threshold) / len(vals)
    summary["metrics"][name] = {
        "avg_score": round(avg, 4),
        "pass_rate": round(pass_rate, 4),
        "n": len(vals),
    }
    print(f"  {name:35s}  avg={avg:.3f}  pass_rate={pass_rate:.1%}  (n={len(vals)})")

if failed_count:
    print(f"\n  ⚠ {failed_count} cases failed judging (error logged above)")
print(f"{'=' * 70}")

# Save final
with open(POSTEVAL_SCORES_PATH, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ Saved post-eval summary → {POSTEVAL_SCORES_PATH}")
print(f"✓ Checkpoint preserved  → {POSTEVAL_CHECKPOINT}")
print(f"\nNext: Run Cell 8 for final delta report (baseline vs fine-tuned)")

# ============================================================
# POST-EVAL CELL 8 — Delta report (baseline vs fine-tuned)
# ============================================================

import json
from pathlib import Path
from datetime import datetime

# --- Load both score files ---
with open(POSTEVAL_SCORES_PATH) as f:
    post_scores = json.load(f)

# Baseline was loaded in Cell 3 as `baseline_scores`

# --- Build delta report ---
delta_report = {
    "report_generated": datetime.now().isoformat(),
    "project": "Car Repair SLM Fine-Tuning",
    "base_model": "google/gemma-4-E2B-it",
    "fine_tuning_method": "QLoRA (4-bit) via Unsloth",
    "judge_model": CFG.eval_model,
    "test_set_size": baseline_scores.get("n_samples"),
    
    "training_details": {
        "framework": "unsloth",
        "lora_r": 16,
        "lora_alpha": 32,
        "epochs": 2,
        "effective_batch_size": 16,
        "learning_rate": 2e-4,
        "precision": "fp16",
        "train_samples": 252,
        "val_samples": 29,
        "runtime_min": 5.5,
        "peak_vram_gb": 8.47,
    },
    
    "metrics": {},
    "overall_summary": {},
}

# --- Compute deltas per metric ---
print("=" * 70)
print("DELTA REPORT — Gemma-4 E2B-it: Baseline vs Fine-Tuned")
print("=" * 70)
print(f"{'Metric':<30} {'Baseline':>10} {'Fine-Tuned':>12} {'Delta':>10} {'Relative':>10}")
print("-" * 70)

for metric_name, post_data in post_scores["metrics"].items():
    base_data = baseline_scores["metrics"].get(metric_name, {})
    
    base_avg = base_data.get("avg_score", 0)
    post_avg = post_data.get("avg_score", 0)
    delta = post_avg - base_avg
    rel_change_pct = (delta / base_avg * 100) if base_avg > 0 else 0
    
    base_pass = base_data.get("pass_rate", 0)
    post_pass = post_data.get("pass_rate", 0)
    pass_delta = post_pass - base_pass
    
    delta_report["metrics"][metric_name] = {
        "baseline": {"avg_score": base_avg, "pass_rate": base_pass},
        "fine_tuned": {"avg_score": post_avg, "pass_rate": post_pass},
        "delta": {
            "avg_score_change": round(delta, 4),
            "relative_change_percent": round(rel_change_pct, 2),
            "pass_rate_change": round(pass_delta, 4),
        }
    }
    
    sign = "+" if delta >= 0 else ""
    rel_sign = "+" if rel_change_pct >= 0 else ""
    print(f"{metric_name:<30} {base_avg:>10.3f} {post_avg:>12.3f} {sign}{delta:>9.3f} {rel_sign}{rel_change_pct:>8.1f}%")

print("-" * 70)

# --- Pass rate summary ---
print(f"\n{'Pass Rate (≥0.7)':<30} {'Baseline':>10} {'Fine-Tuned':>12} {'Delta':>10}")
print("-" * 62)
for metric_name, post_data in post_scores["metrics"].items():
    base_data = baseline_scores["metrics"].get(metric_name, {})
    base_pass = base_data.get("pass_rate", 0)
    post_pass = post_data.get("pass_rate", 0)
    delta_pass = post_pass - base_pass
    sign = "+" if delta_pass >= 0 else ""
    print(f"{metric_name:<30} {base_pass:>9.1%} {post_pass:>11.1%} {sign}{delta_pass:>8.1%}")

# --- Overall summary ---
correctness_delta = delta_report["metrics"].get("Correctness", {}).get("delta", {})
relevancy_delta = delta_report["metrics"].get("AnswerRelevancyMetric", {}).get("delta", {})

delta_report["overall_summary"] = {
    "correctness_improvement_absolute": correctness_delta.get("avg_score_change", 0),
    "correctness_improvement_relative_percent": correctness_delta.get("relative_change_percent", 0),
    "relevancy_change": relevancy_delta.get("avg_score_change", 0),
    "verdict": (
        "SIGNIFICANT IMPROVEMENT" if correctness_delta.get("avg_score_change", 0) >= 0.15
        else "MODERATE IMPROVEMENT" if correctness_delta.get("avg_score_change", 0) >= 0.05
        else "MARGINAL / FLAT"
    ),
}

print(f"\n{'=' * 70}")
print("VERDICT")
print(f"{'=' * 70}")
print(f"  Correctness improved by {correctness_delta.get('avg_score_change', 0):+.3f} "
      f"({correctness_delta.get('relative_change_percent', 0):+.1f}% relative)")
print(f"  Relevancy preserved:     {relevancy_delta.get('avg_score_change', 0):+.3f}")
print(f"  Overall assessment:      {delta_report['overall_summary']['verdict']}")
print(f"{'=' * 70}")

# --- Save delta report ---
with open(DELTA_REPORT_PATH, 'w') as f:
    json.dump(delta_report, f, indent=2)

print(f"\n✓ Saved delta report → {DELTA_REPORT_PATH}")
print(f"\n📊 Full pipeline complete. All artifacts:")
print(f"  - baseline_scores.json        (baseline metrics)")
print(f"  - finetuned_answers.json      (30 generated answers)")
print(f"  - post_eval_scores.json       (fine-tuned metrics)")
print(f"  - delta_report.json           ← ready summary")

# ============================================================
# FINAL STEP — Zip + prepare post-eval artifacts for download
# ============================================================

import shutil, zipfile
from pathlib import Path

# Zip entire results folder
zip_path = "/kaggle/working/post_eval_final_v1"
shutil.make_archive(
    base_name=zip_path,
    format='zip',
    root_dir=str(RESULTS),
)

zip_file = Path(f"{zip_path}.zip")
size_mb = zip_file.stat().st_size / 1024 / 1024

print(f"✓ Zip created: {zip_file}")
print(f"  Size: {size_mb:.2f} MB")
print(f"\nContents:")
with zipfile.ZipFile(zip_file, 'r') as z:
    for f in z.namelist():
        print(f"  {f}")

print(f"\n{'='*60}")
print("NEXT STEPS (DO THIS BEFORE CLOSING KAGGLE):")
print(f"{'='*60}")
print("  1. Kaggle right sidebar → Output section")
print("  2. Find 'post_eval_final_v1.zip' → Download")
print("  3. Upload zip to Google Drive (car_repair_slm/post_eval_results/)")
print("  4. Keep a copy in laptop's permanent folder")
print(f"{'='*60}")
