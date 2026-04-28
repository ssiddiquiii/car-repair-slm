# ============================================================
# POST-EVAL CELL 1 — Verify environment
# ============================================================

import torch, sys, os
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
# !pip install -q -U unsloth #uncomment when run
# !pip install -q -U "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" #uncomment when run

# Supporting
# !pip install -q -U trl peft accelerate bitsandbytes datasets #uncomment when run
# !pip install -q -U huggingface_hub sentencepiece protobuf #uncomment when run

# Evaluation — same as baseline #uncomment when run
# !pip install -q -U deepeval litellm #uncomment when run

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
# POST-EVAL CELL 3 — Config (Patched for Pure LoRA)
# ============================================================

import os
import json
from pathlib import Path
from dataclasses import dataclass
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

@dataclass
class PostEvalConfig:
    base_model: str = "unsloth/gemma-4-E2B-it"
    
    # --- CRITICAL PATCH: Point to your new PURE LORA adapters ---
    # If in the same session: "/kaggle/working/lora_adapters_pure_lora"
    # If in a new session: "/kaggle/input/YOUR_MOUNTED_PURE_LORA_DATASET"
    adapter_dir: str = "/kaggle/input/datasets/sameedsiddiqui0347/lora-adapters-pure-lora" 
    
    # Baseline directory remains the same (v2 artifacts)
    baseline_dir: str = "/kaggle/input/datasets/sameedsiddiqui0347/baseline-v2-artifacts" 
    
    eval_model: str = "groq/llama-3.1-8b-instant"
    eval_threshold: float = 0.7
    max_new_tokens: int = 256
    max_input_length: int = 1024
    work_dir: str = "/kaggle/working"

CFG = PostEvalConfig()
WORK = Path(CFG.work_dir)
RESULTS = WORK / "post_eval_results_pure_lora"
RESULTS.mkdir(parents=True, exist_ok=True)

FINETUNED_ANSWERS_PATH = RESULTS / "finetuned_answers.json"
POSTEVAL_SCORES_PATH = RESULTS / "post_eval_scores.json"
DELTA_REPORT_PATH = RESULTS / "detailed_delta_report.json"

print("=" * 60)
print("LOADING BASELINE ARTIFACTS")
print("=" * 60)

baseline_answers_path = Path(CFG.baseline_dir) / "baseline_answers_v2.json"
baseline_scores_path = Path(CFG.baseline_dir) / "baseline_scores_v2.json"

assert baseline_answers_path.exists(), f"CRITICAL: Cannot find {baseline_answers_path}"
assert baseline_scores_path.exists(), f"CRITICAL: Cannot find {baseline_scores_path}"
assert Path(CFG.adapter_dir).exists(), f"CRITICAL: Cannot find LoRA adapters at {CFG.adapter_dir}"

with open(baseline_answers_path) as f:
    baseline_answers = json.load(f)

with open(baseline_scores_path) as f:
    baseline_scores = json.load(f)

print(f"\n✓ Loaded baseline_answers: {len(baseline_answers)} entries")
print(f"✓ Ready to load 16-bit fine-tuned model")

# ============================================================
# POST-EVAL CELL 4 — Load model + adapters (PURE 16-BIT)
# ============================================================

import torch
import gc
gc.collect()
torch.cuda.empty_cache()

from unsloth import FastModel

print("=" * 60)
print("LOADING FINE-TUNED MODEL (PURE 16-BIT)")
print("=" * 60)

# CRITICAL PATCH: Disabling 4-bit to test the true Pure LoRA weights
model, tokenizer = FastModel.from_pretrained(
    model_name=CFG.adapter_dir,
    max_seq_length=CFG.max_input_length,
    load_in_4bit=False,               # <-- CHANGED
    torch_dtype=torch.float16,        # <-- CHANGED
    full_finetuning=False,
    token=HF_TOKEN,
)

print(f"\n✓ Model + adapters loaded in 16-bit precision")

lora_count = sum(1 for name, _ in model.named_modules() if 'lora' in name.lower())
print(f"\n  LoRA modules detected: {lora_count}")
assert lora_count > 0, "No LoRA modules — adapter not attached"

try:
    FastModel.for_inference(model)
    print(f"  ✓ Inference mode enabled (Unsloth optimized)")
except AttributeError:
    pass

model.eval()

# --- Quick test ---
print(f"\n{'=' * 60}")
print("QUICK INFERENCE TEST")
print(f"{'=' * 60}")

test_question = "What should I check if my car won't start on a cold morning?"

messages = [
    {"role": "system", "content": "You are an expert car repair assistant. Answer the user's question concisely and accurately. Be technically precise about parts, diagnostics, and procedures."},
    {"role": "user", "content": test_question},
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = tokenizer(
    text=prompt, return_tensors="pt", truncation=True, max_length=CFG.max_input_length,
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        do_sample=False,   
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

print(f"\nQ: {test_question}")
print(f"\nA (16-bit fine-tuned):\n{generated}")
print(f"\n{'=' * 60}")

# ============================================================
# POST-EVAL CELL 5 — Generate fine-tuned answers
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
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(
        text=prompt,
        return_tensors="pt",
        truncation=True,
        max_length=CFG.max_input_length,
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=CFG.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    generated = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    
    return generated


print("=" * 60)
print(f"GENERATING FINE-TUNED ANSWERS  ({len(baseline_answers)} questions)")
print("=" * 60)

finetuned_results = []

for i, item in enumerate(tqdm(baseline_answers, desc="Generating")):
    question = item["question"]
    expected_answer = item["expected_answer"]
    baseline_answer = item["generated_answer"]
    context = item.get("context", "")
    
    finetuned_answer = generate_answer(question)
    
    finetuned_results.append({
        "idx": i,
        "question": question,
        "expected_answer": expected_answer,
        "baseline_answer": baseline_answer,
        "finetuned_answer": finetuned_answer,
        "context": context,
    })

with open(FINETUNED_ANSWERS_PATH, 'w') as f:
    json.dump(finetuned_results, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved {len(finetuned_results)} fine-tuned answers → {FINETUNED_ANSWERS_PATH}")

# ============================================================
# POST-EVAL CELL 6 & 7 — Configure and Run DeepEval
# ============================================================

import os, time, threading, re, json
import litellm
from collections import defaultdict
from tqdm.auto import tqdm
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

litellm.suppress_debug_info = True

class LiteLLMJudge(DeepEvalBaseLLM):
    """Groq judge via LiteLLM - Proven v1 implementation."""
    _lock = threading.Lock()
    _call_log = []

    def __init__(self, model_name, api_key, rpm_limit=30, tpm_limit=6000, safety=0.70, max_tokens=1024):
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
                if (len(LiteLLMJudge._call_log) < self.rpm_limit and
                    sum(n for _, n in LiteLLMJudge._call_log) + estimated_tokens <= self.tpm_budget):
                    return
                oldest = LiteLLMJudge._call_log[0][0] if LiteLLMJudge._call_log else time.time()
                wait = max(1.0, min((oldest + 61) - time.time(), 30.0))
            time.sleep(wait)

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

        for attempt in range(retries):
            self._throttle(estimated)
            try:
                resp = litellm.completion(**kwargs)
                actual_tokens = getattr(resp, 'usage', None).total_tokens if getattr(resp, 'usage', None) else estimated
                with LiteLLMJudge._lock: 
                    LiteLLMJudge._call_log.append((time.time(), actual_tokens))
                
                text = resp.choices[0].message.content.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"): text = text[4:]
                    text = text.strip()
                
                if schema is not None: 
                    return schema.model_validate_json(text)
                return text
            except Exception as e:
                msg = str(e).lower()
                if "429" in msg or "rate_limit" in msg: 
                    time.sleep(20)
                    continue
                if attempt == retries - 1: raise RuntimeError(f"Judge failed: {e}")
                time.sleep(5)

    def load_model(self): return self.model_name
    def generate(self, prompt, schema=None): return self._call(prompt, schema)
    async def a_generate(self, prompt, schema=None): return self._call(prompt, schema)
    def get_model_name(self): return self.model_name

# Setup Judge and Metrics
judge = LiteLLMJudge(model_name=CFG.eval_model, api_key=GROQ_API_KEY)

metrics = [
    AnswerRelevancyMetric(threshold=CFG.eval_threshold, model=judge, async_mode=False),
    GEval(
        name="Correctness",
        criteria=(
            "Evaluate whether the actual_output is a factually correct, technically accurate "
            "answer to the input question, using expected_output as ground truth. For car repair, "
            "check: correct parts, diagnostics, and procedures."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=CFG.eval_threshold, 
        model=judge, 
        async_mode=False
    ),
]

# Build test cases from generated answers in Cell 5
with open(FINETUNED_ANSWERS_PATH) as f: 
    finetuned_results = json.load(f)

# --- PATCH: Data Type Normalizer ---
def to_str(value):
    """Convert answer to string — handle list, str, None."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(item).strip() for item in value if item)
    return str(value)

test_cases = [
    LLMTestCase(
        input=to_str(r["question"]),
        actual_output=to_str(r["finetuned_answer"]),
        expected_output=to_str(r["expected_answer"])
    ) for r in finetuned_results
]

# Run Sequential Scoring
POSTEVAL_CHECKPOINT = RESULTS / "post_eval_checkpoint.jsonl"
scores_by_metric = defaultdict(list)
per_case = []
completed_indices = set()

if POSTEVAL_CHECKPOINT.exists():
    with open(POSTEVAL_CHECKPOINT) as f:
        for line in f:
            try:
                entry = json.loads(line)
                completed_indices.add(entry["idx"])
                per_case.append(entry)
                for k, v in entry.get("scores", {}).items(): 
                    scores_by_metric[k].append(v)
            except: pass

print(f"Scoring {len(test_cases)} cases...")
with open(POSTEVAL_CHECKPOINT, 'a') as ckpt:
    for i, tc in enumerate(tqdm(test_cases, desc="Scoring")):
        if i in completed_indices: continue
        case_scores = {}
        for metric in metrics:
            mname = getattr(metric, 'name', None) or metric.__class__.__name__
            try:
                metric.measure(tc)
                score = getattr(metric, 'score', None)
                if score is not None:
                    scores_by_metric[mname].append(score)
                    case_scores[mname] = round(score, 3)
            except Exception as e: 
                print(f"\n  [case {i}] {mname} failed: {str(e)[:100]}")
        
        entry = {"idx": i, "question": finetuned_results[i]["question"][:100], "scores": case_scores}
        per_case.append(entry)
        ckpt.write(json.dumps(entry) + "\n")
        ckpt.flush()

# Final Summary
summary = {
    "model": "google/gemma-4-E2B-it + LoRA",
    "judge": CFG.eval_model,
    "n_samples": len(test_cases),
    "metrics": {
        name: {
            "avg_score": sum(v)/len(v), 
            "pass_rate": sum(1 for x in v if x >= CFG.eval_threshold)/len(v)
        } for name, v in scores_by_metric.items()
    }
}

with open(POSTEVAL_SCORES_PATH, 'w') as f: 
    json.dump(summary, f, indent=2)

print(f"\n✓ Saved post-eval summary → {POSTEVAL_SCORES_PATH}")

# ============================================================
# POST-EVAL CELL 8 — Detailed Delta Report & Zip Artifacts
# ============================================================

import json, shutil
from pathlib import Path

# Fallback pathing
RESULTS = Path("/kaggle/working/post_eval_results_pure_lora")
POSTEVAL_SCORES_PATH = RESULTS / "post_eval_scores.json"

try:
    # Load post-eval scores
    with open(POSTEVAL_SCORES_PATH) as f:
        post_scores = json.load(f)

    # Ensure baseline_scores is available (loads from Cell 3 memory, or reads from disk)
    if 'baseline_scores' not in globals():
        baseline_scores_path = Path(CFG.baseline_dir) / "baseline_scores_v2.json"
        with open(baseline_scores_path) as f:
            baseline_scores = json.load(f)

    delta_report = {"metrics": {}, "overall_summary": {}}

    # --- 1. Averages Table ---
    print("=" * 80)
    print("DELTA REPORT — Gemma-4 E2B-it: Baseline vs Fine-Tuned")
    print("=" * 80)
    print(f"{'Metric':<30} {'Baseline':>10} {'Fine-Tuned':>12} {'Delta':>10} {'Relative':>10}")
    print("-" * 80)

    for metric_name, post_data in post_scores["metrics"].items():
        base_data = baseline_scores["metrics"].get(metric_name, {})
        base_avg = base_data.get("avg_score", 0)
        post_avg = post_data.get("avg_score", 0)
        delta = post_avg - base_avg
        rel_change_pct = (delta / base_avg * 100) if base_avg > 0 else 0

        sign = "+" if delta >= 0 else ""
        rel_sign = "+" if rel_change_pct >= 0 else ""
        print(f"{metric_name:<30} {base_avg:>10.3f} {post_avg:>12.3f} {sign}{delta:>9.3f} {rel_sign}{rel_change_pct:>8.1f}%")

        delta_report["metrics"][metric_name] = {
            "baseline": {"avg_score": base_avg, "pass_rate": base_data.get("pass_rate", 0)},
            "fine_tuned": {"avg_score": post_avg, "pass_rate": post_data.get("pass_rate", 0)},
            "delta": {
                "avg_score_change": round(delta, 4),
                "relative_change_percent": round(rel_change_pct, 2),
            }
        }

    print("-" * 80)

    # --- 2. Pass Rate Table ---
    print(f"\n{'Pass Rate (≥0.7)':<30} {'Baseline':>10} {'Fine-Tuned':>12} {'Delta':>10}")
    print("-" * 72)
    for metric_name, post_data in post_scores["metrics"].items():
        base_data = baseline_scores["metrics"].get(metric_name, {})
        base_pass = base_data.get("pass_rate", 0)
        post_pass = post_data.get("pass_rate", 0)
        delta_pass = post_pass - base_pass

        sign = "+" if delta_pass >= 0 else ""
        print(f"{metric_name:<30} {base_pass:>9.1%} {post_pass:>11.1%} {sign}{delta_pass:>8.1%}")
        delta_report["metrics"][metric_name]["delta"]["pass_rate_change"] = round(delta_pass, 4)

    # --- 3. Verdict ---
    correctness_delta = delta_report["metrics"].get("Correctness", {}).get("delta", {})
    relevancy_delta = delta_report["metrics"].get("AnswerRelevancyMetric", {}).get("delta", {})

    c_abs = correctness_delta.get('avg_score_change', 0)
    c_rel = correctness_delta.get('relative_change_percent', 0)
    r_abs = relevancy_delta.get('avg_score_change', 0)

    if c_abs >= 0.15: verdict = "SIGNIFICANT IMPROVEMENT"
    elif c_abs >= 0.05: verdict = "MODERATE IMPROVEMENT"
    else: verdict = "MARGINAL / FLAT"

    print(f"\n{'=' * 80}")
    print("VERDICT")
    print(f"{'=' * 80}")
    print(f"  Correctness improved by {c_abs:+.3f} ({c_rel:+.1f}% relative)")
    print(f"  Relevancy preserved:     {r_abs:+.3f}")
    print(f"  Overall assessment:      {verdict}")
    print(f"{'=' * 80}")

    delta_report["overall_summary"] = {
        "correctness_improvement_absolute": c_abs,
        "correctness_improvement_relative_percent": c_rel,
        "relevancy_change": r_abs,
        "verdict": verdict
    }

    # Save JSON
    DELTA_REPORT_PATH = RESULTS / "detailed_delta_report.json"
    with open(DELTA_REPORT_PATH, 'w') as f:
        json.dump(delta_report, f, indent=2)

except Exception as e:
    print(f"Error generating delta report: {e}")

# ============================================================
# ==================== END ===================================
# ============================================================