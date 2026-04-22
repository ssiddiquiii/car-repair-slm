# ==================================================================== 
#  ================================= Cell 1
# ==================================================================== 

# Gemma 4 architecture tag is 'gemma4' (released ~2 weeks ago).
# Old transformers versions won't recognize it — force latest.
!pip install -q -U transformers accelerate bitsandbytes peft datasets
!pip install -q -U deepeval litellm huggingface_hub tqdm

import transformers, torch
print(f"transformers: {transformers.__version__}")
print(f"torch:        {torch.__version__}")
print(f"CUDA avail:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"VRAM:         {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

# ==================================================================== 
#  ================================= Cell 2
# ==================================================================== 
from google.colab import drive, userdata
drive.mount('/content/drive')

import os
HF_TOKEN = userdata.get('HF_TOKEN')
GROQ_API_KEY = userdata.get('GROQ_API_KEY')
assert HF_TOKEN and GROQ_API_KEY, "Set HF_TOKEN + GROQ_API_KEY in Colab secrets"

os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['HUGGINGFACE_TOKEN'] = HF_TOKEN
os.environ['GROQ_API_KEY'] = GROQ_API_KEY
os.environ['LITELLM_SUPPRESS_DEBUG_INFO'] = 'true'

from huggingface_hub import login
login(token=HF_TOKEN, add_to_git_credential=False)
print("Authenticated: HF + Groq")

# ==================================================================== 
#  ================================= Cell 3
# ==================================================================== 

from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    model_id: str = "google/gemma-4-E2B-it"

    dataset_repo: str = "ssiddiquii/car-repair-hq"
    train_file: str = "train.json"
    test_file:  str = "test.json"

    eval_n_samples: int = 30

    max_new_tokens: int = 256
    max_input_length: int = 1024

    # Judge — fast, smaller, shorter outputs = better TPM economics
    eval_model: str = "groq/llama-3.1-8b-instant"
    eval_threshold: float = 0.7

    base_dir: str = "/content/drive/MyDrive/car_repair_slm"

CONFIG = Config()
BASE = Path(CONFIG.base_dir)
RESULTS = BASE / "results"
LOGS = BASE / "logs"
for d in [BASE, RESULTS, LOGS]:
    d.mkdir(parents=True, exist_ok=True)

BASELINE_ANSWERS = RESULTS / "baseline_answers.json"
BASELINE_SCORES  = RESULTS / "baseline_scores.json"

print(f"Model:   {CONFIG.model_id}")
print(f"Dataset: {CONFIG.dataset_repo}")
print(f"Judge:   {CONFIG.eval_model}")
print(f"Outputs: {BASE}")

# ==================================================================== 
#  ================================= Cell 4
# ==================================================================== 
import json
from huggingface_hub import hf_hub_download

print("Downloading train.json (held for fine-tuning phase, not used here)...")
train_path = hf_hub_download(
    repo_id=CONFIG.dataset_repo, filename=CONFIG.train_file,
    repo_type="dataset", token=HF_TOKEN,
)
with open(train_path) as f:
    train_data = json.load(f)

print("Downloading test.json (eval set — same questions for pre + post)...")
test_path = hf_hub_download(
    repo_id=CONFIG.dataset_repo, filename=CONFIG.test_file,
    repo_type="dataset", token=HF_TOKEN,
)
with open(test_path) as f:
    test_data = json.load(f)

# Cap eval size
eval_data = test_data[:CONFIG.eval_n_samples] if len(test_data) > CONFIG.eval_n_samples else test_data

print(f"\nTrain pool:  {len(train_data)} samples (reserved for fine-tuning)")
print(f"Test total:  {len(test_data)} samples")
print(f"Eval set:    {len(eval_data)} samples (capped at {CONFIG.eval_n_samples})")
print(f"\nFirst eval item keys: {list(eval_data[0].keys())}")
print(f"\nPreview of first item:")
print(json.dumps(eval_data[0], indent=2, ensure_ascii=False)[:500])


# ==================================================================== 
#  ================================= Cell 5
# ==================================================================== 
sample = eval_data[0]

Q_CANDIDATES = ['question', 'input', 'query', 'prompt', 'instruction']
A_CANDIDATES = ['answer', 'expected_answer', 'output', 'response', 'completion']
C_CANDIDATES = ['context', 'retrieval_context', 'reference', 'passage', 'background']

def find_key(item, candidates):
    for k in candidates:
        if k in item and item[k]:
            return k
    return None

Q_KEY = find_key(sample, Q_CANDIDATES)
A_KEY = find_key(sample, A_CANDIDATES)
C_KEY = find_key(sample, C_CANDIDATES)

assert Q_KEY and A_KEY, f"Missing question/answer field. Got keys: {list(sample.keys())}"

# Context must exist AND be non-empty on most items, else skip context-dependent metrics
has_context = bool(C_KEY) and sum(1 for it in eval_data if it.get(C_KEY)) >= len(eval_data) * 0.8

print(f"Question field: '{Q_KEY}'")
print(f"Answer field:   '{A_KEY}'")
print(f"Context field:  '{C_KEY}'  usable={has_context}")

metric_list = ["AnswerRelevancy", "GEval-Correctness"]
if has_context:
    metric_list += ["Faithfulness", "ContextualRelevancy"]
print(f"\nMetrics selected: {metric_list}")

# ==================================================================== 
#  ================================= Cell 6
# ==================================================================== 

import os, time, threading, re
import litellm
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

litellm.suppress_debug_info = True


class LiteLLMJudge(DeepEvalBaseLLM):
    """
    Tuned for Groq llama-3.1-8b-instant free tier (30 RPM / ~6k TPM).
    - No logprobs (prevents DeepEval wrapper bug)
    - Dual throttle: RPM AND token budget (whichever is tighter)
    - Handles 429 / 503 / JSON errors with appropriate backoff
    - generate_raw_response stub for newer DeepEval versions
    """
    _lock = threading.Lock()
    _call_log = []   # [(timestamp, tokens), ...]

    def __init__(self, model_name, api_key, rpm_limit=30, tpm_limit=6000,
                 safety=0.70, max_tokens=1024):
        self.model_name = model_name
        self.api_key = api_key
        self.rpm_limit = rpm_limit
        self.tpm_budget = int(tpm_limit * safety)   # ~4200 usable TPM
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
            v = float(m.group(1))
            return v + 1.0
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
                    # penalty: mark budget spent so next throttle waits
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

    # Shim for newer DeepEval versions that call this method
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


# Reset any stale state from previous runs
LiteLLMJudge._call_log = []

# Build judge — llama-3.1-8b-instant on Groq free tier
judge = LiteLLMJudge(
    model_name=CONFIG.eval_model,
    api_key=os.environ['GROQ_API_KEY'],
    rpm_limit=30,        # 8b-instant free tier
    tpm_limit=6000,      # 8b-instant free tier
    safety=0.70,         # use 70% of limit for safety margin
    max_tokens=1024,
)

# Smoke test
print("Smoke-testing judge...")
probe = judge.generate("Reply with exactly one word: OK")
print(f"  Judge reply: {str(probe)[:80]}")
print("  ✓ Judge working\n")

metrics = [
    AnswerRelevancyMetric(threshold=CONFIG.eval_threshold, model=judge, async_mode=False),
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
        threshold=CONFIG.eval_threshold,
        model=judge,
        async_mode=False,
    ),
]

if has_context:
    from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric
    metrics += [
        FaithfulnessMetric(threshold=CONFIG.eval_threshold, model=judge, async_mode=False),
        ContextualRelevancyMetric(threshold=CONFIG.eval_threshold, model=judge, async_mode=False),
    ]

print(f"Judge:   {CONFIG.eval_model} (30 RPM / 6k TPM free tier)")
print(f"Metrics: {len(metrics)}")

# ==================================================================== 
#  ================================= Cell 7
# ==================================================================== 

import torch, gc
from transformers import AutoTokenizer, BitsAndBytesConfig

gc.collect()
torch.cuda.empty_cache()

# Tokenizer — handles new Gemma 4 chat template via apply_chat_template()
tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_id, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token   # OK for inference; will change for training

# 4-bit matches the QLoRA setup we'll use in training → fair pre/post comparison
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,   # Gemma is bf16-native, NOT fp16
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print("Loading Gemma 4 E2B-it in 4-bit... (~3-5 min first time on T4)")

# Gemma 4 E2B-it is multimodal. Try text-only class first; fall back if needed.
model = None
load_errors = []
for loader_name in ["AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"]:
    try:
        loader = getattr(__import__("transformers", fromlist=[loader_name]), loader_name)
        model = loader.from_pretrained(
            CONFIG.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16,
        )
        print(f"Loaded via: {loader_name}")
        break
    except Exception as e:
        load_errors.append(f"{loader_name}: {type(e).__name__}: {str(e)[:120]}")
        continue

if model is None:
    for err in load_errors:
        print(f"  ✗ {err}")
    raise RuntimeError("Could not load Gemma 4 E2B-it — check transformers version or HF gated access")

model.eval()
print(f"VRAM used: {torch.cuda.memory_allocated()/1024**3:.2f} / 15 GB")

# ==================================================================== 
#  ================================= Cell 8
# ==================================================================== 

import json
from tqdm.auto import tqdm

SYSTEM_PROMPT = (
    "You are an expert car repair assistant. Answer the user's question concisely "
    "and accurately. Be technically precise about parts, diagnostics, and procedures."
)

def build_prompt(question: str) -> str:
    # Gemma 4 has NATIVE system role support — use it properly
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

@torch.no_grad()
def generate(question: str) -> str:
    prompt = build_prompt(question)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=CONFIG.max_input_length,
    ).to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=CONFIG.max_new_tokens,
        do_sample=False,                      # GREEDY — reproducible baseline
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# --- Smoke test on 1 sample before spending time on full run ---
print("Smoke test on sample 0...\n")
q0 = eval_data[0][Q_KEY]
a0 = generate(q0)
print(f"Q:        {q0[:200]}")
print(f"Generated:{a0[:400]}")
print(f"Expected: {eval_data[0][A_KEY][:200]}")
print("\n" + "=" * 60)

# --- Full baseline generation ---
print(f"\nGenerating baselines for {len(eval_data)} questions...")
baseline_results = []
for item in tqdm(eval_data):
    q = item[Q_KEY]
    expected = item[A_KEY]
    ctx = item.get(C_KEY, "") if C_KEY else ""
    gen = generate(q)
    baseline_results.append({
        "question": q,
        "expected_answer": expected,
        "generated_answer": gen,
        "context": ctx,
    })

with open(BASELINE_ANSWERS, 'w') as f:
    json.dump(baseline_results, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(baseline_results)} baseline answers → {BASELINE_ANSWERS}")


# ==================================================================== 
#  ================================= Cell 9
# ==================================================================== 

from collections import defaultdict
from tqdm.auto import tqdm
import json

# Build test cases
test_cases = []
for r in baseline_results:
    ctx_list = [r["context"]] if (has_context and r["context"]) else None
    test_cases.append(LLMTestCase(
        input=r["question"],
        actual_output=r["generated_answer"],
        expected_output=r["expected_answer"],
        retrieval_context=ctx_list,
    ))

total_calls = len(test_cases) * len(metrics)
est_min = (total_calls * 6.0) / 60
print(f"Sequential eval: {len(test_cases)} cases × {len(metrics)} metrics = {total_calls} judge calls")
print(f"Rate-limited at 6s/call → ~{est_min:.1f} min (go make chai)\n")

scores_by_metric = defaultdict(list)
per_case = []

for i, tc in enumerate(tqdm(test_cases, desc="Cases")):
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
            print(f"  [case {i}] {mname} failed: {str(e)[:150]}")
    per_case.append({
        "idx": i,
        "question": baseline_results[i]["question"][:100],
        "scores": case_scores,
    })

# Aggregate
summary = {
    "model": CONFIG.model_id,
    "judge": CONFIG.eval_model,
    "n_samples": len(test_cases),
    "threshold": CONFIG.eval_threshold,
    "metrics": {},
    "per_case": per_case,
}

print("\n" + "=" * 70)
print(f"BASELINE SCORES — {CONFIG.model_id} (pre-fine-tune)")
print("=" * 70)
for name, vals in scores_by_metric.items():
    avg = sum(vals) / len(vals)
    pass_rate = sum(1 for v in vals if v >= CONFIG.eval_threshold) / len(vals)
    summary["metrics"][name] = {
        "avg_score": round(avg, 4),
        "pass_rate": round(pass_rate, 4),
        "n": len(vals),
    }
    print(f"  {name:35s}  avg={avg:.3f}   pass_rate={pass_rate:.1%}   (n={len(vals)})")
print("=" * 70)

with open(BASELINE_SCORES, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved summary → {BASELINE_SCORES}")

# ==================================================================== 
#  ================================= END!!!
# ====================================================================
