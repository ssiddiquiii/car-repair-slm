"""
Baseline Evaluation - Gemma 4 E2B on Car Repair Q&A
Run on Kaggle with GPU T4 x2

What this does:
- Ask 20 car repair questions to untrained model
- Score the answers using DeepEval
- Save scores as baseline (the "before" snapshot)
"""

# --- Install dependencies ---
# run these once, then comment out
# !pip install -q deepeval google-genai
# !pip install -q git+https://github.com/huggingface/transformers.git


# --- Load the 20 eval questions ---
# these questions are only for testing, never for training
# each has a question + expected correct answer

import json

with open("data/eval_20_qa.json", "r") as f:
    eval_data = json.load(f)

print(f"Loaded {len(eval_data)} eval questions")


# --- Load API keys ---
# pulling keys from Kaggle Secrets (like .env in Node.js)
# GEMINI_API_KEY = for evaluation judge
# HF_TOKEN = for downloading model from HuggingFace

import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
os.environ["GEMINI_API_KEY"] = secrets.get_secret("GEMINI_API_KEY")
hf_token = secrets.get_secret("HF_TOKEN")

print("Keys loaded ✅")


# --- Load the model ---
# tokenizer = converts text to numbers (model only reads numbers)
# model = the actual Gemma 4 E2B brain
# dtype=bfloat16 = half precision, uses less GPU memory
# device_map="auto" = auto-distributes across available GPUs

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

login(token=hf_token)

model_name = "google/gemma-4-E2B-it"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto"
)

print("Gemma 4 E2B loaded ✅")


# --- Generate answers for 20 questions ---
# asking the untrained model each question
# saving both the model's answer and the correct answer

results = []

for item in eval_data:
    # wrap question in chat format
    messages = [{"role": "user", "content": item["question"]}]

    # convert to model-readable format (numbers)
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",        # return PyTorch tensors
        add_generation_prompt=True,  # tells model to start answering
        return_dict=True             # return as dictionary
    ).to(model.device)              # move to GPU

    # save input length to separate question from answer later
    input_len = inputs["input_ids"].shape[1]

    # generate answer without training (no_grad saves memory)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300)

    # decode: numbers back to text, skip question part
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    results.append({
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "expected": item["answer"],
        "actual": response
    })
    print(f"✅ {item['id']}/20 - {item['category']}")

print(f"\nAll {len(results)} answers generated!")

# save answers to file (backup in case eval crashes)
with open("baseline_answers.json", "w") as f:
    json.dump(results, f, indent=2)

print("Answers saved ✅")


# --- Score answers using DeepEval ---
# GEval = uses one AI to judge another AI's answers
# GeminiModel = the judge (free Gemini API)
# LLMTestCase = question + model answer + correct answer
# run_async=False = one request at a time (avoids rate limit)

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval import evaluate
from deepeval.models import GeminiModel
from deepeval.evaluate import AsyncConfig

# setup the judge
gemini_model = GeminiModel(
    model="gemini-2.0-flash",
    api_key=os.environ["GEMINI_API_KEY"]
)

# define what to evaluate
correctness_metric = GEval(
    name="Correctness",
    criteria="Determine if the 'actual output' is factually correct and covers the key points mentioned in the 'expected output' for car repair advice.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.5,
    model=gemini_model
)

# build test cases
test_cases = []
for r in results:
    test_cases.append(LLMTestCase(
        input=r["question"],
        actual_output=r["actual"],
        expected_output=r["expected"]
    ))

# run evaluation
evaluate(
    test_cases=test_cases,
    metrics=[correctness_metric],
    async_config=AsyncConfig(run_async=False)
)

print("\n--- BASELINE EVALUATION COMPLETE ---")
