import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_gemma_model(model_name="google/gemma-4-E2B-it"):
    print("Loading Gemma Model (This may take a moment)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto" 
    )
    print("Gemma 4 E2B loaded ✅")
    return model, tokenizer