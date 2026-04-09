import os
import json
import torch
from src.config import setup_environment
from src.model_loader import load_gemma_model
from src.evaluator import run_deepeval

def main():
    # 1. Initialize Environment
    gemini_key = setup_environment()

    # 2. Load Data
    data_path = os.path.join(os.path.dirname(__file__), "data", "eval_20_qa.json")
    with open(data_path, "r") as f:
        eval_data = json.load(f)
    print(f"Loaded {len(eval_data)} eval questions ✅")

    # 3. Load Model Engine
    model, tokenizer = load_gemma_model()

    # 4. Generate Answers (Inference)
    print("Generating answers...")
    results = []
    
    for item in eval_data:
        messages = [{"role": "user", "content": item["question"]}]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True
        ).to(model.device)
        
        input_len = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=300)
            
        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        results.append({
            "id": item.get("id", "N/A"),
            "question": item["question"],
            "expected": item["answer"], 
            "actual": response
        })
        print(f"Answered: {item['question'][:30]}...")

    # Save Backup Answers
    output_path = os.path.join(os.path.dirname(__file__), "data", "baseline_answers.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Answers saved to data/baseline_answers.json ✅")

    # 5. Run AI Evaluation
    run_deepeval(results, gemini_key)

if __name__ == "__main__":
    main()