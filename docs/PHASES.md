# Project Phases

## Phase 0 — Environment Setup ✅ DONE

- Kaggle account + phone verify + GPU (Tesla T4 x2) confirmed
- HuggingFace account + Read token generated
- Gemma 4 E2B model access confirmed (Apache 2.0, no approval needed)
- HF_TOKEN added to Kaggle Secrets
- Gemini API key generated and added to Kaggle Secrets
- End-to-end verification test passed

## Phase 1 — Data Collection & Pilot Evaluation (In Progress)

### Sprint 1: Pilot Test (20 Q&A)
- 100 Q&A pairs created (10 categories x 10 each)
- 20 eval pairs separated (2 per category) — never used in training
- Baseline test: 20 questions asked to untrained model
- DeepEval scoring with Gemini as judge
- Baseline scores recorded

### Sprint 2: 0 → 2,000 Q&A pairs
- Extract from existing Kaggle datasets

### Sprint 3: 2,000 → 5,000 Q&A pairs
- Extract from Mechanics StackExchange

### Sprint 4: 5,000 → 9,000 Q&A pairs
- Synthetic generation across 10 categories

### Sprint 5: 9,000 → 10,000 (Final Assembly)
- Merge, deduplicate, quality check, export final JSON

## Phase 2 — Model Fine-Tuning

- Install libraries (transformers, trl, peft, bitsandbytes)
- Preprocess and tokenize dataset
- Configure QLoRA (4-bit quantization)
- Train on Kaggle GPU
- Post-training evaluation (same 20 eval questions)
- Compare baseline vs post-training scores
- Save and upload model to HuggingFace

## Phase 3 — Android Deployment

- Convert model to mobile format (GGUF/TFLite/LiteRT)
- Apply mobile quantization (4-bit/2-bit)
- Android app integration
- Offline inference testing
- Performance benchmarking
