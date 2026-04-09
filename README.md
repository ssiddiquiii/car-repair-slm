# Car Repair SLM Fine-Tuning

Fine-tuning **Gemma 4 E2B** (Google DeepMind) on car repair Q&A data for offline Android deployment.

## Project Overview

| Item | Detail |
|------|--------|
| **Model** | Gemma 4 E2B-it (Effective 2B parameters) |
| **Training Data** | 10,000 Q&A pairs (car repair domain) |
| **Training Platform** | Kaggle (Free GPU - Tesla T4 x2) |
| **Evaluation** | DeepEval + Gemini as Judge |
| **Target Deployment** | Android (offline) |

## Project Phases

- **Phase 0**: Environment Setup (Kaggle + HuggingFace + Gemma Access) ✅
- **Phase 1**: Data Collection & Pilot Evaluation (In Progress)
- **Phase 2**: Full Fine-Tuning (10K Q&A pairs)
- **Phase 3**: Android Deployment

## Folder Structure


```text
car-repair-slm/
├── data/
│   ├── car_repair_100_qa.json
│   └── eval_20_qa.json
├── docs/
│   └── PHASES.md
├── scripts/
├── src/
│   ├── config.py
│   ├── evaluator.py
│   ├── main.py
│   └── model_loader.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Categories (10)

Engine, Brakes, Transmission, Electrical System, Cooling System, Tires & Wheels, Oil & Fluids, Body & Exterior, Suspension & Steering, Exhaust System

## Evaluation Strategy

1. Run 20 eval questions on untrained Gemma 4 E2B → **Baseline Score**
2. Fine-tune on training data
3. Run same 20 eval questions on trained model → **Post-Training Score**
4. Compare to detect improvement or erosion

## Tech Stack

- Python, PyTorch, HuggingFace Transformers
- DeepEval (LLM evaluation framework)
- Gemini API (evaluation judge)
- QLoRA / PEFT (for fine-tuning)
