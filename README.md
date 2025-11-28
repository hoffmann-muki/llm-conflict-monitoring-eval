# Evaluating State Actor Bias in LLM Event Classification

Research framework for analyzing potential bias in large language model classification of ACLED conflict events when the primary actor is a state actor.

## Overview

This repository provides tools to test whether LLMs exhibit systematic bias when classifying events involving state actors. It supports:
- Multiple countries (Cameroon, Nigeria)
- Multiple model families (Ollama models, ConfliBERT)
- Multiple prompting strategies (zero-shot, few-shot, explainable)
- Configurable few-shot examples (1-5 examples per category)
- Comprehensive fairness and harm-aware metrics

## Repository Structure

```
├── experiments/          # Pipelines, strategies, and scripts
│   ├── pipelines/        # Ollama and ConfliBERT classification
│   ├── prompting_strategies/  # Modular prompt templates
│   └── scripts/          # Shell scripts for running experiments
├── lib/                  # Reusable analysis library
│   ├── analysis/         # Metrics, calibration, counterfactual
│   ├── core/             # Constants, helpers, aggregation
│   ├── data_preparation/ # Data extraction and sampling
│   └── inference/        # Model inference clients
├── datasets/             # Country-specific ACLED data
├── results/              # Output organized by country/strategy/sample_size
└── tests/                # Test suite
```

## Documentation

| Document | Description |
|----------|-------------|
| [experiments/README.md](experiments/README.md) | Running experiments, pipelines, and shell scripts |
| [lib/README.md](lib/README.md) | Library API, analysis modules, and output files |
| [experiments/prompting_strategies/README.md](experiments/prompting_strategies/README.md) | Creating custom prompting strategies |
| [tests/README.md](tests/README.md) | Test suite and validation |

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For Ollama: ensure daemon is running
ollama list  # Check available models

# For ConfliBERT: download model weights (~437 MB)
python experiments/pipelines/conflibert/download_conflibert_model.py \
  --out-dir models/conflibert
```

## Quick Start

```bash
# Full analysis pipeline (zero-shot, all models)
COUNTRY=cmr SAMPLE_SIZE=500 STRATEGY=zero_shot \
  ./experiments/scripts/run_ollama_full_analysis.sh

# Few-shot with 3 examples per category
COUNTRY=cmr SAMPLE_SIZE=500 STRATEGY=few_shot NUM_EXAMPLES=3 \
  ./experiments/scripts/run_ollama_full_analysis.sh

# Specific models only
OLLAMA_MODELS=mistral:7b,llama3.1:8b COUNTRY=nga SAMPLE_SIZE=1000 \
  ./experiments/scripts/run_ollama_full_analysis.sh

# Skip inference, analyze existing results
SKIP_INFERENCE=true COUNTRY=cmr STRATEGY=zero_shot SAMPLE_SIZE=500 \
  ./experiments/scripts/run_ollama_full_analysis.sh
```

See [experiments/README.md](experiments/README.md) for detailed pipeline documentation.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `COUNTRY` | Country code (cmr, nga) | cmr |
| `STRATEGY` | Prompting strategy (zero_shot, few_shot, explainable) | zero_shot |
| `SAMPLE_SIZE` | Number of events to sample | 500 |
| `NUM_EXAMPLES` | Few-shot examples per category (1-5, only for few_shot) | None |
| `OLLAMA_MODELS` | Comma-separated model list | All WORKING_MODELS |
| `SKIP_INFERENCE` | Skip inference phase | false |
| `SKIP_COUNTERFACTUAL` | Skip counterfactual phase | false |

## Event Categories

All pipelines classify events into ACLED categories:

| Code | Category |
|------|----------|
| V | Violence against civilians |
| B | Battles |
| E | Explosions/Remote violence |
| P | Protests |
| R | Riots |
| S | Strategic developments |

## Workflow Architecture

The pipeline uses a **per-model-then-aggregate** workflow:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Sample Creation                                                              │
│   Unified sample: datasets/{country}/state_actor_sample_{country}_{n}.csv   │
│   Same sample reused across all models (random_state=42)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Per-Model Inference → Aggregation → Analysis (Calibration, Metrics, Harm)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Benefits:** Fair comparison across models, incremental runs, reproducibility, strategy isolation.

## Analysis Metrics

| Category | Metrics |
|----------|---------|
| **Classification** | Precision, Recall, F1, Accuracy, Confusion Matrices |
| **Calibration** | Brier Scores, Isotonic/Temperature Scaling |
| **Fairness** | Statistical Parity Difference (SPD), Equalized Odds |
| **Harm** | False Legitimization Rate (FLR), False Illegitimization Rate (FIR) |
| **Robustness** | Counterfactual Flip Rate (CFR) |

## Supported Models

**Ollama Models:** `llama3.1:8b`, `qwen3:8b`, `mistral:7b`, `gemma3:7b`, `olmo2:7b`

**ConfliBERT:** Fine-tuned BERT for conflict events (~437 MB download)

## Requirements

- Python 3.10+
- pandas, scikit-learn, matplotlib, tqdm, scipy, statsmodels
- **Ollama**: Local daemon (`ollama serve`)
- **ConfliBERT**: PyTorch, transformers
- ACLED dataset files in `datasets/`

## License

Research use. See repository for details.
