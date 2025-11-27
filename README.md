# Evaluating State Actor Bias in LLM Event Classification

Research framework for analyzing potential bias in large language model classification of ACLED conflict events when the primary actor is a state actor.

## Overview

This repository provides tools to test whether LLMs exhibit systematic bias when classifying events involving state actors. It supports:
- Multiple countries (Cameroon, Nigeria)
- Multiple model families (Ollama models, ConfliBERT)
- Multiple prompting strategies (zero-shot, few-shot, explainable)
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
├── results/              # Results by country/strategy
└── tests/                # Test suite
```

**Documentation:**
- [experiments/README.md](experiments/README.md) - Running experiments and pipelines
- [lib/README.md](lib/README.md) - Library components and analysis modules
- [experiments/prompting_strategies/README.md](experiments/prompting_strategies/README.md) - Creating custom strategies
- [tests/README.md](tests/README.md) - Test suite

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For Ollama: ensure daemon is running with required models
ollama list  # Check available models

# For ConfliBERT: download model weights (~437 MB)
python experiments/pipelines/conflibert/download_conflibert_model.py \
  --out-dir models/conflibert
```

## Quick Start

### Full Analysis Pipeline

```bash
# Run complete analysis with all models
COUNTRY=cmr SAMPLE_SIZE=500 STRATEGY=zero_shot \
  ./experiments/scripts/run_ollama_full_analysis.sh

# With specific models
OLLAMA_MODELS=mistral:7b,llama3.1:8b COUNTRY=nga SAMPLE_SIZE=1000 \
  ./experiments/scripts/run_ollama_full_analysis.sh

# Skip inference (analyze existing results)
SKIP_INFERENCE=true COUNTRY=cmr \
  ./experiments/scripts/run_ollama_full_analysis.sh
```

### Individual Components

```bash
# Classification only
python experiments/pipelines/ollama/run_ollama_classification.py cmr \
  --sample-size 500 --strategy zero_shot

# Aggregate per-model results
COUNTRY=cmr STRATEGY=zero_shot python -m lib.core.result_aggregator

# Run specific analyses
COUNTRY=cmr STRATEGY=zero_shot python -m lib.analysis.calibration
COUNTRY=cmr STRATEGY=zero_shot python -m lib.analysis.metrics
COUNTRY=cmr STRATEGY=zero_shot python -m lib.analysis.harm
COUNTRY=cmr STRATEGY=zero_shot python -m lib.analysis.counterfactual --events 20
```

### ConfliBERT Pipeline

```bash
# Download model (one-time)
python experiments/pipelines/conflibert/download_conflibert_model.py \
  --out-dir models/conflibert

# Run classification
python experiments/pipelines/conflibert/run_conflibert_classification.py cmr \
  --model-path models/conflibert --sample-size 500
```

## Workflow Architecture

The pipeline uses a **per-model-then-aggregate** workflow for fair cross-model comparisons:

```
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 1: Per-Model Inference                                        │
│   Each model classifies the SAME sample of events                   │
│   Output: results/{country}/{strategy}/ollama_results_{model}_...   │
├─────────────────────────────────────────────────────────────────────┤
│ Phase 1.5: Aggregation                                              │
│   Combine per-model files into single dataset                       │
│   Output: results/{country}/{strategy}/ollama_results_acled_...     │
├─────────────────────────────────────────────────────────────────────┤
│ Phase 2-5: Analysis                                                 │
│   Calibration, metrics, harm analysis, counterfactual testing       │
│   Output: Various analysis files in same directory                  │
└─────────────────────────────────────────────────────────────────────┘
```

**Benefits:**
- **Fair comparison**: All models classify identical events
- **Incremental runs**: Run one model at a time, aggregate later
- **Reproducibility**: Same sample file reused across model runs
- **Strategy isolation**: Different strategies produce separate result sets

## Results Directory Structure

```
results/
├── cmr/
│   ├── zero_shot/
│   │   ├── ollama_results_mistral-7b_acled_cmr_state_actors.csv
│   │   ├── ollama_results_acled_cmr_state_actors.csv
│   │   ├── ollama_results_calibrated.csv
│   │   ├── metrics_acled_cmr_state_actors.csv
│   │   ├── fairness_metrics_acled_cmr_state_actors.csv
│   │   ├── harm_metrics_detailed.csv
│   │   └── ...
│   └── few_shot/
│       └── ...
└── nga/
    ├── zero_shot/
    │   └── ...
    └── few_shot/
        └── ...
```

## Analysis Metrics

The framework computes comprehensive metrics:

| Category | Metrics |
|----------|---------|
| **Classification** | Precision, Recall, F1, Accuracy, Confusion Matrices |
| **Calibration** | Brier Scores, Isotonic/Temperature Scaling, Reliability Diagrams |
| **Fairness** | Statistical Parity Difference (SPD), Equalized Odds (TPR/FPR) |
| **Harm** | False Legitimization Rate (FLR), False Illegitimization Rate (FIR) |
| **Robustness** | Counterfactual Flip Rate (CFR), Counterfactual Differential Effect (CDE) |
| **Error Analysis** | Source correlations, text length effects, sampled error cases |

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

## Sampling Options

### Proportional Sampling (Default)

Samples reflect the natural class distribution in the data:

```bash
python experiments/pipelines/ollama/run_ollama_classification.py cmr \
  --sample-size 500
```

### Targeted Oversampling

Oversample specific event types for focused analysis:

```bash
python experiments/pipelines/ollama/run_ollama_classification.py cmr \
  --sample-size 500 \
  --primary-group "Violence against civilians" \
  --primary-share 0.6
```

## Requirements

- Python 3.7+
- pandas, scikit-learn, matplotlib, tqdm, scipy
- **Ollama**: Local daemon with models (`ollama serve`)
- **ConfliBERT**: PyTorch, transformers, model weights
- ACLED dataset files in `datasets/`

## Supported Models

### Ollama Models (default)
- `llama3.1:8b`
- `qwen3:8b`
- `mistral:7b`
- `gemma3:7b`
- `olmo2:7b`

### ConfliBERT
- Fine-tuned BERT for conflict event classification
- Requires local model download (~437 MB)

## License

Research use. See repository for details.
