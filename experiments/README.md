# Experiments

Experiment pipelines, prompting strategies, and analysis scripts for LLM state actor bias research.

## Structure

```
experiments/
├── pipelines/
│   ├── ollama/              # Ollama LLM classification pipeline
│   └── conflibert/          # ConfliBERT transformer pipeline
├── prompting_strategies/    # Modular prompting strategies
└── scripts/                 # Shell scripts for running experiments
```

## Pipelines

### Ollama Pipeline

Runs classification using local Ollama models (Mistral, Llama, Gemma, etc.).

```bash
# Basic usage
python experiments/pipelines/ollama/run_ollama_classification.py cmr \
  --sample-size 500 --strategy zero_shot

# With specific models
python experiments/pipelines/ollama/run_ollama_classification.py nga \
  --sample-size 1000 --strategy few_shot \
  --models "mistral:7b,llama3.1:8b"

# Full analysis pipeline (classification + all metrics)
COUNTRY=cmr SAMPLE_SIZE=500 STRATEGY=zero_shot \
  ./experiments/scripts/run_ollama_full_analysis.sh
```

### ConfliBERT Pipeline

Runs classification using the ConfliBERT transformer model (fine-tuned BERT for conflict events).

```bash
# Download model first (one-time, ~437 MB)
python experiments/pipelines/conflibert/download_conflibert_model.py \
  --out-dir models/conflibert

# Run classification
python experiments/pipelines/conflibert/run_conflibert_classification.py cmr \
  --model-path models/conflibert --sample-size 500 --strategy zero_shot

# Via shell script
MODEL_PATH=models/conflibert COUNTRY=cmr SAMPLE_SIZE=500 \
  ./experiments/scripts/run_conflibert_experiment.sh
```

## Prompting Strategies

Modular strategies for generating classification prompts:

| Strategy | Description |
|----------|-------------|
| `zero_shot` | Direct classification without examples (default) |
| `few_shot` | Classification with 1-5 examples per category |
| `explainable` | Chain-of-thought reasoning for transparent decisions |

```python
from experiments.prompting_strategies import ZeroShotStrategy, FewShotStrategy

strategy = ZeroShotStrategy()
prompt = strategy.make_prompt("Military forces attacked civilians in the village")
system_msg = strategy.get_system_message()
```

See [prompting_strategies/README.md](prompting_strategies/README.md) for creating custom strategies.

## Scripts

### run_ollama_full_analysis.sh

Complete analysis pipeline with 5 phases:

1. **Inference** - Run models on sample events
2. **Aggregation** - Combine per-model results
3. **Calibration** - Compute calibrated probabilities
4. **Metrics** - Classification, fairness, and harm metrics
5. **Counterfactual** - Perturbation robustness testing

```bash
# Environment variables
COUNTRY=cmr          # Country code (cmr, nga)
STRATEGY=zero_shot   # Prompting strategy
SAMPLE_SIZE=500      # Number of events to sample
OLLAMA_MODELS=...    # Comma-separated model list (optional)
SKIP_INFERENCE=true  # Skip to analysis phases (if results exist)
SKIP_COUNTERFACTUAL=true  # Skip counterfactual phase

# Full run
COUNTRY=nga SAMPLE_SIZE=1000 STRATEGY=zero_shot ./experiments/scripts/run_ollama_full_analysis.sh

# Analysis only (reuse existing inference results)
SKIP_INFERENCE=true COUNTRY=cmr STRATEGY=zero_shot SAMPLE_SIZE=500 ./experiments/scripts/run_ollama_full_analysis.sh
```

### run_ollama_experiment.sh

Lighter-weight experiment script for quick classification runs.

### run_conflibert_experiment.sh

ConfliBERT experiment with same interface as Ollama scripts.

### run_calibrate_then_apply.sh

Specialized script for calibration-focused experiments.

## Sampling Options

### Proportional Sampling (Default)

Sample reflects natural class distribution in the data:

```bash
python experiments/pipelines/ollama/run_ollama_classification.py cmr \
  --sample-size 500
```

### Targeted Oversampling

Oversample specific event types for focused analysis:

```bash
# 60% Violence against civilians, 40% proportional to other classes
python experiments/pipelines/ollama/run_ollama_classification.py cmr \
  --sample-size 500 \
  --primary-group "Violence against civilians" \
  --primary-share 0.6
```

## Output Structure

Results are organized by country, strategy, and sample size:

```
results/
├── cmr/
│   ├── zero_shot/
│   │   ├── 500/
│   │   │   ├── ollama_results_mistral-7b_acled_cmr_state_actors.csv
│   │   │   ├── ollama_results_acled_cmr_state_actors.csv
│   │   │   ├── ollama_results_calibrated.csv
│   │   │   ├── metrics_acled_cmr_state_actors.csv
│   │   │   └── ...
│   │   └── 1000/
│   │       └── ...
│   └── few_shot/
│       ├── 500/
│       └── 1000/
└── nga/
    └── ...
```

### Output Files

| File | Description |
|------|-------------|
| `ollama_results_{model}_acled_{country}_state_actors.csv` | Per-model inference results |
| `ollama_results_acled_{country}_state_actors.csv` | Combined results from all models |
| `ollama_results_calibrated.csv` | Calibrated predictions |
| `metrics_acled_{country}_state_actors.csv` | Classification metrics (P/R/F1) |
| `fairness_metrics_acled_{country}_state_actors.csv` | SPD, Equalized Odds |
| `harm_metrics_detailed.csv` | False Legitimization/Illegitimization rates |
| `per_class_report.csv` | Per-class performance breakdown |
| `top_disagreements.csv` | High-confidence model disagreements |
| `counterfactual_analysis_{models}.json` | Perturbation test results |

## Requirements

- **Ollama**: Local Ollama daemon with models installed
- **ConfliBERT**: PyTorch, transformers, downloaded model weights
- **Data**: ACLED dataset in `datasets/`
- **Python**: 3.7+ with pandas, scikit-learn, matplotlib, tqdm

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
