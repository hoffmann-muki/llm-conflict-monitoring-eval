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

## Environment Variables

All pipelines support consistent environment variable configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `COUNTRY` | Country code (cmr, nga) | cmr |
| `STRATEGY` | Prompting strategy (zero_shot, few_shot, explainable) | zero_shot |
| `SAMPLE_SIZE` | Number of events to sample | 500 |
| `NUM_EXAMPLES` | Few-shot examples per category (1-5, only for few_shot) | None |
| `OLLAMA_MODELS` | Comma-separated model list | All WORKING_MODELS |

## Pipelines

### Ollama Pipeline

Runs classification using local Ollama models (Mistral, Llama, Gemma, etc.).

```bash
# Basic usage (uses environment variables)
COUNTRY=cmr SAMPLE_SIZE=500 STRATEGY=zero_shot \
  python experiments/pipelines/ollama/run_ollama_classification.py

# Few-shot with 3 examples per category
COUNTRY=cmr SAMPLE_SIZE=500 STRATEGY=few_shot NUM_EXAMPLES=3 \
  python experiments/pipelines/ollama/run_ollama_classification.py

# With specific models
COUNTRY=nga SAMPLE_SIZE=1000 STRATEGY=zero_shot OLLAMA_MODELS="mistral:7b,llama3.1:8b" \
  python experiments/pipelines/ollama/run_ollama_classification.py

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
COUNTRY=cmr SAMPLE_SIZE=500 \
  python experiments/pipelines/conflibert/run_conflibert_classification.py \
    --model-path models/conflibert

# Via shell script
MODEL_PATH=models/conflibert COUNTRY=cmr SAMPLE_SIZE=500 \
  ./experiments/scripts/run_conflibert_experiment.sh
```

## Prompting Strategies

Modular strategies for generating classification prompts:

| Strategy | Description | Extra Config |
|----------|-------------|--------------|
| `zero_shot` | Direct classification without examples (default) | - |
| `few_shot` | Classification with 1-5 examples per category | `NUM_EXAMPLES=1..5` |
| `explainable` | Chain-of-thought reasoning for transparent decisions | - |

```python
from experiments.prompting_strategies import ZeroShotStrategy, FewShotStrategy

# Zero-shot
strategy = ZeroShotStrategy()
prompt = strategy.make_prompt("Military forces attacked civilians in the village")
system_msg = strategy.get_system_message()

# Few-shot with examples
strategy = FewShotStrategy(num_examples=3)
prompt = strategy.make_prompt("Protesters gathered in the capital")
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
COUNTRY=cmr            # Country code (cmr, nga)
STRATEGY=zero_shot     # Prompting strategy
SAMPLE_SIZE=500        # Number of events to sample
NUM_EXAMPLES=3         # Few-shot examples (only for few_shot strategy)
OLLAMA_MODELS=...      # Comma-separated model list (optional)
SKIP_INFERENCE=true    # Skip to analysis phases (if results exist)
SKIP_COUNTERFACTUAL=true  # Skip counterfactual phase

# Full run with zero-shot
COUNTRY=nga SAMPLE_SIZE=1000 STRATEGY=zero_shot \
  ./experiments/scripts/run_ollama_full_analysis.sh

# Full run with few-shot (3 examples)
COUNTRY=cmr SAMPLE_SIZE=500 STRATEGY=few_shot NUM_EXAMPLES=3 \
  ./experiments/scripts/run_ollama_full_analysis.sh

# Analysis only (reuse existing inference results)
SKIP_INFERENCE=true COUNTRY=cmr STRATEGY=zero_shot SAMPLE_SIZE=500 \
  ./experiments/scripts/run_ollama_full_analysis.sh
```

### run_ollama_experiment.sh

Lighter-weight experiment script for quick classification runs.

### run_conflibert_experiment.sh

ConfliBERT experiment with same interface as Ollama scripts.

### run_calibrate_then_apply.sh

Two-stage calibration workflow: calibrate on small sample, apply to larger sample.

```bash
COUNTRY=cmr STRATEGY=zero_shot SMALL_SAMPLE=20 LARGE_SAMPLE=50 \
  ./experiments/scripts/run_calibrate_then_apply.sh
```

## Sample Reuse for Fair Comparison

For fair cross-model comparisons, sample files are created once and reused:

```
datasets/{country}/state_actor_sample_{country}_{sample_size}.csv
```

Example: `datasets/cmr/state_actor_sample_cmr_500.csv`

When running multiple models on the same country/sample_size combination, all models classify the **exact same events** for valid comparison.

## Sampling Options

### Proportional Sampling (Default)

Sample reflects natural class distribution in the data:

```bash
COUNTRY=cmr SAMPLE_SIZE=500 STRATEGY=zero_shot \
  python experiments/pipelines/ollama/run_ollama_classification.py
```

### Targeted Oversampling

Oversample specific event types for focused analysis:

```bash
# 60% Violence against civilians, 40% proportional to other classes
python experiments/pipelines/ollama/run_ollama_classification.py \
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
│       └── 500/
│           ├── 3/       # NUM_EXAMPLES=3
│           └── 5/       # NUM_EXAMPLES=5
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
- **Python**: 3.10+ with pandas, scikit-learn, matplotlib, tqdm

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
