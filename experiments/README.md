# Experiments

This directory contains experimental pipelines for testing different prompting strategies with complete quantitative analysis.

## Structure

```
experiments/
├── prompting_strategies/  # Different prompting approaches
├── pipelines/             # Classification pipeline
└── scripts/               # Automation scripts
```

## Running Experiments

### Quick Start

**Ollama Pipeline:**

```bash
# Run complete experiment with zero-shot strategy
STRATEGY=zero_shot COUNTRY=cmr SAMPLE_SIZE=500 \
  ./experiments/scripts/run_ollama_experiment.sh

# Run with few-shot strategy
STRATEGY=few_shot COUNTRY=cmr SAMPLE_SIZE=500 \
  ./experiments/scripts/run_ollama_experiment.sh

# Run with targeted sampling (60% Violence against civilians)
PRIMARY_GROUP="Violence against civilians" PRIMARY_SHARE=0.6 \
  STRATEGY=zero_shot COUNTRY=cmr SAMPLE_SIZE=500 \
  ./experiments/scripts/run_ollama_experiment.sh
```

**ConfliBERT Pipeline:**

```bash
# Download model first (one-time setup, ~437 MB)
python experiments/pipelines/conflibert/download_conflibert_model.py --out-dir models/conflibert

# Run complete experiment with zero-shot strategy
MODEL_PATH=models/conflibert STRATEGY=zero_shot COUNTRY=cmr SAMPLE_SIZE=500 \
  ./experiments/scripts/run_conflibert_experiment.sh

# Run with targeted sampling
MODEL_PATH=models/conflibert PRIMARY_GROUP="Violence against civilians" PRIMARY_SHARE=0.6 \
  STRATEGY=zero_shot COUNTRY=cmr SAMPLE_SIZE=500 \
  ./experiments/scripts/run_conflibert_experiment.sh
```

### Environment Variables

**Common to both pipelines:**
- `STRATEGY` - Prompting strategy (zero_shot, few_shot, explainable) [default: zero_shot]
- `COUNTRY` - Country code (cmr, nga) [default: cmr]
- `SAMPLE_SIZE` - Number of events [default: 500]
- `PRIMARY_GROUP` - Event type to oversample [default: none (proportional)]
- `PRIMARY_SHARE` - Fraction for primary group (0-1) [default: 0.0]
- `CF_MODELS` - Models for counterfactual [default: all WORKING_MODELS]
- `CF_EVENTS` - Counterfactual event count [default: 50]
- `SKIP_INFERENCE` - Skip inference phase [default: false]
- `SKIP_COUNTERFACTUAL` - Skip counterfactual analysis [default: false]

**Ollama-specific:**
- `OLLAMA_MODELS` - Models for inference [default: all WORKING_MODELS]
- `EXAMPLES_PER_CATEGORY` - Few-shot examples per category (1-5) [default: 1]

**ConfliBERT-specific:**
- `MODEL_PATH` - Path to local ConfliBERT model directory [default: models/conflibert]
- `BATCH_SIZE` - Batch size for inference [default: 16]
- `MAX_LENGTH` - Maximum sequence length [default: 256]
- `DEVICE` - Device (cuda, mps, cpu) [default: auto]

## Results Organization

Results are organized by country and strategy:

```
results/
├── cmr/
│   ├── zero_shot/
│   ├── few_shot/
│   └── explainable/
└── nga/
    └── ...
```

Each strategy folder contains complete quantitative analysis:
- Classification metrics (P/R/F1, confusion matrices)
- Fairness metrics (SPD, Equalized Odds with statistical tests)
- Calibration metrics (Brier scores, reliability diagrams)
- Harm metrics (False Legitimization/Illegitimization rates)
- Error analysis (sampled cases, correlations)
- Counterfactual robustness (perturbation testing)

## Individual Scripts

### Classification Pipeline

**Ollama:**

```bash
# Run classification with proportional sampling (default)
python experiments/pipelines/ollama/run_ollama_classification.py cmr \
  --sample-size 300 --strategy zero_shot

# Run with targeted sampling (e.g., 60% Violence against civilians)
python experiments/pipelines/ollama/run_ollama_classification.py cmr \
  --sample-size 300 \
  --primary-group "Violence against civilians" --primary-share 0.6
```

**ConfliBERT:**

```bash
# Run classification with proportional sampling (default)
python experiments/pipelines/conflibert/run_conflibert_classification.py cmr \
  --model-path models/conflibert --sample-size 300 --strategy zero_shot

# Run with targeted sampling
python experiments/pipelines/conflibert/run_conflibert_classification.py cmr \
  --model-path models/conflibert --sample-size 300 \
  --primary-group "Violence against civilians" --primary-share 0.6
```

**Sampling Options:**
- By default, samples are drawn **proportionally** to reflect natural class distributions
- Use `--primary-group` and `--primary-share` to oversample specific event types
- Proportional sampling recommended for cross-country comparative analysis

### Analysis Scripts

```bash
# Calibration
COUNTRY=cmr RESULTS_DIR=results/cmr/zero_shot \
  .venv/bin/python -m lib.analysis.calibration

# Metrics
COUNTRY=cmr RESULTS_DIR=results/cmr/zero_shot \
  .venv/bin/python -m lib.analysis.metrics

# Harm analysis
COUNTRY=cmr RESULTS_DIR=results/cmr/zero_shot \
  .venv/bin/python -m lib.analysis.harm

# Counterfactual (requires top_disagreements.csv from per_class_metrics)
COUNTRY=cmr RESULTS_DIR=results/cmr/zero_shot \
  .venv/bin/python -m lib.analysis.per_class_metrics  # Run first
COUNTRY=cmr RESULTS_DIR=results/cmr/zero_shot \
  .venv/bin/python -m lib.analysis.counterfactual --events 20  # Uses all WORKING_MODELS

# Or specify models explicitly:
COUNTRY=cmr RESULTS_DIR=results/cmr/zero_shot \
  .venv/bin/python -m lib.analysis.counterfactual \
  --models llama3.2,mistral:7b --events 20

# Or run on a percentage of disagreements (example: top 10% of disagreements):
COUNTRY=cmr RESULTS_DIR=results/cmr/zero_shot \
  .venv/bin/python -m lib.analysis.counterfactual --top-percent 10
```

## Prompting Strategies

See [prompting_strategies/README.md](prompting_strategies/README.md) for details on implementing new strategies.

**Note:** All prompting strategies generate prompts that are passed explicitly to `run_ollama_structured()`. There are no hardcoded prompts in the inference layer - the strategy pattern ensures complete separation of concerns.
