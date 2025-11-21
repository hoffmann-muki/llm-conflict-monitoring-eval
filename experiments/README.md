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

```bash
# Run complete experiment with zero-shot strategy
STRATEGY=zero_shot COUNTRY=cmr SAMPLE_SIZE=500 \
  ./experiments/scripts/run_experiment.sh

# Run with few-shot strategy (1 example per category by default)
STRATEGY=few_shot COUNTRY=cmr SAMPLE_SIZE=500 \
  ./experiments/scripts/run_experiment.sh

# Run with few-shot strategy (3 examples per category)
STRATEGY=few_shot EXAMPLES_PER_CATEGORY=3 COUNTRY=cmr SAMPLE_SIZE=500 \
  ./experiments/scripts/run_experiment.sh

# Run with explainable strategy
STRATEGY=explainable COUNTRY=nga SAMPLE_SIZE=1000 \
  ./experiments/scripts/run_experiment.sh
```

### Environment Variables

- `STRATEGY` - Prompting strategy (zero_shot, few_shot, explainable) [default: zero_shot]
- `COUNTRY` - Country code (cmr, nga) [default: cmr]
- `SAMPLE_SIZE` - Number of events [default: 500]
- `EXAMPLES_PER_CATEGORY` - Few-shot examples per category (1-5) [default: 1]
- `CF_MODELS` - Models for counterfactual [default: llama3.2,mistral:7b]
- `CF_EVENTS` - Counterfactual event count [default: 50]
- `SKIP_INFERENCE` - Skip inference phase [default: false]
- `SKIP_COUNTERFACTUAL` - Skip counterfactual analysis [default: false]

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

```bash
# Run classification with proportional sampling (default)
python experiments/pipelines/ollama/run_ollama_classification.py cmr \
  --sample-size 300 --strategy zero_shot

# Run with targeted sampling (e.g., 60% Violence against civilians)
python experiments/pipelines/ollama/run_ollama_classification.py cmr \
  --sample-size 300 \
  --primary-group "Violence against civilians" --primary-share 0.6

# Using environment variables
STRATEGY=zero_shot COUNTRY=cmr SAMPLE_SIZE=100 \
  python experiments/pipelines/ollama/run_ollama_classification.py
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
  .venv/bin/python -m lib.analysis.counterfactual \
  --models llama3.2,mistral:7b --events 20

# Or run on a percentage of disagreements (example: top 10% of disagreements):
COUNTRY=cmr RESULTS_DIR=results/cmr/zero_shot \
  .venv/bin/python -m lib.analysis.counterfactual \
  --models llama3.2,mistral:7b --top-percent 10
```

## Prompting Strategies

See [prompting_strategies/README.md](prompting_strategies/README.md) for details on implementing new strategies.

**Note:** All prompting strategies generate prompts that are passed explicitly to `run_ollama_structured()`. There are no hardcoded prompts in the inference layer - the strategy pattern ensures complete separation of concerns.
