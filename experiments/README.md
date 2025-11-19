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
STRATEGY=zero_shot COUNTRY=cmr SAMPLE_SIZE=500 ./scripts/run_experiment.sh

# Run with few-shot strategy (once implemented)
STRATEGY=few_shot COUNTRY=cmr SAMPLE_SIZE=500 ./scripts/run_experiment.sh

# Run with explainable strategy (once implemented)
STRATEGY=explainable COUNTRY=nga SAMPLE_SIZE=1000 ./scripts/run_experiment.sh
```

### Environment Variables

- `STRATEGY` - Prompting strategy (zero_shot, few_shot, explainable) [default: zero_shot]
- `COUNTRY` - Country code (cmr, nga) [default: cmr]
- `SAMPLE_SIZE` - Number of events [default: 500]
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
# Run classification with specific strategy
STRATEGY=zero_shot COUNTRY=cmr SAMPLE_SIZE=100 \
  .venv/bin/python pipelines/run_classification.py
```

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

# Counterfactual
COUNTRY=cmr RESULTS_DIR=results/cmr/zero_shot \
  .venv/bin/python -m lib.analysis.counterfactual \
  --models llama3.2,mistral:7b --events 50
```

## Prompting Strategies

See [prompting_strategies/README.md](prompting_strategies/README.md) for details on implementing new strategies.
