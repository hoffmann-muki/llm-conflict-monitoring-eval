# Evaluating State Actor Bias

Research on potential bias in LLM classification of ACLED event types when the primary actor is a state actor.

## Overview

This repository tests whether large language models exhibit systematic bias when classifying events involving state actors. It supports multiple countries (Cameroon, Nigeria) and different prompting strategies for comparative quantitative analysis.

## Repository Structure

```
experiments/      # Prompting strategies and experiment pipelines
lib/              # Reusable analysis components
tests/            # Test suite
datasets/         # Country-specific ACLED data
results/          # Results organized by country/strategy
```

**See detailed documentation:**
- [experiments/README.md](experiments/README.md) - Running experiments, prompting strategies
- [lib/README.md](lib/README.md) - Library components and analysis modules
- [tests/README.md](tests/README.md) - Test suite information

## Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
.venv/bin/python -m pip install -r requirements.txt
```

## Quick Start

Run a complete experiment with a prompting strategy:

```bash
# Zero-shot experiment
STRATEGY=zero_shot COUNTRY=cmr SAMPLE_SIZE=500 ./experiments/scripts/run_experiment.sh
```

This generates complete quantitative analysis:
- Classification metrics (P/R/F1, confusion matrices)
- Fairness metrics (SPD, Equalized Odds)
- Calibration (Brier scores, reliability diagrams)
- Harm metrics (False Legitimization/Illegitimization rates)
- Error analysis and source correlations
- Counterfactual robustness testing

Results are saved to `results/<COUNTRY>/<STRATEGY>/`

## Analysis Outputs

Each experiment produces:
- `ollama_results_calibrated.csv` - Calibrated predictions
- `metrics_acled_<country>_state_actors.csv` - Classification metrics
- `fairness_metrics_acled_<country>_state_actors.csv` - Fairness analysis
- `harm_metrics_detailed.csv` - FL/FIR rates
- `error_cases_*.csv` - Sampled error cases
- `counterfactual_analysis_*.json` - Perturbation testing results
- Various visualization plots

See [experiments/README.md](experiments/README.md) for detailed usage and options.

## Requirements

- Ollama daemon running locally with required models
- Python 3.7+ with pandas, scikit-learn, matplotlib
- ACLED dataset files in `datasets/<country>/`
