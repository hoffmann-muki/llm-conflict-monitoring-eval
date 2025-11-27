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
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Run the Ollama pipeline:

```bash
# Full analysis with all working models
COUNTRY=cmr SAMPLE_SIZE=500 \
  ./experiments/scripts/run_ollama_full_analysis.sh

# Run with specific models only
OLLAMA_MODELS=mistral:7b,llama3.1:8b COUNTRY=nga SAMPLE_SIZE=1000 \
  ./experiments/scripts/run_ollama_full_analysis.sh

# Strategy-based experiment (zero-shot, few-shot, explainable)
STRATEGY=zero_shot COUNTRY=cmr SAMPLE_SIZE=500 \
  ./experiments/scripts/run_ollama_experiment.sh
```

Run the ConfliBERT pipeline:

```bash
# Download ConfliBERT model first (one-time setup, ~437 MB)
python experiments/pipelines/conflibert/download_conflibert_model.py --out-dir models/conflibert

# ConfliBERT experiment with proportional sampling
MODEL_PATH=models/conflibert STRATEGY=zero_shot COUNTRY=cmr SAMPLE_SIZE=500 \
  ./experiments/scripts/run_conflibert_experiment.sh

# Or run the ConfliBERT classification directly
python experiments/pipelines/conflibert/run_conflibert_classification.py cmr \
  --model-path models/conflibert --strategy zero_shot --sample-size 300
```

## Workflow Architecture

The pipeline uses a **per-model-then-aggregate** workflow for fair cross-model comparisons:

```
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 1: Per-Model Inference                                        │
│   - Each model runs on the SAME sample of events                    │
│   - Outputs: ollama_results_{model-slug}_acled_{country}...csv      │
├─────────────────────────────────────────────────────────────────────┤
│ Phase 1.5: Aggregation                                              │
│   - Combines per-model files into single file                       │
│   - Output: ollama_results_acled_{country}_state_actors.csv         │
├─────────────────────────────────────────────────────────────────────┤
│ Phase 2+: Analysis                                                   │
│   - Calibration, metrics, harm analysis                             │
│   - All operate on aggregated file for cross-model analysis         │
└─────────────────────────────────────────────────────────────────────┘
```

This ensures:
- **Fair comparison**: All models classify the exact same events
- **Incremental runs**: Run one model at a time, aggregate later
- **Reproducibility**: Same sample file reused across model runs

**Sampling Configuration:**

By default, both pipelines use **proportional stratified sampling** where samples reflect the natural distribution of event types in your data. This is recommended for cross-country comparative analysis.

To oversample specific event types (e.g., for focused analysis on rare events):

```bash
# Ollama pipeline
python experiments/pipelines/ollama/run_ollama_classification.py cmr --sample-size 300 \
  --primary-group "Violence against civilians" --primary-share 0.6

# ConfliBERT pipeline
python experiments/pipelines/conflibert/run_conflibert_classification.py cmr \
  --model-path models/conflibert --sample-size 300 \
  --primary-group "Violence against civilians" --primary-share 0.6

# Or via environment variables (both scripts)
PRIMARY_GROUP="Violence against civilians" PRIMARY_SHARE=0.6 COUNTRY=cmr \
  ./experiments/scripts/run_ollama_experiment.sh
```

This generates complete quantitative analysis:
- Classification metrics (P/R/F1, confusion matrices)
- Fairness metrics (SPD, Equalized Odds)
- Calibration (Brier scores, reliability diagrams)
- Harm metrics (False Legitimization/Illegitimization rates)
- Error analysis and source correlations
- Counterfactual robustness testing (on top-N disagreement cases)

Results are saved to `results/<COUNTRY>/` or `results/<COUNTRY>/<STRATEGY>/`

## Analysis Outputs

Each experiment produces:

**Per-Model Files (Phase 1):**
- `ollama_results_{model-slug}_acled_{country}_state_actors.csv` - Per-model inference

**Aggregated Files (Phase 1.5+):**
- `ollama_results_acled_{country}_state_actors.csv` - Combined inference results
- `ollama_results_calibrated.csv` - Calibrated predictions
- `calibration_brier_scores.csv` - Calibration metrics (combined + per-model)
- `metrics_acled_{country}_state_actors.csv` - Classification metrics (combined + per-model)
- `fairness_metrics_acled_{country}_state_actors.csv` - Fairness analysis (combined + per-model)
- `harm_metrics_detailed.csv` - FL/FIR rates (combined + per-model)
- `error_cases_*.csv` - Sampled error cases
- `counterfactual_analysis_*.json` - Perturbation testing results
- Various visualization plots

See [experiments/README.md](experiments/README.md) for detailed usage and options.

## Requirements

- For Ollama pipelines: Ollama daemon running locally with required models (see `lib/core/constants.py` for `WORKING_MODELS`)
- For ConfliBERT: PyTorch and `transformers`; download the model with `download_conflibert_model.py` (~437 MB)
- Python 3.7+ with `pandas`, `scikit-learn`, `matplotlib`, `tqdm`
- ACLED dataset files in `datasets/`
