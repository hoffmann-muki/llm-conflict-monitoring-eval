# Evaluating State Actor Bias

This repository contains code and data for evaluating potential bias in LLM classification of ACLED event types when the primary actor is a state actor.

Two country flows are available (Cameroon, Nigeria). The repository namespaces dataset and result artifacts by country under `datasets/<country>/` and `results/<country>/`.

## What this project does
- Extracts country-specific rows from ACLED-like datasets, normalizes actor text and selects usable rows (has notes and known event types).
- Builds stratified samples (configurable size) with a primary-group oversample for Violence against civilians.
- Runs classification with locally-hosted models (via Ollama) and saves structured predictions.
- Calibrates model confidences (isotonic regression + temperature scaling), evaluates thresholding strategies, and produces reliability and accuracy-vs-coverage plots.

## Files of interest
- `political_bias_of_llms_generic.py` — Generic country pipeline: builds stratified sample (env `SAMPLE_SIZE`), runs classifiers, and writes predictions to `results/{country}/`. Supports both Cameroon (`cmr`) and Nigeria (`nga`).
- `tools/calibrate_confidences.py` — fits per-model calibration on labeled predictions and writes calibration params JSON under `results/<COUNTRY>/`.
- `tools/apply_calibration_and_evaluate.py` — applies calibrators to predictions and writes calibrated CSV, threshold metrics and plots under `results/<COUNTRY>/`.
- `tools/compute_metrics_cmr.py` — computes per-model confusion matrices and summary metrics (country-scoped).
- `tools/compare_model_sizes.py` — compares FL/FI across different model sizes within a family (e.g., gemma:2b vs gemma:7b) with McNemar statistical tests and optional inference for missing models.
- `tools/ollama_helpers.py` — centralized utilities for Ollama model inference with structured JSON output parsing, simplified prompts optimized for reliable JSON output, and robust response parsing handling different model output patterns.
- `tools/data_helpers.py` — shared data loading and path management utilities with country-specific configuration handling.
- `tools/metrics_helpers.py` — shared FL/FI computation and aggregation functions for improved code reuse across analysis scripts.
- `tools/constants.py` — shared constants and mappings (ACLED event type labels, etc.).
- `scripts/run_calibrate_then_apply.sh` — unified driver script that accepts a country parameter (cmr or nga).

## Environment
1. Create and activate your venv (example):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies (we use the project's venv python):

```bash
.venv/bin/python -m pip install -r requirements.txt
# or at minimum
.venv/bin/python -m pip install pandas scikit-learn matplotlib
```

## Running the pipeline

### Quick Start Examples
```bash
# Main classification pipelines
COUNTRY=cmr SAMPLE_SIZE=100 .venv/bin/python political_bias_of_llms_generic.py
COUNTRY=nga SAMPLE_SIZE=100 .venv/bin/python political_bias_of_llms_generic.py

# Calibrate-then-apply workflow
COUNTRY=cmr ./scripts/run_calibrate_then_apply.sh
COUNTRY=nga ./scripts/run_calibrate_then_apply.sh

# Model size comparisons
COUNTRY=cmr .venv/bin/python -m tools.compare_model_sizes --family gemma --sizes 2b,7b --run-missing true
COUNTRY=cmr .venv/bin/python -m tools.compare_model_sizes --family qwen3 --sizes 1.7b,4b,8b --run-missing true
```

## Command Line Usage

### Core Pipeline Scripts

#### Main Classification Pipelines
```bash
# Cameroon pipeline - builds sample and runs classification
COUNTRY=cmr SAMPLE_SIZE=100 .venv/bin/python political_bias_of_llms_generic.py

# Nigeria pipeline - builds sample and runs classification  
COUNTRY=nga SAMPLE_SIZE=100 .venv/bin/python political_bias_of_llms_generic.py
```

#### Calibration and Evaluation
```bash
# Fit calibration models on predictions
COUNTRY=cmr .venv/bin/python -m tools.calibrate_confidences

# Apply calibration and generate evaluation metrics/plots
COUNTRY=cmr .venv/bin/python -m tools.apply_calibration_and_evaluate

# Combined calibrate-then-apply workflow
COUNTRY=cmr ./scripts/run_calibrate_then_apply.sh
COUNTRY=nga ./scripts/run_calibrate_then_apply.sh
```

### Analysis Tools

#### Model Size Comparisons
```bash
# Compare FL/FI across model sizes within a family
COUNTRY=cmr .venv/bin/python -m tools.compare_model_sizes --family gemma --sizes 2b,7b [OPTIONS]

# Required arguments:
#   --family FAMILY     Model family prefix (e.g., gemma, qwen3)
#   --sizes SIZES       Comma-separated sizes (e.g., 2b,7b or 1.7b,4b,8b)

# Optional arguments:
#   --run-missing {true,false}  Run inference for missing models (default: true)
#   --out OUTPUT_PATH          Custom output CSV path

# Advanced usage:
COUNTRY=cmr SMALL_SAMPLE=10 LARGE_SAMPLE=100 ./scripts/run_calibrate_then_apply.sh  # override sizes
```

#### Per-Class Analysis and Reporting
```bash
# Generate per-class metrics and disagreement analysis
COUNTRY=cmr .venv/bin/python tools/per_class_and_disagreements.py

# Generate visualizations from analysis reports  
COUNTRY=cmr .venv/bin/python tools/visualize_reports.py
```

### Utility Tools

#### Ollama Inference Helpers
```bash
# Direct model testing (for debugging)
.venv/bin/python -c "
from tools.ollama_helpers import run_ollama_structured
import json
result = run_ollama_structured('gemma:2b', 'Military forces beat civilians')
print(json.dumps(result, indent=2))
"
```

### File Requirements

#### Required Files by Tool

**compare_model_sizes.py:**
- Required: `datasets/<COUNTRY>/state_actor_sample_<COUNTRY>.csv` (sample dataset)
- Optional: `results/<COUNTRY>/ollama_results_calibrated.csv` (existing results; will run inference if missing models)
- Outputs: `results/<COUNTRY>/compare_<family>_sizes.csv`, `results/<COUNTRY>/compare_<family>_sizes_pairwise.csv`, `results/<COUNTRY>/ollama_inference_<family>-<sizes>.csv`

**calibrate_confidences.py:**
- Required: `results/<COUNTRY>/ollama_results_<dataset>.csv` (raw predictions)
- Outputs: `results/<COUNTRY>/calibration_params_<dataset>.json`

**apply_calibration_and_evaluate.py:**
- Required: `results/<COUNTRY>/ollama_results_<dataset>.csv`, `results/<COUNTRY>/calibration_params_<dataset>.json`
- Outputs: `results/<COUNTRY>/ollama_results_calibrated.csv`, plots, metrics

**per_class_and_disagreements.py:**
- Required: `results/<COUNTRY>/ollama_results_calibrated.csv`
- Outputs: `results/<COUNTRY>/per_class_report.csv`, `results/<COUNTRY>/top_disagreements.csv`

**visualize_reports.py:**
- Required: `results/<COUNTRY>/per_class_report.csv`, `results/<COUNTRY>/top_disagreements.csv`
- Outputs: `results/<COUNTRY>/per_class_metrics.png`, `results/<COUNTRY>/top_disagreements_table.png`

## Outputs and File Structure
- All outputs are organized under `results/<COUNTRY>/` (e.g., `results/cmr/` or `results/nga/`)
- **Core pipeline outputs**: `ollama_results_acled_<country>_state_actors.csv`, `ollama_results_calibrated.csv`, `calibration_params_acled_<country>_state_actors.json`, `metrics_thresholds_calibrated.csv`, `reliability_diagrams.png`, `accuracy_vs_coverage.png`
- **Model size comparisons**: `compare_<family>_sizes.csv` (FL/FI summary), `compare_<family>_sizes_pairwise.csv` (McNemar test results), `ollama_inference_<family>-<sizes>.csv` (inference-only results)
- **Analysis reports**: `per_class_report.csv`, `top_disagreements.csv`, `per_class_metrics.png`, `top_disagreements_table.png`

## Model size comparisons

The `tools/compare_model_sizes.py` script enables systematic comparison of false legitimization (FL) and false illegitimization (FI) rates across different model sizes within the same family (e.g., gemma:2b vs gemma:7b). Key features:

- **Statistical testing**: Uses McNemar's test for paired comparisons on the same events to determine if differences between models are statistically significant
- **Automatic inference**: Can run inference on missing models using the `--run-missing` flag, generating inference-only CSV files for traceability and debugging
- **FL/FI metrics**: Computes false legitimization (classifying illegitimate state actions as legitimate) and false illegitimization (classifying legitimate state actions as illegitimate) rates
- **Country-scoped**: Reads from country-specific calibrated results and sample datasets

## Requirements and Notes
- **Ollama setup**: Ollama daemon must be running locally and the required models pulled to run classification
- **Data dependencies**: Scripts expect country-scoped calibrated CSV (e.g., `results/cmr/ollama_results_calibrated.csv`) with calibrated probability column `pred_conf_temp`
- **Statistical considerations**: Small-sample thresholds are noisy; use larger calibration sets or bootstrapping for robust thresholds
- **Model comparisons**: Require consistent sample datasets across models for valid statistical testing  
- **Inference optimization**: Pipeline uses simplified, direct prompts optimized for JSON output to avoid parsing failures and focus on core classification performance rather than explanation/reasoning extraction
