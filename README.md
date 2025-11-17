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
- `tools/apply_calibration_and_evaluate.py` — fits calibration models AND applies them to predictions; writes calibrated CSV, threshold metrics and plots under `results/<COUNTRY>/`.
- `tools/compute_metrics.py` — computes per-model confusion matrices and summary metrics (country-generic).
- `tools/compute_thresholds_per_class.py` — computes per-class thresholds for improved classification decisions based on calibrated probabilities.
- `tools/compare_model_sizes.py` — compares FL/FI across different model sizes within a family (e.g., gemma:2b vs gemma:7b) with McNemar statistical tests and optional inference for missing models.
- `tools/counterfactual_analysis.py` — counterfactual analysis framework for understanding model disagreements through hypothesis-driven perturbations.
- `tools/visualize_counterfactual.py` — visualization suite for counterfactual analysis results with statistical plots and summary tables.
- `tools/per_class_metrics_and_disagreements.py` — generates per-class metrics and extracts top disagreement examples between models.
- `tools/visualize_reports.py` — creates visualization plots from analysis reports (per-class metrics, disagreements).
- `tools/compute_fl_fi.py` — computes False Legitimacy and False Illegitimacy metrics per model.
- `tools/ollama_helpers.py` — centralized utilities for Ollama model inference with structured JSON output parsing, simplified prompts optimized for reliable JSON output, and robust response parsing handling different model output patterns.
- `tools/data_helpers.py` — shared data loading and path management utilities with country-specific configuration handling and setup_country_environment() function.
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

### Development Environment
The repository includes VS Code configuration (`.vscode/settings.json`) that adds the `tools/` directory to the Python analysis path for improved import resolution and IntelliSense support.

## Running the pipeline

### ⚡ Quick Start: Complete Analysis (Recommended)

Run the full analysis pipeline with a single command:

```bash
# Complete analysis for Cameroon (default)
COUNTRY=cmr SAMPLE_SIZE=500 ./scripts/run_full_analysis.sh

# Complete analysis for Nigeria
COUNTRY=nga SAMPLE_SIZE=1000 ./scripts/run_full_analysis.sh

# Skip inference if predictions already exist
COUNTRY=cmr SKIP_INFERENCE=true ./scripts/run_full_analysis.sh

# Skip counterfactual analysis (faster)
COUNTRY=cmr SKIP_COUNTERFACTUAL=true ./scripts/run_full_analysis.sh

# Customize counterfactual analysis
COUNTRY=cmr CF_MODELS="llama3.2,qwen3:8b" CF_EVENTS=100 ./scripts/run_full_analysis.sh
```

**This single script runs all analysis phases:**
1. Model inference (predictions generation)
2. Calibration and core metrics (Brier scores, P/R/F1, fairness, error correlations)
3. Bias and harm analysis (FL/FI rates, error case sampling)
4. Counterfactual perturbation analysis (CFR, CDE, validity metrics)
5. Visualization and summary report generation

**All metrics computed:**
- Classification: Precision, Recall, F1, Accuracy, Confusion matrices
- Calibration: Brier scores, reliability diagrams
- Fairness: Statistical Parity Difference (SPD) with 95% bootstrap CI
- Fairness: Equalized Odds (TPR/FPR differences) with permutation tests
- Harm: False Legitimization Rate (FLR), False Illegitimization Rate (FIR)
- Source Analysis: Error correlation with ACLED notes length
- Counterfactual: Flip rates (CFR), Differential effects (CDE) with t-tests/Wilcoxon
- Counterfactual: Soft-validity metrics (edit distance, fluency)

---

### Alternative: Individual Scripts

For fine-grained control, run individual steps:

```bash
# 1. Main classification pipelines
COUNTRY=cmr SAMPLE_SIZE=100 .venv/bin/python political_bias_of_llms_generic.py
COUNTRY=nga SAMPLE_SIZE=100 .venv/bin/python political_bias_of_llms_generic.py

# 2. Calibrate-then-apply workflow (partial automation)
COUNTRY=cmr ./scripts/run_calibrate_then_apply.sh
COUNTRY=nga ./scripts/run_calibrate_then_apply.sh

# 3. Model size comparisons
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
# Apply calibration and generate evaluation metrics/plots (combines calibration fitting + application)
COUNTRY=cmr .venv/bin/python -m tools.apply_calibration_and_evaluate

# Compute per-class thresholds for improved classification
COUNTRY=cmr .venv/bin/python -m tools.compute_thresholds_per_class

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
COUNTRY=cmr .venv/bin/python -m tools.per_class_metrics_and_disagreements

# Generate visualizations from analysis reports  
COUNTRY=cmr .venv/bin/python -m tools.visualize_reports

# Compute FL/FI metrics by model
COUNTRY=cmr .venv/bin/python -m tools.compute_fl_fi

# Generate confusion matrices and summary metrics
COUNTRY=cmr .venv/bin/python -m tools.compute_metrics
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

**apply_calibration_and_evaluate.py:**
- Required: `results/<COUNTRY>/ollama_results_acled_<COUNTRY>_state_actors.csv` (raw predictions)
- Outputs: `results/<COUNTRY>/ollama_results_calibrated.csv`, `results/<COUNTRY>/calibration_params_acled_<COUNTRY>_state_actors.json`, plots, metrics

**compute_thresholds_per_class.py:**
- Required: `results/<COUNTRY>/ollama_results_calibrated.csv`
- Outputs: `results/<COUNTRY>/selected_thresholds_per_class.csv`, `results/<COUNTRY>/selected_thresholds.json`

**per_class_metrics_and_disagreements.py:**
- Required: `results/<COUNTRY>/ollama_results_calibrated.csv`
- Outputs: `results/<COUNTRY>/per_class_report.csv`, `results/<COUNTRY>/top_disagreements.csv`

**visualize_reports.py:**
- Required: `results/<COUNTRY>/per_class_report.csv`, `results/<COUNTRY>/top_disagreements.csv`
- Outputs: `results/<COUNTRY>/per_class_metrics.png`, `results/<COUNTRY>/top_disagreements_table.png`

**counterfactual_analysis.py:**
- Required: `datasets/<COUNTRY>/state_actor_sample_<COUNTRY>.csv`, `results/<COUNTRY>/ollama_results_calibrated.csv`
- Outputs: `results/<COUNTRY>/counterfactual_analysis_<models>.json`, `results/<COUNTRY>/counterfactual_analysis_<models>_summary.csv`

**visualize_counterfactual.py:**
- Required: `results/<COUNTRY>/counterfactual_analysis_<models>.json`
- Outputs: Multiple visualization plots and `counterfactual_report.txt`

## Complete Analysis Workflow

### Execution Order (Automated by `run_full_analysis.sh`)

**Phase 1: Model Inference**
1. `political_bias_of_llms_generic.py` → Generates predictions

**Phase 2: Calibration & Core Metrics**
2. `tools.apply_calibration_and_evaluate` → Calibration + Brier scores
3. `tools.compute_metrics` → Classification metrics + fairness + error correlations
4. `tools.compute_thresholds_per_class` → Per-class decision thresholds

**Phase 3: Bias & Harm Analysis**
5. `tools.compute_fl_fi` → False Legitimization/Illegitimization rates
6. `tools.per_class_metrics_and_disagreements` → Error case sampling (N≤200 per type)
7. `tools.visualize_reports` → Generate visualization plots

**Phase 4: Counterfactual Analysis** (Optional)
8. `tools.counterfactual_analysis` → Perturbation testing (CFR, CDE, validity)
9. `tools.visualize_counterfactual` → Counterfactual visualizations

### Environment Variables

**`run_full_analysis.sh` supports:**
- `COUNTRY` - Country code (cmr, nga) [default: cmr]
- `SAMPLE_SIZE` - Number of events [default: 500]
- `CF_MODELS` - Models for counterfactual [default: llama3.2,mistral:7b]
- `CF_EVENTS` - Counterfactual event count [default: 50]
- `SKIP_INFERENCE` - Skip phase 1 [default: false]
- `SKIP_COUNTERFACTUAL` - Skip phase 4 [default: false]

## Outputs and File Structure

All outputs are organized under `results/<COUNTRY>/` (e.g., `results/cmr/` or `results/nga/`)

### Core Predictions & Calibration
- `ollama_results_acled_<country>_state_actors.csv` - Raw model predictions
- `ollama_results_calibrated.csv` - Calibrated predictions with isotonic/temperature scaling
- `calibration_params_acled_<country>_state_actors.json` - Calibration parameters
- `calibration_brier_scores.csv` - **NEW**: Brier scores (raw, isotonic, temperature)
- `reliability_diagrams.png` - Calibration quality plots
- `accuracy_vs_coverage.png` - Selective prediction analysis
- `isotonic_mappings.json` - Isotonic regression mappings

### Classification & Fairness Metrics
- `metrics_acled_<country>_state_actors.csv` - Precision, Recall, F1, Accuracy per model
- `confusion_matrices_acled_<country>_state_actors.json` - Per-model confusion matrices
- `fairness_metrics_acled_<country>_state_actors.csv` - **NEW**: SPD, Equalized Odds, bootstrap CI, permutation tests
- `per_class_report.csv` - Per-class performance metrics
- `selected_thresholds_per_class.csv` - Optimized decision thresholds per class
- `metrics_thresholds_calibrated.csv` - Threshold analysis

### Bias & Harm Analysis
- `fl_fi_by_model.csv` - False Legitimacy/Illegitimacy counts
- `harm_metrics_detailed.csv` - **NEW**: FLR, FIR rates, harm ratios
- `error_cases_false_legitimization.csv` - **NEW**: N≤200 error samples for analysis
- `error_cases_false_illegitimization.csv` - **NEW**: N≤200 error samples for analysis
- `top_disagreements.csv` - High-confidence model disagreements

### Source & Correlation Analysis
- `error_correlations_acled_<country>_state_actors.csv` - **NEW**: Error rate vs notes length, Spearman correlations

### Counterfactual Analysis
- `counterfactual_analysis_<models>.json` - **NEW**: Full counterfactual results with CFR, CDE, soft-validity
- `counterfactual_analysis_<models>_summary.csv` - Summary table
- Various counterfactual visualization plots

### Visualizations
- `per_class_metrics.png` - Per-class performance visualization
- `top_disagreements_table.png` - Disagreement table visualization
- Additional counterfactual visualizations (if phase 4 runs)

## Counterfactual Analysis

The repository includes a counterfactual analysis framework for understanding why models make different classifications:

- **Hypothesis-driven perturbations**: Tests specific hypotheses about model sensitivity (actor substitution, intensity modifiers, action synonyms, negation, sufficiency)
- **Statistical analysis**: Includes McNemar tests for paired model comparisons and clustering of sensitivity patterns
- **Comprehensive reporting**: Generates detailed JSON reports, summary CSV files, and multiple visualization plots
- **Model disagreement analysis**: Focuses on events where models disagree to understand decision boundaries

### Usage
```bash
# Run counterfactual analysis to understand model disagreements
COUNTRY=nga .venv/bin/python -m tools.counterfactual_analysis --models llama3.2,mistral:7b --events 10

# Generate visualizations for counterfactual results
COUNTRY=nga .venv/bin/python -m tools.visualize_counterfactual --input results/nga/counterfactual_analysis.json
```

## Testing

The repository includes a test suite under `tests/` to validate components:

```bash
# Test the generic pipeline components
.venv/bin/python tests/test_generic_pipeline.py

# Test the counterfactual analysis framework  
.venv/bin/python tests/test_counterfactual.py
```

**Available Tests:**
- `tests/test_generic_pipeline.py` — Validates the generic pipeline imports, country setup, and basic functionality
- `tests/test_counterfactual.py` — Tests the counterfactual analysis framework with sample data

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
