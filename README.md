# Evaluating State Actor Bias

This repository contains code and data for evaluating potential bias in LLM classification of ACLED event types when the primary actor is a state actor.

Two country flows are available (Cameroon, Nigeria). The repository namespaces dataset and result artifacts by country under `datasets/<country>/` and `results/<country>/<strategy>/`.

## Repository Structure

```
experiments/              # Experimental pipelines for different prompting strategies
├── prompting_strategies/ # Different prompting approaches (zero-shot, few-shot, explainable)
├── pipelines/            # Classification pipeline (strategy-agnostic)
└── scripts/              # Automation scripts for running experiments

lib/                      # Reusable library components
├── data_preparation/     # Data extraction, normalization, sampling
├── inference/            # Ollama client and inference utilities
├── analysis/             # Metrics, calibration, fairness, counterfactual analysis
└── core/                 # Constants, helpers, shared utilities

tests/                    # Test suite
datasets/                 # Country-specific datasets
results/                  # Experiment results organized by country/strategy
```

## What this project does

- Extracts country-specific rows from ACLED-like datasets, normalizes actor text and selects usable rows (has notes and known event types).
- Builds stratified samples (configurable size) with a primary-group oversample for Violence against civilians.
- Runs classification experiments with **different prompting strategies** (zero-shot, few-shot, explainable) to compare model performance.
- Generates **complete quantitative analysis** for each strategy: classification metrics, fairness analysis, harm metrics, counterfactual robustness.
- Calibrates model confidences (isotonic regression + temperature scaling), evaluates thresholding strategies, and produces reliability and accuracy-vs-coverage plots.

## Key Components

### Experiments
- `experiments/pipelines/run_classification.py` — Strategy-agnostic classification pipeline; accepts `STRATEGY` env var
- `experiments/prompting_strategies/` — Modular prompting strategies:
  - `zero_shot.py` — Direct classification without examples (current default)
  - `few_shot.py` — Classification with example demonstrations (ready for implementation)
  - `explainable.py` — Chain-of-thought reasoning prompts (ready for implementation)
- `experiments/scripts/run_experiment.sh` — Main experiment runner with strategy selection
- `experiments/scripts/run_full_analysis.sh` — Complete analysis pipeline automation
- `experiments/scripts/run_calibrate_then_apply.sh` — Calibration workflow

### Library (lib/)
- `lib/data_preparation/` — Data extraction, normalization, and sampling utilities
- `lib/inference/ollama_client.py` — Ollama model inference with structured JSON output
- `lib/analysis/calibration.py` — Calibration (isotonic + temperature scaling) and Brier scores
- `lib/analysis/metrics.py` — Classification metrics, confusion matrices, fairness analysis
- `lib/analysis/harm.py` — False Legitimization/Illegitimization rates and harm metrics
- `lib/analysis/counterfactual.py` — Counterfactual perturbation analysis framework
- `lib/analysis/thresholds.py` — Per-class decision threshold optimization
- `lib/analysis/per_class.py` — Per-class metrics and error case sampling
- `lib/analysis/visualize_reports.py` — Visualization plots for analysis reports
- `lib/analysis/visualize_counterfactual.py` — Counterfactual analysis visualizations
- `lib/analysis/compare_models.py` — Model size comparisons with McNemar tests
- `lib/core/constants.py` — Shared constants and ACLED event type mappings
- `lib/core/data_helpers.py` — Path management and country-specific configuration
- `lib/core/metrics_helpers.py` — FL/FI computation and aggregation functions

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
The repository includes VS Code configuration (`.vscode/settings.json`) that adds the `lib/` and `experiments/` directories to the Python analysis path for improved import resolution and IntelliSense support.

## Running Experiments

### ⚡ Quick Start: Complete Experiment with Strategy Selection

Run experiments with different prompting strategies to compare quantitative results:

```bash
# Zero-shot experiment (current default approach)
STRATEGY=zero_shot COUNTRY=cmr SAMPLE_SIZE=500 ./experiments/scripts/run_experiment.sh

# Few-shot experiment with examples (once implemented)
STRATEGY=few_shot COUNTRY=cmr SAMPLE_SIZE=500 ./experiments/scripts/run_experiment.sh

# Explainable experiment with chain-of-thought (once implemented)
STRATEGY=explainable COUNTRY=nga SAMPLE_SIZE=1000 ./experiments/scripts/run_experiment.sh

# Skip inference if predictions already exist
STRATEGY=zero_shot COUNTRY=cmr SKIP_INFERENCE=true ./experiments/scripts/run_experiment.sh

# Skip counterfactual analysis (faster)
STRATEGY=zero_shot COUNTRY=cmr SKIP_COUNTERFACTUAL=true ./experiments/scripts/run_experiment.sh

# Customize counterfactual analysis
STRATEGY=zero_shot COUNTRY=cmr CF_MODELS="llama3.2,qwen3:8b" CF_EVENTS=100 ./experiments/scripts/run_experiment.sh
```

**Each experiment generates complete quantitative analysis:**
- Classification metrics (P/R/F1, accuracy, confusion matrices)
- Fairness metrics (SPD, Equalized Odds with statistical tests)
- Calibration metrics (Brier scores, reliability diagrams)
- Harm metrics (False Legitimization/Illegitimization rates)
- Error analysis (N≤200 sampled cases per type)
- Source correlations (error rates vs text features)
- Counterfactual robustness (perturbation testing, CFR, CDE)

**Results are organized by strategy:**
```
results/
├── cmr/
│   ├── zero_shot/          # Zero-shot strategy results
│   ├── few_shot/           # Few-shot strategy results  
│   └── explainable/        # Explainable strategy results
└── nga/
    └── ...
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
- Fairness: Statistical Parity Difference (SPD) with 95% bootstrap CI, Equalized Odds (TPR/FPR differences) with permutation tests
- Harm: False Legitimization Rate (FLR), False Illegitimization Rate (FIR)
- Source Analysis: Error correlation with ACLED notes length
- Counterfactual: Flip rates (CFR), Differential effects (CDE) with t-tests/Wilcoxon, Soft-validity metrics (edit distance, fluency)

---

### Alternative: Individual Steps

For fine-grained control, run individual pipeline steps:

```bash
# 1. Classification with specific strategy
STRATEGY=zero_shot COUNTRY=cmr SAMPLE_SIZE=100 .venv/bin/python experiments/pipelines/run_classification.py

# 2. Calibration and evaluation
COUNTRY=cmr RESULTS_DIR=results/cmr/zero_shot .venv/bin/python -m lib.analysis.calibration

# 3. Compute metrics and fairness analysis
COUNTRY=cmr RESULTS_DIR=results/cmr/zero_shot .venv/bin/python -m lib.analysis.metrics

# 4. Harm analysis
COUNTRY=cmr RESULTS_DIR=results/cmr/zero_shot .venv/bin/python -m lib.analysis.harm

# 5. Counterfactual analysis
COUNTRY=cmr RESULTS_DIR=results/cmr/zero_shot .venv/bin/python -m lib.analysis.counterfactual --models llama3.2,mistral:7b --events 50

# 6. Model size comparisons
COUNTRY=cmr .venv/bin/python -m lib.analysis.compare_models --family gemma --sizes 2b,7b --run-missing true
```

## Command Line Usage

### Experiment Runner (Recommended)

```bash
# Run complete experiment with strategy selection
STRATEGY=zero_shot COUNTRY=cmr SAMPLE_SIZE=500 ./experiments/scripts/run_experiment.sh

# Environment variables:
#   STRATEGY             - Prompting strategy (zero_shot, few_shot, explainable) [default: zero_shot]
#   COUNTRY              - Country code (cmr, nga) [default: cmr]
#   SAMPLE_SIZE          - Number of events [default: 500]
#   CF_MODELS            - Models for counterfactual [default: llama3.2,mistral:7b]
#   CF_EVENTS            - Counterfactual event count [default: 50]
#   SKIP_INFERENCE       - Skip phase 1 [default: false]
#   SKIP_COUNTERFACTUAL  - Skip phase 4 [default: false]
```

### Individual Pipeline Components

#### Classification Pipelines
```bash
# Cameroon pipeline with zero-shot strategy
STRATEGY=zero_shot COUNTRY=cmr SAMPLE_SIZE=100 .venv/bin/python experiments/pipelines/run_classification.py

# Nigeria pipeline with few-shot strategy (once implemented)
STRATEGY=few_shot COUNTRY=nga SAMPLE_SIZE=100 .venv/bin/python experiments/pipelines/run_classification.py
```

#### Calibration and Evaluation
```bash
# Apply calibration and generate evaluation metrics/plots
COUNTRY=cmr RESULTS_DIR=results/cmr/zero_shot .venv/bin/python -m lib.analysis.calibration

# Compute per-class thresholds for improved classification
COUNTRY=cmr RESULTS_DIR=results/cmr/zero_shot .venv/bin/python -m lib.analysis.thresholds

# Combined calibrate-then-apply workflow
COUNTRY=cmr ./experiments/scripts/run_calibrate_then_apply.sh
```

### Analysis Tools

#### Model Size Comparisons
```bash
# Compare FL/FI across model sizes within a family
COUNTRY=cmr .venv/bin/python -m lib.analysis.compare_models --family gemma --sizes 2b,7b [OPTIONS]

# Required arguments:
#   --family FAMILY     Model family prefix (e.g., gemma, qwen3)
#   --sizes SIZES       Comma-separated sizes (e.g., 2b,7b or 1.7b,4b,8b)

# Optional arguments:
#   --run-missing {true,false}  Run inference for missing models (default: true)
#   --out OUTPUT_PATH          Custom output CSV path
```

#### Per-Class Analysis and Reporting
```bash
# Generate per-class metrics and disagreement analysis
COUNTRY=cmr RESULTS_DIR=results/cmr/zero_shot .venv/bin/python -m lib.analysis.per_class

# Generate visualizations from analysis reports  
COUNTRY=cmr RESULTS_DIR=results/cmr/zero_shot .venv/bin/python -m lib.analysis.visualize_reports

# Compute FL/FI metrics by model
COUNTRY=cmr RESULTS_DIR=results/cmr/zero_shot .venv/bin/python -m lib.analysis.harm

# Generate confusion matrices and summary metrics
COUNTRY=cmr RESULTS_DIR=results/cmr/zero_shot .venv/bin/python -m lib.analysis.metrics
```

### Utility Tools

#### Ollama Inference Helpers
```bash
# Direct model testing (for debugging)
.venv/bin/python -c "
from lib.inference.ollama_client import run_ollama_structured
import json
result = run_ollama_structured('gemma:2b', 'Military forces beat civilians')
print(json.dumps(result, indent=2))
"
```

### File Requirements

#### Required Files by Tool

**lib.analysis.compare_models:**
- Required: `datasets/<COUNTRY>/state_actor_sample_<COUNTRY>.csv` (sample dataset)
- Optional: `results/<COUNTRY>/<STRATEGY>/ollama_results_calibrated.csv` (existing results; will run inference if missing models)
- Outputs: `results/<COUNTRY>/<STRATEGY>/compare_<family>_sizes.csv`, pairwise comparisons, inference logs

**lib.analysis.calibration:**
- Required: `results/<COUNTRY>/<STRATEGY>/ollama_results_acled_<COUNTRY>_state_actors.csv` (raw predictions)
- Outputs: `results/<COUNTRY>/<STRATEGY>/ollama_results_calibrated.csv`, calibration parameters, reliability plots, Brier scores

**lib.analysis.thresholds:**
- Required: `results/<COUNTRY>/<STRATEGY>/ollama_results_calibrated.csv`
- Outputs: `results/<COUNTRY>/<STRATEGY>/selected_thresholds_per_class.csv`, threshold configurations

**lib.analysis.per_class:**
- Required: `results/<COUNTRY>/<STRATEGY>/ollama_results_calibrated.csv`
- Outputs: `results/<COUNTRY>/<STRATEGY>/per_class_report.csv`, error case samples, top disagreements

**lib.analysis.visualize_reports:**
- Required: Per-class reports and disagreements from previous step
- Outputs: Visualization plots (per-class metrics, disagreement tables)

**lib.analysis.counterfactual:**
- Required: `datasets/<COUNTRY>/state_actor_sample_<COUNTRY>.csv`, calibrated predictions
- Outputs: `results/<COUNTRY>/<STRATEGY>/counterfactual_analysis_<models>.json`, summary tables

**lib.analysis.visualize_counterfactual:**
- Required: Counterfactual analysis JSON from previous step
- Outputs: Multiple visualization plots and analysis report

## Complete Analysis Workflow

### Execution Order

The `run_full_analysis.sh` script automates the following phases:

**Phase 1: Model Inference**

1. `political_bias_of_llms_generic.py` → Generates predictions

**Phase 2: Calibration & Core Metrics**

2. `lib.analysis.calibration` → Calibration + Brier scores
3. `lib.analysis.metrics` → Classification metrics + fairness + error correlations
4. `lib.analysis.thresholds` → Per-class decision thresholds

**Phase 3: Bias & Harm Analysis**

5. `lib.analysis.harm` → False Legitimization/Illegitimization rates
6. `lib.analysis.per_class` → Error case sampling (N≤200 per type)
7. `lib.analysis.visualize_reports` → Generate visualization plots

**Phase 4: Counterfactual Analysis** (Optional)

8. `lib.analysis.counterfactual` → Perturbation testing (CFR, CDE, validity)
9. `lib.analysis.visualize_counterfactual` → Counterfactual visualizations

### Environment Variables

**Supported environment variables:**

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
- `calibration_brier_scores.csv` - Brier scores (raw, isotonic, temperature)
- `reliability_diagrams.png` - Calibration quality plots
- `accuracy_vs_coverage.png` - Selective prediction analysis
- `isotonic_mappings.json` - Isotonic regression mappings

### Classification & Fairness Metrics

- `metrics_acled_<country>_state_actors.csv` - Precision, Recall, F1, Accuracy per model
- `confusion_matrices_acled_<country>_state_actors.json` - Per-model confusion matrices
- `fairness_metrics_acled_<country>_state_actors.csv` - SPD, Equalized Odds, bootstrap CI, permutation tests
- `per_class_report.csv` - Per-class performance metrics
- `selected_thresholds_per_class.csv` - Optimized decision thresholds per class
- `metrics_thresholds_calibrated.csv` - Threshold analysis

### Bias & Harm Analysis

- `fl_fi_by_model.csv` - False Legitimacy/Illegitimacy counts
- `harm_metrics_detailed.csv` - FLR, FIR rates, harm ratios
- `error_cases_false_legitimization.csv` - N≤200 error samples for analysis
- `error_cases_false_illegitimization.csv` - N≤200 error samples for analysis
- `top_disagreements.csv` - High-confidence model disagreements

### Source & Correlation Analysis

- `error_correlations_acled_<country>_state_actors.csv` - Error rate vs notes length, Spearman correlations

### Counterfactual Analysis

- `counterfactual_analysis_<models>.json` - Full counterfactual results with CFR, CDE, soft-validity
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

**Usage:**

```bash
# Run counterfactual analysis to understand model disagreements
COUNTRY=nga RESULTS_DIR=results/nga/zero_shot .venv/bin/python -m lib.analysis.counterfactual --models llama3.2,mistral:7b --events 10

# Generate visualizations for counterfactual results
COUNTRY=nga .venv/bin/python -m lib.analysis.visualize_counterfactual --input results/nga/zero_shot/counterfactual_analysis.json
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

## Model Size Comparisons

The `lib.analysis.compare_models` module enables systematic comparison of false legitimization (FL) and false illegitimization (FI) rates across different model sizes within the same family (e.g., gemma:2b vs gemma:7b). Key features:

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
