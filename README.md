# Evaluating State Actor Bias

This repository contains code and data for evaluating potential bias in LLM classification of ACLED event types when the primary actor is a state actor.

Two country flows are available; the current active pipeline includes Cameroon.

## What this project does
- Extracts country-specific rows from ACLED-like datasets, normalizes actor text and selects usable rows (has notes and known event types).
- Builds stratified samples (configurable size) with a primary-group oversample for Violence against civilians.
- Runs classification with locally-hosted models (via Ollama) and saves structured predictions.
- Calibrates model confidences (isotonic + temperature scaling), evaluates thresholding strategies, and produces reliability and accuracy-vs-coverage plots.

## Files of interest
- `political_bias_of_llms_cmr.py` — Cameroon pipeline: builds stratified sample (env `SAMPLE_SIZE`), runs classifiers, and writes predictions to `results/`.
- `tools/calibrate_confidences.py` — fits per-model calibration on labeled predictions and writes calibration params JSON.
- `tools/apply_calibration_and_evaluate.py` — applies calibrators to predictions, writes calibrated CSV, threshold metrics CSV and plots (reliability, accuracy-vs-coverage).
- `tools/compute_metrics_cmr.py` — computes per-model confusion matrices and summary metrics.
- `scripts/run_calibrate_then_apply.sh` — driver script: runs small-sample calibration, selects per-model thresholds, runs large-sample classification and applies thresholds.

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

- Classify a sample (default SAMPLE_SIZE=100):

```bash
.venv/bin/python political_bias_of_llms_cmr.py
```

- Run the calibrate-then-apply driver (small sample -> large sample):

```bash
./scripts/run_calibrate_then_apply.sh

# override sizes
SMALL_SAMPLE=10 LARGE_SAMPLE=100 ./scripts/run_calibrate_then_apply.sh
```

## Outputs
- Predictions, calibrated CSV, calibration params, threshold metrics and plots are written under `results/`.
- Key files: `results/ollama_results_acled_cameroon_state_actors.csv`, `results/ollama_results_calibrated.csv`, `results/calibration_params_acled_cameroon_state_actors.json`, `results/metrics_thresholds_calibrated.csv`, `results/reliability_diagrams.png`, `results/accuracy_vs_coverage.png`.

## Reporting and visualization

- `tools/per_class_and_disagreements.py` — generate two audit CSVs from calibrated predictions (`results/ollama_results_calibrated.csv`):
	- `results/per_class_report.csv` — per-model, per-class precision/recall/f1/support
	- `results/top_disagreements.csv` — top-N event rows where model predictions disagree (sorted by max calibrated probability)

- `tools/visualize_reports.py` — create visual artifacts from the above CSVs:
	- `results/per_class_metrics.png` — bar chart of per-class F1 by model
	- `results/top_disagreements_table.png` — rendered table of the top disagreement rows

Usage examples:

```bash
.venv/bin/python tools/per_class_and_disagreements.py   # creates the CSV reports
.venv/bin/python tools/visualize_reports.py            # creates PNG visualizations
```

Notes:
- The scripts expect `results/ollama_results_calibrated.csv` to exist and to include the calibrated probability column `pred_conf_temp`.

## Notes
- Ollama daemon must be running locally and the required models pulled to run classification.
- Small-sample thresholds are noisy; use larger calibration sets or bootstrapping for robust thresholds.
