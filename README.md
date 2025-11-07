# Evaluating State Actor Bias

This repository contains code and data for evaluating potential bias in LLM classification of ACLED event types when the primary actor is a state actor.

Two country flows are available (Cameroon, Nigeria). The repository namespaces dataset and result artifacts by country under `datasets/<country>/` and `results/<country>/`.

## What this project does
- Extracts country-specific rows from ACLED-like datasets, normalizes actor text and selects usable rows (has notes and known event types).
- Builds stratified samples (configurable size) with a primary-group oversample for Violence against civilians.
- Runs classification with locally-hosted models (via Ollama) and saves structured predictions.
- Calibrates model confidences (isotonic regression + temperature scaling), evaluates thresholding strategies, and produces reliability and accuracy-vs-coverage plots.

## Files of interest
- `political_bias_of_llms_cmr.py` — Cameroon pipeline: builds stratified sample (env `SAMPLE_SIZE`), runs classifiers, and writes predictions to `results/cmr/`.
- `political_bias_of_llms_nga.py` — Nigeria pipeline: same as above but writes to `results/nga/`.
- `tools/calibrate_confidences.py` — fits per-model calibration on labeled predictions and writes calibration params JSON under `results/<COUNTRY>/`.
- `tools/apply_calibration_and_evaluate.py` — applies calibrators to predictions and writes calibrated CSV, threshold metrics and plots under `results/<COUNTRY>/`.
- `tools/compute_metrics_cmr.py` — computes per-model confusion matrices and summary metrics (country-scoped).
- `scripts/run_calibrate_then_apply_cmr.sh` and `scripts/run_calibrate_then_apply_nga.sh` — driver scripts for Cameroon and Nigeria respectively.

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
# Run Cameroon sample
COUNTRY=cmr SAMPLE_SIZE=100 .venv/bin/python political_bias_of_llms_cmr.py
# Run Nigeria sample
COUNTRY=nga SAMPLE_SIZE=100 .venv/bin/python political_bias_of_llms_nga.py
```

- Run the calibrate-then-apply driver (small sample -> large sample):

```bash
# Run country-specific driver (examples)
COUNTRY=cmr ./scripts/run_calibrate_then_apply_cmr.sh
COUNTRY=nga ./scripts/run_calibrate_then_apply_nga.sh

# override sizes
COUNTRY=cmr SMALL_SAMPLE=10 LARGE_SAMPLE=100 ./scripts/run_calibrate_then_apply_cmr.sh
```

## Outputs
- Predictions, calibrated CSV, calibration params, threshold metrics and plots are written under `results/<COUNTRY>/` (e.g. `results/cmr/` or `results/nga/`).
- Key files (example for Cameroon): `results/cmr/ollama_results_acled_cmr_state_actors.csv`, `results/cmr/ollama_results_calibrated.csv`, `results/cmr/calibration_params_acled_cmr_state_actors.json`, `results/cmr/metrics_thresholds_calibrated.csv`, `results/cmr/reliability_diagrams.png`, `results/cmr/accuracy_vs_coverage.png`.

## Reporting and visualization

- `tools/per_class_and_disagreements.py` — generate two audit CSVs from calibrated predictions (`results/ollama_results_calibrated.csv`):
	- `results/per_class_report.csv` — per-model, per-class precision/recall/f1/support
	- `results/top_disagreements.csv` — top-N event rows where model predictions disagree (sorted by max calibrated probability)

- `tools/visualize_reports.py` — create visual artifacts from the above CSVs:
	- `results/per_class_metrics.png` — bar chart of per-class F1 by model
	- `results/top_disagreements_table.png` — rendered table of the top disagreement rows

Usage examples:

```bash
# run for a specific country folder (set COUNTRY=cmr or COUNTRY=nga)
COUNTRY=cmr .venv/bin/python tools/per_class_and_disagreements.py   # creates CSV reports under results/cmr/
COUNTRY=cmr .venv/bin/python tools/visualize_reports.py            # creates PNG visualizations under results/cmr/
```

Notes:
- The scripts expect a country-scoped calibrated CSV, e.g. `results/cmr/ollama_results_calibrated.csv`, to exist and to include the calibrated probability column `pred_conf_temp`.

## Notes
- Ollama daemon must be running locally and the required models pulled to run classification.
- Small-sample thresholds are noisy; use larger calibration sets or bootstrapping for robust thresholds.
