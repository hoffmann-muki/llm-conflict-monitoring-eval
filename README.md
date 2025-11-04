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

## Notes
- Ollama daemon must be running locally and the required models pulled to run classification.
- Small-sample thresholds are noisy; use larger calibration sets or bootstrapping for robust thresholds.
