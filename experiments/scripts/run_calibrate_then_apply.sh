#!/usr/bin/env bash
set -euo pipefail

# Generic calibration script that works for any country
# Usage: ./run_calibrate_then_apply.sh [COUNTRY]
# COUNTRY defaults to cmr if not specified

COUNTRY=${1:-${COUNTRY:-cmr}}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "Warning: project venv python not found at $VENV_PY. Falling back to system python3."
  VENV_PY="$(which python3)"
fi

# Config
SMALL_SAMPLE=${SMALL_SAMPLE:-20}
LARGE_SAMPLE=${LARGE_SAMPLE:-50}
# minimum coverage required when choosing a threshold from the small sample (0..1)
MIN_COVERAGE=${MIN_COVERAGE:-0.5}

echo "Using python: $VENV_PY"
echo "Country: $COUNTRY"
echo "Small sample=$SMALL_SAMPLE Large sample=$LARGE_SAMPLE MIN_COVERAGE=$MIN_COVERAGE"

cd "$REPO_ROOT"

echo "--- Running small sample (SAMPLE_SIZE=$SMALL_SAMPLE) ---"
# Run the classification pipeline
STRATEGY="zero_shot" SAMPLE_SIZE=$SMALL_SAMPLE COUNTRY=$COUNTRY \
    "$VENV_PY" experiments/pipelines/run_classification.py

# Run calibration and evaluation
COUNTRY=$COUNTRY "$VENV_PY" -m lib.analysis.calibration
COUNTRY=$COUNTRY "$VENV_PY" -m lib.analysis.thresholds

# Country-specific file paths
METRICS_FILE="results/${COUNTRY}/metrics_thresholds_calibrated.csv"
THRESH_JSON="results/selected_thresholds_${COUNTRY}.json"

if [ ! -f "$METRICS_FILE" ]; then
  echo "Expected metrics file not found: $METRICS_FILE"
  exit 1
fi

echo "--- Selecting thresholds from small-sample metrics ---"
"$VENV_PY" - <<PY > /dev/null
import pandas as pd, json
df = pd.read_csv('$METRICS_FILE')
# Choose the threshold with best accuracy>=MIN_COVERAGE per model
selected = {}
for m in df['model'].unique():
    sub = df[df['model']==m]
    # filter to those with coverage >= MIN_COVERAGE
    candidates = sub[sub['coverage'] >= $MIN_COVERAGE]
    if len(candidates) == 0:
        print(f"Warning: no thresholds for {m} meet MIN_COVERAGE={$MIN_COVERAGE}; using most permissive")
        candidates = sub
    # pick the one with highest accuracy
    best_row = candidates.loc[candidates['accuracy'].idxmax()]
    selected[m] = float(best_row['threshold'])
    print(f"{m}: threshold={best_row['threshold']:.3f} acc={best_row['accuracy']:.3f} cov={best_row['coverage']:.3f}")

with open('$THRESH_JSON', 'w') as f:
    json.dump(selected, f, indent=2)
print("Wrote thresholds to $THRESH_JSON")
PY

echo "--- Running large sample (SAMPLE_SIZE=$LARGE_SAMPLE) ---"
# Run the large sample with the classification pipeline
STRATEGY="zero_shot" SAMPLE_SIZE=$LARGE_SAMPLE COUNTRY=$COUNTRY \
    "$VENV_PY" experiments/pipelines/run_classification.py

# Run calibration and evaluation
COUNTRY=$COUNTRY "$VENV_PY" -m lib.analysis.calibration

CALIBRATED_CSV="results/${COUNTRY}/ollama_results_calibrated.csv"

if [ ! -f "$CALIBRATED_CSV" ]; then
    echo "Expected calibrated CSV not found: $CALIBRATED_CSV"
    exit 1
fi

echo "--- Applying thresholds to large calibrated results and reporting ---"
"$VENV_PY" - <<PY
import pandas as pd, json
df = pd.read_csv('$CALIBRATED_CSV')
th = json.load(open('$THRESH_JSON'))
labels = ['V','B','E','P','R','S']
rows = []
for m, info in th.items():
    sub = df[df['model']==m].copy()
    valid = sub[sub['true_label'].isin(labels) & sub['pred_label'].isin(labels)]
    total = len(valid)
    accepted = pd.DataFrame()
    # detect per-class thresholds: mapping of label->threshold
    if isinstance(info, dict) and all(lab in info for lab in labels):
        parts = []
        for lab in labels:
            t = float(info.get(lab, 0.0))
            col = f'prob_{lab}'
            if col not in sub.columns:
                continue
            sel = sub[(sub['pred_label']==lab) & sub['true_label'].isin(labels)].copy()
            sel['prob_for_pred'] = pd.to_numeric(sel[col], errors='coerce').fillna(0.0)
            sel_acc = sel[sel['prob_for_pred'] >= t]
            parts.append(sel_acc)
        if len(parts) > 0:
            accepted = pd.concat(parts, ignore_index=True)
        else:
            accepted = sub.iloc[0:0]
        thr_repr = 'per-class'
    else:
        # legacy single threshold format
        prob_col = info.get('prob_col', 'pred_conf') if isinstance(info, dict) else 'pred_conf'
        t = float(info.get('threshold', 0.0)) if isinstance(info, dict) else float(info)
        sub['prob'] = pd.to_numeric(sub.get(prob_col, sub.get('pred_conf')), errors='coerce').fillna(0.0)
        accepted = sub[sub['prob'] >= t]
        thr_repr = t

    acc = None
    if len(accepted) > 0:
        acc = (accepted['pred_label']==accepted['true_label']).mean()
    rows.append({'model':m, 'threshold':thr_repr, 'accepted': len(accepted), 'coverage': (len(accepted)/total if total>0 else 0), 'accuracy': (None if acc is None else float(acc)), 'total_valid': total})

final_df = pd.DataFrame(rows)
out_path = 'results/${COUNTRY}/final_threshold_performance.csv'
final_df.to_csv(out_path, index=False)
print(f"Final threshold performance (saved to {out_path}):")
print(final_df.to_string(index=False))
PY

echo "--- Generating reports ---"
COUNTRY=$COUNTRY "$VENV_PY" -m lib.analysis.per_class_metrics
COUNTRY=$COUNTRY "$VENV_PY" -m lib.analysis.visualize_reports

echo "Done! Check results/$COUNTRY/ for outputs."
