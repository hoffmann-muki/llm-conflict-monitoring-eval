#!/usr/bin/env bash
set -euo pipefail

# Run a small-sample calibration, pick per-model thresholds, then run on a larger sample
# and apply those thresholds to the larger sample's calibrated probabilities.

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
echo "Small sample=$SMALL_SAMPLE Large sample=$LARGE_SAMPLE MIN_COVERAGE=$MIN_COVERAGE"

cd "$REPO_ROOT"

echo "--- Running small sample (SAMPLE_SIZE=$SMALL_SAMPLE) ---"
SAMPLE_SIZE=$SMALL_SAMPLE CMR_POSTPROCESS=1 "$VENV_PY" political_bias_of_llms_cmr.py

BASE_METRICS="results/metrics_thresholds_calibrated.csv"
# country-specific metrics filename to avoid clashes
METRICS_FILE="results/metrics_thresholds_calibrated_cmr.csv"
THRESH_JSON="results/selected_thresholds_cmr.json"

if [ ! -f "$METRICS_FILE" ]; then
  echo "Expected metrics file not found: $METRICS_FILE"
  exit 1
fi

echo "--- Selecting thresholds from small-sample metrics ---"
"$VENV_PY" - <<PY > /dev/null
import pandas as pd, json
df = pd.read_csv('$METRICS_FILE')
# prefer isotonic calibrated column if present
for pref in ['pred_conf_iso','pred_conf_temp','pred_conf']:
    if pref in df['prob_col'].unique():
        pref_col = pref
        break
    pref_col = 'pred_conf'

out = {}
for m in df['model'].unique():
    sub = df[(df['model']==m) & (df['prob_col']==pref_col)].copy()
    if sub.empty:
        continue
    # choose threshold with coverage >= MIN_COVERAGE and max accuracy
    cand = sub[sub['coverage'] >= float($MIN_COVERAGE)]
    if cand.empty:
        best = sub.sort_values(['accuracy','coverage'], ascending=[False,False]).iloc[0]
    else:
        best = cand.sort_values(['accuracy','coverage'], ascending=[False,False]).iloc[0]
    out[m] = {'prob_col': pref_col, 'threshold': float(best['threshold']), 'coverage': float(best['coverage']), 'accuracy': None if pd.isna(best['accuracy']) else float(best['accuracy'])}

with open('$THRESH_JSON','w') as f:
    json.dump(out, f, indent=2)
print('wrote', '$THRESH_JSON')
PY

echo "Selected thresholds written to $THRESH_JSON"

echo "--- Running large sample (SAMPLE_SIZE=$LARGE_SAMPLE) ---"
SAMPLE_SIZE=$LARGE_SAMPLE CMR_POSTPROCESS=1 "$VENV_PY" political_bias_of_llms_cmr.py

# calibrated CSV that the apply step reads; prefer country-specific filename
BASE_CAL="results/ollama_results_calibrated.csv"
CALIBRATED_CSV="results/ollama_results_calibrated_cmr.csv"
APPLY_OUT_CSV="results/threshold_application_on_large_cmr.csv"

if [ -f "$BASE_CAL" ] && [ ! -f "$CALIBRATED_CSV" ]; then
    mv "$BASE_CAL" "$CALIBRATED_CSV"
    echo "Moved $BASE_CAL -> $CALIBRATED_CSV"
fi

if [ ! -f "$CALIBRATED_CSV" ]; then
    echo "Calibrated results not found: $CALIBRATED_CSV"
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

out = pd.DataFrame(rows)
out.to_csv('$APPLY_OUT_CSV', index=False)
print('Wrote', '$APPLY_OUT_CSV')
print(out.to_string(index=False))
PY

echo "Done."
