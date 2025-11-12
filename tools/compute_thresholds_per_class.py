#!/usr/bin/env python3
"""Compute per-model per-class thresholds from calibrated per-class probabilities.

Writes:
- results/selected_thresholds_per_class.csv
- results/selected_thresholds.json (model -> {label: threshold})
"""
import os
import json
import numpy as np
import pandas as pd
from data_helpers import setup_country_environment

COUNTRY, RESULTS_DIR = setup_country_environment()

CAL_CSV = os.path.join(RESULTS_DIR, 'ollama_results_calibrated.csv')
OUT_CSV = os.path.join(RESULTS_DIR, 'selected_thresholds_per_class.csv')
OUT_JSON = os.path.join(RESULTS_DIR, 'selected_thresholds.json')

labels = ['V','B','E','P','R','S']
candidates = np.concatenate((np.linspace(0,0.9,10), [0.95,0.97,0.99]))

def choose_threshold_for_label(sub, prob_col, label, target_acc=0.8):
    # sub: rows for a single model where true/pred exist
    best = None
    total_valid = int((sub['true_label'].isin(labels)).sum())
    for t in candidates:
        sel = sub[(sub[prob_col] >= t) & (sub['true_label']==label) & (sub['pred_label'].isin(labels))]
        accepted = len(sel)
        if accepted == 0:
            acc = None
        else:
            acc = (sel['pred_label']==sel['true_label']).sum() / accepted
        if acc is None:
            continue
        if acc >= target_acc:
            # prefer higher t (more strict) while meeting target_acc
            best = t if (best is None or t>best) else best
    return best

def main():
    if not os.path.exists(CAL_CSV):
        raise SystemExit(f"Missing {CAL_CSV}")
    df = pd.read_csv(CAL_CSV)
    # require per-class prob columns
    for lab in labels:
        col = f'prob_{lab}'
        if col not in df.columns:
            print(f"Missing per-class column {col}; cannot compute per-class thresholds.")
            return

    rows = []
    selected = {}
    for m in df['model'].unique():
        sub = df[(df['model']==m) & df['true_label'].isin(labels) & df['pred_label'].isin(labels)].copy()
        selected[m] = {}
        for lab in labels:
            col = f'prob_{lab}'
            thr = choose_threshold_for_label(sub, col, lab, target_acc=0.8)
            if thr is None:
                # fallback to a permissive threshold
                thr = 0.0
            rows.append({'model': m, 'label': lab, 'threshold': float(thr)})
            selected[m][lab] = float(thr)

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, 'w') as f:
        json.dump(selected, f, indent=2)
    print('Wrote per-class thresholds to', OUT_CSV, 'and', OUT_JSON)

if __name__ == '__main__':
    main()
