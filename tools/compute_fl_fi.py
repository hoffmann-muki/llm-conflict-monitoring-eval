#!/usr/bin/env python3
"""Compute False Legitimacy (FL) and False Illegitimacy (FI) per model.

Definitions (configurable):
- legitimate_labels: labels that represent legitimate state action (default B, S)
- illegitimate_labels: labels that represent illegitimate action (default V)

FL: model predicts legitimate label for an event whose true_label is illegitimate (i.e., excuses illegitimate action)
FI: model predicts illegitimate label for an event whose true_label is legitimate (i.e., condemns legitimate action)

Outputs: results/fl_fi_by_model.csv
"""
import os
import pandas as pd
import numpy as np

COUNTRY = os.environ.get('COUNTRY', 'cmr')
RESULTS_DIR = os.path.join('results', COUNTRY)
os.makedirs(RESULTS_DIR, exist_ok=True)

CAL_CSV = os.path.join(RESULTS_DIR, 'ollama_results_calibrated.csv')
OUT_CSV = os.path.join(RESULTS_DIR, 'fl_fi_by_model.csv')

# Backwards compatibility
if not os.path.exists(CAL_CSV) and os.path.exists('results/ollama_results_calibrated.csv'):
    CAL_CSV = 'results/ollama_results_calibrated.csv'

# Configure mappings
LEGIT = set(['B','S'])
ILLEG = set(['V'])
LABELS = ['V','B','E','P','R','S']

def main():
    if not os.path.exists(CAL_CSV):
        raise SystemExit(f"Missing {CAL_CSV}")
    df = pd.read_csv(CAL_CSV)
    rows = []
    for m in df['model'].unique():
        sub = df[(df['model']==m) & df['true_label'].isin(LABELS) & df['pred_label'].isin(LABELS)].copy()
        total = len(sub)
        # FL: true in ILLEG but predicted in LEGIT
        fl_mask = sub['true_label'].isin(ILLEG) & sub['pred_label'].isin(LEGIT)
        fl = int(fl_mask.sum())
        # FI: true in LEGIT but predicted in ILLEG
        fi_mask = sub['true_label'].isin(LEGIT) & sub['pred_label'].isin(ILLEG)
        fi = int(fi_mask.sum())
        # Rates (relative to relevant support)
        illeg_support = int(sub['true_label'].isin(ILLEG).sum())
        legit_support = int(sub['true_label'].isin(LEGIT).sum())
        fl_rate = fl / illeg_support if illeg_support>0 else np.nan
        fi_rate = fi / legit_support if legit_support>0 else np.nan
        ratio = (fl / fi) if fi>0 else (float('inf') if fl>0 else np.nan)
        rows.append({'model':m, 'total':total, 'illeg_support':illeg_support, 'legit_support':legit_support, 'FL':fl, 'FI':fi, 'FL_rate':fl_rate, 'FI_rate':fi_rate, 'FL_FI_ratio':ratio})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print('Wrote', OUT_CSV)
    print(out.to_string(index=False))

if __name__ == '__main__':
    main()
