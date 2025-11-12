#!/usr/bin/env python3
"""Compute False Legitimacy (FL) and False Illegitimacy (FI) per model.

Definitions:
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

import os
import pandas as pd
from tools.metrics_helpers import aggregate_fl_fi
from tools.data_helpers import paths_for_country

COUNTRY = os.environ.get('COUNTRY', 'cmr')
paths = paths_for_country(COUNTRY)
os.makedirs(paths['results_dir'], exist_ok=True)

CAL_CSV = paths['calibrated_csv']
OUT_CSV = os.path.join(paths['results_dir'], 'fl_fi_by_model.csv')

def main():
    if not os.path.exists(CAL_CSV):
        raise SystemExit(f"Missing {CAL_CSV}")
    df = pd.read_csv(CAL_CSV)
    out = aggregate_fl_fi(df, by='model')
    out.to_csv(OUT_CSV, index=False)
    print('Wrote', OUT_CSV)
    print(out.to_string(index=False))

if __name__ == '__main__':
    main()
