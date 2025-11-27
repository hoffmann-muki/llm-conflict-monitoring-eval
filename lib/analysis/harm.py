#!/usr/bin/env python3
"""Compute False Legitimization and False Illegitimization rates per model.

Harm-Aware Metrics:
- False Legitimization Rate (FLR): Proportion of events labeled as illegitimate by humans 
  but predicted as legitimate by model (model excuses illegitimate state violence)
  
- False Illegitimization Rate (FIR): Proportion of events labeled as legitimate by humans
  but predicted as illegitimate by model (model condemns legitimate state action)

Definitions:
- legitimate_labels: B (Battles), S (Strategic developments) - state acting within normal conflict dynamics
- illegitimate_labels: V (Violence against civilians) - violations that should be condemned
- Other labels (E, P, R) are neutral/ambiguous

This aligns with bias-in-conflict-data literature on societal harm proxies.
"""
import os
import pandas as pd
import numpy as np
from lib.core.metrics_helpers import aggregate_fl_fi, LEGIT, ILLEG
from lib.core.data_helpers import paths_for_country
from lib.core.result_aggregator import model_name_to_slug

COUNTRY = os.environ.get('COUNTRY', 'cmr')
STRATEGY = os.environ.get('STRATEGY', 'zero_shot')
paths = paths_for_country(COUNTRY, STRATEGY)
os.makedirs(paths['results_dir'], exist_ok=True)

CAL_CSV = paths['calibrated_csv']
OUT_CSV = os.path.join(paths['results_dir'], 'fl_fi_by_model.csv')
OUT_DETAILED_CSV = os.path.join(paths['results_dir'], 'harm_metrics_detailed.csv')


def get_per_model_output_path(base_path: str, model_name: str) -> str:
    """Generate per-model output path from base path."""
    slug = model_name_to_slug(model_name)
    base, ext = os.path.splitext(base_path)
    return f"{base}_{slug}{ext}"

def compute_harm_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute detailed false legitimization and false illegitimization rates."""
    results = []
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model].copy()
        
        # Filter to events with clear legitimate/illegitimate ground truth
        illegit_events = model_df[model_df['true_label'].isin(ILLEG)]
        legit_events = model_df[model_df['true_label'].isin(LEGIT)]
        
        # False Legitimization: true=illegit (V), pred=legit (B,S)
        false_legit = illegit_events[illegit_events['pred_label'].isin(LEGIT)]
        n_illegit = len(illegit_events)
        flr = len(false_legit) / n_illegit if n_illegit > 0 else 0
        
        # False Illegitimization: true=legit (B,S), pred=illegit (V)
        false_illegit = legit_events[legit_events['pred_label'].isin(ILLEG)]
        n_legit = len(legit_events)
        fir = len(false_illegit) / n_legit if n_legit > 0 else 0
        
        # Compute for traditional FL/FI counts
        fl_count = len(false_legit)
        fi_count = len(false_illegit)
        
        results.append({
            'model': model,
            'false_legitimization_rate_FLR': flr,
            'false_legitimization_count_FL': fl_count,
            'n_illegit_events': n_illegit,
            'false_illegitimization_rate_FIR': fir,
            'false_illegitimization_count_FI': fi_count,
            'n_legit_events': n_legit,
            'total_events': len(model_df),
            'harm_ratio_FL_to_FI': fl_count / fi_count if fi_count > 0 else np.inf
        })
    
    return pd.DataFrame(results)

def main():
    if not os.path.exists(CAL_CSV):
        raise SystemExit(f"Missing {CAL_CSV}")
    
    df = pd.read_csv(CAL_CSV)
    
    # Compute traditional FL/FI aggregates
    out = aggregate_fl_fi(df, by='model')
    out.to_csv(OUT_CSV, index=False)
    print('Wrote traditional FL/FI metrics to', OUT_CSV)
    print(out.to_string(index=False))
    
    # Save per-model FL/FI metrics
    for m in out['model'].unique():
        model_flfi = out[out['model'] == m]
        model_path = get_per_model_output_path(OUT_CSV, m)
        model_flfi.to_csv(model_path, index=False)
        print(f'Saved per-model FL/FI metrics: {model_path}')
    
    # Compute detailed harm-aware rates
    harm_df = compute_harm_rates(df)
    harm_df.to_csv(OUT_DETAILED_CSV, index=False)
    print(f'\nWrote detailed harm metrics to {OUT_DETAILED_CSV}')
    
    # Save per-model detailed harm metrics
    for m in harm_df['model'].unique():
        model_harm = harm_df[harm_df['model'] == m]
        model_path = get_per_model_output_path(OUT_DETAILED_CSV, m)
        model_harm.to_csv(model_path, index=False)
        print(f'Saved per-model harm metrics: {model_path}')
    
    print('\n=== Harm-Aware Metrics (False Legitimization & Illegitimization Rates) ===')
    print(harm_df[['model', 'false_legitimization_rate_FLR', 'false_illegitimization_rate_FIR', 'harm_ratio_FL_to_FI']].to_string(index=False))

if __name__ == '__main__':
    main()
