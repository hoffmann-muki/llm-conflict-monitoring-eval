#!/usr/bin/env python3
"""Aggregate SPD values across fairness CSVs and plot with CI error bars.

Usage:
  PYTHONPATH=. python scripts/aggregate_spd_and_plot.py --countries cmr nga --strategy zero_shot

Outputs per country/strategy (under results/{country}/{strategy}/):
  - spd_aggregate.csv            : compact CSV with model, SPD, CI, sample_size, path
  - spd_plot.png                 : bar plot of SPD with CI error bars
  - spd_significant_models.csv   : small table of models with SPD CI not crossing zero

"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os


def find_fairness_files(country: str, strategy: str) -> list:
    # pattern: results/{country}/{strategy}/**/fairness_metrics_acled_{country}_actors.csv
    pattern = f"results/{country}/{strategy}/**/fairness_metrics_acled_{country}_actors.csv"
    files = glob.glob(pattern, recursive=True)
    # Also check direct path without nested model folders
    direct = Path(f"results/{country}/{strategy}/fairness_metrics_acled_{country}_actors.csv")
    if direct.exists():
        files.append(str(direct))
    return sorted(set(files))


def aggregate_files(files: list, country: str, strategy: str) -> pd.DataFrame:
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            print(f"Warning: failed to read {f}")
            continue
        # attempt to infer sample_size from path components
        parts = Path(f).parts
        sample_size = None
        for p in parts:
            if p.isdigit():
                sample_size = int(p)
                break
        for _, r in df.iterrows():
            rows.append({
                'country': country,
                'strategy': strategy,
                'sample_size': sample_size,
                'file': f,
                'model': r.get('model'),
                'target_label': r.get('target_label'),
                'n_state': r.get('n_state'),
                'n_nonstate': r.get('n_nonstate'),
                'SPD': r.get('SPD'),
                'SPD_CI_lower': r.get('SPD_CI_lower'),
                'SPD_CI_upper': r.get('SPD_CI_upper'),
                'TPR_pvalue': r.get('TPR_pvalue'),
                'FPR_pvalue': r.get('FPR_pvalue'),
            })
    return pd.DataFrame(rows)


def write_outputs(df: pd.DataFrame, country: str, strategy: str):
    out_root = Path('results') / country / strategy
    out_root.mkdir(parents=True, exist_ok=True)
    agg_path = out_root / 'spd_aggregate.csv'
    df.to_csv(agg_path, index=False)
    print(f'Wrote aggregate CSV: {agg_path}')

    # Focus on target_label == 'V' (Violence against civilians) if present
    df_v = df[df['target_label'] == 'V'] if 'target_label' in df.columns else df.copy()

    if df_v.empty:
        print('No SPD rows for target_label V; skipping plot')
        return

    # Group by model and average across sample_size/files if duplicates
    plot_df = df_v.groupby('model').agg(
        SPD_mean=('SPD', 'mean'),
        SPD_ci_low=('SPD_CI_lower', 'mean'),
        SPD_ci_high=('SPD_CI_upper', 'mean'),
        n_state=('n_state', 'mean')
    ).reset_index()

    # Determine error bars (mean SPD - lower/upper)
    plot_df['err_low'] = plot_df['SPD_mean'] - plot_df['SPD_ci_low']
    plot_df['err_high'] = plot_df['SPD_ci_high'] - plot_df['SPD_mean']

    # Sort models for plot by SPD
    plot_df = plot_df.sort_values('SPD_mean')

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(plot_df))
    ax.bar(x, plot_df['SPD_mean'], yerr=[plot_df['err_low'], plot_df['err_high']], capsize=6)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['model'], rotation=45, ha='right')
    ax.set_ylabel('Statistical Parity Difference (SPD)')
    ax.set_title(f'SPD by model — {country} / {strategy}')
    plt.tight_layout()
    out_plot = out_root / 'spd_plot.png'
    fig.savefig(out_plot)
    plt.close(fig)
    print(f'Wrote SPD plot: {out_plot}')

    # Significant models: CI not crossing zero
    sig_mask = (plot_df['SPD_ci_low'] > 0) | (plot_df['SPD_ci_high'] < 0)
    sig_df = plot_df.loc[sig_mask, ['model', 'SPD_mean', 'SPD_ci_low', 'SPD_ci_high']].copy()
    sig_path = out_root / 'spd_significant_models.csv'
    sig_df.to_csv(sig_path, index=False)
    print(f'Wrote significant models table: {sig_path} ({len(sig_df)} rows)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--countries', nargs='+', required=True)
    parser.add_argument('--strategy', required=True)
    args = parser.parse_args()

    for c in args.countries:
        files = find_fairness_files(c, args.strategy)
        if not files:
            print(f'No fairness files found for {c}/{args.strategy}')
            continue
        agg = aggregate_files(files, c, args.strategy)
        if agg.empty:
            print(f'No aggregated rows for {c}/{args.strategy}')
            continue
        write_outputs(agg, c, args.strategy)


if __name__ == '__main__':
    main()
