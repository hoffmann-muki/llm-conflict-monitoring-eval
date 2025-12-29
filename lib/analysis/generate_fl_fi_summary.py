#!/usr/bin/env python3
"""
Consolidated FL/FI (False Legitimation/Illegitimation) analysis generator.
Works for both Cameroon and Nigeria data.

FL: V → B (illegitimate action is excused)
FI: B → V (legitimate action is condemned)

Usage:
    python generate_fl_fi_summary.py --country cmr
    python generate_fl_fi_summary.py --country nga
"""

import pandas as pd
import math
import argparse
from pathlib import Path
from typing import Tuple


def wilson_ci(count: int, total: int) -> Tuple[float, float]:
    """Calculate Wilson score 95% confidence interval."""
    if total == 0:
        return (0.0, 0.0)

    p = count / total
    z = 1.96  # 95% CI

    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom

    lower = max(0, center - margin) * 100
    upper = min(1, center + margin) * 100

    return (lower, upper)


def binomial_test_pvalue(k: int, n: int, p: float = 0.5) -> float:
    """
    Simplified binomial test p-value calculation.
    Tests H0: probability = p (default 0.5 for FL vs FI).
    """
    if n == 0:
        return 1.0

    # Normal approximation (good for n > 20)
    if n > 20:
        mean = n * p
        std = math.sqrt(n * p * (1 - p))

        if std == 0:
            return 1.0 if k == mean else 0.0

        z = (k - mean) / std
        abs_z = abs(z)

        # Conservative p-value approximation
        if abs_z > 3:
            p_value = 0.001
        elif abs_z > 2.576:
            p_value = 0.01
        elif abs_z > 1.96:
            p_value = 0.05
        elif abs_z > 1:
            p_value = 0.2
        else:
            p_value = 1.0

        return p_value

    # For small n, conservative estimate
    return 1.0 if abs(k - n * p) < 2 else 0.05


def analyze_model_condition(results_file: Path) -> dict:
    """Analyze FL/FI for one model/condition."""

    df = pd.read_csv(results_file)

    # Total analysis
    total_v = len(df[df['true_label'] == 'V'])
    total_b = len(df[df['true_label'] == 'B'])

    total_fl = len(df[(df['true_label'] == 'V') & (df['pred_label'] == 'B')])
    total_fi = len(df[(df['true_label'] == 'B') & (df['pred_label'] == 'V')])

    total_fl_pct = (total_fl / total_v * 100) if total_v > 0 else 0
    total_fi_pct = (total_fi / total_b * 100) if total_b > 0 else 0

    fl_ci = wilson_ci(total_fl, total_v)
    fi_ci = wilson_ci(total_fi, total_b)

    # p-value from binomial test
    p_value = binomial_test_pvalue(total_fl, total_fl + total_fi, 0.5)

    delta_lb = total_fi_pct - total_fl_pct

    return {
        'true_v': total_v,
        'true_b': total_b,
        'fl_count': total_fl,
        'fl_pct': total_fl_pct,
        'fl_ci': fl_ci,
        'fi_count': total_fi,
        'fi_pct': total_fi_pct,
        'fi_ci': fi_ci,
        'delta_lb': delta_lb,
        'p_value': p_value
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate FL/FI summary for country (all conditions)'
    )
    parser.add_argument(
        '--country',
        type=str,
        required=True,
        choices=['cmr', 'nga'],
        help='Country code (cmr or nga)'
    )

    args = parser.parse_args()

    base_dir = Path('results') / args.country
    output_dir = Path('results/analysis/fl_fi') / args.country
    output_dir.mkdir(parents=True, exist_ok=True)

    # Conditions to analyze
    conditions = [
        ('zero_shot', None),
        ('few_shot', 1),
        ('few_shot', 3),
        ('few_shot', 5)
    ]

    models = ['llama3.2_3b', 'mistral_7b', 'olmo2_7b', 'gemma3_4b']

    all_results = []

    for strategy, shots in conditions:
        condition_name = strategy if shots is None else f'{strategy}_{shots}'

        if shots is None:
            strategy_dir = base_dir / strategy / '1000'
        else:
            strategy_dir = base_dir / strategy / '1000' / str(shots)

        print(f'\nAnalyzing {condition_name}...')

        if not strategy_dir.exists():
            print(f'  Directory not found, skipping')
            continue

        for model in models:
            model_dir = strategy_dir / model
            results_files = list(model_dir.glob(f'ollama_results_*_acled_{args.country}_actors.csv'))

            if not results_files:
                print(f'  {model}: no results file')
                continue

            results_file = results_files[0]
            print(f'  {model}: analyzing...')

            result = analyze_model_condition(results_file)

            all_results.append({
                'Model': model,
                'Condition': condition_name,
                'n_FL': result['fl_count'],
                'ε_FL': result['fl_pct'],
                'FL_CI_lower': result['fl_ci'][0],
                'FL_CI_upper': result['fl_ci'][1],
                'n_FI': result['fi_count'],
                'ε_FI': result['fi_pct'],
                'FI_CI_lower': result['fi_ci'][0],
                'FI_CI_upper': result['fi_ci'][1],
                'Δ_LB': result['delta_lb'],
                'p_value': result['p_value'],
                'True_V': result['true_v'],
                'True_B': result['true_b']
            })

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Sort by model then condition
    condition_order = {'zero_shot': 0, 'few_shot_1': 1, 'few_shot_3': 2, 'few_shot_5': 3}
    df['sort_key'] = df['Condition'].map(condition_order)
    df = df.sort_values(['Model', 'sort_key']).drop('sort_key', axis=1)

    # Save
    output_file = output_dir / 'fl_fi_summary_all_conditions.csv'
    df.to_csv(output_file, index=False)

    print(f'\n✓ Saved to: {output_file}')
    print(f'\nGenerated {len(df)} rows for {args.country.upper()}')


if __name__ == '__main__':
    main()
