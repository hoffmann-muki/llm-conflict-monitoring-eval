#!/usr/bin/env python3
"""
Generate per-actor FL/FI analysis (Police, Military, Total) for both countries.

Usage:
    python generate_per_actor_fl_fi.py --country cmr --condition zero_shot
    python generate_per_actor_fl_fi.py --country nga --condition zero_shot
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


def analyze_actor(df: pd.DataFrame, actor_filter: str = None) -> dict:
    """Analyze FL/FI for a specific actor or all actors."""

    # Filter by actor if specified
    if actor_filter:
        df = df[df['actor_norm'].str.contains(actor_filter, case=False, na=False)]

    total_v = len(df[df['true_label'] == 'V'])
    total_b = len(df[df['true_label'] == 'B'])

    total_fl = len(df[(df['true_label'] == 'V') & (df['pred_label'] == 'B')])
    total_fi = len(df[(df['true_label'] == 'B') & (df['pred_label'] == 'V')])

    total_fl_pct = (total_fl / total_v * 100) if total_v > 0 else 0
    total_fi_pct = (total_fi / total_b * 100) if total_b > 0 else 0

    fl_ci = wilson_ci(total_fl, total_v)
    fi_ci = wilson_ci(total_fi, total_b)

    delta_lb = total_fi_pct - total_fl_pct

    # p-value from binomial test
    p_value = binomial_test_pvalue(total_fl, total_fl + total_fi, 0.5)

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
        description='Generate per-actor FL/FI analysis'
    )
    parser.add_argument(
        '--country',
        type=str,
        required=True,
        choices=['cmr', 'nga'],
        help='Country code (cmr or nga)'
    )
    parser.add_argument(
        '--condition',
        type=str,
        required=True,
        help='Condition (e.g., zero_shot, few_shot_1, few_shot_3, few_shot_5)'
    )

    args = parser.parse_args()

    # Parse condition
    if args.condition == 'zero_shot':
        strategy = 'zero_shot'
        shots = None
    elif args.condition.startswith('few_shot_'):
        strategy = 'few_shot'
        shots = args.condition.split('_')[2]
    else:
        print(f'Unknown condition: {args.condition}')
        return

    # Build path
    base_dir = Path('results') / args.country
    if shots is None:
        strategy_dir = base_dir / strategy / '1000'
    else:
        strategy_dir = base_dir / strategy / '1000' / shots

    output_dir = Path('results/analysis/fl_fi') / args.country
    output_dir.mkdir(parents=True, exist_ok=True)

    models = ['llama3.2_3b', 'mistral_7b', 'olmo2_7b', 'gemma3_4b']
    actors = ['Police', 'Military']

    all_results = []

    print(f'\nAnalyzing {args.condition} for {args.country.upper()}...\n')

    for model in models:
        model_dir = strategy_dir / model
        results_files = list(model_dir.glob(f'ollama_results_*_acled_{args.country}_actors.csv'))

        if not results_files:
            print(f'  {model}: no results file')
            continue

        results_file = results_files[0]
        df = pd.read_csv(results_file)

        print(f'  {model}:')

        # Per-actor analysis
        for actor in actors:
            result = analyze_actor(df, actor)

            print(f'    {actor}: {result["true_v"]}V/{result["true_b"]}B, '
                  f'FL={result["fl_count"]} ({result["fl_pct"]:.1f}%), '
                  f'FI={result["fi_count"]} ({result["fi_pct"]:.1f}%), '
                  f'Δ={result["delta_lb"]:.1f}%, p={result["p_value"]:.3f}')

            all_results.append({
                'Model': model,
                'Actor': actor,
                'Condition': args.condition,
                'n_V': result['true_v'],
                'n_B': result['true_b'],
                'n_FL': result['fl_count'],
                'ε_FL': result['fl_pct'],
                'FL_CI_lower': result['fl_ci'][0],
                'FL_CI_upper': result['fl_ci'][1],
                'n_FI': result['fi_count'],
                'ε_FI': result['fi_pct'],
                'FI_CI_lower': result['fi_ci'][0],
                'FI_CI_upper': result['fi_ci'][1],
                'Δ_LB': result['delta_lb'],
                'p_value': result['p_value']
            })

        # Total analysis
        result = analyze_actor(df, None)

        print(f'    Total: {result["true_v"]}V/{result["true_b"]}B, '
              f'FL={result["fl_count"]} ({result["fl_pct"]:.1f}%), '
              f'FI={result["fi_count"]} ({result["fi_pct"]:.1f}%), '
              f'Δ={result["delta_lb"]:.1f}%, p={result["p_value"]:.3f}')

        all_results.append({
            'Model': model,
            'Actor': 'Total',
            'Condition': args.condition,
            'n_V': result['true_v'],
            'n_B': result['true_b'],
            'n_FL': result['fl_count'],
            'ε_FL': result['fl_pct'],
            'FL_CI_lower': result['fl_ci'][0],
            'FL_CI_upper': result['fl_ci'][1],
            'n_FI': result['fi_count'],
            'ε_FI': result['fi_pct'],
            'FI_CI_lower': result['fi_ci'][0],
            'FI_CI_upper': result['fi_ci'][1],
            'Δ_LB': result['delta_lb'],
            'p_value': result['p_value']
        })

    # Save
    df = pd.DataFrame(all_results)
    output_file = output_dir / f'fl_fi_per_actor_{args.condition}.csv'
    df.to_csv(output_file, index=False)

    print(f'\n✓ Saved to: {output_file}')
    print(f'\nGenerated {len(df)} rows for {args.country.upper()}')


if __name__ == '__main__':
    main()
