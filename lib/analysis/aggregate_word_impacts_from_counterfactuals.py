#!/usr/bin/env python3
"""
Aggregate word impacts across multiple models from zero-shot counterfactual analysis.

This script:
1. Loads counterfactual JSON files from zero_shot results
2. Extracts word impacts from each model
3. Aggregates statistics across all models
4. Computes p-values using t-tests
5. Saves aggregated CSV for figure generation

Usage:
    python lib/analysis/aggregate_word_impacts.py

Output:
    results/cmr/explainable/1000/word_impacts.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import json
from typing import List, Dict, Tuple
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.plot_word_impact import WordImpactAnalyzer


def aggregate_across_models(word_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate word impacts across multiple models.

    Args:
        word_dfs: List of DataFrames from different models

    Returns:
        Aggregated DataFrame with cross-model statistics
    """
    # Concatenate all model data
    all_data = pd.concat(word_dfs, ignore_index=True)

    # Group by word and category
    grouped = all_data.groupby(['word', 'category'])

    aggregated = []

    for (word, category), group in grouped:
        n_total = len(group)

        # Calculate aggregated metrics
        delta_mean = group['confidence_delta'].mean()
        delta_std = group['confidence_delta'].std(ddof=1) if len(group) > 1 else 0
        count = n_total

        # Flip rate
        n_flips = group['label_flipped'].sum()
        flip_rate = n_flips / n_total if n_total > 0 else 0

        # Statistical test: H0: mean confidence_delta = 0
        if len(group) > 1:
            t_stat, p_value = stats.ttest_1samp(group['confidence_delta'], 0)
        else:
            p_value = 1.0

        significant = p_value < 0.05

        # Score shift percentage
        score_shift_pct = delta_mean * 100

        # Impact magnitude (|effect| × √n)
        impact_magnitude = abs(score_shift_pct) * np.sqrt(count)

        aggregated.append({
            'word': word,
            'category': category,
            'delta_mean': delta_mean,
            'delta_std': delta_std,
            'count': count,
            'flip_rate': flip_rate,
            'n_total': n_total,
            'p_value': p_value,
            'significant': significant,
            'score_shift_pct': score_shift_pct,
            'impact_magnitude': impact_magnitude
        })

    df = pd.DataFrame(aggregated)
    return df.sort_values('impact_magnitude', ascending=False)


def main():
    """Generate aggregated word impacts from zero-shot counterfactual data."""

    # Paths
    zero_shot_dir = Path("results/cmr/zero_shot/1000")
    output_file = Path("results/cmr/zero_shot/1000/word_impacts.csv")

    # Models to include (matching perturbation table)
    models = ['mistral_7b', 'llama3.2_3b', 'olmo2_7b', 'gemma3_4b']

    print("=" * 80)
    print("AGGREGATING WORD IMPACTS ACROSS MODELS")
    print("=" * 80)
    print(f"\nSource: {zero_shot_dir}")
    print(f"Models: {', '.join(models)}\n")

    analyzer = WordImpactAnalyzer()
    all_word_dfs = []

    for model_dir in models:
        json_file = zero_shot_dir / model_dir / f"counterfactual_analysis_{model_dir.replace('_', '-').replace('.', '_')}.json"

        if not json_file.exists():
            print(f"⚠ Skipping {model_dir}: JSON not found")
            continue

        print(f"Processing {model_dir}...")

        # Load detailed results
        with open(json_file, 'r') as f:
            data = json.load(f)
            detailed_results = data.get('detailed_results', [])

        if not detailed_results:
            print(f"  ⚠ No detailed results")
            continue

        # Extract model name from JSON
        model_name = data['metadata']['models'][0] if data['metadata'].get('models') else model_dir

        # Extract word impacts for this model
        word_df = analyzer.extract_word_impacts(detailed_results, model_name)

        if len(word_df) > 0:
            print(f"  ✓ Extracted {len(word_df)} word impact instances")
            all_word_dfs.append(word_df)
        else:
            print(f"  ⚠ No word impacts extracted")

    if not all_word_dfs:
        print("\n✗ No data to aggregate")
        return

    print(f"\n✓ Loaded data from {len(all_word_dfs)} models")
    print(f"  Total instances: {sum(len(df) for df in all_word_dfs)}")

    # Aggregate across models
    print("\nAggregating statistics...")
    agg_df = aggregate_across_models(all_word_dfs)

    print(f"✓ Aggregated {len(agg_df)} unique words/phrases")
    print(f"  Max n_total: {agg_df['n_total'].max()} (expected: {len(all_word_dfs)} models × 20 events)")

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(output_file, index=False)

    print(f"\n✓ Saved: {output_file}")
    print("\nTop 10 words by impact magnitude:")
    print(agg_df[['word', 'category', 'flip_rate', 'n_total', 'impact_magnitude']].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
