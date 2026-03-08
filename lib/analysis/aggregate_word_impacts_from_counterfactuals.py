#!/usr/bin/env python3
"""
Aggregate word impacts across multiple models from counterfactual analysis.

This script:
1. Loads counterfactual JSON files from the results directory
2. Extracts word/phrase impact rows from each model's detailed_results
3. Aggregates statistics across all models
4. Computes p-values using t-tests
5. Saves aggregated CSV for figure generation

Usage (via pipeline scripts with env vars):
    COUNTRY=cmr STRATEGY=zero_shot SAMPLE_SIZE=1000 python -m lib.analysis.aggregate_word_impacts_from_counterfactuals

Output:
    results/{country}/{strategy}/{sample_size}/word_impacts.csv
"""

import os
import re
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import json
from typing import List, Dict, Tuple
import sys


class WordImpactAnalyzer:
    """Extract per-word/-phrase impact rows from counterfactual detailed_results.

    For each perturbation in a counterfactual JSON's 'detailed_results' section,
    this class extracts the tokens that were modified (the 'original' and
    'replacement' surface strings) and associates them with the confidence_delta
    and label_flipped values produced by the given model.  The result is a
    flat DataFrame suitable for cross-model aggregation.
    """

    # Category label used for perturbations that don't carry a specific word token
    # (e.g. legitimation/provenance additions insert whole phrases)
    PHRASE_CATEGORY = 'phrase'

    def extract_word_impacts(self, detailed_results: list, model_name: str) -> pd.DataFrame:
        """Extract word-level impact rows from one model's counterfactual results.

        Args:
            detailed_results: List of event-level dicts from counterfactual JSON.
            model_name:       Model key used in 'model_results' sub-dicts.

        Returns:
            DataFrame with columns:
                word, category, confidence_delta, label_flipped, event_id, model
        """
        rows = []
        for event in detailed_results:
            event_id = event.get('event_id', '')
            for pert_result in event.get('perturbations', []):
                pert = pert_result.get('perturbation', {})
                pert_type = pert.get('type', 'unknown')
                model_res = pert_result.get('model_results', {}).get(model_name, {})

                if not model_res.get('success', False):
                    continue
                if 'confidence_delta' not in model_res:
                    continue

                conf_delta = float(model_res['confidence_delta'])
                flipped = bool(model_res.get('label_flipped', False))

                # Extract the modified token(s) from the perturbation dict.
                # Different generator types store the token under different keys.
                tokens = self._extract_tokens(pert)
                for word, category in tokens:
                    rows.append({
                        'word': word,
                        'category': category if category else pert_type,
                        'confidence_delta': conf_delta,
                        'label_flipped': flipped,
                        'event_id': event_id,
                        'model': model_name,
                    })

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=['word', 'category', 'confidence_delta', 'label_flipped', 'event_id', 'model']
        )

    @staticmethod
    def _extract_tokens(pert: dict) -> List[Tuple[str, str]]:
        """Return a list of (token_string, category) pairs from a perturbation dict."""
        pert_type = pert.get('type', 'unknown')
        tokens = []

        # Substitution-style perturbations: have explicit 'original' and 'replacement'
        if 'original' in pert and 'replacement' in pert:
            tokens.append((str(pert['replacement']), pert_type))
            return tokens

        # Phrase-insertion perturbations (legitimation_add, delegitimation_add,
        # provenance_add, neutral_control) carry the phrase under a key like
        # 'phrase', 'source', or 'modifier'.
        for key in ('phrase', 'source', 'modifier'):
            if key in pert:
                # Use first 5 words as the representative token to keep CSV readable
                phrase = ' '.join(str(pert[key]).split()[:5])
                tokens.append((phrase, pert_type))
                return tokens

        # Fallback: use description text up to first comma
        desc = pert.get('description', '')
        if desc:
            token = desc.split(',')[0].strip()
            tokens.append((token, pert_type))
        return tokens


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
    """Generate aggregated word impacts from counterfactual data.

    Reads COUNTRY, STRATEGY, SAMPLE_SIZE, NUM_EXAMPLES from environment to
    construct the results directory path, then auto-discovers all model
    subdirectories that contain a counterfactual JSON file.
    """
    country     = os.environ.get('COUNTRY',     'cmr')
    strategy    = os.environ.get('STRATEGY',    'zero_shot')
    sample_size = os.environ.get('SAMPLE_SIZE', '1000')
    num_examples = os.environ.get('NUM_EXAMPLES')

    if strategy == 'few_shot' and num_examples:
        results_base = Path(f"results/{country}/{strategy}/{sample_size}/{num_examples}")
    else:
        results_base = Path(f"results/{country}/{strategy}/{sample_size}")

    output_file = results_base / 'word_impacts.csv'

    # Auto-discover model directories that contain a counterfactual JSON
    json_pattern = str(results_base / '*' / 'counterfactual_analysis_*.json')
    json_files   = sorted(glob.glob(json_pattern))

    if not json_files:
        print(f"No counterfactual JSON files found under {results_base}/*/")
        print("Run counterfactual analysis first.")
        sys.exit(0)

    # Map model_dir_slug -> json_path (one JSON per model directory)
    model_json_map: Dict[str, Path] = {}
    for jf in json_files:
        model_dir = Path(jf).parent.name
        if model_dir not in model_json_map:    # keep first match per directory (lexicographic order)
            model_json_map[model_dir] = Path(jf)

    models = list(model_json_map.keys())

    print("=" * 80)
    print("AGGREGATING WORD IMPACTS ACROSS MODELS")
    print("=" * 80)
    print(f"\nSource: {results_base}")
    print(f"Models: {', '.join(models)}\n")

    analyzer = WordImpactAnalyzer()
    all_word_dfs = []

    for model_dir in models:
        json_file = model_json_map[model_dir]

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
