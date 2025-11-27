#!/usr/bin/env python3
"""Result aggregator for per-model inference results.

This module scans for per-model result files and aggregates them into a single
combined file for cross-model analysis (calibration, metrics, harm, disagreements).

Directory structure: results/{country}/{strategy}/
Per-model files follow the pattern:
    ollama_results_{model_slug}_acled_{country}_state_actors.csv

Combined file (for downstream analysis):
    ollama_results_acled_{country}_state_actors.csv
"""

import os
import glob
import re
import pandas as pd
from lib.core.data_helpers import setup_country_environment, get_strategy


def model_name_to_slug(model_name: str) -> str:
    """Convert model name to filesystem-safe slug.
    
    Examples:
        'mistral:7b' -> 'mistral-7b'
        'llama3.1:8b' -> 'llama3.1-8b'
    """
    return model_name.replace(':', '-').replace('/', '_')


def slug_to_model_name(slug: str) -> str:
    """Convert filesystem slug back to model name.
    
    Examples:
        'mistral-7b' -> 'mistral:7b'
        'llama3.1-8b' -> 'llama3.1:8b'
    """
    # Find the last dash followed by a size indicator (e.g., 7b, 8b, 1.7b)
    # and replace it with a colon
    match = re.match(r'^(.+)-(\d+\.?\d*b)$', slug, re.IGNORECASE)
    if match:
        return f"{match.group(1)}:{match.group(2)}"
    return slug


def get_per_model_results_pattern(country: str, results_dir: str = None, 
                                   strategy: str = None) -> str:
    """Get glob pattern for per-model result files."""
    if strategy is None:
        strategy = get_strategy()
    if results_dir is None:
        results_dir = os.path.join('results', country, strategy)
    return os.path.join(results_dir, f'ollama_results_*_acled_{country}_state_actors.csv')


def get_combined_results_path(country: str, results_dir: str = None,
                              strategy: str = None) -> str:
    """Get path to combined results file."""
    if strategy is None:
        strategy = get_strategy()
    if results_dir is None:
        results_dir = os.path.join('results', country, strategy)
    return os.path.join(results_dir, f'ollama_results_acled_{country}_state_actors.csv')


def get_per_model_result_path(country: str, model_name: str, results_dir: str = None,
                              strategy: str = None) -> str:
    """Get path to a specific model's result file."""
    if strategy is None:
        strategy = get_strategy()
    if results_dir is None:
        results_dir = os.path.join('results', country, strategy)
    slug = model_name_to_slug(model_name)
    return os.path.join(results_dir, f'ollama_results_{slug}_acled_{country}_state_actors.csv')


def list_per_model_files(country: str, results_dir: str = None,
                         strategy: str = None) -> list:
    """List all per-model result files for a country and strategy.
    
    Returns:
        List of (model_slug, file_path) tuples
    """
    if strategy is None:
        strategy = get_strategy()
    if results_dir is None:
        results_dir = os.path.join('results', country, strategy)
    pattern = get_per_model_results_pattern(country, results_dir, strategy)
    files = glob.glob(pattern)
    
    # Extract model slug from filename
    # Pattern: ollama_results_{slug}_acled_{country}_state_actors.csv
    results = []
    for f in files:
        basename = os.path.basename(f)
        match = re.match(rf'ollama_results_(.+)_acled_{country}_state_actors\.csv', basename)
        if match:
            slug = match.group(1)
            results.append((slug, f))
    
    return results


def aggregate_model_results(country: str = None, results_dir: str = None, 
                           strategy: str = None,
                           verbose: bool = True) -> pd.DataFrame:
    """Aggregate all per-model result files into a single DataFrame.
    
    Args:
        country: Country code (e.g., 'nga', 'cmr'). If None, reads from COUNTRY env var.
        results_dir: Results directory. If None, uses default 'results/{country}'.
        strategy: Prompting strategy. If None, reads from STRATEGY env var.
        verbose: Print progress messages.
    
    Returns:
        Combined DataFrame with all models' results.
    """
    if country is None:
        country, results_dir = setup_country_environment()
    elif results_dir is None:
        results_dir = os.path.join('results', country)
    
    if strategy is None:
        strategy = get_strategy()
    
    per_model_files = list_per_model_files(country, results_dir, strategy)
    
    if not per_model_files:
        if verbose:
            print(f"No per-model result files found for strategy '{strategy}' in {results_dir}")
            print(f"Pattern: {get_per_model_results_pattern(country, results_dir, strategy)}")
        return pd.DataFrame()
    
    if verbose:
        print(f"Found {len(per_model_files)} per-model result files (strategy={strategy}):")
        for slug, path in per_model_files:
            print(f"  - {slug}: {path}")
    
    # Load and concatenate all files
    dfs = []
    for slug, path in per_model_files:
        try:
            df = pd.read_csv(path)
            if verbose:
                print(f"  Loaded {len(df)} rows from {slug}")
            dfs.append(df)
        except Exception as e:
            if verbose:
                print(f"  Error loading {path}: {e}")
    
    if not dfs:
        if verbose:
            print("No valid data loaded from per-model files")
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Deduplicate by (model, event_id) - keep last in case of re-runs
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=['model', 'event_id'], keep='last')
    after_dedup = len(combined)
    
    if verbose:
        if before_dedup != after_dedup:
            print(f"Deduplicated: {before_dedup} -> {after_dedup} rows")
        print(f"Combined total: {len(combined)} rows")
        print(f"Models: {combined['model'].unique().tolist()}")
    
    return combined


def write_combined_results(country: str = None, results_dir: str = None,
                          strategy: str = None,
                          verbose: bool = True) -> str:
    """Aggregate per-model files and write combined results file.
    
    Returns:
        Path to the combined results file.
    """
    if country is None:
        country, results_dir = setup_country_environment()
    elif results_dir is None:
        results_dir = os.path.join('results', country)
    
    if strategy is None:
        strategy = get_strategy()
    
    combined = aggregate_model_results(country, results_dir, strategy, verbose)
    
    if combined.empty:
        if verbose:
            print("No data to write")
        return None
    
    out_path = get_combined_results_path(country, results_dir, strategy)
    combined.to_csv(out_path, index=False)
    
    if verbose:
        print(f"\nWrote combined results to: {out_path}")
    
    return out_path


def main():
    """CLI entry point for result aggregation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Aggregate per-model inference results into a combined file'
    )
    parser.add_argument('--country', default=os.environ.get('COUNTRY', None),
                       help='Country code (e.g., cmr, nga). Default: COUNTRY env var')
    parser.add_argument('--strategy', default=os.environ.get('STRATEGY', None),
                       help='Prompting strategy (zero_shot, few_shot, explainable). Default: STRATEGY env var')
    parser.add_argument('--results-dir', default=None,
                       help='Results directory. Default: results/{country}')
    parser.add_argument('--list-only', action='store_true',
                       help='List per-model files without aggregating')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    strategy = args.strategy
    
    if args.list_only:
        country = args.country or os.environ.get('COUNTRY', 'cmr')
        files = list_per_model_files(country, args.results_dir, strategy)
        if files:
            print(f"Per-model result files for {country} (strategy={strategy or get_strategy()}):")
            for slug, path in files:
                print(f"  {slug}: {path}")
        else:
            print(f"No per-model result files found for {country} (strategy={strategy or get_strategy()})")
    else:
        write_combined_results(args.country, args.results_dir, strategy, verbose)


if __name__ == '__main__':
    main()
