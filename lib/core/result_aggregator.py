#!/usr/bin/env python3
"""Result aggregator for per-model inference results.

This module scans for per-model result files and aggregates them into a single
combined file for cross-model analysis (calibration, metrics, harm, disagreements).

Directory structure: results/{country}/{strategy}/{sample_size}/{model_slug}/
Per-model files follow the pattern:
    {model_slug}/ollama_results_{model_slug}_acled_{country}_state_actors.csv

Combined file (for downstream analysis, at parent level):
    ollama_results_acled_{country}_state_actors.csv
"""

import os
import glob
import re
import pandas as pd
from lib.core.data_helpers import (
    setup_country_environment, get_strategy, get_sample_size, get_num_examples,
    model_name_to_dir_slug, get_model_results_dir
)


def model_name_to_slug(model_name: str) -> str:
    """Convert model name to filesystem-safe slug.
    
    Examples:
        'mistral:7b' -> 'mistral-7b'
        'llama3.2:3b' -> 'llama3.1-3b'
    """
    return model_name.replace(':', '-').replace('/', '_')


def slug_to_model_name(slug: str) -> str:
    """Convert filesystem slug back to model name.
    
    Examples:
        'mistral-7b' -> 'mistral:7b'
        'llama3.1-3b' -> 'llama3.2:3b'
    """
    # Find the last dash followed by a size indicator (e.g., 7b, 8b, 1.7b)
    # and replace it with a colon
    match = re.match(r'^(.+)-(\d+\.?\d*b)$', slug, re.IGNORECASE)
    if match:
        return f"{match.group(1)}:{match.group(2)}"
    return slug


def get_per_model_results_pattern(country: str, results_dir: str = None, 
                                   strategy: str = None, sample_size: str = None,
                                   num_examples: int = None) -> str:
    """Get glob pattern for per-model result files in model subdirectories."""
    if strategy is None:
        strategy = get_strategy()
    if sample_size is None:
        sample_size = get_sample_size()
    if num_examples is None:
        num_examples = get_num_examples()
    if results_dir is None:
        _, results_dir = setup_country_environment(country, strategy, str(sample_size), num_examples)
    # Pattern matches model subdirectories: results_dir/*/ollama_results_*_...
    return os.path.join(results_dir, '*', f'ollama_results_*_acled_{country}_state_actors.csv')


def get_combined_results_path(country: str, results_dir: str = None,
                              strategy: str = None, sample_size: str = None,
                              num_examples: int = None) -> str:
    """Get path to combined results file."""
    if strategy is None:
        strategy = get_strategy()
    if sample_size is None:
        sample_size = get_sample_size()
    if num_examples is None:
        num_examples = get_num_examples()
    if results_dir is None:
        _, results_dir = setup_country_environment(country, strategy, str(sample_size), num_examples)
    return os.path.join(results_dir, f'ollama_results_acled_{country}_state_actors.csv')


def get_per_model_result_path(country: str, model_name: str, results_dir: str = None,
                              strategy: str = None, sample_size: str = None,
                              num_examples: int = None) -> str:
    """Get path to a specific model's result file within its model subdirectory."""
    if strategy is None:
        strategy = get_strategy()
    if sample_size is None:
        sample_size = get_sample_size()
    if num_examples is None:
        num_examples = get_num_examples()
    if results_dir is None:
        _, results_dir = setup_country_environment(country, strategy, str(sample_size), num_examples)
    
    # Create model subdirectory path
    model_results_dir = get_model_results_dir(results_dir, model_name)
    os.makedirs(model_results_dir, exist_ok=True)
    
    slug = model_name_to_slug(model_name)
    return os.path.join(model_results_dir, f'ollama_results_{slug}_acled_{country}_state_actors.csv')


def list_per_model_files(country: str, results_dir: str = None,
                         strategy: str = None, sample_size: str = None,
                         num_examples: int = None) -> list:
    """List all per-model result files for a country, strategy, and sample size.
    
    Scans model subdirectories within results_dir for per-model result files.
    
    Returns:
        List of (model_slug, file_path) tuples
    """
    if strategy is None:
        strategy = get_strategy()
    if sample_size is None:
        sample_size = get_sample_size()
    if num_examples is None:
        num_examples = get_num_examples()
    if results_dir is None:
        _, results_dir = setup_country_environment(country, strategy, str(sample_size), num_examples)
    pattern = get_per_model_results_pattern(country, results_dir, strategy, sample_size, num_examples)
    files = glob.glob(pattern)
    
    # Extract model slug from filename (consistent with how we write them)
    # Pattern: {model_dir}/ollama_results_{slug}_acled_{country}_state_actors.csv
    results = []
    for f in files:
        basename = os.path.basename(f)
        match = re.match(rf'ollama_results_(.+)_acled_{country}_state_actors\.csv', basename)
        if match:
            slug = match.group(1)
            results.append((slug, f))
    
    return results


def aggregate_model_results(country: str = None, results_dir: str = None, 
                           strategy: str = None, sample_size: str = None,
                           num_examples: int = None,
                           verbose: bool = True) -> pd.DataFrame:
    """Aggregate all per-model result files into a single DataFrame.
    
    Args:
        country: Country code (e.g., 'nga', 'cmr'). If None, reads from COUNTRY env var.
        results_dir: Results directory. If None, uses default 'results/{country}/{strategy}/{sample_size}'.
        strategy: Prompting strategy. If None, reads from STRATEGY env var.
        sample_size: Sample size. If None, reads from SAMPLE_SIZE env var.
        num_examples: Number of few-shot examples. If None, reads from NUM_EXAMPLES env var.
        verbose: Print progress messages.
    
    Returns:
        Combined DataFrame with all models' results.
    """
    if strategy is None:
        strategy = get_strategy()
    if sample_size is None:
        sample_size = get_sample_size()
    if num_examples is None:
        num_examples = get_num_examples()
    
    if country is None:
        country, results_dir = setup_country_environment(None, strategy, sample_size, num_examples)
    elif results_dir is None:
        _, results_dir = setup_country_environment(country, strategy, str(sample_size), num_examples)
    
    per_model_files = list_per_model_files(country, results_dir, strategy, sample_size, num_examples)
    
    if not per_model_files:
        if verbose:
            print(f"No per-model result files found for strategy '{strategy}', sample_size '{sample_size}' in {results_dir}")
            print(f"Pattern: {get_per_model_results_pattern(country, results_dir, strategy, sample_size, num_examples)}")
        return pd.DataFrame()
    
    if verbose:
        print(f"Found {len(per_model_files)} per-model result files (strategy={strategy}, sample_size={sample_size}):")
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
                          strategy: str = None, sample_size: str = None,
                          num_examples: int = None,
                          verbose: bool = True) -> str:
    """Aggregate per-model files and write combined results file.
    
    Returns:
        Path to the combined results file.
    """
    if strategy is None:
        strategy = get_strategy()
    if sample_size is None:
        sample_size = get_sample_size()
    if num_examples is None:
        num_examples = get_num_examples()
    
    if country is None:
        country, results_dir = setup_country_environment(None, strategy, sample_size, num_examples)
    elif results_dir is None:
        _, results_dir = setup_country_environment(country, strategy, str(sample_size), num_examples)
    
    combined = aggregate_model_results(country, results_dir, strategy, sample_size, num_examples, verbose)
    
    if combined.empty:
        if verbose:
            print("No data to write")
        return None
    
    out_path = get_combined_results_path(country, results_dir, strategy, sample_size, num_examples)
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
    parser.add_argument('--sample-size', default=os.environ.get('SAMPLE_SIZE', None),
                       help='Sample size (e.g., 500, 1000). Default: SAMPLE_SIZE env var')
    parser.add_argument('--num-examples', type=int, default=None,
                       help='Number of few-shot examples (1-5). Only used with few_shot strategy. '
                            'Default: NUM_EXAMPLES env var')
    parser.add_argument('--results-dir', default=None,
                       help='Results directory. Default: results/{country}/{strategy}/{sample_size}')
    parser.add_argument('--list-only', action='store_true',
                       help='List per-model files without aggregating')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    strategy = args.strategy
    sample_size = args.sample_size
    num_examples = args.num_examples
    
    if args.list_only:
        country = args.country or os.environ.get('COUNTRY', 'cmr')
        files = list_per_model_files(country, args.results_dir, strategy, sample_size, num_examples)
        if files:
            print(f"Per-model result files for {country} (strategy={strategy or get_strategy()}, sample_size={sample_size or get_sample_size()}):"
                  + (f", num_examples={num_examples}" if strategy == 'few_shot' and num_examples else "") + ":")
            for slug, path in files:
                print(f"  {slug}: {path}")
        else:
            print(f"No per-model result files found for {country} (strategy={strategy or get_strategy()}, sample_size={sample_size or get_sample_size()})")
    else:
        write_combined_results(args.country, args.results_dir, strategy, sample_size, num_examples, verbose)


if __name__ == '__main__':
    main()
