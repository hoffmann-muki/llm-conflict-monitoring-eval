import os
from typing import Dict, Tuple

def get_strategy() -> str:
    """Get the prompting strategy from environment variable.
    
    Returns:
        Strategy name (e.g., 'zero_shot', 'few_shot', 'explainable').
        Defaults to 'zero_shot' if STRATEGY env var is not set.
    """
    return os.environ.get('STRATEGY', 'zero_shot')


def get_sample_size() -> str:
    """Get the sample size from environment variable.
    
    Returns:
        Sample size as string (e.g., '500', '1000').
        Defaults to '500' if SAMPLE_SIZE env var is not set.
    """
    return os.environ.get('SAMPLE_SIZE', '500')


def setup_country_environment(country: str | None = None, strategy: str | None = None,
                              sample_size: str | None = None) -> Tuple[str, str]:
    """Standard country, strategy, and sample size environment setup used across tools.
    
    Returns:
        Tuple of (country_code, results_dir_path)
        
    Note: results_dir is now results/{country}/{strategy}/{sample_size}/
    """
    country = country or os.environ.get('COUNTRY', 'cmr')
    if strategy is None:
        strategy = get_strategy()
    if sample_size is None:
        sample_size = get_sample_size()
    results_dir = os.path.join('results', country, strategy, str(sample_size))
    os.makedirs(results_dir, exist_ok=True)
    return country, results_dir

def paths_for_country(country: str, strategy: str = None, sample_size: str = None) -> Dict[str, str]:
    """Get standard paths for a country, strategy, and sample size.
    
    Args:
        country: Country code (e.g., 'cmr', 'nga')
        strategy: Prompting strategy. If None, reads from STRATEGY env var.
        sample_size: Sample size. If None, reads from SAMPLE_SIZE env var.
    
    Returns:
        Dictionary with paths for results_dir, datasets_dir, sample_path, calibrated_csv
        
    Note: results_dir is now results/{country}/{strategy}/{sample_size}/
    """
    if strategy is None:
        strategy = get_strategy()
    if sample_size is None:
        sample_size = get_sample_size()
    
    results_dir = os.path.join('results', country, strategy, str(sample_size))
    datasets_dir = os.path.join('datasets', country)
    sample_path = os.path.join(datasets_dir, f'state_actor_sample_{country}.csv')
    calibrated_csv = os.path.join(results_dir, 'ollama_results_calibrated.csv')
    return {
        'results_dir': results_dir,
        'datasets_dir': datasets_dir,
        'sample_path': sample_path,
        'calibrated_csv': calibrated_csv
    }

def resolve_columns(df, candidates):
    """Resolve column names case-insensitively.

    candidates: iterable of column names to find; returns a dict mapping the
    canonical name -> actual column present in df (or None).
    """
    cols_lower = {c.lower(): c for c in df.columns}
    out = {}
    for name in candidates:
        out[name] = cols_lower.get(name.lower(), None)
    return out

def write_sample(country: str, sample_df, sample_name: str = 'state_actor_sample') -> str:
    paths = paths_for_country(country)
    os.makedirs(paths['datasets_dir'], exist_ok=True)
    path = os.path.join(paths['datasets_dir'], f'{sample_name}_{country}.csv')
    sample_df.to_csv(path, index=False)
    return path
