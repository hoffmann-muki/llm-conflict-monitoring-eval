import os
import re
from typing import Dict, Tuple, Optional


def model_name_to_dir_slug(model_name: str) -> str:
    """Convert model name to directory-safe slug using underscores.
    
    Examples:
        "mistral:7b" -> "mistral_7b"
        "gemma3:4b" -> "gemma3_4b"
        "qwen3:1.7b" -> "qwen3_1.7b"
    
    This differs from model_name_to_slug() in result_aggregator which uses 
    hyphens for filenames. Underscores are more conventional for directory names.
    """
    return re.sub(r'[^a-zA-Z0-9._]', '_', model_name)


def get_model_results_dir(results_dir: str, model_name: str) -> str:
    """Get the model-specific subdirectory within a results directory.
    
    Args:
        results_dir: Base results directory (e.g., results/cmr/zero_shot/1000/)
        model_name: Model name (e.g., 'mistral:7b')
        
    Returns:
        Path to model subdirectory (e.g., results/cmr/zero_shot/1000/mistral_7b/)
    """
    model_dir_slug = model_name_to_dir_slug(model_name)
    return os.path.join(results_dir, model_dir_slug)


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


def get_num_examples() -> Optional[int]:
    """Get the number of few-shot examples from environment variable.
    
    Returns:
        Number of examples (1-5) for few-shot strategy, or None if not set.
        Only relevant when strategy is 'few_shot'.
    """
    val = os.environ.get('NUM_EXAMPLES')
    if val is not None:
        try:
            n = int(val)
            if 1 <= n <= 5:
                return n
        except ValueError:
            pass
    return None


def _build_results_dir(country: str, strategy: str, sample_size: str, 
                       num_examples: Optional[int] = None) -> str:
    """Build the results directory path.
    
    For few_shot strategy with num_examples specified, appends /{num_examples}
    to create paths like: results/nga/few_shot/1000/3
    
    For other strategies or when num_examples is not specified:
    results/{country}/{strategy}/{sample_size}
    """
    base_dir = os.path.join('results', country, strategy, str(sample_size))
    
    # For few_shot strategy, add num_examples subdirectory if specified
    if strategy == 'few_shot' and num_examples is not None:
        return os.path.join(base_dir, str(num_examples))
    
    return base_dir


def setup_country_environment(country: str | None = None, strategy: str | None = None,
                              sample_size: str | None = None,
                              num_examples: int | None = None) -> Tuple[str, str]:
    """Standard country, strategy, and sample size environment setup used across tools.
    
    Args:
        country: Country code. If None, reads from COUNTRY env var.
        strategy: Prompting strategy. If None, reads from STRATEGY env var.
        sample_size: Sample size. If None, reads from SAMPLE_SIZE env var.
        num_examples: Number of few-shot examples (1-5). If None, reads from NUM_EXAMPLES env var.
                     Only used when strategy is 'few_shot'.
    
    Returns:
        Tuple of (country_code, results_dir_path)
        
    Note: 
        - For most strategies: results/{country}/{strategy}/{sample_size}/
        - For few_shot with num_examples: results/{country}/few_shot/{sample_size}/{num_examples}/
    """
    country = country or os.environ.get('COUNTRY', 'cmr')
    if strategy is None:
        strategy = get_strategy()
    if sample_size is None:
        sample_size = get_sample_size()
    if num_examples is None:
        num_examples = get_num_examples()
    
    results_dir = _build_results_dir(country, strategy, str(sample_size), num_examples)
    os.makedirs(results_dir, exist_ok=True)
    return country, results_dir


def paths_for_country(country: str, strategy: str = None, sample_size: str = None,
                      num_examples: Optional[int] = None) -> Dict[str, str]:
    """Get standard paths for a country, strategy, and sample size.
    
    Args:
        country: Country code (e.g., 'cmr', 'nga')
        strategy: Prompting strategy. If None, reads from STRATEGY env var.
        sample_size: Sample size. If None, reads from SAMPLE_SIZE env var.
        num_examples: Number of few-shot examples (1-5). If None, reads from NUM_EXAMPLES env var.
                     Only used when strategy is 'few_shot'.
    
    Returns:
        Dictionary with paths for results_dir, datasets_dir, sample_path, calibrated_csv
        
    Note: 
        - For most strategies: results/{country}/{strategy}/{sample_size}/
        - For few_shot with num_examples: results/{country}/few_shot/{sample_size}/{num_examples}/
    """
    if strategy is None:
        strategy = get_strategy()
    if sample_size is None:
        sample_size = get_sample_size()
    if num_examples is None:
        num_examples = get_num_examples()
    
    results_dir = _build_results_dir(country, strategy, str(sample_size), num_examples)
    datasets_dir = os.path.join('datasets', country)
    # Include sample_size in filename for fair cross-model comparison at different sample sizes
    sample_path = os.path.join(datasets_dir, f'state_actor_sample_{country}_{sample_size}.csv')
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

def write_sample(country: str, sample_df, sample_size: str = None, 
                 sample_name: str = 'state_actor_sample') -> str:
    """Write sample DataFrame to CSV for cross-model consistency.
    
    Args:
        country: Country code (e.g., 'cmr', 'nga')
        sample_df: DataFrame containing the sample
        sample_size: Sample size (included in filename for fair comparison)
        sample_name: Base name for the sample file
    
    Returns:
        Path to the written sample file
    """
    if sample_size is None:
        sample_size = get_sample_size()
    paths = paths_for_country(country, sample_size=sample_size)
    os.makedirs(paths['datasets_dir'], exist_ok=True)
    # Use sample_path which already includes sample_size
    path = paths['sample_path']
    sample_df.to_csv(path, index=False)
    return path
