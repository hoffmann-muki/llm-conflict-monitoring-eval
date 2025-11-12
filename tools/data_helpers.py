import os
from typing import Dict

def paths_for_country(country: str) -> Dict[str, str]:
    results_dir = os.path.join('results', country)
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
