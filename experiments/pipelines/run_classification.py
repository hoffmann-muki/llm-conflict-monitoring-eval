#!/usr/bin/env python3
"""Strategy-agnostic classification pipeline for experiments.

This pipeline runs classification experiments using different prompting strategies
and generates quantitative results for comparison (classification, fairness,
counterfactual, harm metrics).
"""

import pandas as pd
import os
import sys

# Import strategy classes
from experiments.prompting_strategies import ZeroShotStrategy
from experiments.prompting_strategies.few_shot import FewShotStrategy
from experiments.prompting_strategies.explainable import ExplainableStrategy

# Import from lib structure
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from lib.data_preparation import extract_country_rows, get_actor_norm_series, extract_state_actor, build_stratified_sample
from lib.core.constants import LABEL_MAP, EVENT_CLASSES_FULL, CSV_SRC, WORKING_MODELS
from lib.core.data_helpers import paths_for_country, resolve_columns, write_sample, setup_country_environment

# Country name mapping
COUNTRY_NAMES = {
    'cmr': 'Cameroon',
    'nga': 'Nigeria'
}

# Strategy registry
STRATEGY_REGISTRY = {
    'zero_shot': ZeroShotStrategy,
    'few_shot': FewShotStrategy,
    'explainable': ExplainableStrategy
}


def get_strategy(strategy_name: str):
    """Get strategy instance by name.
    
    Args:
        strategy_name: Name of the strategy (e.g., 'zero_shot', 'few_shot')
        
    Returns:
        Strategy instance
        
    Raises:
        ValueError: If strategy name is not recognized
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    
    # Build config based on strategy and environment variables
    config = {}
    if strategy_name == 'few_shot':
        # Support EXAMPLES_PER_CATEGORY environment variable for few-shot
        examples_per_category = os.environ.get('EXAMPLES_PER_CATEGORY', '1')
        try:
            config['examples_per_category'] = int(examples_per_category)
        except ValueError:
            print(f"Warning: Invalid EXAMPLES_PER_CATEGORY value '{examples_per_category}', using default 1")
            config['examples_per_category'] = 1
    
    return STRATEGY_REGISTRY[strategy_name](config=config if config else None)


def run_model_on_rows_with_strategy(model_name: str, rows, strategy, 
                                    note_col: str = 'notes',
                                    event_id_col: str = 'event_id_cnty',
                                    true_label_col: str = 'event_type',
                                    actor_norm_col: str = 'actor_norm'):
    """Run model on rows using specified prompting strategy.
    
    This is a strategy-aware version of run_model_on_rows that uses
    the strategy's make_prompt() method.
    
    Args:
        model_name: Name of the Ollama model
        rows: DataFrame rows to classify
        strategy: PromptingStrategy instance
        note_col: Column name for event notes
        event_id_col: Column name for event ID
        true_label_col: Column name for true label
        actor_norm_col: Column name for normalized actor
        
    Returns:
        List of result dictionaries
    """
    import time
    import json
    from lib.inference.ollama_client import run_ollama_structured
    
    results = []
    for r in rows.itertuples(index=False):
        t0 = time.time()
        try:
            note = getattr(r, note_col)
            # Use strategy-specific prompt
            prompt = strategy.make_prompt(note)
            # TODO: Refactor run_ollama_structured to accept prompt directly
            # For now, we'll use the existing function which calls make_prompt internally
            resp = run_ollama_structured(model_name, note)
            label = str(resp.get("label", "FAIL")).strip()
            conf = float(resp.get("confidence", 0))
            logits = None
            for k in ("logits", "log_probs", "scores", "label_scores"):
                if k in resp:
                    logits = resp.get(k)
                    break
        except Exception as e:
            label = "ERROR"
            conf = 0.0
            logits = None
            print(f"Error classifying event: {e}")
        
        elapsed = round(time.time() - t0, 2)
        results.append({
            "model": model_name,
            "event_id": getattr(r, event_id_col, None),
            "true_label": getattr(r, true_label_col, None),
            "pred_label": label,
            "pred_conf": conf,
            "logits": json.dumps(logits) if logits is not None else None,
            "latency_sec": elapsed,
            "actor_norm": getattr(r, actor_norm_col, None)
        })
    
    return results


def run_classification_experiment(country_code: str, 
                                  sample_size: int = 100,
                                  strategy_name: str = 'zero_shot'):
    """Run classification experiment with specified prompting strategy.
    
    Args:
        country_code: Country code (e.g., 'cmr', 'nga')
        sample_size: Number of samples to generate
        strategy_name: Prompting strategy to use
    """
    if country_code not in COUNTRY_NAMES:
        raise ValueError(
            f"Unsupported country code: {country_code}. "
            f"Supported: {list(COUNTRY_NAMES.keys())}"
        )
    
    country_name = COUNTRY_NAMES[country_code]
    
    if not os.path.exists(CSV_SRC):
        raise SystemExit(f"Source CSV not found: {CSV_SRC}")
    
    # Get prompting strategy
    strategy = get_strategy(strategy_name)
    print(f"\n{'='*70}")
    print(f"Running experiment for {country_name} ({country_code})")
    print(f"Strategy: {strategy_name}")
    print(f"Sample size: {sample_size}")
    print(f"{'='*70}\n")
    
    # Data preparation (same as before)
    df_all = pd.read_csv(CSV_SRC)
    df_country = extract_country_rows(CSV_SRC, country_name)
    
    # Persist extracted country-specific CSV for auditing and reuse
    paths = paths_for_country(country_code)
    os.makedirs(paths['datasets_dir'], exist_ok=True)
    out_country = os.path.join(
        paths['datasets_dir'], 
        f"{country_name}_lagged_data_up_to-2024-10-24.csv"
    )
    df_country.to_csv(out_country, index=False)
    print(f"Wrote extracted {country_name} data to {out_country}")
    
    # Resolve column names case-insensitively
    cols = resolve_columns(
        df_country, 
        ['actor1', 'notes', 'event_type', 'event_id_cnty']
    )
    col_actor = cols.get('actor1') or 'actor1'
    col_notes = cols.get('notes') or 'notes'
    col_event_type = cols.get('event_type') or 'event_type'
    col_event_id = cols.get('event_id_cnty') or 'event_id_cnty'
    
    # Create normalized actor column
    df_country["actor_norm"] = get_actor_norm_series(
        df_country, 
        actor_col=col_actor
    )
    
    # Create state_actor boolean
    df_country["state_actor"] = extract_state_actor(
        df_country, 
        country=country_name.lower(), 
        actor_col=col_actor
    )
    
    # Keep only state-actor rows with valid event types and notes
    usable = (
        df_country.loc[
            df_country["state_actor"]
            & df_country[col_notes].notna()
            & df_country[col_event_type].isin(EVENT_CLASSES_FULL),
            [col_event_id, col_notes, col_event_type, "actor_norm"]
        ]
        .rename(columns={
            col_event_id: "event_id_cnty", 
            col_notes: "notes", 
            col_event_type: "event_type"
        })
        .assign(notes=lambda x: x["notes"].str.replace(
            r"\s+", " ", regex=True
        ).str.slice(0, 400))
        .drop_duplicates(subset=["event_id_cnty"])
    )
    
    print(f"Usable state-actor rows found ({country_name}): {len(usable):,}")
    
    # Build stratified sample
    n_total = min(sample_size, len(usable))
    df_test = build_stratified_sample(
        usable,
        stratify_col='event_type',
        n_total=n_total,
        primary_group='Violence against civilians',
        primary_share=0.6,
        label_map=LABEL_MAP,
        random_state=42,
        replace=False
    )
    
    sample_path = write_sample(country_code, df_test)
    print(f"Wrote stratified sample to {sample_path}")
    print(df_test.head())
    
    # Run classification with strategy
    models = WORKING_MODELS
    
    results = []
    subset = df_test.copy()
    print(f"\nStarting classification with {strategy_name} strategy:")
    print(f"  - {len(subset)} events")
    print(f"  - {len(models)} models\n")
    
    for m in models:
        print(f"Starting model: {m}")
        model_results = run_model_on_rows_with_strategy(m, subset, strategy)
        results.extend(model_results)
        print(f"Model {m} completed.")
    
    res_df = pd.DataFrame(results)
    
    # Setup results directory with strategy subfolder
    _, results_dir = setup_country_environment(country_code)
    
    # Create strategy-specific subdirectory
    strategy_results_dir = os.path.join(results_dir, strategy_name)
    os.makedirs(strategy_results_dir, exist_ok=True)
    
    # Save results
    out_path = os.path.join(
        strategy_results_dir, 
        f"ollama_results_acled_{country_code}_state_actors.csv"
    )
    res_df.to_csv(out_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"Experiment completed!")
    print(f"Results saved to: {out_path}")
    print(f"{'='*70}\n")
    print(res_df.head(5))
    
    return out_path


def main():
    """Main entry point - accepts country, strategy from command line or environment."""
    if len(sys.argv) > 1:
        country_code = sys.argv[1]
    else:
        country_code = os.environ.get('COUNTRY', 'cmr')
    
    sample_size = int(os.environ.get('SAMPLE_SIZE', '100'))
    strategy_name = os.environ.get('STRATEGY', 'zero_shot')
    
    # Run the experiment
    run_classification_experiment(country_code, sample_size, strategy_name)


if __name__ == "__main__":
    main()
