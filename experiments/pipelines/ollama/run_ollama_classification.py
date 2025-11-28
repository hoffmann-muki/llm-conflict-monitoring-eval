#!/usr/bin/env python3
"""Strategy-agnostic classification pipeline for experiments.

This pipeline runs classification experiments using different prompting strategies
and generates quantitative results for comparison (classification, fairness,
counterfactual, harm metrics).
"""

import pandas as pd
import os
import sys
import time
import json
import argparse

# Strategy helper imported from the core helpers
from lib.core.strategy_helpers import get_strategy
from lib.core.constants import COUNTRY_NAMES

# Import from lib structure
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from lib.data_preparation import extract_country_rows, get_actor_norm_series, extract_state_actor, build_stratified_sample
from lib.core.constants import LABEL_MAP, EVENT_CLASSES_FULL, CSV_SRC, WORKING_MODELS, COUNTRY_NAMES as _COUNTRY_NAMES
from lib.inference.ollama_client import run_ollama_structured
from lib.core.data_helpers import paths_for_country, resolve_columns, write_sample, setup_country_environment
from lib.core.result_aggregator import model_name_to_slug, get_per_model_result_path

# get_strategy and COUNTRY_NAMES are provided by lib.core.constants


def run_model_on_rows_with_strategy(model_name: str, rows, strategy, 
                                    note_col: str = 'notes',
                                    event_id_col: str = 'event_id_cnty',
                                    true_label_col: str = 'gold_label',
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
    results = []
    for r in rows.itertuples(index=False):
        t0 = time.time()
        try:
            note = getattr(r, note_col)
            # Generate strategy-specific prompt
            prompt = strategy.make_prompt(note)
            system_msg = strategy.get_system_message()
            # Run with strategy prompt and system message
            resp = run_ollama_structured(
                model_name, 
                prompt=prompt,
                system_msg=system_msg
            )
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
                                  strategy_name: str = 'zero_shot',
                                  primary_group: str | None = None,
                                  primary_share: float = 0.0,
                                  num_examples: int | None = None,
                                  models: list | None = None):
    """Run classification experiment with specified prompting strategy.
    
    Args:
        country_code: Country code (e.g., 'cmr', 'nga')
        sample_size: Number of samples to generate
        strategy_name: Prompting strategy to use
        primary_group: Optional event type to oversample (default: None for proportional sampling)
        primary_share: Fraction of sample reserved for primary_group (0-1, default: 0.0)
        num_examples: Number of few-shot examples (1-5). Only used when strategy_name='few_shot'.
    """
    if country_code not in COUNTRY_NAMES:
        raise ValueError(
            f"Unsupported country code: {country_code}. "
            f"Supported: {list(COUNTRY_NAMES.keys())}"
        )
    
    country_name = COUNTRY_NAMES[country_code]
    
    if not os.path.exists(CSV_SRC):
        raise SystemExit(f"Source CSV not found: {CSV_SRC}")
    
    # Get prompting strategy (pass num_examples for few_shot)
    strategy = get_strategy(strategy_name, num_examples)
    print(f"\n{'='*70}")
    print(f"Running experiment for {country_name} ({country_code})")
    print(f"Strategy: {strategy_name}")
    if strategy_name == 'few_shot' and num_examples:
        print(f"Few-shot examples: {num_examples}")
    print(f"Sample size: {sample_size}")
    print(f"{'='*70}\n")
    
    # Data preparation (same as before)
    df_all = pd.read_csv(CSV_SRC)
    df_country = extract_country_rows(CSV_SRC, country_name)
    
    # Persist extracted country-specific CSV for auditing and reuse
    # Note: paths_for_country now takes sample_size as string and num_examples for few_shot
    paths = paths_for_country(country_code, strategy_name, str(sample_size), num_examples)
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
    
    # Check if sample file already exists (for consistent cross-model comparison)
    sample_path = paths['sample_path']
    if os.path.exists(sample_path):
        print(f"Reusing existing sample file for cross-model consistency: {sample_path}")
        df_test = pd.read_csv(sample_path)
        print(f"Loaded {len(df_test)} events from existing sample")
    else:
        # Build stratified sample
        n_total = min(sample_size, len(usable))
        
        # Log sampling configuration
        if primary_group:
            print(f"Using targeted sampling: {primary_share*100:.0f}% {primary_group}, "
                  f"{(1-primary_share)*100:.0f}% proportional to other classes")
        else:
            print("Using proportional sampling: sample reflects natural class distribution")
        
        df_test = build_stratified_sample(
            usable,
            stratify_col='event_type',
            n_total=n_total,
            primary_group=primary_group,
            primary_share=primary_share,
            label_map=LABEL_MAP,
            random_state=42,
            replace=False
        )
        
        sample_path = write_sample(country_code, df_test)
        print(f"Wrote stratified sample to {sample_path}")
    
    print(df_test.head())
    
    # Run classification with strategy
    # Priority: explicit `models` argument -> OLLAMA_MODELS env var -> WORKING_MODELS constant
    if models is None:
        env_models = os.environ.get('OLLAMA_MODELS')
        if env_models:
            models = [m.strip() for m in env_models.split(',') if m.strip()]
        else:
            models = WORKING_MODELS
    
    results = []
    subset = df_test.copy()
    print(f"\nStarting classification with {strategy_name} strategy:")
    print(f"  - {len(subset)} events")
    print(f"  - {len(models)} models\n")
    
    # Setup results directory (includes strategy, sample_size, and num_examples for few_shot)
    _, results_dir = setup_country_environment(country_code, strategy_name, str(sample_size), num_examples)
    
    for m in models:
        print(f"Starting model: {m}")
        model_results = run_model_on_rows_with_strategy(m, subset, strategy)
        results.extend(model_results)
        print(f"Model {m} completed.")
        
        # Save per-model results immediately (allows incremental runs)
        model_df = pd.DataFrame(model_results)
        model_out_path = get_per_model_result_path(
            country_code, m, results_dir, 
            strategy=strategy_name, 
            sample_size=str(sample_size)
        )
        model_df.to_csv(model_out_path, index=False)
        print(f"Saved {m} results to: {model_out_path}")
    
    res_df = pd.DataFrame(results)
    
    # Note: Per-model files already saved above. 
    # The combined file will be created by result_aggregator before analysis phases.
    
    print(f"\n{'='*70}")
    print(f"Experiment completed!")
    print(f"Per-model results saved to: {results_dir}/ollama_results_*_acled_{country_code}_state_actors.csv")
    print(f"Run 'python -m lib.core.result_aggregator' to combine results for analysis.")
    print(f"{'='*70}\n")
    print(res_df.head(5))
    
    return results_dir


def main():
    """Main entry point - accepts country, strategy from command line or environment."""
    parser = argparse.ArgumentParser(
        description='Run classification experiment with configurable sampling'
    )
    parser.add_argument('country', nargs='?', default=os.environ.get('COUNTRY', 'cmr'),
                       help='Country code (e.g., cmr, nga)')
    parser.add_argument('--sample-size', type=int, 
                       default=int(os.environ.get('SAMPLE_SIZE', '100')),
                       help='Number of events to sample (default: 100)')
    parser.add_argument('--strategy', default=os.environ.get('STRATEGY', 'zero_shot'),
                       help='Prompting strategy: zero_shot, few_shot, explainable (default: zero_shot)')
    parser.add_argument('--primary-group', default=None,
                       help='Event type to oversample (e.g., "Violence against civilians"). '
                            'Default: None (proportional sampling)')
    parser.add_argument('--primary-share', type=float, default=0.0,
                       help='Fraction for primary group (0-1). Only used if --primary-group is set. '
                            'Default: 0.0')
    parser.add_argument('--models', default=os.environ.get('OLLAMA_MODELS', None),
                       help='Comma-separated list of Ollama models to run. Overrides WORKING_MODELS. '
                           'Example: --models "llama3.1:8b,mistral:7b"')
    parser.add_argument('--num-examples', type=int, default=None,
                       help='Number of few-shot examples (1-5). Only used with --strategy few_shot. '
                            'Default: reads from NUM_EXAMPLES env var, or 1 if not set.')
    
    args = parser.parse_args()
    
    # Validate primary_share
    if args.primary_share < 0 or args.primary_share > 1:
        parser.error('--primary-share must be between 0 and 1')
    
    if args.primary_group and args.primary_share == 0:
        parser.error('--primary-share must be > 0 when --primary-group is specified')
    
    # Validate num_examples
    if args.num_examples is not None:
        if not 1 <= args.num_examples <= 5:
            parser.error('--num-examples must be between 1 and 5')
        if args.strategy != 'few_shot':
            parser.error('--num-examples is only valid with --strategy few_shot')
    
    # Run the experiment
    models_arg = None
    if args.models:
        models_arg = [m.strip() for m in args.models.split(',') if m.strip()]

    run_classification_experiment(
        country_code=args.country,
        sample_size=args.sample_size,
        strategy_name=args.strategy,
        primary_group=args.primary_group,
        primary_share=args.primary_share,
        num_examples=args.num_examples,
        models=models_arg
    )


if __name__ == "__main__":
    main()
