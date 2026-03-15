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
import re

import torch

# Strategy helper imported from the core helpers
from lib.core.strategy_helpers import get_strategy
from lib.core.constants import COUNTRY_NAMES

# Import from lib structure
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from lib.data_preparation import extract_country_rows, get_actor_norm_series, extract_state_actor, build_stratified_sample, build_balanced_actor_sample
from lib.core.constants import LABEL_MAP, EVENT_CLASSES_FULL, CSV_SRC, WORKING_MODELS, COUNTRY_NAMES as _COUNTRY_NAMES
from lib.inference.ollama_client import run_ollama_structured
from lib.inference.hf_causal_client import (
    _extract_json_object,
    _build_generation_prompt,
    _load_hf_runtime,
    is_hf_inference_model,
    resolve_hf_model_path,
    resolve_hf_device,
    resolve_hf_max_new_tokens,
)
from lib.core.data_helpers import paths_for_country, resolve_columns, write_sample, setup_country_environment
from lib.core.result_aggregator import model_name_to_slug, get_per_model_result_path

# get_strategy and COUNTRY_NAMES are provided by lib.core.constants


VALID_LABELS = {"V", "B", "E", "P", "R", "S"}

def _normalize_label(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "ERROR"
    upper = raw.upper()
    if upper in VALID_LABELS:
        return upper
    return LABEL_MAP.get(raw, LABEL_MAP.get(raw.title(), "ERROR"))


def run_model_on_rows_with_strategy(model_name: str, rows, strategy,
                                    note_col: str = 'notes',
                                    event_id_col: str = 'event_id_cnty',
                                    true_label_col: str = 'gold_label',
                                    actor_norm_col: str = 'actor_norm',
                                    use_hf: bool = False):
    """Run model on rows using specified prompting strategy.
    
    This is a strategy-aware version of run_model_on_rows that uses
    the strategy's make_prompt() method.
    
    Args:
        model_name: Name of the Ollama model or HF-backed model
        rows: DataFrame rows to classify
        strategy: PromptingStrategy instance
        note_col: Column name for event notes
        event_id_col: Column name for event ID
        true_label_col: Column name for true label
        actor_norm_col: Column name for normalized actor
        use_hf: Whether to use HF inference backend
        
    Returns:
        List of result dictionaries
    """
    results = []
    hf_runtime = None
    hf_device = None
    hf_max_new_tokens = None
    
    if use_hf:
        hf_device = resolve_hf_device()
        hf_max_new_tokens = resolve_hf_max_new_tokens(default=96)
        model_path = resolve_hf_model_path(model_name)
        print(f"Loading HF checkpoint for {model_name}: {model_path}")
        hf_runtime = _load_hf_runtime(model_path, hf_device)

    for r in rows.itertuples(index=False):
        t0 = time.time()
        note = None
        try:
            note = getattr(r, note_col)
            # Generate strategy-specific prompt
            prompt = strategy.make_prompt(note)
            system_msg = strategy.get_system_message()
            if use_hf:
                tokenizer, model = hf_runtime  # type: ignore[misc]
                generation_prompt = _build_generation_prompt(prompt, system_msg)
                inputs = tokenizer(generation_prompt, return_tensors="pt").to(hf_device)

                with torch.no_grad():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=hf_max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                gen_tokens = generated[0][inputs["input_ids"].shape[1]:]
                raw_output = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                resp = _extract_json_object(raw_output)
                if resp is None:
                    label_match = re.search(r'"label"\s*:\s*"([VBEPRS])"', raw_output)
                    conf_match = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', raw_output)
                    resp = {
                        "label": label_match.group(1) if label_match else "ERROR",
                        "confidence": float(conf_match.group(1)) if conf_match else 0.0,
                    }
            else:
                # Run with strategy prompt and system message via Ollama API
                resp = run_ollama_structured(
                    model_name,
                    prompt=prompt,
                    system_msg=system_msg,
                    schema=strategy.get_schema()
                )

            label = _normalize_label(resp.get("label", "ERROR"))
            conf = float(resp.get("confidence", 0))
            logits = None
            for k in ("logits", "log_probs", "scores", "label_scores"):
                if k in resp:
                    logits = resp.get(k)
                    break
            # Capture reasoning for explainable strategy
            reasoning = resp.get("reasoning") if "reasoning" in resp else None
        except Exception as e:
            label = "ERROR"
            conf = 0.0
            logits = None
            reasoning = None
            print(f"Error classifying event: {e}")
        
        elapsed = round(time.time() - t0, 2)
        results.append({
            "model": model_name,
            "event_id": getattr(r, event_id_col, None),
            "true_label": getattr(r, true_label_col, None),
            "pred_label": label,
            "pred_conf": conf,
            "logits": json.dumps(logits) if logits is not None else None,
            "reasoning": json.dumps(reasoning) if reasoning is not None else None,
            "latency_sec": elapsed,
            "actor_norm": getattr(r, actor_norm_col, None),
            "notes": note  # Include notes for downstream counterfactual analysis
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
        ['actor1', 'notes', 'event_type', 'event_id_cnty', 'inter1']
    )
    col_actor = cols.get('actor1') or 'actor1'
    col_notes = cols.get('notes') or 'notes'
    col_event_type = cols.get('event_type') or 'event_type'
    col_event_id = cols.get('event_id_cnty') or 'event_id_cnty'
    col_inter1 = cols.get('inter1') or 'INTER1'
    
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
        # Build balanced actor sample for fairness analysis
        n_total = min(sample_size, len(usable))
        
        # Log sampling configuration
        if primary_group:
            print(f"Using balanced actor sampling with primary event: {primary_group}")
            print(f"  {primary_share*100:.0f}% {primary_group}, {(1-primary_share)*100:.0f}% other classes")
            print(f"  Ensuring equal representation of state and non-state actors")
        else:
            print("Using balanced actor sampling: 50% state actors, 50% non-state actors")
            print("  Sample stratified by event type within each actor group")
        
        df_test = build_balanced_actor_sample(
            df_country,
            n_total=n_total,
            balance_ratio=0.5,
            event_types=EVENT_CLASSES_FULL,
            event_col=col_event_type,
            actor_code_col=col_inter1,  # ACLED INTER1 column (resolved)
            min_per_cell=10,
            primary_event=primary_group,
            primary_share=primary_share if primary_group else None,
            label_map=LABEL_MAP,
            random_state=42,
            verbose=True
        )
        
        sample_path = write_sample(country_code, df_test, sample_size=str(sample_size))
        print(f"Wrote balanced actor sample to {sample_path}")
    
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
        use_hf = is_hf_inference_model(m)
        if use_hf:
            device = resolve_hf_device()
            print(f"  Inference backend: transformers (HF local checkpoint) on {device}")
        else:
            print("  Inference backend: Ollama API")

        model_results = run_model_on_rows_with_strategy(
            m,
            subset,
            strategy,
            use_hf=use_hf,
        )
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
    print(f"Per-model results saved to: {results_dir}/ollama_results_*_acled_{country_code}_actors.csv")
    print(f"Run 'python -m lib.core.result_aggregator' to combine results for analysis.")
    print(f"{'='*70}\n")
    print(res_df.head(5))
    
    return results_dir


def main():
    """Main entry point - accepts country, strategy from command line or environment."""
    # Read environment variable defaults (shell script sets these)
    env_country = os.environ.get('COUNTRY', 'cmr')
    env_sample_size = int(os.environ.get('SAMPLE_SIZE', '100'))
    env_strategy = os.environ.get('STRATEGY', 'zero_shot')
    # Only read NUM_EXAMPLES from env if strategy is few_shot (shell always sets it)
    env_num_examples = None
    if env_strategy == 'few_shot':
        env_num_examples_str = os.environ.get('NUM_EXAMPLES')
        env_num_examples = int(env_num_examples_str) if env_num_examples_str else None

    parser = argparse.ArgumentParser(
        description='Run classification experiment with configurable sampling'
    )
    parser.add_argument('country', nargs='?', default=env_country,
                       help=f'Country code (e.g., cmr, nga). Default: {env_country} (from COUNTRY env var)')
    parser.add_argument('--sample-size', type=int, default=env_sample_size,
                       help=f'Number of events to sample (default: {env_sample_size} from SAMPLE_SIZE env var)')
    parser.add_argument('--strategy', default=env_strategy,
                       help=f'Prompting strategy: zero_shot, few_shot, explainable (default: {env_strategy} from STRATEGY env var)')
    parser.add_argument('--primary-group', default=None,
                       help='Event type to oversample (e.g., "Violence against civilians"). '
                            'Default: None (proportional sampling)')
    parser.add_argument('--primary-share', type=float, default=0.0,
                       help='Fraction for primary group (0-1). Only used if --primary-group is set. '
                            'Default: 0.0')
    parser.add_argument('--models', default=None,
                       help='Comma-separated list of Ollama models to run. Overrides WORKING_MODELS. '
                           'Example: --models "llama3.2:3b,mistral:7b"')
    parser.add_argument('--num-examples', type=int, default=env_num_examples,
                       help='Number of few-shot examples (1-5). Only used with --strategy few_shot. '
                            'Default: from NUM_EXAMPLES env var (only when strategy=few_shot).')
    
    args = parser.parse_args()
    
    # Validate primary_share
    if args.primary_share < 0 or args.primary_share > 1:
        parser.error('--primary-share must be between 0 and 1')
    
    if args.primary_group and args.primary_share == 0:
        parser.error('--primary-share must be > 0 when --primary-group is specified')
    
    # Validate num_examples - only check if explicitly provided
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
        models=models_arg,
    )


if __name__ == "__main__":
    main()
