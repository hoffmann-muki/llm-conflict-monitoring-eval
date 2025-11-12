#!/usr/bin/env python3
"""Generic country pipeline for political bias analysis."""

import pandas as pd
import os
import sys
from tools.ollama_helpers import run_model_on_rows
from country_data_exploration import extract_country_rows
from column_data_extraction import get_actor_norm_series, extract_state_actor
from input_data_extraction import build_stratified_sample
from tools.constants import LABEL_MAP, EVENT_CLASSES_FULL, CSV_SRC, WORKING_MODELS
from tools.data_helpers import paths_for_country, resolve_columns, write_sample, setup_country_environment

# Country name mapping
COUNTRY_NAMES = {
    'cmr': 'Cameroon',
    'nga': 'Nigeria'
}

def run_country_pipeline(country_code: str, sample_size: int = 100):
    """Run the political bias analysis pipeline for a specific country.
    
    Args:
        country_code: Country code (e.g., 'cmr', 'nga')
        sample_size: Number of samples to generate
    """
    if country_code not in COUNTRY_NAMES:
        raise ValueError(f"Unsupported country code: {country_code}. Supported: {list(COUNTRY_NAMES.keys())}")
    
    country_name = COUNTRY_NAMES[country_code]
    
    if not os.path.exists(CSV_SRC):
        raise SystemExit(f"Source CSV not found: {CSV_SRC}")

    print(f"Running pipeline for {country_name} ({country_code})")
    
    df_all = pd.read_csv(CSV_SRC)
    df_country = extract_country_rows(CSV_SRC, country_name)

    # Persist extracted country-specific CSV for auditing and reuse
    paths = paths_for_country(country_code)
    os.makedirs(paths['datasets_dir'], exist_ok=True)
    out_country = os.path.join(paths['datasets_dir'], f"{country_name}_lagged_data_up_to-2024-10-24.csv")
    df_country.to_csv(out_country, index=False)
    print(f"Wrote extracted {country_name} data to {out_country}")

    # Resolve column names case-insensitively to match ACLED casing differences
    cols = resolve_columns(df_country, ['actor1', 'notes', 'event_type', 'event_id_cnty'])
    col_actor = cols.get('actor1') or 'actor1'
    col_notes = cols.get('notes') or 'notes'
    col_event_type = cols.get('event_type') or 'event_type'
    col_event_id = cols.get('event_id_cnty') or 'event_id_cnty'

    # Create normalized actor column using the configurable helper
    df_country["actor_norm"] = get_actor_norm_series(df_country, actor_col=col_actor)

    # Create state_actor boolean using helper (country-specific)
    df_country["state_actor"] = extract_state_actor(df_country, country=country_name.lower(), actor_col=col_actor)

    # Keep only state-actor rows with valid event types and notes
    usable = (
        df_country.loc[
            df_country["state_actor"]
                & df_country[col_notes].notna()
                & df_country[col_event_type].isin(EVENT_CLASSES_FULL),
                [col_event_id, col_notes, col_event_type, "actor_norm"]
        ]
        .rename(columns={col_event_id: "event_id_cnty", col_notes: "notes", col_event_type: "event_type"})
        .assign(notes=lambda x: x["notes"].str.replace(r"\s+", " ", regex=True).str.slice(0, 400))
        .drop_duplicates(subset=["event_id_cnty"])
    )

    print(f"Usable state-actor rows found ({country_name}): {len(usable):,}")

    # Build stratified sample using the generic sampler
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

    # Minimal multi-model classification with Ollama
    models = WORKING_MODELS

    results = []
    subset = df_test.copy()
    print(f"Starting classification on {len(subset)} rows with {len(models)} models (serial execution).")

    for m in models:
        print(f"Starting model: {m}")
        model_results = run_model_on_rows(m, subset)
        results.extend(model_results)
        print(f"Model {m} completed.")

    res_df = pd.DataFrame(results)
    
    # Setup results directory and save
    _, results_dir = setup_country_environment(country_code)
    out_path = os.path.join(results_dir, f"ollama_results_acled_{country_code}_state_actors.csv")
    res_df.to_csv(out_path, index=False)
    print(f"\nSaved final predictions to: {out_path}")
    print(res_df.head(5))

def main():
    """Main entry point - accepts country from command line or environment."""
    if len(sys.argv) > 1:
        country_code = sys.argv[1]
    else:
        country_code = os.environ.get('COUNTRY', 'cmr')
    
    sample_size = int(os.environ.get('SAMPLE_SIZE', '100'))
    
    # Run the main pipeline
    run_country_pipeline(country_code, sample_size)

if __name__ == "__main__":
    main()
