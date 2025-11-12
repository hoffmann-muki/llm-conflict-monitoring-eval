import pandas as pd
import os
from tools.ollama_helpers import run_model_on_rows
from country_data_exploration import extract_country_rows
from column_data_extraction import get_actor_norm_series, extract_state_actor
from input_data_extraction import build_stratified_sample
from tools.constants import LABEL_MAP, EVENT_CLASSES_FULL, CSV_SRC, WORKING_MODELS
from tools.data_helpers import paths_for_country, resolve_columns, write_sample

if not os.path.exists(CSV_SRC):
    raise SystemExit(f"Source CSV not found: {CSV_SRC}")

df_all = pd.read_csv(CSV_SRC)
df_nga = extract_country_rows(CSV_SRC, "Nigeria")

# Persist extracted country-specific CSV for auditing and reuse under datasets/nga/
paths = paths_for_country('nga')
os.makedirs(paths['datasets_dir'], exist_ok=True)
out_country = os.path.join(paths['datasets_dir'], "Nigeria_lagged_data_up_to-2024-10-24.csv")
df_nga.to_csv(out_country, index=False)
print(f"Wrote extracted Nigeria data to {out_country}")

# Resolve column names case-insensitively to match ACLED casing differences
cols = resolve_columns(df_nga, ['actor1', 'notes', 'event_type', 'event_id_cnty'])
col_actor = cols.get('actor1') or 'actor1'
col_notes = cols.get('notes') or 'notes'
col_event_type = cols.get('event_type') or 'event_type'
col_event_id = cols.get('event_id_cnty') or 'event_id_cnty'

# Create normalized actor column using the configurable helper
df_nga["actor_norm"] = get_actor_norm_series(df_nga, actor_col=col_actor)

# Create state_actor boolean using helper (country-specific)
df_nga["state_actor"] = extract_state_actor(df_nga, country="nigeria", actor_col=col_actor)

# Keep only state-actor rows with valid event types and notes
usable = (
    df_nga.loc[
        df_nga["state_actor"]
            & df_nga[col_notes].notna()
            & df_nga[col_event_type].isin(EVENT_CLASSES_FULL),
            [col_event_id, col_notes, col_event_type, "actor_norm"]
    ]
    .rename(columns={col_event_id: "event_id_cnty", col_notes: "notes", col_event_type: "event_type"})
    .assign(notes=lambda x: x["notes"].str.replace(r"\s+", " ", regex=True).str.slice(0, 400))
    .drop_duplicates(subset=["event_id_cnty"])
)

print(f"Usable state-actor rows found (Nigeria): {len(usable):,}")

# Build stratified sample using the generic sampler (default: 100 samples) #
SAMPLE_SIZE = int(os.environ.get('SAMPLE_SIZE', '100'))
N_TOTAL = min(SAMPLE_SIZE, len(usable))
df_test = build_stratified_sample(
    usable,
    stratify_col='event_type',
    n_total=N_TOTAL,
    primary_group='Violence against civilians',
    primary_share=0.6,
    label_map=LABEL_MAP,
    random_state=42,
    replace=False
)

SAMPLE_PATH = write_sample('nga', df_test)
print(f"Wrote stratified sample to {SAMPLE_PATH}")
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
COUNTRY = os.environ.get('COUNTRY', 'nga')
RESULTS_DIR = f'results/{COUNTRY}'
os.makedirs(RESULTS_DIR, exist_ok=True)
out_path = os.path.join(RESULTS_DIR, f"ollama_results_acled_{COUNTRY}_state_actors.csv")
res_df.to_csv(out_path, index=False)
print(f"\nSaved final predictions to: {out_path}")
print(res_df.head(5))
