import pandas as pd
from country_data_exploration import extract_country_rows
from column_data_extraction import get_actor_norm_series, extract_state_actor
import os

# ACLED top-level event types we’ll classify between:
EVENT_CLASSES_FULL = [
    "Violence against civilians",
    "Battles",
    "Explosions/Remote violence",
    "Protests",
    "Riots",
    "Strategic developments"
]

CSV_SRC = "datasets/Africa_lagged_data_up_to-2024-10-24.csv"
OUT_CAM = "datasets/Cameroon_lagged_data_up_to-2024-10-24.csv"

if not os.path.exists(CSV_SRC):
    raise SystemExit(f"Source CSV not found: {CSV_SRC}")

df_all = pd.read_csv(CSV_SRC)
df_cam = extract_country_rows(CSV_SRC, "Cameroon")

# Resolve column names case-insensitively to match ACLED casing differences
cols_lower = {c.lower(): c for c in df_cam.columns}
col_actor = cols_lower.get('actor1', 'actor1')
col_notes = cols_lower.get('notes', 'notes')
col_event_type = cols_lower.get('event_type', 'event_type')
col_event_id = cols_lower.get('event_id_cnty', 'event_id_cnty')

# Create normalized actor column using the configurable helper
df_cam["actor_norm"] = get_actor_norm_series(df_cam, actor_col=col_actor)

# Create state_actor boolean using helper (country-specific)
df_cam["state_actor"] = extract_state_actor(df_cam, country="cameroon", actor_col=col_actor)

# Keep only state-actor rows with valid event types and notes
usable = (
    df_cam.loc[
        df_cam["state_actor"]
            & df_cam[col_notes].notna()
            & df_cam[col_event_type].isin(EVENT_CLASSES_FULL),
            [col_event_id, col_notes, col_event_type, "actor_norm"]
    ]
    .rename(columns={col_event_id: "event_id_cnty", col_notes: "notes", col_event_type: "event_type"})
    .assign(notes=lambda x: x["notes"].str.replace(r"\s+", " ", regex=True).str.slice(0, 400))
    .drop_duplicates(subset=["event_id_cnty"])
)

print(f"Usable state-actor rows found (Cameroon): {len(usable):,}")
