"""Small helper to extract country-specific rows from the Africa lagged dataset.
Usage: import `extract_country_rows` from this module to get country-specific dataframes.
"""
from __future__ import annotations
import os
import pandas as pd

def extract_country_rows(csv_path: str, country: str, country_col: str = "country") -> pd.DataFrame:
    """Load a CSV and return rows where `country_col` equals `country` (case-insensitive).
    Args:
        csv_path: path to the Africa lagged CSV.
        country: country name to filter, e.g. "Cameroon".
        country_col: column name to match against (default: 'country').
    Returns:
        pd.DataFrame with only rows for the requested country.
    """
    df = pd.read_csv(csv_path)
    # Auto-detect country column if the provided country_col is not present.
    cols_lower = {c.lower(): c for c in df.columns}
    if country_col not in df.columns:
        if country_col.lower() in cols_lower:
            country_col = cols_lower[country_col.lower()]
        else:
            # try common alternatives
            for alt in ("country", "COUNTRY", "Country"):
                if alt in df.columns:
                    country_col = alt
                    break
    if country_col not in df.columns:
        raise ValueError(f"Column '{country_col}' not found in CSV: columns={df.columns.tolist()}")
    # Normalize and filter (case-insensitive, strip whitespace)
    mask = df[country_col].fillna("").astype(str).str.strip().str.lower() == country.strip().lower()
    return df.loc[mask].reset_index(drop=True)
