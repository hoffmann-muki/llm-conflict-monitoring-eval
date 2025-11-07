"""Column-oriented data extraction helpers.
"""
from __future__ import annotations
import re
import pandas as pd

def unique_values_for_column(csv_path: str, column: str = "ACTOR1", nrows: int | None = None) -> list:
    """Return sorted unique values for any column (case-insensitive match).
    Args:
        csv_path: path to the CSV file.
        column: column name to retrieve uniques for (case-insensitive).
        nrows: optional number of rows to read for a faster check.
    Returns:
        Sorted list of unique strings (empty string for missing values).
    """
    df = pd.read_csv(csv_path, nrows=nrows)
    cols_lower = {c.lower(): c for c in df.columns}
    found = cols_lower.get(column.lower())
    if found is None:
        matches = [c for c in df.columns if c.lower() == column.lower()]
        if matches:
            found = matches[0]
    if found is None:
        raise ValueError(f"Column '{column}' not found in {csv_path}; available columns: {df.columns.tolist()}")
    uniques = pd.Series(df[found].fillna("")).astype(str).str.strip().unique()
    return sorted(uniques)

def unique_values_for_actor1(csv_path: str, nrows: int | None = None) -> list:
    """Backward-compatible wrapper returning unique `ACTOR1` values."""
    return unique_values_for_column(csv_path, column="ACTOR1", nrows=nrows)

def strip_parens(s: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", str(s)).strip()

def is_state_actor(name: str, country: str) -> bool:
    s = strip_parens(name).lower()
    return (f"military forces of {country}" in s) or (f"police forces of {country}" in s)

def extract_state_actor(dataframe: pd.DataFrame, country: str, actor_col: str = "actor1") -> pd.Series:
    """Extracts a boolean Series indicating state actors from the dataframe.
    Parameters
    - dataframe: input DataFrame
    - country: country name used to detect 'military forces of {country}' and 'police forces of {country}'
    - actor_col: column name to use for actor strings (case-insensitive)
    """
    if actor_col not in dataframe.columns:
        cols_lower = {c.lower(): c for c in dataframe.columns}
        actor_col = cols_lower.get(actor_col.lower(), actor_col)
    return dataframe[actor_col].fillna("").apply(is_state_actor, country=country)

def _default_actor_groups(country: str = "Cameroon"):
    """Return the default ordered list of (label, matcher_fn) tuples for a given country.
    The `country` argument is used to build the state-forces matcher and label so callers
    can get country-specific grouping (e.g. 'State forces (Nigeria)').
    """
    country_l = country.lower()
    state_label = f"State forces ({country})"
    state_matcher = lambda s: (f"military forces of {country_l}" in s) or (f"police forces of {country_l}" in s)
    return [
        ("Communal Militia", lambda s: "communal militia" in s),
        ("Ambazonia", lambda s: "ambazon" in s or "ambazonia" in s),
        ("Boko Haram / ISWAP", lambda s: any(k in s for k in ("boko haram", "iswap", "islamic state west africa"))),
        (state_label, state_matcher),
        ("Protesters/Rioters/Civilians", lambda s: any(k in s for k in ("protest", "riot", "civilian"))),
        ("MNJTF", lambda s: "mnjtf" in s or "multinational joint task force" in s),
        ("Pirates", lambda s: "pirate" in s),
    ]

def normalize_actor1(s: str, groups: list | None = None, country: str | None = None) -> str:
    """Normalize a single ACTOR1 string into canonical actor groups.
    Parameters
    - s: raw actor string
        - groups: optional ordered list of (label, matcher) where matcher is a callable taking the lowercased string and returning bool.
            If None, uses the built-in default groups for `country` (defaults to Cameroon when country is None).
        - country: optional country name to select country-specific default groups (only used when `groups` is None).
    """
    if s is None:
        return "Other Actors"
    s0 = strip_parens(s)
    sl = s0.lower()
    if groups is None:
                groups = _default_actor_groups(country or "Cameroon")
    for label, matcher in groups:
        try:
            if matcher(sl):
                return label
        except Exception:
            # If a custom matcher fails, skip it
            continue
    return "Other Actors"

def get_actor_norm_series(df: pd.DataFrame, actor_col: str = "actor1", groups: list | None = None, country: str | None = None) -> pd.Series:
    """Return a pandas Series with normalized actor group for each row.
    Parameters
    - df: input DataFrame
    - actor_col: column name to use (default 'actor1')
    - groups: optional ordered list of (label, matcher) to customize grouping
    """
    if actor_col not in df.columns:
        # try case-insensitive match
        cols_lower = {c.lower(): c for c in df.columns}
        actor_col = cols_lower.get(actor_col.lower(), actor_col)
    # Pass country through to normalize_actor1 so default groups are country-specific when groups is None
    return df[actor_col].fillna("").astype(str).apply(lambda s: normalize_actor1(s, groups=groups, country=country))
