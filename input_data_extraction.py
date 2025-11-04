"""Generic input data extraction helpers.
"""
from __future__ import annotations
import math
import pandas as pd

def _resolve_col(df: pd.DataFrame, col: str) -> str:
    if col in df.columns:
        return col
    cols_lower = {c.lower(): c for c in df.columns}
    return cols_lower.get(col.lower(), col)

def _largest_remainder_alloc(counts: dict[str, int], n_alloc: int) -> dict[str, int]:
    """Allocate n_alloc slots to groups proportionally using largest-remainder.
    counts: mapping group -> available count (positive ints). If sum(counts)==0,
    returns zeros.
    """
    total = sum(counts.values())
    if total == 0 or n_alloc <= 0:
        return {g: 0 for g in counts}
    # compute ideals
    ideals = {g: (counts[g] / total) * n_alloc for g in counts}
    floor_alloc = {g: math.floor(ideals[g]) for g in counts}
    alloc = dict(floor_alloc)
    remainder = n_alloc - sum(alloc.values())
    if remainder > 0:
        # sort by fractional part descending
        remainders = sorted(counts.keys(), key=lambda g: (ideals[g] - math.floor(ideals[g])), reverse=True)
        for g in remainders[:remainder]:
            alloc[g] += 1
    return alloc

def build_stratified_sample(
    df: pd.DataFrame,
    stratify_col: str = "event_type",
    n_total: int = 100,
    primary_group: str | None = "Violence against civilians",
    primary_share: float = 0.6,
    label_map: dict | None = None,
    random_state: int | None = 42,
    replace: bool = False,
    keep_columns: list | None = None,
) -> pd.DataFrame:
    """Return a stratified sample from `df`.
    Args:
        df: source DataFrame.
        stratify_col: column to stratify by (case-insensitive lookup).
        n_total: desired total sample size.
        primary_group: optional group to reserve `primary_share` for.
        primary_share: fraction (0..1) of n_total reserved for primary_group.
        label_map: optional mapping from full label -> short label; if provided,
            the output will include a `gold_label` column with mapped values.
        random_state: seed for reproducible sampling.
        replace: sample with replacement when a group lacks enough rows.
        keep_columns: list of columns to return; default will include common ones.
    Returns:
        A DataFrame with sampled rows, shuffled.
    """
    strat_col = _resolve_col(df, stratify_col)
    if strat_col not in df.columns:
        raise ValueError(f"Stratify column '{stratify_col}' not found in DataFrame")
    if keep_columns is None:
        # choose canonical columns but resolve case-insensitively
        want = ["event_id_cnty", "notes", strat_col, "actor_norm"]
        cols_lower = {c.lower(): c for c in df.columns}
        keep_columns = [cols_lower.get(c.lower(), c) for c in want if cols_lower.get(c.lower(), c) in df.columns]

    # compute available counts per group
    counts = df[strat_col].fillna("").astype(str).value_counts().to_dict()

    # determine primary allocation
    n_total = min(n_total, len(df))
    n_primary = 0
    if primary_group is not None:
        available_primary = counts.get(primary_group, 0)
        desired_primary = int(math.floor(n_total * primary_share))
        n_primary = min(desired_primary, available_primary)
    n_other = n_total - n_primary

    # sample primary group
    samples = []
    if n_primary > 0:
        primary_df = df[df[strat_col].astype(str) == primary_group]
        samples.append(primary_df.sample(n=n_primary, replace=replace, random_state=random_state))

    # prepare other groups (exclude primary_group)
    other_counts = {g: counts[g] for g in counts if g != primary_group}
    if n_other > 0 and other_counts:
        alloc = _largest_remainder_alloc(other_counts, n_other)

        # Now sample according to alloc, clamping to available if replace=False
        unfilled = 0
        other_samples = []
        for g, n_req in alloc.items():
            grp_df = df[df[strat_col].astype(str) == g]
            avail = len(grp_df)
            if n_req <= 0:
                continue
            if n_req <= avail:
                other_samples.append(grp_df.sample(n=n_req, replace=False, random_state=random_state))
            else:
                if replace:
                    other_samples.append(grp_df.sample(n=n_req, replace=True, random_state=random_state))
                else:
                    # take all available and mark deficit
                    other_samples.append(grp_df.sample(n=avail, replace=False, random_state=random_state))
                    unfilled += (n_req - avail)

        # If there are unfilled slots and replacement is False, attempt to redistribute
        if unfilled > 0 and not replace:
            # find groups with spare capacity
            spare = {g: max(0, other_counts[g] - alloc.get(g, 0)) for g in other_counts}
            spare_total = sum(spare.values())
            if spare_total > 0:
                extra_alloc = _largest_remainder_alloc(spare, unfilled)
                for g, extra in extra_alloc.items():
                    if extra <= 0:
                        continue
                    grp_df = df[df[strat_col].astype(str) == g]
                    avail = len(grp_df)
                    take_n = min(extra, max(0, avail - alloc.get(g, 0)))
                    if take_n > 0:
                        other_samples.append(grp_df.sample(n=take_n, replace=False, random_state=random_state))
                        unfilled -= take_n
        samples.extend(other_samples)

    # concat samples, shuffle, prepare output
    if len(samples) == 0:
        return pd.DataFrame(columns=keep_columns)
    result = pd.concat(samples, ignore_index=True)
    result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # rename strat_col to gold_label_full for compatibility if present in result
    # (resolve case-insensitively)
    cols_lower = {c.lower(): c for c in result.columns}
    strat_actual = cols_lower.get(strat_col.lower())
    if strat_actual and strat_actual in result.columns:
        result = result.rename(columns={strat_actual: 'gold_label_full'})
    if label_map:
        result['gold_label'] = result['gold_label_full'].map(label_map)
    desired = ['event_id_cnty', 'notes', 'gold_label', 'gold_label_full', 'actor_norm']
    # resolve desired against result columns case-insensitively
    res_cols_lower = {c.lower(): c for c in result.columns}
    final_cols = [res_cols_lower.get(c.lower()) for c in desired if res_cols_lower.get(c.lower()) in result.columns]
    return result.loc[:, final_cols]
