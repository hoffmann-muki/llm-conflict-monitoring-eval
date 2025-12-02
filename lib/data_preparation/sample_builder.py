"""Generic input data extraction helpers.
"""
from __future__ import annotations
import math
import warnings
from typing import Literal
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


def classify_actor_type(actor_code: int) -> Literal['state', 'non-state']:
    """Classify ACLED INTER1 code as state or non-state actor.
    
    ACLED INTER1 codes:
        1 = State Forces
        2 = Rebel Groups
        3 = Political Militias
        4 = Identity Militias
        5 = Rioters
        6 = Protesters
        7 = Civilians
        8 = External/Other Forces
    """
    return 'state' if actor_code == 1 else 'non-state'


def build_balanced_actor_sample(
    df: pd.DataFrame,
    n_total: int = 1000,
    balance_ratio: float = 0.5,
    event_types: list[str] | None = None,
    event_col: str = "EVENT_TYPE",
    actor_code_col: str = "INTER1",
    min_per_cell: int = 30,
    primary_event: str | None = "Violence against civilians",
    primary_share: float | None = None,
    label_map: dict | None = None,
    random_state: int | None = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build a balanced sample with equal state/non-state actors for fairness analysis.
    
    This function creates a sample stratified by both actor type (state vs non-state)
    and event type, enabling computation of fairness metrics like Statistical Parity
    Difference (SPD) and Equalized Odds.
    
    Args:
        df: Source DataFrame with ACLED data.
        n_total: Total desired sample size.
        balance_ratio: Proportion for state actors (0.5 = balanced). 
            E.g., 0.5 means 50% state, 50% non-state.
        event_types: List of event types to include. If None, auto-detects from data
            but excludes types with insufficient state actor data (Protests, Riots).
        event_col: Column name for event type (case-insensitive).
        actor_code_col: Column name for ACLED INTER1 actor code (case-insensitive).
        min_per_cell: Minimum samples required per (event_type × actor_type) cell.
            Cells below this threshold trigger warnings.
        primary_event: Optional event type to prioritize (e.g., "Violence against civilians").
        primary_share: If set, allocate this share to primary_event within each actor group.
            E.g., 0.6 means 60% of state samples are Violence, 60% of non-state are Violence.
        label_map: Mapping from full event type name to short label (e.g., {'Violence...': 'V'}).
        random_state: Seed for reproducible sampling.
        verbose: Print allocation summary.
    
    Returns:
        DataFrame with balanced sample, including columns:
        - event_id_cnty: Event identifier
        - notes: Event description text
        - gold_label: Short label (if label_map provided)
        - gold_label_full: Full event type name
        - actor_type: 'state' or 'non-state'
        - actor_norm: Normalized actor name (ACTOR1 from ACLED)
    
    Example:
        >>> sample = build_balanced_actor_sample(
        ...     df, n_total=1000, balance_ratio=0.5,
        ...     primary_event="Violence against civilians", primary_share=0.6
        ... )
        >>> print(sample['actor_type'].value_counts())
        state        500
        non-state    500
    """
    # Resolve column names (case-insensitive)
    event_col = _resolve_col(df, event_col)
    actor_code_col = _resolve_col(df, actor_code_col)
    
    if event_col not in df.columns:
        raise ValueError(f"Event column '{event_col}' not found in DataFrame")
    if actor_code_col not in df.columns:
        raise ValueError(f"Actor code column '{actor_code_col}' not found in DataFrame")
    
    # Create actor_type column
    df = df.copy()
    df['actor_type'] = df[actor_code_col].apply(classify_actor_type)
    
    # Auto-detect usable event types if not specified
    if event_types is None:
        # Find event types that have both state and non-state actors with sufficient data
        event_counts = pd.crosstab(df[event_col], df['actor_type'])
        usable_events = []
        for event in event_counts.index:
            state_n = int(event_counts.loc[event, 'state']) if 'state' in event_counts.columns else 0 # type: ignore
            nonstate_n = int(event_counts.loc[event, 'non-state']) if 'non-state' in event_counts.columns else 0 # type: ignore
            if state_n >= min_per_cell and nonstate_n >= min_per_cell:
                usable_events.append(event)
        
        if not usable_events:
            raise ValueError(f"No event types have sufficient data (>= {min_per_cell}) for both actor types")
        
        event_types = usable_events
        if verbose:
            print(f"Auto-detected usable event types: {event_types}")
    
    # Filter to selected event types
    df_filtered = df[df[event_col].isin(event_types)].copy()
    
    # Compute available counts per (event_type × actor_type) cell
    cell_counts = {}
    for event in event_types:
        for actor in ['state', 'non-state']:
            mask = (df_filtered[event_col] == event) & (df_filtered['actor_type'] == actor)
            cell_counts[(event, actor)] = mask.sum()
    
    # Determine allocation per actor group
    n_state = int(math.floor(n_total * balance_ratio))
    n_nonstate = n_total - n_state
    
    # Allocate within each actor group
    allocations = {'state': {}, 'non-state': {}}
    
    for actor, n_actor in [('state', n_state), ('non-state', n_nonstate)]:
        available = {event: cell_counts[(event, actor)] for event in event_types}
        
        if primary_event and primary_share and primary_event in available:
            # Reserve primary_share for primary event
            n_primary = int(math.floor(n_actor * primary_share))
            n_primary = min(n_primary, available[primary_event])
            n_other = n_actor - n_primary
            
            # Allocate remaining to other events proportionally
            other_available = {e: available[e] for e in event_types if e != primary_event}
            other_alloc = _largest_remainder_alloc(other_available, n_other)
            
            allocations[actor] = {primary_event: n_primary, **other_alloc}
        else:
            # Allocate proportionally to all events
            allocations[actor] = _largest_remainder_alloc(available, n_actor)
        
        # Clamp to available and warn if below minimum
        for event in event_types:
            requested = allocations[actor].get(event, 0)
            avail = available[event]
            if requested > avail:
                allocations[actor][event] = avail
                if verbose:
                    warnings.warn(f"Reduced {actor}/{event} from {requested} to {avail} (limited data)")
            if allocations[actor][event] < min_per_cell and allocations[actor][event] > 0:
                if verbose:
                    warnings.warn(f"Cell {actor}/{event} has only {allocations[actor][event]} samples (< {min_per_cell})")
    
    # Print allocation summary
    if verbose:
        print(f"\n=== Sample Allocation (n_total={n_total}, balance_ratio={balance_ratio}) ===")
        print(f"{'Event Type':<35} {'State':>8} {'Non-State':>10} {'Total':>8}")
        print("-" * 65)
        for event in event_types:
            s = allocations['state'].get(event, 0)
            ns = allocations['non-state'].get(event, 0)
            print(f"{event:<35} {s:>8} {ns:>10} {s+ns:>8}")
        total_s = sum(allocations['state'].values())
        total_ns = sum(allocations['non-state'].values())
        print("-" * 65)
        print(f"{'TOTAL':<35} {total_s:>8} {total_ns:>10} {total_s+total_ns:>8}")
        print()
    
    # Sample from each cell
    samples = []
    for actor in ['state', 'non-state']:
        for event in event_types:
            n_sample = allocations[actor].get(event, 0)
            if n_sample <= 0:
                continue
            
            cell_df = df_filtered[(df_filtered[event_col] == event) & 
                                   (df_filtered['actor_type'] == actor)]
            
            if len(cell_df) >= n_sample:
                sampled = cell_df.sample(n=n_sample, replace=False, random_state=random_state)
            else:
                # Take all available
                sampled = cell_df.sample(n=len(cell_df), replace=False, random_state=random_state)
            
            samples.append(sampled)
    
    if not samples:
        raise ValueError("No samples collected - check event_types and data availability")
    
    # Combine and shuffle
    result = pd.concat(samples, ignore_index=True)
    result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Rename event column to gold_label_full
    result = result.rename(columns={event_col: 'gold_label_full'})
    
    # Add gold_label if label_map provided
    if label_map:
        result['gold_label'] = result['gold_label_full'].map(label_map)
    
    # Resolve actor name column (ACTOR1 -> actor_norm)
    actor_name_col = _resolve_col(df, 'ACTOR1')
    if actor_name_col in result.columns:
        result = result.rename(columns={actor_name_col: 'actor_norm'})
    
    # Resolve event_id column
    event_id_col = _resolve_col(df, 'EVENT_ID_CNTY')
    if event_id_col in result.columns:
        result = result.rename(columns={event_id_col: 'event_id_cnty'})
    
    # Resolve notes column
    notes_col = _resolve_col(df, 'NOTES')
    if notes_col in result.columns:
        result = result.rename(columns={notes_col: 'notes'})
    
    # Select final columns
    desired_cols = ['event_id_cnty', 'notes', 'gold_label', 'gold_label_full', 'actor_type', 'actor_norm']
    final_cols = [c for c in desired_cols if c in result.columns]
    
    return result[final_cols]
