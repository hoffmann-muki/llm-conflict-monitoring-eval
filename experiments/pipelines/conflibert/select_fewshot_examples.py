#!/usr/bin/env python3
"""Select high-quality few-shot examples across countries.

Creates stratified, high-quality train/validation CSVs under
`experiments/data/few_shot/{tag}/{country}/` and prints summary.

Heuristics for quality:
 - Non-empty `notes` column
 - Notes length between 80 and 600 characters preferred (medium-long)
 - Deduplicate by `event_id_cnty`
 - Balanced by event_type and actor_norm (state vs non-state)

Usage:
  python experiments/pipelines/conflibert/select_fewshot_examples.py \
    --countries cmr nga --per-class-train 5 --per-class-val 2 --tag fewshot_v1

"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np

from lib.core.constants import COUNTRY_NAMES, EVENT_CLASSES_FULL, CSV_SRC
from lib.data_preparation import extract_country_rows, get_actor_norm_series, extract_state_actor


def select_high_quality(df: pd.DataFrame, n_per_class: int, seed: int = 42) -> pd.DataFrame:
    # Ensure notes present
    df = df[df['notes'].notna()].copy()
    df['notes_len'] = df['notes'].str.len().fillna(0)

    # Prefer medium-long notes (80-600 chars) by scoring
    def score_row(r):
        score = 0
        if 80 <= r['notes_len'] <= 600:
            score += 2
        elif 40 <= r['notes_len'] < 80 or 600 < r['notes_len'] <= 1200:
            score += 1
        # prefer entries with actor_norm set
        if pd.notna(r.get('actor_norm')) and str(r.get('actor_norm')).strip():
            score += 1
        return score

    df['quality_score'] = df.apply(score_row, axis=1)

    # For each event_type, sample top-scoring rows
    rows = []
    rng = np.random.default_rng(seed)
    for et in EVENT_CLASSES_FULL:
        bucket = df[df['event_type'] == et].copy()
        if bucket.empty:
            continue
        # sort by quality_score then random tie-breaker
        bucket = bucket.sample(frac=1, random_state=seed).sort_values(['quality_score', 'notes_len'], ascending=False)
        take = min(n_per_class, len(bucket))
        rows.append(bucket.head(take))

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out


def build_and_write(countries, per_class_train, per_class_val, tag, out_root='experiments/data/few_shot'):
    out_root = Path(out_root)
    summary = {}
    for c in countries:
        if c not in COUNTRY_NAMES:
            print(f"Skipping unknown country code: {c}")
            continue
        country_name = COUNTRY_NAMES[c]
        # Extract country rows using shared helper (uses CSV_SRC constant)
        # This returns dataframe with columns including event_id_cnty, notes, event_type
        try:
            df_country = extract_country_rows(CSV_SRC, country_name)
        except Exception:
            # Fallback: try dataset path under datasets/{country}
            cand = Path(f"datasets/{c}/{country_name}_lagged_data_up_to-2024-10-24.csv")
            if not cand.exists():
                print(f"Data file not found for {c}: {cand}")
                continue
            df_country = pd.read_csv(cand)

        # normalize column names to lowercase for consistent access (NOTES -> notes, EVENT_TYPE -> event_type)
        df_country.columns = [c.lower() for c in df_country.columns]
        cols_map = {col.lower(): col for col in df_country.columns}
        # Ensure event_id_cnty exists
        if 'event_id_cnty' not in df_country.columns and 'event_id' in df_country.columns:
            df_country = df_country.rename(columns={'event_id': 'event_id_cnty'})

        # Create actor_norm if missing
        if 'actor_norm' not in df_country.columns:
            df_country['actor_norm'] = get_actor_norm_series(df_country, country=country_name)

        # Normalize event_type (some datasets use EVENT_TYPE or event_type_full)
        if 'event_type' not in df_country.columns and 'event_type_full' in df_country.columns:
            df_country = df_country.rename(columns={'event_type_full': 'event_type'})

        usable = (
            df_country.loc[
                df_country['notes'].notna() & df_country['event_type'].isin(EVENT_CLASSES_FULL),
                ['event_id_cnty', 'notes', 'event_type', 'actor_norm']
            ]
            .drop_duplicates(subset=['event_id_cnty'])
        )

        # Select train first, then select validation from the remaining pool to ensure disjoint splits
        usable_renamed = usable.rename(columns={'event_id_cnty': 'event_id'})
        train_df = select_high_quality(usable_renamed, per_class_train)
        # remove train ids from usable before selecting val
        remaining = usable_renamed[~usable_renamed['event_id'].isin(train_df['event_id'])] if not train_df.empty else usable_renamed
        val_df = select_high_quality(remaining, per_class_val, seed=per_class_train+1)

        # Write files
        out_dir = out_root / tag / c
        out_dir.mkdir(parents=True, exist_ok=True)
        train_path = out_dir / 'train.csv'
        val_path = out_dir / 'val.csv'
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        summary[c] = {'train': len(train_df), 'val': len(val_df), 'train_path': str(train_path), 'val_path': str(val_path)}

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--countries', nargs='+', default=['cmr', 'nga'], help='Country codes to draw from')
    parser.add_argument('--per-class-train', type=int, default=5)
    parser.add_argument('--per-class-val', type=int, default=2)
    parser.add_argument('--tag', default='fewshot_v1')
    args = parser.parse_args()

    summary = build_and_write(args.countries, args.per_class_train, args.per_class_val, args.tag)
    print('Selection summary:')
    for c, v in summary.items():
        print(f"  {c}: train={v['train']} val={v['val']} -> {v['train_path']}, {v['val_path']}")


if __name__ == '__main__':
    main()
