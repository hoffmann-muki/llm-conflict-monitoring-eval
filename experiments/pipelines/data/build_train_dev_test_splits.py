#!/usr/bin/env python3
"""Build leak-safe train/dev/test splits for ACLED model fine-tuning baselines.

This script creates a reproducible split bundle used by both ConfliBERT and
small-LLM fine-tuning pipelines.

Design goals
------------
1) No train/eval leakage by event ID.
2) Explicit holdout evaluation on Cameroon and Nigeria.
3) Optional class balancing for train/dev.
4) Reproducible output with manifest metadata.

Outputs
-------
data/processed/splits/{split_version}/
  - train.csv
  - dev.csv
  - test_cmr.csv
  - test_nga.csv
  - manifest.json

Usage
-----
python experiments/pipelines/data/build_train_dev_test_splits.py \
  --split-version acled_v1 \
  --eval-countries cmr,nga \
  --dev-ratio 0.1 \
  --balance-train \
  --train-max-per-class 8000 \
  --test-max-per-country 3000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from lib.core.constants import COUNTRY_NAMES, CSV_SRC, EVENT_CLASSES_FULL
from lib.core.data_helpers import resolve_columns
from lib.data_preparation import get_actor_norm_series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leak-safe ACLED split bundle")
    parser.add_argument("--source-csv", default=CSV_SRC, help="Source ACLED CSV path")
    parser.add_argument("--split-version", required=True, help="Split identifier (e.g. acled_v1)")
    parser.add_argument("--out-root", default="data/processed/splits", help="Root output directory")
    parser.add_argument("--eval-countries", default="cmr,nga", help="Comma-separated eval country codes")
    parser.add_argument("--dev-ratio", type=float, default=0.1, help="Dev ratio from training pool")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--include-eval-countries-in-train", action="store_true",
                        help="Allow CMR/NGA rows in train/dev pool (not recommended)")
    parser.add_argument("--balance-train", action="store_true", help="Class-balance train split")
    parser.add_argument("--balance-dev", action="store_true", help="Class-balance dev split")
    parser.add_argument("--balance-test", action="store_true", help="Class-balance test country splits")

    parser.add_argument("--train-max-per-class", type=int, default=0,
                        help="Cap per class for train (0 = no cap)")
    parser.add_argument("--dev-max-per-class", type=int, default=0,
                        help="Cap per class for dev (0 = no cap)")
    parser.add_argument("--test-max-per-country", type=int, default=0,
                        help="Cap total rows per test country (0 = no cap)")
    parser.add_argument("--min-notes-len", type=int, default=10,
                        help="Minimum text length to keep")
    return parser.parse_args()


def _country_codes_to_names(codes: List[str]) -> List[str]:
    names = []
    for c in codes:
        code = c.strip().lower()
        if code not in COUNTRY_NAMES:
            raise ValueError(f"Unsupported country code '{code}'. Supported: {sorted(COUNTRY_NAMES.keys())}")
        names.append(COUNTRY_NAMES[code])
    return names


def _standardize(df: pd.DataFrame, min_notes_len: int) -> pd.DataFrame:
    cols = resolve_columns(df, ["event_id_cnty", "event_id", "notes", "event_type", "country", "actor1"])

    event_id_col = cols.get("event_id_cnty") or cols.get("event_id")
    notes_col = cols.get("notes")
    label_col = cols.get("event_type")
    country_col = cols.get("country")
    actor_col = cols.get("actor1")

    missing = [n for n, col in {
        "event_id": event_id_col,
        "notes": notes_col,
        "event_type": label_col,
        "country": country_col,
    }.items() if col is None]
    if missing:
        raise ValueError(f"Missing required columns in source CSV: {missing}")

    out = df[[event_id_col, notes_col, label_col, country_col] + ([actor_col] if actor_col else [])].copy()
    out = out.rename(columns={
        event_id_col: "event_id",
        notes_col: "notes",
        label_col: "event_type",
        country_col: "country",
        **({actor_col: "actor1"} if actor_col else {}),
    })

    out["notes"] = out["notes"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    out["country"] = out["country"].fillna("").astype(str).str.strip()

    out = out[out["event_type"].isin(EVENT_CLASSES_FULL)].copy()
    out = out[out["notes"].str.len() >= min_notes_len].copy()
    out = out.drop_duplicates(subset=["event_id"]).reset_index(drop=True)

    if "actor1" in out.columns:
        out["actor_norm"] = get_actor_norm_series(out, actor_col="actor1")
    else:
        out["actor_norm"] = ""

    return out[["event_id", "notes", "event_type", "country", "actor_norm"]]


def _sample_per_class(df: pd.DataFrame, max_per_class: int, seed: int) -> pd.DataFrame:
    if max_per_class <= 0:
        return df
    sampled = []
    for _, grp in df.groupby("event_type", sort=True):
        n = min(len(grp), max_per_class)
        sampled.append(grp.sample(n=n, random_state=seed, replace=False))
    return pd.concat(sampled, ignore_index=True)


def _balance_classes(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if df.empty:
        return df
    counts = df["event_type"].value_counts()
    target = int(counts.min())
    if target <= 0:
        return df
    sampled = []
    for _, grp in df.groupby("event_type", sort=True):
        sampled.append(grp.sample(n=target, random_state=seed, replace=False))
    return pd.concat(sampled, ignore_index=True)


def _stratified_train_dev_split(df: pd.DataFrame, dev_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    dev_parts = []

    for _, grp in df.groupby("event_type", sort=True):
        grp = grp.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_dev = int(round(len(grp) * dev_ratio))
        n_dev = min(max(n_dev, 1), max(len(grp) - 1, 1)) if len(grp) > 1 else 0
        dev_parts.append(grp.iloc[:n_dev])
        train_parts.append(grp.iloc[n_dev:])

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    dev_df = pd.concat(dev_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, dev_df


def _check_no_overlap(name_a: str, a: pd.DataFrame, name_b: str, b: pd.DataFrame) -> None:
    overlap = set(a["event_id"]) & set(b["event_id"])
    if overlap:
        raise ValueError(f"Leakage detected: {len(overlap)} overlapping event_id values between {name_a} and {name_b}")


def _counts(df: pd.DataFrame) -> Dict[str, int]:
    if df.empty:
        return {}
    return {k: int(v) for k, v in df["event_type"].value_counts().sort_index().items()}


def main() -> None:
    args = parse_args()

    eval_codes = [c.strip().lower() for c in args.eval_countries.split(",") if c.strip()]
    eval_country_names = _country_codes_to_names(eval_codes)

    source = Path(args.source_csv)
    if not source.exists():
        raise SystemExit(f"Source CSV not found: {source}")

    df_raw = pd.read_csv(source)
    df = _standardize(df_raw, args.min_notes_len)

    eval_mask = df["country"].str.lower().isin({n.lower() for n in eval_country_names})
    test_pool = df[eval_mask].copy()
    train_pool = df.copy() if args.include_eval_countries_in_train else df[~eval_mask].copy()

    train_df, dev_df = _stratified_train_dev_split(train_pool, args.dev_ratio, args.seed)

    if args.balance_train:
        train_df = _balance_classes(train_df, args.seed)
    if args.train_max_per_class > 0:
        train_df = _sample_per_class(train_df, args.train_max_per_class, args.seed)

    if args.balance_dev:
        dev_df = _balance_classes(dev_df, args.seed)
    if args.dev_max_per_class > 0:
        dev_df = _sample_per_class(dev_df, args.dev_max_per_class, args.seed)

    test_by_code: Dict[str, pd.DataFrame] = {}
    for code, country_name in zip(eval_codes, eval_country_names):
        tdf = test_pool[test_pool["country"].str.lower() == country_name.lower()].copy()
        if args.balance_test:
            tdf = _balance_classes(tdf, args.seed)
        if args.test_max_per_country > 0 and len(tdf) > args.test_max_per_country:
            tdf = tdf.sample(n=args.test_max_per_country, random_state=args.seed, replace=False)
        tdf = tdf.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        test_by_code[code] = tdf

    # Leakage checks
    _check_no_overlap("train", train_df, "dev", dev_df)
    for code, tdf in test_by_code.items():
        _check_no_overlap("train", train_df, f"test_{code}", tdf)
        _check_no_overlap("dev", dev_df, f"test_{code}", tdf)

    out_dir = Path(args.out_root) / args.split_version
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    dev_path = out_dir / "dev.csv"
    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)

    test_paths = {}
    for code, tdf in test_by_code.items():
        p = out_dir / f"test_{code}.csv"
        tdf.to_csv(p, index=False)
        test_paths[code] = str(p)

    manifest = {
        "split_version": args.split_version,
        "source_csv": str(source),
        "seed": args.seed,
        "dev_ratio": args.dev_ratio,
        "eval_countries": eval_codes,
        "eval_country_names": eval_country_names,
        "include_eval_countries_in_train": args.include_eval_countries_in_train,
        "balance_train": args.balance_train,
        "balance_dev": args.balance_dev,
        "balance_test": args.balance_test,
        "train_max_per_class": args.train_max_per_class,
        "dev_max_per_class": args.dev_max_per_class,
        "test_max_per_country": args.test_max_per_country,
        "n_rows": {
            "train": int(len(train_df)),
            "dev": int(len(dev_df)),
            **{f"test_{k}": int(len(v)) for k, v in test_by_code.items()},
        },
        "class_distribution": {
            "train": _counts(train_df),
            "dev": _counts(dev_df),
            **{f"test_{k}": _counts(v) for k, v in test_by_code.items()},
        },
        "paths": {
            "train": str(train_path),
            "dev": str(dev_path),
            **{f"test_{k}": v for k, v in test_paths.items()},
        },
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 80)
    print("ACLED SPLIT BUNDLE CREATED")
    print("=" * 80)
    print(f"Output directory: {out_dir}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Dev rows:   {len(dev_df):,}")
    for code in eval_codes:
        print(f"Test {code}:   {len(test_by_code[code]):,}")
    print(f"Manifest:   {manifest_path}")


if __name__ == "__main__":
    main()
