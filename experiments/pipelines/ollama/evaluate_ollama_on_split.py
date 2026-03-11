#!/usr/bin/env python3
"""Evaluate an Ollama model on a fixed split CSV.

Uses the existing structured classification interface and writes a predictions
CSV in repository-standard format.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from experiments.prompting_strategies import ZeroShotStrategy
from lib.core.constants import LABEL_MAP
from lib.core.data_helpers import resolve_columns
from lib.inference.ollama_client import run_ollama_structured


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Ollama model on fixed split CSV")
    parser.add_argument("--model", required=True, help="Ollama model name")
    parser.add_argument("--input-csv", required=True, help="Split CSV path (train/dev/test_*.csv)")
    parser.add_argument("--output-csv", required=True, help="Output predictions CSV")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit evaluation to first N rows (default: None, use all)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    cols = resolve_columns(df, ["event_id", "event_id_cnty", "notes", "event_type", "gold_label", "actor_norm"])

    # Optionally limit to first N samples
    if args.max_samples is not None:
        df = df.head(args.max_samples)
        print(f"Limiting evaluation to first {args.max_samples} samples (total in CSV: {len(df)})")

    event_id_col = cols.get("event_id") or cols.get("event_id_cnty")
    notes_col = cols.get("notes")
    true_label_col = cols.get("event_type") or cols.get("gold_label")
    actor_col = cols.get("actor_norm")

    missing = [name for name, col in {
        "event_id": event_id_col,
        "notes": notes_col,
        "true_label": true_label_col,
    }.items() if col is None]
    if missing:
        raise SystemExit(f"Missing required columns in input CSV: {missing}")

    strategy = ZeroShotStrategy()
    rows = []

    for _, r in df.iterrows():
        note = str(r[notes_col]) if pd.notna(r[notes_col]) else ""
        t0 = time.time()
        try:
            prompt = strategy.make_prompt(note)
            system_msg = strategy.get_system_message()
            resp = run_ollama_structured(args.model, prompt, system_msg, schema=strategy.get_schema())
            pred_label = str(resp.get("label", "ERROR")).strip()
            pred_conf = float(resp.get("confidence", 0.0))
            logits = resp.get("logits")
        except Exception:
            pred_label = "ERROR"
            pred_conf = 0.0
            logits = None

        true_raw = str(r[true_label_col])
        true_code = LABEL_MAP.get(true_raw, true_raw)

        rows.append({
            "model": args.model,
            "event_id": r[event_id_col],
            "true_label": true_code,
            "pred_label": pred_label,
            "pred_conf": pred_conf,
            "logits": json.dumps(logits) if logits is not None else None,
            "notes": note,
            "latency_sec": round(time.time() - t0, 4),
            "actor_norm": r[actor_col] if actor_col else "",
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Wrote predictions: {output_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
