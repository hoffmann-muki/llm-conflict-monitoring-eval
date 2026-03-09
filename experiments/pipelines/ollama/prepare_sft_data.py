#!/usr/bin/env python3
"""Prepare instruction-style JSONL data for small-LLM ACLED fine-tuning.

Input CSV must contain `notes` and a label column (`event_type` full label
or `gold_label` short code). Output JSONL follows one-example-per-line with a
single `text` field for straightforward causal-LM SFT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lib.core.constants import LABEL_MAP
from lib.core.data_helpers import resolve_columns


TASK_PROMPT = (
    "Classify the ACLED event into one of six labels: V, B, E, P, R, S. "
    "Return only JSON with fields label and confidence."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SFT JSONL for small LLM fine-tuning")
    parser.add_argument("--input-csv", required=True, help="Train/dev CSV path")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL path")
    parser.add_argument("--include-confidence", action="store_true",
                        help="Include fixed confidence=1.0 in target JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_csv)
    out_path = Path(args.output_jsonl)

    if not in_path.exists():
        raise SystemExit(f"Input CSV not found: {in_path}")

    df = pd.read_csv(in_path)
    cols = resolve_columns(df, ["notes", "event_type", "gold_label"])
    notes_col = cols.get("notes")
    label_col = cols.get("gold_label") or cols.get("event_type")

    if notes_col is None or label_col is None:
        raise SystemExit("Input CSV must contain notes and one of [gold_label, event_type]")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0
    with open(out_path, "w") as f:
        for _, r in df.iterrows():
            note = str(r[notes_col]).strip() if pd.notna(r[notes_col]) else ""
            raw_label = str(r[label_col]).strip()
            label = raw_label if raw_label in {"V", "B", "E", "P", "R", "S"} else LABEL_MAP.get(raw_label)
            if not note or label not in {"V", "B", "E", "P", "R", "S"}:
                n_skipped += 1
                continue

            target = {"label": label}
            if args.include_confidence:
                target["confidence"] = 1.0

            text = (
                "### Instruction\n"
                f"{TASK_PROMPT}\n\n"
                "### Input\n"
                f"{note}\n\n"
                "### Output\n"
                f"{json.dumps(target, ensure_ascii=False)}"
            )
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n_written += 1

    if n_skipped:
        print(f"Warning: skipped {n_skipped} rows with missing notes or unrecognised labels")
    print(f"Wrote SFT JSONL: {out_path} ({n_written} examples)")


if __name__ == "__main__":
    main()
