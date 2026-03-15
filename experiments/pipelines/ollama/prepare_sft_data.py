#!/usr/bin/env python3
"""Prepare strategy-aligned JSONL data for small-LLM ACLED fine-tuning.

Input CSV must contain `notes` and a label column (`event_type` full label
or `gold_label` short code). Output JSONL includes:
- prompt: model input text (aligned to production prompting strategy)
- response: assistant JSON answer string
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lib.core.strategy_helpers import get_strategy
from lib.core.constants import LABEL_MAP
from lib.core.data_helpers import resolve_columns


TASK_PROMPT = (
    "Classify the ACLED event into one of six labels: V, B, E, P, R, S. "
    "Return only JSON with fields label and confidence."
)

LABELS = ["V", "B", "E", "P", "R", "S"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SFT JSONL for small LLM fine-tuning")
    parser.add_argument("--input-csv", required=True, help="Train/dev CSV path")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL path")
    parser.add_argument("--strategy", default="zero_shot",
                        choices=["zero_shot", "few_shot", "explainable"],
                        help="Prompting strategy used during production inference")
    parser.add_argument("--num-examples", type=int, default=3,
                        help="Few-shot examples per category (1-5). Used only when strategy=few_shot")
    parser.add_argument("--include-confidence", action="store_true",
                        help="Always include confidence in target JSON (even if schema does not require it)")
    return parser.parse_args()


def _build_reasoning(note: str, label: str) -> list[str]:
    """Generate compact structured reasoning for explainable strategy supervision."""
    label_desc = {
        "V": "direct harm to civilians",
        "B": "armed clash between organized actors",
        "E": "explosive or remote attack",
        "P": "organized demonstration",
        "R": "violent crowd action",
        "S": "strategic operation without direct battle",
    }
    short_note = note[:120].replace("\n", " ")
    return [
        f"Actors: inferred from event text snippet '{short_note}'.",
        "Actions: identify core conflict behavior and event dynamics.",
        f"Rationale: label {label} best matches {label_desc.get(label, 'the observed pattern')}.",
    ]


def _build_target(label: str, note: str, schema: dict, include_confidence: bool) -> dict:
    """Build supervision target that matches strategy schema requirements."""
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required = set(schema.get("required", [])) if isinstance(schema, dict) else set()

    target = {"label": label}

    if "confidence" in props and ("confidence" in required or include_confidence):
        target["confidence"] = 1.0

    if "logits" in props:
        target["logits"] = {lab: (1.0 if lab == label else 0.0) for lab in LABELS}

    if "reasoning" in props:
        target["reasoning"] = _build_reasoning(note, label)

    return target


def _build_prompt(note: str, strategy_name: str, num_examples: int) -> tuple[str, dict]:
    """Construct training prompt aligned with production prompting strategy."""
    strategy = get_strategy(strategy_name, num_examples if strategy_name == "few_shot" else None)
    user_prompt = strategy.make_prompt(note)
    system_msg = strategy.get_system_message()

    if system_msg:
        prompt = (
            "### System\n"
            f"{system_msg}\n\n"
            "### User\n"
            f"{user_prompt}\n\n"
            "### Assistant\n"
        )
    else:
        prompt = (
            "### User\n"
            f"{user_prompt}\n\n"
            "### Assistant\n"
        )

    return prompt, strategy.get_schema()


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

    if args.strategy == "few_shot" and not 1 <= args.num_examples <= 5:
        raise SystemExit("--num-examples must be between 1 and 5 when --strategy=few_shot")

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

            prompt, schema = _build_prompt(note, args.strategy, args.num_examples)
            target = _build_target(label, note, schema, args.include_confidence)
            response = json.dumps(target, ensure_ascii=False)

            record = {
                "prompt": prompt,
                "response": response,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    if n_skipped:
        print(f"Warning: skipped {n_skipped} rows with missing notes or unrecognised labels")
    print(f"Strategy: {args.strategy}")
    if args.strategy == "few_shot":
        print(f"Few-shot examples per category: {args.num_examples}")
    print(f"Wrote SFT JSONL: {out_path} ({n_written} examples)")


if __name__ == "__main__":
    main()
