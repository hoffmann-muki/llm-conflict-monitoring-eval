#!/usr/bin/env python3
"""Evaluate a fine-tuned ConfliBERT classifier on a fixed split CSV.

Writes predictions in repository-standard schema so downstream calibration and
analysis tools can consume outputs directly.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from lib.core.constants import LABEL_MAP
from lib.core.data_helpers import resolve_columns


def _build_id_to_short_code(model) -> dict:
    """Derive index→short-code mapping from the loaded model's own config.

    The model config stores `id2label` with *full* event-type names as set
    during training (e.g., ``{0: "Violence against civilians", 1: "Battles", …}``
    using ``EVENT_CLASSES_FULL`` order). Mapping through LABEL_MAP yields the
    canonical short codes (V/B/E/P/R/S) used throughout the repository.
    """
    id2label = model.config.id2label  # {int: str}, full event-type names
    return {idx: LABEL_MAP.get(label, label) for idx, label in id2label.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ConfliBERT on a fixed split CSV")
    parser.add_argument("--model-path", required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--input-csv", required=True, help="Split CSV path (train/dev/test_*.csv)")
    parser.add_argument("--output-csv", required=True, help="Output predictions CSV")
    parser.add_argument("--model-name", default="conflibert_finetuned", help="Model identifier in output")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")
    if not Path(args.model_path).exists():
        raise SystemExit(f"Model path not found: {args.model_path}")

    df = pd.read_csv(input_csv)
    cols = resolve_columns(df, ["event_id", "event_id_cnty", "notes", "event_type", "gold_label", "actor_norm"])

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

    # Use local_files_only when the path is a local directory; allow HF hub
    # resolution (e.g., for the unmodified base checkpoint) otherwise.
    is_local = Path(args.model_path).exists()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, local_files_only=is_local)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, local_files_only=is_local)
    model.to(args.device)
    model.eval()

    # Build the label mapping from the model's own config so the index order
    # matches exactly how the model was trained, regardless of label ordering
    # conventions used elsewhere in the codebase.
    id_to_short_code = _build_id_to_short_code(model)

    rows = []
    texts = df[notes_col].fillna("").astype(str).tolist()

    for start in range(0, len(df), args.batch_size):
        end = min(start + args.batch_size, len(df))
        batch_texts = texts[start:end]

        t0 = time.time()
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(args.device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)
            logits = out.logits.detach().cpu().numpy()

        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        pred_ids = np.argmax(probs, axis=1)
        elapsed = round(time.time() - t0, 4)

        for i, row_idx in enumerate(range(start, end)):
            pred_code = id_to_short_code.get(int(pred_ids[i]), "INVALID")
            true_raw = str(df.iloc[row_idx][true_label_col])
            true_code = LABEL_MAP.get(true_raw, true_raw)
            rows.append({
                "model": args.model_name,
                "event_id": df.iloc[row_idx][event_id_col],
                "true_label": true_code,
                "pred_label": pred_code,
                "pred_conf": float(probs[i][int(pred_ids[i])]),
                "logits": json.dumps([float(x) for x in logits[i].tolist()]),
                "notes": batch_texts[i],
                "latency_sec": elapsed,
                "actor_norm": df.iloc[row_idx][actor_col] if actor_col else "",
            })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Wrote predictions: {output_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
