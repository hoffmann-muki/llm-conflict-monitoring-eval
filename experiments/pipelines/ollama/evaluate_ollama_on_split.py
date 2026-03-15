#!/usr/bin/env python3
"""Evaluate a merged HuggingFace causal LLM on a fixed split CSV.

Runs direct local inference through transformers, generates JSON outputs for
ACLED label classification, and writes a repository-standard predictions CSV.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.prompting_strategies import ZeroShotStrategy
from lib.core.constants import LABEL_MAP
from lib.core.data_helpers import resolve_columns

VALID_LABELS = {"V", "B", "E", "P", "R", "S"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate merged HF model on fixed split CSV")
    parser.add_argument("--model-path", required=True, help="Path or HF id of merged causal LM")
    parser.add_argument("--model-name", default="", help="Display name written to output CSV")
    parser.add_argument("--input-csv", required=True, help="Split CSV path (train/dev/test_*.csv)")
    parser.add_argument("--output-csv", required=True, help="Output predictions CSV")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit evaluation to first N rows (default: None, use all)")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Maximum new tokens to generate per sample")
    return parser.parse_args()


def _extract_json_object(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start:i + 1]
                try:
                    parsed = json.loads(snippet)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    return None
    return None


def _normalize_label(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "ERROR"
    upper = raw.upper()
    if upper in VALID_LABELS:
        return upper
    return LABEL_MAP.get(raw, LABEL_MAP.get(raw.title(), "ERROR"))


def _build_generation_prompt(note: str, strategy: ZeroShotStrategy) -> str:
    user_prompt = strategy.make_prompt(note)
    system_msg = strategy.get_system_message()
    if system_msg:
        return (
            "### System\n"
            f"{system_msg}\n\n"
            "### User\n"
            f"{user_prompt}\n\n"
            "### Assistant\n"
        )
    return (
        "### User\n"
        f"{user_prompt}\n\n"
        "### Assistant\n"
    )


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    model_name = args.model_name or Path(args.model_path).name or args.model_path

    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    local_files_only = os.environ.get("HF_HUB_OFFLINE", "") == "1"
    if local_files_only and not os.path.isdir(args.model_path):
        raise SystemExit(
            f"HF_HUB_OFFLINE=1 is set but model path does not exist locally: {args.model_path}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=local_files_only,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model = model.to(device)  # type: ignore[call-arg]
    model.eval()

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

    # Runtime checks above guarantee these are valid column names.
    event_id_col = cast(str, event_id_col)
    notes_col = cast(str, notes_col)
    true_label_col = cast(str, true_label_col)

    strategy = ZeroShotStrategy()
    rows = []

    for _, r in df.iterrows():
        note = str(r[notes_col]) if pd.notna(r[notes_col]) else ""
        t0 = time.time()
        try:
            prompt = _build_generation_prompt(note, strategy)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            gen_tokens = generated[0][inputs["input_ids"].shape[1]:]
            raw_output = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            resp = _extract_json_object(raw_output)
            if resp is None:
                label_match = re.search(r'"label"\s*:\s*"([VBEPRS])"', raw_output)
                conf_match = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', raw_output)
                resp = {
                    "label": label_match.group(1) if label_match else "ERROR",
                    "confidence": float(conf_match.group(1)) if conf_match else 0.0,
                }

            pred_label = _normalize_label(resp.get("label", "ERROR"))
            pred_conf = float(resp.get("confidence", 0.0))
            logits = resp.get("logits")
        except Exception:
            pred_label = "ERROR"
            pred_conf = 0.0
            logits = None

        true_raw = str(r[true_label_col])
        true_code = LABEL_MAP.get(true_raw, true_raw)

        rows.append({
            "model": model_name,
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
