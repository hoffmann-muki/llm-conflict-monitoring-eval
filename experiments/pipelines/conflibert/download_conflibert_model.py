#!/usr/bin/env python3
"""Download and cache ConfliBERT tokenizer + model.

This small utility prefetches the tokenizer and sequence-classification
model (default: "snowood1/ConfliBERT-scr-uncased") and saves them to a
local directory so they can be loaded offline by the pipeline.

Usage:
    python experiments/pipelines/conflibert/download_conflibert_model.py \
        --model-id snowood1/ConfliBERT-scr-uncased --out-dir models/conflibert

Notes:
- If the model is private, set the env var `HUGGINGFACE_HUB_TOKEN` (or
  run `huggingface-cli login`) to allow download.
- The classification pipeline will automatically download the model when
  you run it; this script is only necessary if you want an explicit
  local copy ahead of time.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer


LOGGER = logging.getLogger("download_conflibert")


def download_model(model_id: str, out_dir: str, cache_dir: str | None = None) -> Path:
    """Download tokenizer and model and save to `out_dir`.

    Returns the path to the output directory.
    """
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    load_kwargs = {}
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    LOGGER.info("Downloading tokenizer for %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, **load_kwargs)
    tokenizer.save_pretrained(out_path)
    LOGGER.info("Saved tokenizer -> %s", out_path)

    LOGGER.info("Downloading model for %s", model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, **load_kwargs)
    model.save_pretrained(out_path)
    LOGGER.info("Saved model -> %s", out_path)

    return out_path


def parse_args():
    p = argparse.ArgumentParser(description="Prefetch ConfliBERT tokenizer + model")
    p.add_argument("--model-id", default=os.environ.get("CONFLIBERT_MODEL", "snowood1/ConfliBERT-scr-uncased"),
                   help="HuggingFace model id (default: snowood1/ConfliBERT-scr-uncased)")
    p.add_argument("--out-dir", default="models/conflibert",
                   help="Directory to save tokenizer and model (default: models/conflibert)")
    p.add_argument("--cache-dir", default=None,
                   help="Optional transformers cache dir")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="[%(levelname)s] %(message)s")

    try:
        out = download_model(args.model_id, args.out_dir, cache_dir=args.cache_dir)
        print(f"Model + tokenizer saved to: {out}")
    except Exception as e:
        LOGGER.exception("Failed to download model: %s", e)
        raise


if __name__ == "__main__":
    main()
