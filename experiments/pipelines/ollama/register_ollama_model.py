#!/usr/bin/env python3
"""Register a merged HuggingFace model directory as an Ollama model.

This utility is intentionally separate from fine-tuning so model training and
model serving remain decoupled.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register merged HF model with Ollama")
    parser.add_argument("--merged-model-dir", required=True, help="Path to merged HF model directory")
    parser.add_argument("--ollama-model-name", required=True, help="Target Ollama model name")
    parser.add_argument("--modelfile", default="", help="Optional Modelfile output path")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    merged_dir = Path(args.merged_model_dir)
    if not merged_dir.exists() or not merged_dir.is_dir():
        raise SystemExit(f"Merged model directory not found: {merged_dir}")

    modelfile = Path(args.modelfile) if args.modelfile else merged_dir / "Modelfile"
    modelfile.parent.mkdir(parents=True, exist_ok=True)

    with open(modelfile, "w") as f:
        f.write(f"FROM {merged_dir.resolve()}\n")

    cmd = ["ollama", "create", args.ollama_model_name, "-f", str(modelfile)]
    print("Prepared Modelfile:", modelfile)
    print("Command:", " ".join(cmd))

    if args.dry_run:
        return

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise SystemExit("ollama CLI not found in PATH") from exc

    print(f"Created Ollama model: {args.ollama_model_name}")


if __name__ == "__main__":
    main()
