#!/usr/bin/env python3
"""Fine-tune a small causal LLM for ACLED label classification (LoRA SFT).

This script consumes JSONL files produced by `prepare_sft_data.py` with
`prompt` and `response` fields and performs supervised fine-tuning using PEFT LoRA.
Loss is computed only on response tokens (prompt tokens are masked).

Outputs
-------
- Adapter directory (`--output-dir`)
- Optional merged model directory (`--merged-output-dir`)
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import inspect
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


def _import_hf_datasets():
    """Import Hugging Face datasets package without local-folder shadowing."""
    removed = []
    cwd = os.getcwd()
    try:
        repo_root = str(Path(__file__).resolve().parents[3])
    except Exception:
        repo_root = None

    for p in ("", cwd, os.path.abspath(cwd), repo_root):
        if not p:
            continue
        while p in sys.path:
            sys.path.remove(p)
            removed.append(p)

    try:
        return importlib.import_module("datasets")
    finally:
        for p in reversed(removed):
            sys.path.insert(0, p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune small LLM on ACLED SFT JSONL")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--dev-jsonl", default="")
    parser.add_argument("--base-model", required=True,
                        help="HF base model id/path (e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0)")
    parser.add_argument("--output-dir", required=True, help="Output adapter directory")
    parser.add_argument("--merged-output-dir", default="", help="Optional merged model output dir")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def _find_lora_target_modules(model) -> list[str]:
    candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    found = set()
    for name, _ in model.named_modules():
        for c in candidates:
            if name.endswith(c):
                found.add(c)
    if found:
        return sorted(found)

    # Conservative fallback for GPT-style models
    fallback = []
    for name, _ in model.named_modules():
        if name.endswith("c_attn"):
            fallback.append("c_attn")
        if name.endswith("c_proj"):
            fallback.append("c_proj")
    return sorted(set(fallback)) or ["q_proj", "v_proj"]


def _load_causal_lm(model_id: str, dtype, local_files_only: bool):
    """Load a causal LM with a stable attention backend for training.

    Some HPC software stacks hit CUDA/CUBLAS errors through SDPA kernels.
    Prefer eager attention and fall back if unsupported by the installed
    transformers version/model class.
    """
    common_kwargs = {
        "torch_dtype": dtype,
        "local_files_only": local_files_only,
    }
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="eager",
            **common_kwargs,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_id, **common_kwargs)


def main() -> None:
    args = parse_args()

    datasets = _import_hf_datasets()

    train_path = Path(args.train_jsonl)
    dev_path = Path(args.dev_jsonl) if args.dev_jsonl else None
    if not train_path.exists():
        raise SystemExit(f"Train JSONL not found: {train_path}")
    if dev_path and not dev_path.exists():
        raise SystemExit(f"Dev JSONL not found: {dev_path}")

    data_files = {"train": str(train_path)}
    if dev_path:
        data_files["validation"] = str(dev_path)

    raw_ds = datasets.load_dataset("json", data_files=data_files)
    train_columns = set(raw_ds["train"].column_names)

    if not {"prompt", "response"}.issubset(train_columns):
        raise SystemExit(
            "SFT JSONL must contain 'prompt' and 'response' fields. "
            "Legacy 'text'-only format is no longer supported."
        )

    local_files_only = os.environ.get('HF_HUB_OFFLINE', '') == '1'
    if local_files_only and not os.path.isdir(args.base_model):
        raise FileNotFoundError(
            f"HF_HUB_OFFLINE=1 is set but local model directory not found: {args.base_model}"
        )

    # Determine precision mode.
    # Default to fp16 on CUDA because some HPC stacks hit CUBLAS failures with bf16.
    # Set SMALL_LLM_PRECISION=bf16 to opt in explicitly.
    precision_mode = os.environ.get("SMALL_LLM_PRECISION", "fp16").strip().lower()
    use_fp16 = False
    use_bf16 = False
    model_dtype = torch.float32
    if torch.cuda.is_available():
        if precision_mode == "bf16":
            use_bf16 = True
            model_dtype = torch.bfloat16
        elif precision_mode == "fp32":
            model_dtype = torch.float32
        else:
            use_fp16 = True
            model_dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_causal_lm(args.base_model, model_dtype, local_files_only)
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    target_modules = _find_lora_target_modules(model)
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_cfg)

    def tokenize(examples):
        """Tokenize prompt/response pairs with output-only loss masking."""
        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        prompts = examples["prompt"]
        responses = examples["response"]

        for prompt, response in zip(prompts, responses):
            full_text = f"{prompt}{response}"

            full_enc = tokenizer(
                full_text,
                truncation=True,
                max_length=args.max_length,
                padding=False,
            )
            prompt_enc = tokenizer(
                prompt,
                truncation=True,
                max_length=args.max_length,
                padding=False,
            )

            input_ids = full_enc["input_ids"]
            attn_mask = full_enc["attention_mask"]
            labels = input_ids.copy()

            # Mask prompt tokens so loss is computed only on response
            prompt_len = min(len(prompt_enc["input_ids"]), len(labels))
            for i in range(prompt_len):
                labels[i] = -100

            all_input_ids.append(input_ids)
            all_attention_masks.append(attn_mask)
            all_labels.append(labels)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels,
        }

    tokenized = raw_ds.map(tokenize, batched=True, remove_columns=raw_ds["train"].column_names)

    collator = default_data_collator

    has_validation = "validation" in tokenized

    ta_sig = inspect.signature(TrainingArguments.__init__).parameters
    ta_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=20,
        save_strategy="epoch",
        load_best_model_at_end=has_validation,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none",
        seed=args.seed,
    )

    # Set warmup using explicit steps to avoid deprecated warmup_ratio path.
    total_steps = max(1, (len(tokenized["train"]) // args.batch_size) * args.epochs)
    warmup_steps = max(1, int(total_steps * 0.03))
    if "warmup_steps" in ta_sig:
        ta_kwargs["warmup_steps"] = warmup_steps

    if "overwrite_output_dir" in ta_sig:
        ta_kwargs["overwrite_output_dir"] = True

    if "evaluation_strategy" in ta_sig:
        ta_kwargs["evaluation_strategy"] = "epoch" if has_validation else "no"
    elif "eval_strategy" in ta_sig:
        ta_kwargs["eval_strategy"] = "epoch" if has_validation else "no"
    else:
        ta_kwargs["do_eval"] = has_validation
        ta_kwargs["do_train"] = True

    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=collator,
    )

    trainer.train()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    metadata = {
        "base_model": args.base_model,
        "target_modules": target_modules,
        "train_jsonl": str(train_path),
        "dev_jsonl": str(dev_path) if dev_path else None,
        "sft_input_format": "prompt_response",
        "output_only_loss_masking": True,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
    }
    with open(out_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved LoRA adapter: {out_dir}")

    merged_dir = None
    if args.merged_output_dir:
        merged_dir = Path(args.merged_output_dir)
        merged_dir.mkdir(parents=True, exist_ok=True)

        base = _load_causal_lm(args.base_model, model_dtype, local_files_only)
        merged_model = PeftModel.from_pretrained(base, out_dir).merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Saved merged model: {merged_dir}")


if __name__ == "__main__":
    main()
