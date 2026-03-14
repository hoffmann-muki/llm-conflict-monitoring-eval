#!/usr/bin/env python3
"""ConfliBERT classification pipeline.

Produces results compatible with the repository's analysis tooling
(`per_class_metrics`, `counterfactual`, etc.).

Usage examples:
    python experiments/pipelines/conflibert/run_conflibert_classification.py cmr --model-path models/conflibert --strategy zero_shot --sample-size 100
    python experiments/pipelines/conflibert/run_conflibert_classification.py nga --model-path models/conflibert --strategy few_shot --sample-size 200
"""
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import json
import os
import time

from lib.core.constants import LABEL_MAP, COUNTRY_NAMES, EVENT_CLASSES_FULL, CSV_SRC
from lib.core.strategy_helpers import get_strategy
from lib.core.data_helpers import paths_for_country, setup_country_environment, resolve_columns
from lib.data_preparation import (
    extract_country_rows,
    get_actor_norm_series,
    extract_state_actor,
    build_stratified_sample,
    build_balanced_actor_sample
)

def _build_id_to_code_from_model(model):
    """Build prediction id -> short label code mapping from model config.

    This avoids assuming any fixed class index order and keeps inference
    consistent with whatever mapping the fine-tuned checkpoint was trained with.
    """
    id2code = {}
    raw_id2label = getattr(model.config, "id2label", None) or {}
    for raw_id, full_label in raw_id2label.items():
        try:
            class_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        code = LABEL_MAP.get(str(full_label))
        if code is not None:
            id2code[class_id] = code

    # Fallback for checkpoints without id2label metadata
    if not id2code:
        id2code = {i: LABEL_MAP[label] for i, label in enumerate(EVENT_CLASSES_FULL)}

    return id2code


class TextDataset(Dataset):
    """Dataset for batched inference."""
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        t = self.texts[idx]
        enc = self.tokenizer(
            t,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        return item


def parse_args():
    """Parse command-line arguments matching Ollama classification interface."""
    parser = argparse.ArgumentParser(
        description='ConfliBERT classification with prompting strategies'
    )
    parser.add_argument('country', nargs='?', default='cmr',
                       help='Country code (e.g., cmr, nga). Default: cmr')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='Number of events to sample (default: 100)')
    parser.add_argument('--strategy', default='zero_shot',
                       help='Prompting strategy: zero_shot, few_shot, explainable (default: zero_shot)')
    parser.add_argument('--model-path', required=True,
                       help='Path to local ConfliBERT model directory (use download_conflibert_model.py to obtain)')
    parser.add_argument('--primary-group', default=None,
                       help='Event type to oversample (e.g., "Violence against civilians"). '
                            'Default: None (proportional sampling)')
    parser.add_argument('--primary-share', type=float, default=0.0,
                       help='Fraction for primary group (0-1). Only used if --primary-group is set. '
                            'Default: 0.0')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for inference (default: 16)')
    parser.add_argument('--max-length', type=int, default=256,
                       help='Maximum sequence length (default: 256)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device for inference (default: cuda if available, else cpu)')
    parser.add_argument('--num-examples', type=int, default=None,
                       help='Number of few-shot examples (1-5). Only used with --strategy few_shot. '
                            'Default: 1 for few_shot strategy.')
    parser.add_argument('--model-name', default=None,
                       help='Optional canonical model name to write into results and filename. '
                            'If omitted, derived from --model-path basename (lowercased).')
    
    return parser.parse_args()

def run_conflibert_classification(country_code: str, strategy_name: str, 
                                 sample_size: int, model_path: str,
                                 batch_size: int, max_length: int, device: str,
                                 primary_group: str | None = None, primary_share: float = 0.0,
                                 num_examples: int | None = None, model_name: str | None = None):
    """Run ConfliBERT classification with independent stratified sampling.
    
    This function:
    1. Creates a stratified sample from the source data.
    2. Runs ConfliBERT inference with strategy-aware prompting.
    3. Writes results in the repository-standard format for downstream analysis.
    
    Args:
        country_code: Country code (e.g., 'cmr', 'nga')
        strategy_name: Strategy name (for output organization)
        sample_size: Number of samples to generate
        model_path: Path to local ConfliBERT model directory
        batch_size: Batch size for inference
        max_length: Max sequence length
        device: Device for inference
        primary_group: Optional event type to oversample (default: None)
        primary_share: Fraction of sample reserved for primary_group (0-1)
        num_examples: Number of few-shot examples (1-5). Only used when strategy_name='few_shot'.
    """
    if country_code not in COUNTRY_NAMES:
        raise ValueError(
            f"Unsupported country code: {country_code}. "
            f"Supported: {list(COUNTRY_NAMES.keys())}"
        )
    
    country_name = COUNTRY_NAMES[country_code]
    strategy = get_strategy(strategy_name, num_examples)
    
    if not os.path.exists(CSV_SRC):
        raise SystemExit(f"Source CSV not found: {CSV_SRC}")
    
    print(f"\n{'='*70}")
    print(f"ConfliBERT Classification: {country_name} ({country_code})")
    print(f"Strategy: {strategy_name}")
    if strategy_name == 'few_shot' and num_examples:
        print(f"Few-shot examples: {num_examples}")
    print(f"Model path: {model_path}")
    print(f"Sample size: {sample_size}")
    print(f"{'='*70}\n")
    
    # Data preparation - create stratified sample
    df_all = pd.read_csv(CSV_SRC)
    df_country = extract_country_rows(CSV_SRC, country_name)
    
    # Persist extracted country-specific CSV
    paths = paths_for_country(country_code, strategy_name, str(sample_size), num_examples)
    os.makedirs(paths['datasets_dir'], exist_ok=True)
    out_country = os.path.join(
        paths['datasets_dir'],
        f"{country_name}_lagged_data_up_to-2024-10-24.csv"
    )
    df_country.to_csv(out_country, index=False)
    print(f"Wrote extracted {country_name} data to {out_country}")
    
    # Resolve column names case-insensitively
    cols = resolve_columns(
        df_country, 
        ['actor1', 'notes', 'event_type', 'event_id_cnty', 'inter1']
    )
    col_actor = cols.get('actor1') or 'actor1'
    col_notes = cols.get('notes') or 'notes'
    col_event_type = cols.get('event_type') or 'event_type'
    col_event_id = cols.get('event_id_cnty') or 'event_id_cnty'
    col_inter1 = cols.get('inter1') or 'INTER1'
    
    # Create normalized actor column
    df_country["actor_norm"] = get_actor_norm_series(
        df_country,
        actor_col=col_actor
    )
    
    # Create state_actor boolean
    df_country["state_actor"] = extract_state_actor(
        df_country,
        country=country_name.lower(),
        actor_col=col_actor
    )
    
    # Keep only state-actor rows with valid event types and notes
    usable = (
        df_country.loc[
            df_country["state_actor"]
            & df_country[col_notes].notna()
            & df_country[col_event_type].isin(EVENT_CLASSES_FULL),
            [col_event_id, col_notes, col_event_type, "actor_norm"]
        ]
        .rename(columns={
            col_event_id: "event_id_cnty",
            col_notes: "notes",
            col_event_type: "event_type"
        })
        .assign(notes=lambda x: x["notes"].str.replace(
            r"\s+", " ", regex=True
        ).str.slice(0, 400))
        .drop_duplicates(subset=["event_id_cnty"])
    )
    
    print(f"Usable state-actor rows found ({country_name}): {len(usable):,}")
    
    # Check if sample file already exists (for consistent cross-model/cross-pipeline comparison)
    # Uses unified sample path shared with Ollama pipeline
    sample_path = paths['sample_path']
    if os.path.exists(sample_path):
        print(f"Reusing existing sample file for cross-model consistency: {sample_path}")
        df = pd.read_csv(sample_path)
        print(f"Loaded {len(df)} events from existing sample")
    else:
        # Build balanced actor sample for fairness analysis
        n_total = min(sample_size, len(usable))
        
        if primary_group:
            print(f"Using balanced actor sampling with primary event: {primary_group}")
            print(f"  {primary_share*100:.0f}% {primary_group}, {(1-primary_share)*100:.0f}% other classes")
            print(f"  Ensuring equal representation of state and non-state actors")
        else:
            print("Using balanced actor sampling: 50% state actors, 50% non-state actors")
            print("  Sample stratified by event type within each actor group")
        
        df = build_balanced_actor_sample(
            df_country,
            n_total=n_total,
            balance_ratio=0.5,
            event_types=EVENT_CLASSES_FULL,
            event_col=col_event_type,
            actor_code_col=col_inter1,
            min_per_cell=10,
            primary_event=primary_group,
            primary_share=primary_share if primary_group else None,
            label_map=LABEL_MAP,
            random_state=42,
            verbose=True
        )
        
        # Save sample for reproducibility (unified path for cross-pipeline comparison)
        df.to_csv(sample_path, index=False)
        print(f"Wrote balanced actor sample to {sample_path}")
        print(f"Sample size: {len(df)} events")
    
    # Extract data
    texts = df['notes'].astype(str).tolist()
    event_ids = df['event_id_cnty'].tolist()
    # Use gold_label_full if available (from balanced sampler), else event_type (legacy)
    true_label_col = 'gold_label_full' if 'gold_label_full' in df.columns else 'event_type'
    true_labels = df[true_label_col].tolist()
    actor_norms = df['actor_norm'].tolist()
    
    # Map labels to codes
    true_label_codes = [LABEL_MAP.get(lab) for lab in true_labels]
    
    # Validate model path exists
    if not os.path.exists(model_path):
        raise SystemExit(
            f"Model path not found: {model_path}\n"
            f"Run: python experiments/pipelines/conflibert/download_conflibert_model.py --out-dir {model_path}"
        )
    
    # Load model and tokenizer from local path
    print(f"Loading model/tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    model.to(device)
    model.eval()
    
    # Verify model outputs match expected labels
    expected_num_labels = len(EVENT_CLASSES_FULL)
    if model.config.num_labels != expected_num_labels:
        print(f"Warning: model.num_labels={model.config.num_labels} "
              f"but label mapping has {expected_num_labels} classes.")

    id_to_code = _build_id_to_code_from_model(model)
    
    # Create dataset and loader
    dataset = TextDataset(texts, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size)

    # Determine model name to use in results rows and filename
    derived_model_name = model_name if model_name else os.path.basename(os.path.normpath(model_path)).lower()
    
    # Run inference
    results = []
    idx = 0
    
    print(f"\nRunning inference on {len(df)} events...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="ConfliBERT inference"):
            t0 = time.time()
            batch_inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch_inputs)
            logits = outputs.logits.detach().cpu().numpy()
            
            # Softmax for probabilities
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp / exp.sum(axis=1, keepdims=True)
            pred_ids = logits.argmax(axis=1)
            
            elapsed = time.time() - t0
            batch_latency = elapsed / len(pred_ids)
            
            # Build results matching Ollama pipeline format
            for i, (pred_id, prob_vec) in enumerate(zip(pred_ids, probs)):
                if idx >= len(event_ids):
                    break
                
                pred_code = id_to_code.get(int(pred_id), "UNKNOWN")
                confidence = float(prob_vec[pred_id])
                
                results.append({
                    # Use canonical model name (derived from --model-name or model_path basename)
                    "model": derived_model_name,
                    "event_id": event_ids[idx],
                    "true_label": true_label_codes[idx],
                    "pred_label": pred_code,
                    "pred_conf": confidence,
                    "logits": json.dumps([float(x) for x in prob_vec]),
                    "notes": texts[idx],
                    "latency_sec": round(batch_latency, 3),
                    "actor_norm": actor_norms[idx]
                })
                idx += 1
    
    # Create results DataFrame
    res_df = pd.DataFrame(results)
    
    # Setup results directory with strategy subfolder (matching Ollama pipeline)
    _, results_dir = setup_country_environment(country_code, strategy_name, str(sample_size), num_examples)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results (include derived_model_name in filename to avoid accidental overwrites)
    out_path = os.path.join(
        results_dir,
        f"{derived_model_name}_results_acled_{country_code}_actors.csv"
    )
    res_df.to_csv(out_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"ConfliBERT classification completed!")
    print(f"Results saved to: {out_path}")
    print(f"{'='*70}\n")
    print(res_df.head(5))
    
    # Basic accuracy report
    correct = (res_df['true_label'] == res_df['pred_label']).sum()
    total = len(res_df)
    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.1%})")
    
    return out_path


def main():
    """Main entry point matching Ollama classification interface."""
    args = parse_args()
    
    # Validate primary_share
    if args.primary_share < 0 or args.primary_share > 1:
        raise ValueError('--primary-share must be between 0 and 1')
    
    if args.primary_group and args.primary_share == 0:
        raise ValueError('--primary-share must be > 0 when --primary-group is specified')
    
    # Validate num_examples
    if args.num_examples is not None:
        if not 1 <= args.num_examples <= 5:
            raise ValueError('--num-examples must be between 1 and 5')
        if args.strategy != 'few_shot':
            raise ValueError('--num-examples is only valid with --strategy few_shot')

    # ConfliBERT is a supervised, fine-tuned classifier. For evaluation clarity
    # we force it to run only as the supervised baseline. If a different
    # strategy was requested via CLI, override it and warn the user.
    if args.strategy != 'zero_shot':
        print(f"Note: overriding requested strategy '{args.strategy}' to 'zero_shot' for ConfliBERT (supervised baseline).")
    args.strategy = 'zero_shot'
    # Ignore num_examples when forcing zero_shot
    args.num_examples = None

    run_conflibert_classification(
        country_code=args.country,
        strategy_name=args.strategy,
        sample_size=args.sample_size,
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        primary_group=args.primary_group,
        primary_share=args.primary_share,
        num_examples=args.num_examples,
        model_name=args.model_name
    )


if __name__ == '__main__':
    main()