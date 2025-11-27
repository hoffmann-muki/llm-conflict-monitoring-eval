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
    build_stratified_sample
)

# We'll produce a stable ID mapping for the model classes
CODE_TO_ID = {c: i for i, c in enumerate(sorted(set(LABEL_MAP.values())))}
ID_TO_CODE = {v: k for k, v in CODE_TO_ID.items()}


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
    parser.add_argument('country', nargs='?', default=os.environ.get('COUNTRY', 'cmr'),
                       help='Country code (e.g., cmr, nga)')
    parser.add_argument('--sample-size', type=int,
                       default=int(os.environ.get('SAMPLE_SIZE', '100')),
                       help='Number of events to sample (default: 100)')
    parser.add_argument('--strategy', default=os.environ.get('STRATEGY', 'zero_shot'),
                       help='Prompting strategy: zero_shot, few_shot, explainable (default: zero_shot)')
    parser.add_argument('--model-path', required=True,
                       help='Path to local ConfliBERT model directory (use download_conflibert_model.py to obtain)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for inference (default: 16)')
    parser.add_argument('--max-length', type=int, default=256,
                       help='Maximum sequence length (default: 256)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device for inference (default: cuda if available, else cpu)')
    
    return parser.parse_args()

def run_conflibert_classification(country_code: str, strategy_name: str, 
                                 sample_size: int, model_path: str,
                                 batch_size: int, max_length: int, device: str,
                                 primary_group: str | None = None, primary_share: float = 0.0):
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
    """
    if country_code not in COUNTRY_NAMES:
        raise ValueError(
            f"Unsupported country code: {country_code}. "
            f"Supported: {list(COUNTRY_NAMES.keys())}"
        )
    
    country_name = COUNTRY_NAMES[country_code]
    strategy = get_strategy(strategy_name)
    
    if not os.path.exists(CSV_SRC):
        raise SystemExit(f"Source CSV not found: {CSV_SRC}")
    
    print(f"\n{'='*70}")
    print(f"ConfliBERT Classification: {country_name} ({country_code})")
    print(f"Strategy: {strategy_name}")
    print(f"Model path: {model_path}")
    print(f"Sample size: {sample_size}")
    print(f"{'='*70}\n")
    
    # Data preparation - create stratified sample
    df_all = pd.read_csv(CSV_SRC)
    df_country = extract_country_rows(CSV_SRC, country_name)
    
    # Persist extracted country-specific CSV
    paths = paths_for_country(country_code)
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
        ['actor1', 'notes', 'event_type', 'event_id_cnty']
    )
    col_actor = cols.get('actor1') or 'actor1'
    col_notes = cols.get('notes') or 'notes'
    col_event_type = cols.get('event_type') or 'event_type'
    col_event_id = cols.get('event_id_cnty') or 'event_id_cnty'
    
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
    
    # Build stratified sample
    n_total = min(sample_size, len(usable))
    
    if primary_group:
        print(f"Using targeted sampling: {primary_share*100:.0f}% {primary_group}, "
              f"{(1-primary_share)*100:.0f}% proportional to other classes")
    else:
        print("Using proportional sampling: sample reflects natural class distribution")
    
    df = build_stratified_sample(
        usable,
        stratify_col='event_type',
        n_total=n_total,
        primary_group=primary_group,
        primary_share=primary_share,
        label_map=LABEL_MAP,
        random_state=42,
        replace=False
    )
    
    # Save sample for reproducibility
    sample_path = os.path.join(paths['datasets_dir'], f"conflibert_sample_{country_code}_state_actors.csv")
    df.to_csv(sample_path, index=False)
    print(f"Wrote stratified sample to {sample_path}")
    print(f"Loaded sample size: {len(df)} events")
    
    # Extract data
    texts = df['notes'].astype(str).tolist()
    event_ids = df['event_id_cnty'].tolist()
    true_labels = df['event_type'].tolist()
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
    expected_num_labels = len(CODE_TO_ID)
    if model.config.num_labels != expected_num_labels:
        print(f"Warning: model.num_labels={model.config.num_labels} "
              f"but label mapping has {expected_num_labels} classes.")
    
    # Create dataset and loader
    dataset = TextDataset(texts, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size)
    
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
                
                pred_code = ID_TO_CODE.get(pred_id, "UNKNOWN")
                confidence = float(prob_vec[pred_id])
                
                results.append({
                    "model": f"conflibert_{os.path.basename(model_path.rstrip('/'))}",
                    "event_id": event_ids[idx],
                    "true_label": true_label_codes[idx],
                    "pred_label": pred_code,
                    "pred_conf": confidence,
                    "logits": json.dumps([float(x) for x in prob_vec]),
                    "latency_sec": round(batch_latency, 3),
                    "actor_norm": actor_norms[idx]
                })
                idx += 1
    
    # Create results DataFrame
    res_df = pd.DataFrame(results)
    
    # Setup results directory with strategy subfolder (matching Ollama pipeline)
    _, results_dir = setup_country_environment(country_code)
    strategy_results_dir = os.path.join(results_dir, strategy_name)
    os.makedirs(strategy_results_dir, exist_ok=True)
    
    # Save results
    out_path = os.path.join(
        strategy_results_dir,
        f"conflibert_results_acled_{country_code}_state_actors.csv"
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
    
    run_conflibert_classification(
        country_code=args.country,
        strategy_name=args.strategy,
        sample_size=args.sample_size,
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device
    )


if __name__ == '__main__':
    main()