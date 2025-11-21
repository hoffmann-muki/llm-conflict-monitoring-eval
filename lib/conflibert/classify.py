#!/usr/bin/env python3
"""ConfliBERT classification pipeline.

Produces results compatible with the repository's analysis tooling
(`per_class_metrics`, `counterfactual`, etc.).

Usage examples:
    python -m lib.conflibert.classify --country cmr --strategy zero_shot --sample-size 100
    python -m lib.conflibert.classify --country nga --strategy few_shot --sample-size 200
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
import sys
import time

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from lib.core.constants import LABEL_MAP, EVENT_CLASSES_FULL, COUNTRY_NAMES
from lib.core.strategy_helpers import get_strategy
from lib.core.data_helpers import paths_for_country, setup_country_environment

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
    """Parse command-line arguments matching run_classification.py interface."""
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
    parser.add_argument('--model', default='snowood1/ConfliBERT-scr-uncased',
                       help='HuggingFace model ID or path (default: snowood1/ConfliBERT-scr-uncased)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for inference (default: 16)')
    parser.add_argument('--max-length', type=int, default=256,
                       help='Maximum sequence length (default: 256)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device for inference (default: cuda if available, else cpu)')
    
    return parser.parse_args()

def run_conflibert_classification(country_code: str, strategy_name: str, 
                                 sample_size: int, model_name: str,
                                 batch_size: int, max_length: int, device: str):
    """Run ConfliBERT classification matching the Ollama pipeline interface.
    
    This function:
    1. Loads the stratified sample created by the sampling pipeline.
    2. Runs ConfliBERT inference.
    3. Writes results in the repository-standard format for downstream analysis.
    
    Args:
        country_code: Country code (e.g., 'cmr', 'nga')
        strategy_name: Strategy name (for output organization)
        sample_size: Expected sample size
        model_name: HuggingFace model ID
        batch_size: Batch size for inference
        max_length: Max sequence length
        device: Device for inference
    """
    if country_code not in COUNTRY_NAMES:
        raise ValueError(
            f"Unsupported country code: {country_code}. "
            f"Supported: {list(COUNTRY_NAMES.keys())}"
        )
    
    country_name = COUNTRY_NAMES[country_code]
    strategy = get_strategy(strategy_name)
    
    print(f"\n{'='*70}")
    print(f"ConfliBERT Classification: {country_name} ({country_code})")
    print(f"Strategy: {strategy_name}")
    print(f"Model: {model_name}")
    print(f"{'='*70}\n")
    
    # Load the stratified sample (created by run_classification.py or similar)
    paths = paths_for_country(country_code)
    sample_path = os.path.join(
        paths['datasets_dir'],
        f"sample_{country_code}_state_actors.csv"
    )
    
    if not os.path.exists(sample_path):
        raise SystemExit(
            f"Sample file not found: {sample_path}\n"
            f"Please run the Ollama pipeline first to create the stratified sample:\n"
            f"  python -m experiments.pipelines.run_classification {country_code} "
            f"--sample-size {sample_size} --strategy {strategy_name}"
        )
    
    df = pd.read_csv(sample_path)
    print(f"Loaded sample from {sample_path}")
    print(f"Sample size: {len(df)}")
    
    # Verify required columns exist
    required_cols = ['event_id_cnty', 'notes', 'event_type', 'actor_norm']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    
    # Extract data
    texts = df['notes'].astype(str).tolist()
    event_ids = df['event_id_cnty'].tolist()
    true_labels = df['event_type'].tolist()
    actor_norms = df['actor_norm'].tolist()
    
    # Map labels to codes
    true_label_codes = [LABEL_MAP.get(lab) for lab in true_labels]
    
    # Load model and tokenizer
    print(f"Loading model/tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
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
                    "model": f"conflibert_{model_name.split('/')[-1]}",
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
    """Main entry point matching run_classification.py interface."""
    args = parse_args()
    
    run_conflibert_classification(
        country_code=args.country,
        strategy_name=args.strategy,
        sample_size=args.sample_size,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device
    )


if __name__ == '__main__':
    main()