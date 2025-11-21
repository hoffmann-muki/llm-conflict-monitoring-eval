#!/usr/bin/env python3
"""
classify.py

Usage:
    python classify.py --csv acled_notes.csv --text-col notes --label-col event_type \
        --output predictions.csv --batch-size 32 --max-length 256

Notes:
- By default this script uses the Hugging Face model `snowood1/ConfliBERT-scr-uncased`.
- You may override the model with `--model <model_id_or_path>` to point to a different HF model or local path.
"""
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import json
import os

class TextDataset(Dataset):
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
        # squeeze to remove batch dim
        item = {k: v.squeeze(0) for k, v in enc.items()}
        return item

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='CSV with ACLED rows')
    p.add_argument('--text-col', default='notes', help='column with notes text')
    p.add_argument('--label-col', default='event_type', help='column with human label')
    p.add_argument('--model', default='snowood1/ConfliBERT-scr-uncased', help='model id or path (default: snowood1/ConfliBERT-scr-uncased)')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', default='predictions.csv')
    p.add_argument('--skip-unmapped', action='store_true', help='skip rows whose label cannot be mapped')
    return p.parse_args()

# Edit this mapping to match your dataset strings
LABEL_TEXT_TO_CODE = {
    "Violence against civilians": "V",
    "Battles": "B",
    "Explosions/Remote violence": "E",
    "Protests": "P",
    "Riots": "R",
    "Strategic developments": "S"
}

# We'll produce a stable ID mapping for the model classes
CODE_TO_ID = {c: i for i, c in enumerate(sorted(set(LABEL_TEXT_TO_CODE.values())))}
ID_TO_CODE = {v: k for k, v in CODE_TO_ID.items()}

def main():
    args = parse_args()
    df = pd.read_csv(args.csv, dtype=str).fillna('')
    texts = df[args.text_col].astype(str).tolist()
    human_label_texts = df[args.label_col].astype(str).tolist()

    # Map human labels to codes (V/B/E/P/R/S)
    mapped_codes = []
    mapped_ids = []
    unmapped_idx = []
    for i, lab in enumerate(human_label_texts):
        code = LABEL_TEXT_TO_CODE.get(lab)
        if code is None:
            mapped_codes.append(None)
            mapped_ids.append(None)
            unmapped_idx.append(i)
        else:
            mapped_codes.append(code)
            mapped_ids.append(CODE_TO_ID[code])

    # Optionally drop unmapped rows
    if args.skip_unmapped and len(unmapped_idx) > 0:
        keep_mask = [i not in unmapped_idx for i in range(len(texts))]
        texts = [t for k, t in enumerate(texts) if keep_mask[k]]
        human_label_texts = [t for k, t in enumerate(human_label_texts) if keep_mask[k]]
        mapped_codes = [t for k, t in enumerate(mapped_codes) if keep_mask[k]]
        mapped_ids = [t for k, t in enumerate(mapped_ids) if keep_mask[k]]

    # Load tokenizer and model
    print("Loading model/tokenizer:", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(args.device)
    model.eval()

    # If model label space differs from our CODE_TO_ID, warn (we assume model.num_labels matches).
    expected_num_labels = len(CODE_TO_ID)
    if model.config.num_labels != expected_num_labels:
        print(f"Warning: model.num_labels={model.config.num_labels} but your mapping has {expected_num_labels}.")
        # If model has more labels, we'll still run inference but remap predictions to nearest if needed.

    dataset = TextDataset(texts, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    preds = []
    probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.detach().cpu().numpy()
            # softmax
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            p = exp / exp.sum(axis=1, keepdims=True)
            pred_ids = logits.argmax(axis=1)
            preds.extend(pred_ids.tolist())
            probs.extend(p.tolist())

    # Map model prediction ids to your codes. If model.num_labels == expected_num_labels we assume same ordering.
    if model.config.num_labels == expected_num_labels:
        pred_codes = [ID_TO_CODE.get(i, None) for i in preds]
    else:
        # fallback: map model id -> code by index order if you know it; otherwise save raw ids
        pred_codes = [str(i) for i in preds]

    # Build output dataframe
    out = df.copy()
    out['_pred_id'] = preds
    out['_pred_code'] = pred_codes
    out['_pred_probs'] = [json.dumps(list(map(float, p))) for p in probs]
    out['_gold_code'] = mapped_codes
    out['_gold_id'] = mapped_ids

    out.to_csv(args.output, index=False)
    print("Wrote predictions to", args.output)

    # If gold is available for rows, compute metrics
    have_gold = any([g is not None for g in mapped_ids])
    if have_gold:
        # filter rows where gold exists
        eval_indices = [i for i, g in enumerate(mapped_ids) if g is not None]
        y_true = [mapped_ids[i] for i in eval_indices]
        y_pred = [preds[i] for i in eval_indices]

        # If model ids don't align with our ids, map pred ids to our ids only when equal number of labels
        if model.config.num_labels != expected_num_labels:
            print("Model label count doesn't match your label count; classification_report below will use raw model ids.")
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, target_names=[ID_TO_CODE[i] for i in sorted(ID_TO_CODE.keys())], zero_division=0))
        print("Confusion matrix:")
        print(confusion_matrix(y_true, y_pred))
    else:
        print("No gold labels available / mapped for evaluation.")

if __name__ == '__main__':
    main()