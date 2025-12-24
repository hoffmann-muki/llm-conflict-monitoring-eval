#!/usr/bin/env python3
"""Full fine-tuning script for ConfliBERT using Hugging Face Trainer.

Trains a sequence classification model from a provided few-shot CSV and
saves the model under `models/conflibert_finetuned_{tag}`. After training
the script runs evaluation/inference on provided validation CSVs and writes
predictions into segregated results folders:

  results/{country}/few_shot/{sample_size}/{tag}/conflibert_predictions_{country}.csv

Expected CSV format for train/val inputs: at minimum columns
`event_id` (or `event_id_cnty`), `notes`, and `event_type` (one of EVENT_CLASSES_FULL).

Usage example:
  python experiments/pipelines/conflibert/finetune_conflibert.py \
    --train-csv experiments/data/few_shot/fewshot_v1/cmr/train.csv,experiments/data/few_shot/fewshot_v1/nga/train.csv \
    --val-csv experiments/data/few_shot/fewshot_v1/cmr/val.csv,experiments/data/few_shot/fewshot_v1/nga/val.csv \
    --model-id bert-base-uncased --tag fewshot_v1 --epochs 6 --per-device-train-batch-size 8

"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from lib.core.constants import EVENT_CLASSES_FULL


def load_csvs_to_dataset(csv_paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in csv_paths:
        if not p:
            continue
        df = pd.read_csv(p)
        if 'event_id_cnty' in df.columns and 'event_id' not in df.columns:
            df = df.rename(columns={'event_id_cnty': 'event_id'})
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def prepare_datasets(train_csvs, val_csvs, label_list: List[str]):
    train_df = load_csvs_to_dataset(train_csvs)
    val_df = load_csvs_to_dataset(val_csvs)

    if train_df.empty:
        raise ValueError('No training data found')

    # Ensure event_type column name
    if 'event_type' not in train_df.columns and 'event_type_full' in train_df.columns:
        train_df = train_df.rename(columns={'event_type_full': 'event_type'})
    if 'event_type' not in val_df.columns and 'event_type_full' in val_df.columns:
        val_df = val_df.rename(columns={'event_type_full': 'event_type'})

    # Map labels to ids
    label2id = {lab: i for i, lab in enumerate(label_list)}
    id2label = {i: lab for lab, i in label2id.items()}

    train_df = train_df[train_df['event_type'].isin(label_list)].copy()
    train_df['label'] = train_df['event_type'].map(label2id)

    if not val_df.empty:
        val_df = val_df[val_df['event_type'].isin(label_list)].copy()
        val_df['label'] = val_df['event_type'].map(label2id)

    ds = DatasetDict({'train': Dataset.from_pandas(train_df)})
    if not val_df.empty:
        ds['validation'] = Dataset.from_pandas(val_df)
    return ds, label2id, id2label


def tokenize_fn(examples, tokenizer, text_column='notes'):
    return tokenizer(examples[text_column], truncation=True, max_length=512)


def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {'accuracy': acc, 'f1_macro': f1}


def run_training(
    train_csvs: List[str],
    val_csvs: List[str],
    model_id: str,
    out_dir: str,
    tag: str,
    epochs: int = 4,
    per_device_train_batch_size: int = 8,
    learning_rate: float = 5e-5,
    fp16: bool = True,
):
    label_list = EVENT_CLASSES_FULL
    ds, label2id, id2label = prepare_datasets(train_csvs, val_csvs, label_list)

    config = AutoConfig.from_pretrained(model_id, num_labels=len(label_list), id2label=id2label, label2id=label2id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, config=config)

    tokenized = ds.map(lambda x: tokenize_fn(x, tokenizer), batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        evaluation_strategy='epoch' if 'validation' in tokenized else 'no',
        save_strategy='epoch',
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        fp16=fp16 and torch.cuda.is_available(),
        load_best_model_at_end=True if 'validation' in tokenized else False,
        metric_for_best_model='f1_macro',
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized.get('validation', None),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if 'validation' in tokenized else None,
    )

    trainer.train()

    # Save final model + tokenizer
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    return out_dir, tokenizer, trainer.model, id2label


def predict_and_write(model, tokenizer, id2label, val_csvs: List[str], tag: str, out_results_root='results'):
    # For each country-specific val csv, generate predictions and write to results path
    for p in val_csvs:
        if not p or not Path(p).exists():
            print(f"Skipping missing val file: {p}")
            continue
        df = pd.read_csv(p)
        if 'event_id_cnty' in df.columns and 'event_id' not in df.columns:
            df = df.rename(columns={'event_id_cnty': 'event_id'})

        country = None
        # try to infer country code from file path (folder name or parent)
        parts = Path(p).parts
        for part in parts[::-1]:
            if part.lower() in ['cmr', 'nga']:
                country = part.lower()
                break

        sample_size = len(df)
        out_dir = Path(out_results_root) / (country or 'misc') / 'few_shot' / str(sample_size) / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        model.eval()
        device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else torch.device('cpu')
        for _, r in df.iterrows():
            text = r.get('notes', '')
            eid = r.get('event_id', '')
            actor = r.get('actor_norm', '')
            t0 = time.time()
            inputs = tokenizer(text, truncation=True, max_length=512, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            latency = time.time() - t0
            logits = out.logits[0].cpu().numpy()
            probs = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
            pred_i = int(np.argmax(probs).item())
            pred_label = id2label[pred_i]
            true_label = r.get('event_type', '')
            rows.append({
                'model': f'conflibert_finetuned_{tag}',
                'event_id': eid,
                'true_label': true_label,
                'pred_label': pred_label,
                'pred_conf': float(probs[pred_i]),
                'logits': json.dumps(logits.tolist()),
                'notes': text,
                'latency_sec': latency,
                'actor_norm': actor,
            })

        out_path = out_dir / f'conflibert_predictions_{country or "misc"}.csv'
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f'Wrote predictions to {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True, help='Comma-separated train CSV paths')
    parser.add_argument('--val-csv', required=False, default='', help='Comma-separated val CSV paths')
    parser.add_argument('--model-id', default='bert-base-uncased', help='base model id')
    parser.add_argument('--tag', default='fewshot_v1', help='tag used for outputs and model name')
    parser.add_argument('--out-root', default='models', help='where to save the fine-tuned model')
    parser.add_argument('--results-root', default='results', help='root for storing prediction CSVs')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--per-device-train-batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()

    train_csvs = [p.strip() for p in args.train_csv.split(',') if p.strip()]
    val_csvs = [p.strip() for p in args.val_csv.split(',') if p.strip()]

    out_dir = Path(args.out_root) / f'conflibert_finetuned_{args.tag}'
    out_dir = str(out_dir)

    model_dir, tokenizer, model, id2label = run_training(
        train_csvs,
        val_csvs,
        args.model_id,
        out_dir,
        args.tag,
        epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
    )

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if val_csvs:
        predict_and_write(model, tokenizer, id2label, val_csvs, args.tag, out_results_root=args.results_root)

    print(f'Fine-tuned model saved to {model_dir}')


if __name__ == '__main__':
    main()
