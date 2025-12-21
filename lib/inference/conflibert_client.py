"""Lightweight single-text inference helper for ConfliBERT.

This module provides a small wrapper to run a single text through a local
ConfliBERT transformers model and return a result dict compatible with the
format used by `lib.analysis.counterfactual` (i.e. keys: `label`, `confidence`).

The helper loads the tokenizer/model lazily and caches them for subsequent
calls. It tries a few sensible default paths for the model directory and
supports a `model_name` token like `conflibert_conflibert` produced by the
ConfliBERT pipeline.
"""
from __future__ import annotations

import os
import numpy as np
from typing import Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from lib.core.constants import LABEL_MAP

# Cached model/tokenizer
_MODEL_CACHE = {
    'model_dir': None,
    'tokenizer': None,
    'model': None,
    'device': None
}


def _build_id_mappings():
    # Recreate the same stable id mapping used in the training pipeline
    codes = sorted(set(LABEL_MAP.values()))
    code_to_id = {c: i for i, c in enumerate(codes)}
    id_to_code = {v: k for k, v in code_to_id.items()}
    return id_to_code


def _resolve_model_dir(model_token: Optional[str] = None) -> str:
    """Try to resolve a local model directory for ConfliBERT.

    Heuristics:
    - If `models/conflibert` exists, prefer it.
    - If `model_token` is like `conflibert_name`, try `models/<name>`.
    - Otherwise, fall back to `models/conflibert`.
    """
    # Preferred default
    preferred = os.path.join('models', 'conflibert')
    if os.path.isdir(preferred):
        return preferred

    if model_token and model_token.startswith('conflibert'):
        parts = model_token.split('_', 1)
        if len(parts) == 2 and parts[1]:
            candidate = os.path.join('models', parts[1])
            if os.path.isdir(candidate):
                return candidate

    # Last resort: current working dir `models/<token>`
    if model_token:
        candidate = os.path.join('models', model_token)
        if os.path.isdir(candidate):
            return candidate

    # If none found, return the preferred path (may not exist)
    return preferred


def _load_model(model_dir: str, device: str = 'cpu'):
    if _MODEL_CACHE['model_dir'] == model_dir and _MODEL_CACHE['model'] is not None:
        return _MODEL_CACHE['tokenizer'], _MODEL_CACHE['model'], _MODEL_CACHE['device']

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.to(device)
    model.eval()

    _MODEL_CACHE['model_dir'] = model_dir
    _MODEL_CACHE['tokenizer'] = tokenizer
    _MODEL_CACHE['model'] = model
    _MODEL_CACHE['device'] = device

    return tokenizer, model, device


def run_conflibert_single(model_token: Optional[str], text: str, device: Optional[str] = None) -> dict:
    """Run a single text through ConfliBERT and return {'label', 'confidence'}.

    Args:
        model_token: token/name from calibrated CSV (e.g., 'conflibert_conflibert')
        text: the event text to classify
        device: torch device string, defaults to CUDA if available else CPU

    Returns:
        dict with keys 'label' (str) and 'confidence' (float). On failure returns {}.
    """
    try:
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model_dir = _resolve_model_dir(model_token)
        tokenizer, model, dev = _load_model(model_dir, device)

        # Tokenize and run
        enc = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits.detach().cpu().numpy().squeeze(0)

        # Softmax to probabilities
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        pred_id = int(np.argmax(probs))

        id_to_code = _build_id_mappings()
        label = id_to_code.get(pred_id, 'INVALID')
        confidence = float(probs[pred_id])

        return {'label': label, 'confidence': confidence}
    except Exception:
        # Return empty dict on any failure — caller should handle this
        return {}
