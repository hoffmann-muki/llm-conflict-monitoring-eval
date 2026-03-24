"""Lightweight single-text inference helper for ConfliBERT.

This module provides a small wrapper to run a single text through a local
ConfliBERT transformers model and return a result dict compatible with the
format used by `lib.analysis.counterfactual` (i.e. keys: `label`, `confidence`).

The helper loads the tokenizer/model lazily and caches them for subsequent
calls. It tries a few sensible default paths for the model directory and
supports a `model_name` token like `conflibert` produced by the
ConfliBERT pipeline.
"""
from __future__ import annotations

import os
import numpy as np
from typing import Optional, Tuple, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from lib.core.constants import LABEL_MAP, EVENT_CLASSES_FULL

# Cached model/tokenizer
_MODEL_CACHE = {
    'model_dir': None,
    'tokenizer': None,
    'model': None,
    'device': None
}


def _build_id_mappings(model):
    """Build id -> short code mapping from model config labels.

    Uses model.config.id2label when present, and falls back to the canonical
    EVENT_CLASSES_FULL order used in fine-tuning.
    """
    id_to_code = {}
    raw_id2label = getattr(model.config, 'id2label', None) or {}
    for raw_id, full_label in raw_id2label.items():
        try:
            class_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        code = LABEL_MAP.get(str(full_label))
        if code is not None:
            id_to_code[class_id] = code

    if not id_to_code:
        id_to_code = {i: LABEL_MAP[label] for i, label in enumerate(EVENT_CLASSES_FULL)}

    return id_to_code


def _parse_model_path_map(raw: str) -> Dict[str, str]:
    """Parse CONFLIBERT_MODEL_PATH_MAP values.

    Supported formats per entry:
      model_token=/abs/path
      model_token:/abs/path
    Entries are comma-separated.
    """
    mapping: Dict[str, str] = {}
    for chunk in (raw or "").split(','):
        entry = chunk.strip()
        if not entry:
            continue
        if '=' in entry:
            k, v = entry.split('=', 1)
        elif ':' in entry:
            k, v = entry.split(':', 1)
        else:
            continue
        k = k.strip()
        v = v.strip()
        if k and v:
            mapping[k] = v
    return mapping


def _resolve_model_dir(model_token: Optional[str] = None) -> str:
    """Try to resolve a local model directory for ConfliBERT.

    Resolution order:
    - If `model_token` is an existing absolute/relative directory, use it.
    - If CONFLIBERT_MODEL_PATH_MAP contains the token, use mapped path.
    - If model_token exists under models/<model_token>, use it.
    - If model_token is like conflibert_name, try models/<name>.
    - Else, fallback to models/conflibert.
    """
    preferred = os.path.join('models', 'conflibert')

    # Explicit path provided directly as token
    if model_token and os.path.isdir(model_token):
        return model_token

    # Explicit mapping via env var
    path_map = _parse_model_path_map(os.environ.get('CONFLIBERT_MODEL_PATH_MAP', ''))
    if model_token and model_token in path_map and os.path.isdir(path_map[model_token]):
        return path_map[model_token]

    # Optional generic fallback path via env var
    env_fallback = os.environ.get('CONFLIBERT_MODEL_PATH', '').strip()
    if model_token and env_fallback and os.path.isdir(env_fallback):
        return env_fallback

    if model_token and model_token.startswith('conflibert'):
        candidate = os.path.join('models', model_token)
        if os.path.isdir(candidate):
            return candidate

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

    # For non-baseline ConfliBERT tokens, never silently fall back to base.
    # Return token-specific path even if missing so model load fails loudly.
    if model_token and model_token.startswith('conflibert') and model_token != 'conflibert':
        return os.path.join('models', model_token)

    # If none found, return default base path (may not exist)
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


def run_conflibert_with_attribution(
    model_token: Optional[str],
    text: str,
    n_steps: int = 50,
    device: Optional[str] = None,
) -> Optional[dict]:
    """Run ConfliBERT inference and compute Layer Integrated Gradients attribution.

    Uses Captum's ``LayerIntegratedGradients`` against the model's embedding
    layer, with an all-[PAD] baseline (token id 0).  Attribution scores are
    summed over the embedding dimension to yield a scalar per subword token.

    References
    ----------
    Sundararajan, Taly & Yan (2017).  Axiomatic Attribution for Deep Networks.
        ICML 2017.

    Args:
        model_token: ConfliBERT model identifier (e.g. 'conflibert').
        text:        Event text to classify and attribute.
        n_steps:     Number of Riemann-sum approximation steps for IG.
                     50 is adequate for token-level attribution; use 100 for
                     publication-quality figures.
        device:      Torch device string; defaults to CUDA if available.

    Returns:
        dict with keys:
            label           str           Predicted ACLED event type code.
            confidence      float         Softmax probability of predicted class.
            pred_class_idx  int           Index of predicted class in id_to_code.
            tokens          List[str]     Subword tokens (including special tokens).
            attributions    List[float]   Per-token attribution scores (same length
                                          as ``tokens``), summed over embedding dim.
            convergence_delta float       IG convergence diagnostic (should be
                                          close to 0; large values indicate
                                          insufficient n_steps).
        Returns None if model loading or inference fails.
    """
    try:
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model_dir = _resolve_model_dir(model_token)
        tokenizer, model, dev = _load_model(model_dir, device)

        enc = tokenizer(
            text,
            truncation=True,
            padding=True,
            return_tensors='pt',
        )
        input_ids      = enc['input_ids'].to(dev)
        attention_mask = enc['attention_mask'].to(dev)

        # Determine predicted class (forward pass without gradient tracking)
        with torch.no_grad():
            outputs    = model(input_ids=input_ids, attention_mask=attention_mask)
            logits_np  = outputs.logits.detach().cpu().numpy().squeeze(0)
        exp_logits  = np.exp(logits_np - np.max(logits_np))
        probs        = exp_logits / exp_logits.sum()
        pred_class_idx = int(np.argmax(probs))

        # Forward function over raw inputs (LayerIntegratedGradients interpolates
        # the embedding layer's output, so input_ids remain as-is here)
        def _forward(ids, mask):
            out = model(input_ids=ids, attention_mask=mask)
            return out.logits

        lig = LayerIntegratedGradients(_forward, model.get_input_embeddings())

        # Baseline: all [PAD] tokens (index 0 for both BERT and RoBERTa variants)
        baseline_ids = torch.zeros_like(input_ids)

        attributions, convergence_delta = lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            target=pred_class_idx,
            n_steps=n_steps,
            return_convergence_delta=True,
        )

        # Sum over embedding dimension → scalar per token; detach before numpy
        token_scores = (
            attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy().tolist()
        )
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())

        id_to_code = _build_id_mappings(model)
        label      = id_to_code.get(pred_class_idx, 'INVALID')
        confidence = float(probs[pred_class_idx])

        return {
            'label':             label,
            'confidence':        confidence,
            'pred_class_idx':    pred_class_idx,
            'tokens':            tokens,
            'attributions':      [round(float(s), 6) for s in token_scores],
            'convergence_delta': float(convergence_delta.mean().item()),
        }

    except Exception as e:
        print(f"[ERROR] run_conflibert_with_attribution failed: {type(e).__name__}: {e}")
        return None


def run_conflibert_single(model_token: Optional[str], text: str, device: Optional[str] = None) -> dict:
    """Run a single text through ConfliBERT and return {'label', 'confidence'}.

    Args:
        model_token: token/name from calibrated CSV (e.g., 'conflibert')
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

        id_to_code = _build_id_mappings(model)
        label = id_to_code.get(pred_id, 'INVALID')
        confidence = float(probs[pred_id])

        return {'label': label, 'confidence': confidence}
    except Exception:
        # Return empty dict on any failure — caller should handle this
        return {}
