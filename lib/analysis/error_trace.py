#!/usr/bin/env python3
"""Error tracing for model-induced bias analysis.

Two complementary attribution methods are implemented, matched to each
model type's capabilities:

Ollama LLMs — Rationale-Flip Concordance (RFC)
    Re-runs each flipped perturbation with ExplainableStrategy and compares
    the structured three-item rationale (actor / action / category-rationale)
    against the rationale produced for the original, unperturbed text.

    A flip is classified as *concordant* when the rationale changes and the
    changed item explicitly references the perturbed token — indicating that
    the model's stated reasoning tracks the evidence.  A flip is classified as
    *discordant* when the label changes but the rationale is unchanged or does
    not mention the perturbation — a confabulation signal consistent with
    Turpin et al. (NeurIPS 2023).

    Discordant flips on low-ambiguity events (EAS tier = Low) are the primary
    indicator of model-induced bias: the model changed its decision on an
    unambiguous event but cannot articulate a reason.

ConfliBERT — Layer Integrated Gradients (LIG)
    Uses Captum's LayerIntegratedGradients against the model's embedding layer
    to compute a scalar attribution score for each input token with respect to
    the predicted class logit (Sundararajan et al. 2017).

    For events where a perturbation caused a label flip, the function also
    computes an attribution delta: the change in per-token attribution when
    the perturbed text is classified instead of the original.  The top delta
    tokens identify which token substitution most drove the decision shift.

    ConfliBERT has no generative capacity, so RFC is not applicable.

Both traces are stratified by Event Ambiguity Tier (EAS) to support the
core research question: whether discordant flips or high per-token
attributions concentrate in Low-ambiguity events (model-induced bias) or
High-ambiguity events (task-inherent uncertainty).

References
----------
Turpin et al. (2023). Language Models Don't Always Say What They Think:
    Unfaithful Explanations in Chain-of-Thought Prompting. NeurIPS 2023.
Sundararajan, Taly & Yan (2017). Axiomatic Attribution for Deep Networks.
    ICML 2017.
Ye & Durrett (2022). The Unreliability of Explanations in Few-Shot Prompting
    for Textual Reasoning. NeurIPS 2022.
Ferrando et al. (2024). A Primer on the Inner Workings of Transformer-based
    Language Models. ACL 2024.
Feder et al. (2021). CausaLM: Causal Model Explanation Through Counterfactual
    Language Models. Computational Linguistics 47(2).

Usage (pipeline, via env vars):
    COUNTRY=cmr STRATEGY=zero_shot SAMPLE_SIZE=1000 \\
        python -m lib.analysis.error_trace

Usage (standalone):
    from lib.analysis.error_trace import ErrorTraceAnalyzer
"""
from __future__ import annotations

import os
import sys
import glob
import json
import re
import csv
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients

from lib.inference.ollama_client import run_ollama_structured
from lib.inference.hf_causal_client import is_hf_inference_model, run_hf_structured
from lib.inference.conflibert_client import run_conflibert_with_attribution
from experiments.prompting_strategies import ExplainableStrategy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Number of top-attributed tokens to retain per event in the report
TOP_K_TOKENS = 10

# Minimum Jaccard similarity below which two rationale strings are considered
# to have changed (used in RFC concordance check)
RATIONALE_CHANGE_THRESHOLD = 0.85

# Maximum number of flipped events for which the Ollama rationale re-pass is
# executed (a hard cap to bound inference cost)
MAX_RATIONALE_EVENTS = 50


# ---------------------------------------------------------------------------
# Rationale-Flip Concordance (RFC) utilities
# ---------------------------------------------------------------------------

def _tokenise_for_overlap(text: str) -> set:
    """Tokenise text to lowercase vocabulary set, excluding common stopwords."""
    _STOPWORDS = {
        'a', 'an', 'the', 'is', 'was', 'are', 'were', 'in', 'on', 'at',
        'of', 'to', 'and', 'or', 'for', 'with', 'by', 'from', 'that',
        'this', 'it', 'its', 'as', 'be', 'been', 'being', 'has', 'have',
        'had', 'do', 'does', 'did', 'not', 'no', 'but', 'if', 'so',
    }
    tokens = re.findall(r'\b[a-z]+\b', text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def _jaccard(a: set, b: set) -> float:
    """Compute Jaccard similarity of two token sets: |a ∩ b| / |a ∪ b|."""
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _rationale_changed(original_items: List[str], perturbed_items: List[str]) -> bool:
    """Return True if any paired rationale items diverged meaningfully.

    Uses RATIONALE_CHANGE_THRESHOLD to determine when Jaccard similarity
    falls below the threshold, indicating substantive change.
    """
    if not original_items or not perturbed_items:
        return False
    for orig, pert in zip(original_items, perturbed_items):
        if _jaccard(_tokenise_for_overlap(orig), _tokenise_for_overlap(pert)) < RATIONALE_CHANGE_THRESHOLD:
            return True
    return False


def _rationale_mentions_token(rationale_items: List[str], token: str) -> bool:
    """Return True if any rationale item contains the token (case-insensitive)."""
    if not token:
        return False
    token_lower = token.lower().strip()
    for item in rationale_items:
        if token_lower in item.lower():
            return True
    return False


def compute_rfc(
    original_reasoning: List[str],
    perturbed_reasoning: List[str],
    pert_original_word: Optional[str],
    pert_replacement_word: Optional[str],
) -> Dict[str, Any]:
    """Compute Rationale-Flip Concordance score for one flipped perturbation.

    A flip is concordant when the rationale changed AND the new rationale
    explicitly references the replacement token, indicating the model's stated
    reasoning aligns with the actual input evidence. A flip is discordant when
    the label changed but the rationale remained unchanged or does not mention
    the perturbation — a confabulation signal per Turpin et al. (NeurIPS 2023).

    Args:
        original_reasoning:    Three-item rationale (actor / action / category).
        perturbed_reasoning:   Rationale after perturbation applied.
        pert_original_word:    Token that was replaced (None for phrase insertions).
        pert_replacement_word: Replacement token or inserted phrase.

    Returns:
        Dict with keys:
            concordant          bool   Rationale changed AND mentions replacement
            rationale_changed   bool   Any rationale item diverged
            mentions_change     bool   Replacement token appears in new rationale
            rfc_score           float  1.0 if concordant, else 0.0
    """
    rationale_changed = _rationale_changed(original_reasoning, perturbed_reasoning)
    mentions_change = _rationale_mentions_token(
        perturbed_reasoning,
        pert_replacement_word or pert_original_word or ''
    )
    concordant = rationale_changed and mentions_change
    return {
        'concordant': concordant,
        'rationale_changed': rationale_changed,
        'mentions_change': mentions_change,
        'rfc_score': 1.0 if concordant else 0.0,
    }


def _fetch_rationale(model: str, text: str) -> Optional[List[str]]:
    """Invoke ExplainableStrategy on text via Ollama and extract reasoning list.

    Returns the three-item rationale [actor, action, category_rationale] on
    success, None if any step fails (missing fields, model error, etc.).
    """
    try:
        strategy = ExplainableStrategy()
        prompt = strategy.make_prompt(text)
        system_msg = strategy.get_system_message()
        schema = strategy.get_schema()
        if is_hf_inference_model(model):
            max_tokens = int(os.environ.get('HF_MAX_NEW_TOKENS_RATIONALE', os.environ.get('HF_MAX_NEW_TOKENS', '160')))
            result = run_hf_structured(model, prompt, system_msg, schema=schema, max_new_tokens=max_tokens)
        else:
            result = run_ollama_structured(model, prompt, system_msg, schema=schema)
        if result and isinstance(result.get('reasoning'), list):
            return [str(r) for r in result['reasoning']]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Integrated Gradients helpers (ConfliBERT)
# ---------------------------------------------------------------------------

def _aggregate_subword_attributions(
    tokens: List[str],
    scores: List[float],
) -> List[Tuple[str, float]]:
    """Aggregate subword token attributions to word level.

    Handles BERT-style continuations (##-prefix) and RoBERTa / SentencePiece
    word boundaries (Ġ or ▁ prefix). Subword scores are summed to the word
    level and the result is sorted by descending absolute attribution.

    Returns list of (word_string, aggregated_score) tuples.
    """
    words: List[str] = []
    word_scores: List[float] = []

    for token, score in zip(tokens, scores):
        # Skip special tokens
        if token in ('[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>'):
            continue

        clean = token.lstrip('Ġ▁')   # RoBERTa / SentencePiece space prefix

        if token.startswith('##'):
            # BERT continuation token: merge with previous word
            clean = token[2:]
            if words:
                words[-1] = words[-1] + clean
                word_scores[-1] += score
            else:
                words.append(clean)
                word_scores.append(score)
        elif token.startswith('Ġ') or token.startswith('▁'):
            # RoBERTa / SentencePiece: new word
            words.append(clean)
            word_scores.append(score)
        else:
            # First token of a word (BERT) or already-clean token
            words.append(clean)
            word_scores.append(score)

    # Sort by absolute attribution (descending)
    paired = sorted(zip(words, word_scores), key=lambda x: abs(x[1]), reverse=True)
    return [(w, round(float(s), 6)) for w, s in paired if w]


# ---------------------------------------------------------------------------
# Core analyser
# ---------------------------------------------------------------------------

class ErrorTraceAnalyzer:
    """Orchestrate RFC (Ollama) and LIG (ConfliBERT) error tracing.

    Parameters
    ----------
    country:      ACLED country code (used for result path construction)
    strategy:     Prompting strategy string ('zero_shot', 'few_shot', etc.)
    sample_size:  Dataset sample size string
    num_examples: Few-shot example count string (or None)
    """

    def __init__(
        self,
        country: str,
        strategy: str,
        sample_size: str,
        num_examples: Optional[str] = None,
    ) -> None:
        self.country = country
        self.strategy = strategy
        self.sample_size = sample_size
        self.num_examples = num_examples

        if strategy == 'few_shot' and num_examples:
            self.results_base = Path(
                f"results/{country}/{strategy}/{sample_size}/{num_examples}"
            )
        else:
            self.results_base = Path(f"results/{country}/{strategy}/{sample_size}")

    # ------------------------------------------------------------------
    # RFC analysis (Ollama models)
    # ------------------------------------------------------------------

    def analyse_rfc(
        self,
        detailed_results: List[Dict],
        models: List[str],
    ) -> Dict[str, Any]:
        """Compute Rationale-Flip Concordance for all Ollama models.

        For each model, flipped perturbations are re-run with ExplainableStrategy
        and the rationale is compared with the original. Capped at MAX_RATIONALE_EVENTS
        per model to bound cost. Aggregates concordance rates by perturbation type
        and by EAS ambiguity tier.

        Returns nested dict keyed by model, each containing 'records' and 'aggregate'.
        """
        rfc_by_model: Dict[str, Any] = {}

        for model in models:
            print(f"  RFC analysis — model: {model}")
            records = []
            n_rationale_calls = 0

            for event in detailed_results:
                event_id    = event.get('event_id', '')
                orig_text   = event.get('original_text', '')
                amb_tier    = event.get('ambiguity_tier', 'Low')

                # Collect flipped perturbations for this model
                flipped_perts = [
                    pr for pr in event.get('perturbations', [])
                    if pr.get('model_results', {})
                       .get(model, {})
                       .get('label_flipped', False)
                ]
                if not flipped_perts:
                    continue

                if n_rationale_calls >= MAX_RATIONALE_EVENTS:
                    print(f"    Reached MAX_RATIONALE_EVENTS ({MAX_RATIONALE_EVENTS}); "
                          "skipping remaining events for this model.")
                    break

                # Fetch rationale for the original text
                orig_reasoning = _fetch_rationale(model, orig_text)
                n_rationale_calls += 1

                for pr in flipped_perts:
                    pert        = pr['perturbation']
                    pert_type   = pert.get('type', 'unknown')
                    pert_text   = pert.get('text', '')
                    orig_word   = pert.get('original')
                    repl_word   = pert.get('replacement') or pert.get('phrase', '')

                    pert_reasoning = _fetch_rationale(model, pert_text)

                    if orig_reasoning and pert_reasoning:
                        rfc = compute_rfc(orig_reasoning, pert_reasoning,
                                         orig_word, repl_word)
                    else:
                        # Rationale fetch failed — mark as unresolvable
                        rfc = {
                            'concordant': None,
                            'rationale_changed': None,
                            'mentions_change': None,
                            'rfc_score': None,
                        }

                    records.append({
                        'event_id':            event_id,
                        'ambiguity_tier':      amb_tier,
                        'perturbation_type':   pert_type,
                        'original_word':       orig_word,
                        'replacement_word':    repl_word,
                        'original_reasoning':  orig_reasoning,
                        'perturbed_reasoning': pert_reasoning,
                        **rfc,
                    })

            rfc_by_model[model] = {
                'records': records,
                'aggregate': _aggregate_rfc_stats(records),
            }
            _print_rfc_summary(model, rfc_by_model[model]['aggregate'])

        return rfc_by_model

    # ------------------------------------------------------------------
    # LIG analysis (ConfliBERT)
    # ------------------------------------------------------------------

    def analyse_lig(
        self,
        detailed_results: List[Dict],
        model_token: str,
        n_steps: int = 50,
    ) -> List[Dict]:
        """Compute Layer Integrated Gradients attribution for ConfliBERT.

        Computes per-token attributions via LayerIntegratedGradients against
        the embedding layer (Sundararajan et al., ICML 2017). For flipped
        perturbations, computes attribution delta to identify which token
        substitution most shifted the model's decision.

        Args:
            detailed_results: 'detailed_results' list from counterfactual JSON.
            model_token:      ConfliBERT model identifier.
            n_steps:          IG approximation resolution (50 adequate for token-level;
                              100+ for publication-quality figures).

        Returns:
            List of per-event attribution records.
        """
        event_records = []

        for event in detailed_results:
            event_id   = event.get('event_id', '')
            orig_text  = event.get('original_text', '')
            amb_tier   = event.get('ambiguity_tier', 'Low')

            print(f"    LIG attribution — event {event_id}")

            # Attribution on the original text
            orig_attr = run_conflibert_with_attribution(
                model_token, orig_text, n_steps=n_steps
            )
            if orig_attr is None:
                print(f"      [SKIP] {event_id}: run_conflibert_with_attribution returned None")
                continue

            top_original = _aggregate_subword_attributions(
                orig_attr['tokens'], orig_attr['attributions']
            )[:TOP_K_TOKENS]

            # Attribution delta for each flipped perturbation
            flip_deltas = []
            for pr in event.get('perturbations', []):
                mr = pr.get('model_results', {}).get(model_token, {})
                if not mr.get('label_flipped', False):
                    continue

                pert      = pr['perturbation']
                pert_text = pert.get('text', '')
                pert_type = pert.get('type', 'unknown')
                orig_word = pert.get('original')
                repl_word = pert.get('replacement') or pert.get('phrase', '')

                pert_attr = run_conflibert_with_attribution(
                    model_token, pert_text, n_steps=n_steps
                )
                if pert_attr is None:
                    continue

                # Build word-level attribution maps for both texts
                orig_map = dict(_aggregate_subword_attributions(
                    orig_attr['tokens'], orig_attr['attributions']
                ))
                pert_map = dict(_aggregate_subword_attributions(
                    pert_attr['tokens'], pert_attr['attributions']
                ))

                # Compute delta for words present in both
                all_words = set(orig_map) | set(pert_map)
                deltas = {
                    w: round(pert_map.get(w, 0.0) - orig_map.get(w, 0.0), 6)
                    for w in all_words
                }
                top_delta = sorted(
                    deltas.items(), key=lambda x: abs(x[1]), reverse=True
                )[:TOP_K_TOKENS]

                flip_deltas.append({
                    'perturbation_type':         pert_type,
                    'original_word':             orig_word,
                    'replacement_word':          repl_word,
                    'attribution_delta_top_tokens': [
                        {'token': w, 'delta': d} for w, d in top_delta
                    ],
                })

            event_records.append({
                'model':            model_token,
                'event_id':         event_id,
                'ambiguity_tier':   amb_tier,
                'top_tokens_original': [
                    {'token': t, 'attribution': s} for t, s in top_original
                ],
                'flipped_perturbations': flip_deltas,
            })

        print(f"[DEBUG] analyse_lig built {len(event_records)} records")
        return event_records

    def analyse_lig_from_disagreements(
        self,
        disagreements_csv_path: Path,
        model_token: str,
        n_steps: int = 50,
    ) -> List[Dict]:
        """Compute LIG for a ConfliBERT model directly from top_disagreements.csv.

        This mode removes dependency on counterfactual JSON/RFC artifacts and
        attributes ConfliBERT predictions on disagreement events already mined
        from calibrated inference outputs.
        """
        df = pd.read_csv(disagreements_csv_path)
        if 'notes' not in df.columns:
            print(f"Skipping LIG from disagreements: missing 'notes' in {disagreements_csv_path}")
            return []

        event_records: List[Dict] = []
        for _, row in df.iterrows():
            event_id = str(row.get('event_id', ''))
            text = str(row.get('notes', '') or '').strip()
            amb_tier = str(row.get('ambiguity_tier', 'Unknown'))
            if not text:
                continue

            print(f"    LIG attribution from disagreements — event {event_id}")
            attr = run_conflibert_with_attribution(model_token, text, n_steps=n_steps)
            if attr is None:
                continue

            top_original = _aggregate_subword_attributions(
                attr['tokens'], attr['attributions']
            )[:TOP_K_TOKENS]

            event_records.append({
                'model': model_token,
                'event_id': event_id,
                'ambiguity_tier': amb_tier,
                'top_tokens_original': [
                    {'token': t, 'attribution': s} for t, s in top_original
                ],
                'flipped_perturbations': [],
            })

        return event_records

    def _analyse_single_counterfactual(
        self,
        counterfactual_json_path: Path,
        lig_n_steps: int = 50,
    ) -> Dict[str, Any]:
        """Analyse one counterfactual JSON and return structured results."""
        with open(counterfactual_json_path) as f:
            cf_data = json.load(f)

        metadata      = cf_data.get('metadata', {})
        all_models    = metadata.get('models', [])
        detailed      = cf_data.get('detailed_results', [])
        n_events      = len(detailed)

        print(f"\nSource:  {counterfactual_json_path}")
        if not detailed:
            print("No detailed_results in counterfactual JSON — nothing to trace.")
            return {
                'source': str(counterfactual_json_path),
                'cf_data': cf_data,
                'rfc_results': {},
                'lig_results': [],
                'n_events': 0,
                'all_models': all_models,
            }

        generative_models = [m for m in all_models if not m.lower().startswith('conflibert')]
        conflibert_models = [m for m in all_models if m.lower().startswith('conflibert')]

        print(f"Events:             {n_events}")
        print(f"Generative models:  {generative_models or '—'}")
        print(f"ConfliBERT models:  {conflibert_models or '—'}")

        # ----- RFC (Generative LLMs: Ollama and/or HF local) --------------
        rfc_results: Dict[str, Any] = {}
        if generative_models:
            print("\n[1/2] Rationale-Flip Concordance (RFC) — Generative models")
            rfc_results = self.analyse_rfc(detailed, generative_models)
        else:
            print("\n[1/2] RFC skipped — no generative models in this source.")

        # ----- LIG (ConfliBERT) -------------------------------------------
        lig_results: List[Dict] = []
        if conflibert_models:
            print("\n[2/2] Layer Integrated Gradients (LIG) — ConfliBERT")
            # Analyse each ConfliBERT model present in this source JSON.
            for model_token in conflibert_models:
                lig_batch = self.analyse_lig(detailed, model_token, n_steps=lig_n_steps)
                print(f"[DEBUG] analyse_lig returned {len(lig_batch)} records for {model_token}")
                lig_results.extend(lig_batch)
        elif not generative_models:
            print("\n[2/2] LIG skipped — no models found in this source.")
        else:
            print("\n[2/2] LIG skipped — no ConfliBERT models in this source.")

        return {
            'source': str(counterfactual_json_path),
            'cf_data': cf_data,
            'rfc_results': rfc_results,
            'lig_results': lig_results,
            'n_events': n_events,
            'all_models': all_models,
        }

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def run(
        self,
        counterfactual_json_path: Path,
        lig_n_steps: int = 50,
    ) -> None:
        """Execute the complete error trace pipeline; write outputs to disk.

        Args:
            counterfactual_json_path: Path to counterfactual_analysis_*.json.
            lig_n_steps:              IG approximation steps for LIG (higher =
                                      more accurate; 50 is standard).
        """
        self.run_many([counterfactual_json_path], lig_n_steps=lig_n_steps)

    def run_many(
        self,
        counterfactual_json_paths: List[Path],
        lig_n_steps: int = 50,
    ) -> None:
        """Execute error trace across one or more counterfactual JSON files."""
        print("=" * 80)
        print("ERROR TRACE ANALYSIS")
        print("=" * 80)

        analyses: List[Dict[str, Any]] = []
        for cf_path in counterfactual_json_paths:
            a = self._analyse_single_counterfactual(cf_path, lig_n_steps)
            print(f"[DEBUG] _analyse_single_counterfactual returned {len(a.get('lig_results', []))} LIG records")
            analyses.append(a)

        # Merge RFC by model across all sources
        merged_rfc: Dict[str, Dict[str, Any]] = {}
        merged_lig: List[Dict] = []
        total_events = 0
        sources_meta = []

        for a in analyses:
            total_events += int(a.get('n_events', 0))
            sources_meta.append({
                'source': a.get('source', ''),
                'n_events': a.get('n_events', 0),
                'models': a.get('all_models', []),
            })

            for model, data in a.get('rfc_results', {}).items():
                merged_rfc.setdefault(model, {'records': [], 'aggregate': {}})
                merged_rfc[model]['records'].extend(data.get('records', []))

            lig_batch = a.get('lig_results', [])
            print(f"[DEBUG] Merging {len(lig_batch)} LIG records from analysis")
            merged_lig.extend(lig_batch)

        for model, data in merged_rfc.items():
            data['aggregate'] = _aggregate_rfc_stats(data['records'])

        print(f"[DEBUG] Before write: merged_rfc={len(merged_rfc)} models, merged_lig={len(merged_lig)} records")
        self._write_report_multi(
            analyses=analyses,
            merged_rfc=merged_rfc,
            merged_lig=merged_lig,
            total_events=total_events,
            sources_meta=sources_meta,
            lig_n_steps=lig_n_steps,
        )
        self._write_summary_csv(merged_rfc, merged_lig)

    def run_lig_from_disagreements(
        self,
        disagreements_csv_path: Path,
        conflibert_models: List[str],
        lig_n_steps: int = 50,
    ) -> None:
        """Run standalone LIG from top_disagreements.csv without RFC dependency."""
        print("=" * 80)
        print("ERROR TRACE ANALYSIS (LIG from disagreements)")
        print("=" * 80)
        print(f"Source: {disagreements_csv_path}")

        merged_lig: List[Dict] = []
        for model_token in conflibert_models:
            print(f"  LIG analysis — model: {model_token}")
            merged_lig.extend(
                self.analyse_lig_from_disagreements(
                    disagreements_csv_path=disagreements_csv_path,
                    model_token=model_token,
                    n_steps=lig_n_steps,
                )
            )

        sources_meta = [{
            'source': str(disagreements_csv_path),
            'n_events': len(merged_lig),
            'models': conflibert_models,
        }]

        self._write_report_multi(
            analyses=[],
            merged_rfc={},
            merged_lig=merged_lig,
            total_events=len(merged_lig),
            sources_meta=sources_meta,
            lig_n_steps=lig_n_steps,
        )
        self._write_summary_csv({}, merged_lig)

    def _write_report(
        self,
        cf_data:     Dict,
        rfc_results: Dict,
        lig_results: List[Dict],
    ) -> None:
        metadata = cf_data.get('metadata', {})
        report = {
            'metadata': {
                'country':          self.country,
                'strategy':         self.strategy,
                'sample_size':      self.sample_size,
                'n_events_analysed': len(cf_data.get('detailed_results', [])),
                'n_flip_events':    sum(
                    1 for e in cf_data.get('detailed_results', [])
                    if any(
                        mr.get('label_flipped', False)
                        for pr in e.get('perturbations', [])
                        for mr in pr.get('model_results', {}).values()
                    )
                ),
                'methods': {
                    'ollama': 'rationale_flip_concordance (Turpin et al. NeurIPS 2023)',
                    'conflibert': 'layer_integrated_gradients (Sundararajan et al. ICML 2017)',
                },
                'references': [
                    'Turpin M, et al. (2023). Language Models Don\'t Always Say What They Think. NeurIPS.',
                    'Sundararajan M, Taly A, Yan Q (2017). Axiomatic Attribution for Deep Networks. ICML.',
                    'Ye X, Durrett G (2022). Unreliability of Explanations in Few-Shot Prompting. NeurIPS.',
                    'Ferrando J, et al. (2024). A Primer on the Inner Workings of Transformer-based LMs. ACL.',
                    'Feder A, et al. (2021). CausaLM: Causal Model Explanation via Counterfactual LMs. CL 47(2).',
                ],
            },
            'ollama_rationale_analysis': {
                'by_model': {
                    model: {
                        'n_flips_analysed': len(data['records']),
                        'aggregate':        data['aggregate'],
                        'records':          data['records'],
                    }
                    for model, data in rfc_results.items()
                },
                'interpretation': (
                    'Discordant flips (label changed, rationale unchanged or irrelevant) '
                    'on Low-ambiguity events (EAS tier = Low) are the primary indicator '
                    'of model-induced bias per Turpin et al. 2023.'
                ),
            },
            'conflibert_attribution': {
                'method': (
                    'LayerIntegratedGradients against the embedding layer, '
                    f'{50} approximation steps, [PAD]-token baseline.'
                ),
                'per_event': lig_results,
            },
        }

        out_path = self.results_base / 'error_trace_report.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Smart merge: replace model-specific data, append new models
        if out_path.exists():
            with open(out_path, 'r') as f:
                existing_report = json.load(f)
            
            # Get list of models being updated
            new_models = set(report.get('ollama_rationale_analysis', {}).get('by_model', {}).keys())
            
            # Remove old data for models being re-run
            existing_models = existing_report.get('ollama_rationale_analysis', {}).get('by_model', {})
            for model in new_models:
                if model in existing_models:
                    del existing_models[model]
            
            # Add new model data
            for model, data in report.get('ollama_rationale_analysis', {}).get('by_model', {}).items():
                existing_models[model] = data
            
            report['ollama_rationale_analysis']['by_model'] = existing_models
            
            # Replace conflibert LIG results if present
            if report.get('conflibert_attribution', {}).get('per_event'):
                existing_lig = existing_report.get('conflibert_attribution', {}).get('per_event', [])
                # Filter out old conflibert results
                existing_lig = [e for e in existing_lig if e.get('model') != 'conflibert']
                existing_lig.extend(report.get('conflibert_attribution', {}).get('per_event', []))
                report['conflibert_attribution']['per_event'] = existing_lig
        
        with open(out_path, 'w') as f:
            json.dump(report, f, indent=2, default=_json_safe)
        print(f"\n✓ Report saved: {out_path}")

    def _write_report_multi(
        self,
        analyses: List[Dict[str, Any]],
        merged_rfc: Dict[str, Dict[str, Any]],
        merged_lig: List[Dict],
        total_events: int,
        sources_meta: List[Dict[str, Any]],
        lig_n_steps: int,
    ) -> None:
        """Write a combined report covering all discovered counterfactual sources."""
        n_flip_events = 0
        for a in analyses:
            cf_data = a.get('cf_data', {})
            n_flip_events += sum(
                1 for e in cf_data.get('detailed_results', [])
                if any(
                    mr.get('label_flipped', False)
                    for pr in e.get('perturbations', [])
                    for mr in pr.get('model_results', {}).values()
                )
            )

        report = {
            'metadata': {
                'country': self.country,
                'strategy': self.strategy,
                'sample_size': self.sample_size,
                'num_sources': len(analyses),
                'sources': sources_meta,
                'n_events_analysed': total_events,
                'n_flip_events': n_flip_events,
                'methods': {
                    'ollama': 'rationale_flip_concordance (Turpin et al. NeurIPS 2023)',
                    'conflibert': 'layer_integrated_gradients (Sundararajan et al. ICML 2017)',
                },
                'references': [
                    'Turpin M, et al. (2023). Language Models Don\'t Always Say What They Think. NeurIPS.',
                    'Sundararajan M, Taly A, Yan Q (2017). Axiomatic Attribution for Deep Networks. ICML.',
                    'Ye X, Durrett G (2022). Unreliability of Explanations in Few-Shot Prompting. NeurIPS.',
                    'Ferrando J, et al. (2024). A Primer on the Inner Workings of Transformer-based LMs. ACL.',
                    'Feder A, et al. (2021). CausaLM: Causal Model Explanation via Counterfactual LMs. CL 47(2).',
                ],
            },
            'ollama_rationale_analysis': {
                'by_model': {
                    model: {
                        'n_flips_analysed': len(data.get('records', [])),
                        'aggregate': data.get('aggregate', {}),
                        'records': data.get('records', []),
                    }
                    for model, data in merged_rfc.items()
                },
                'interpretation': (
                    'Discordant flips (label changed, rationale unchanged or irrelevant) '
                    'on Low-ambiguity events (EAS tier = Low) are the primary indicator '
                    'of model-induced bias per Turpin et al. 2023.'
                ),
            },
            'conflibert_attribution': {
                'method': (
                    'LayerIntegratedGradients against the embedding layer, '
                    f'{lig_n_steps} approximation steps, [PAD]-token baseline.'
                ),
                'per_event': merged_lig,
            },
        }

        out_path = self.results_base / 'error_trace_report.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Smart merge: replace model-specific data, append new models
        if out_path.exists():
            with open(out_path, 'r') as f:
                existing_report = json.load(f)
            
            # Get list of models being updated
            new_models = set(report.get('ollama_rationale_analysis', {}).get('by_model', {}).keys())
            
            # Remove old data for models being re-run
            existing_models = existing_report.get('ollama_rationale_analysis', {}).get('by_model', {})
            for model in new_models:
                if model in existing_models:
                    del existing_models[model]
            
            # Add new model data
            for model, data in report.get('ollama_rationale_analysis', {}).get('by_model', {}).items():
                existing_models[model] = data
            
            report['ollama_rationale_analysis']['by_model'] = existing_models
            
            # Replace LIG results for conflibert if present
            if report.get('conflibert_attribution', {}).get('per_event'):
                existing_lig = existing_report.get('conflibert_attribution', {}).get('per_event', [])
                # Filter out old conflibert results
                existing_lig = [e for e in existing_lig if e.get('model') != 'conflibert']
                existing_lig.extend(report.get('conflibert_attribution', {}).get('per_event', []))
                report['conflibert_attribution']['per_event'] = existing_lig
        
        with open(out_path, 'w') as f:
            json.dump(report, f, indent=2, default=_json_safe)
        print(f"\n✓ Report saved: {out_path}")

    def _write_summary_csv(
        self,
        rfc_results: Dict,
        lig_results: List[Dict],
    ) -> None:
        """Write a flat summary CSV for easy inspection and downstream analysis."""
        rows = []

        # RFC rows (one row per (event, model, perturbation))
        for model, data in rfc_results.items():
            for rec in data['records']:
                rows.append({
                    'source':            'rfc',
                    'model':             model,
                    'event_id':          rec.get('event_id') or '',
                    'ambiguity_tier':    rec.get('ambiguity_tier') or '',
                    'perturbation_type': rec.get('perturbation_type') or '',
                    'original_word':     rec.get('original_word') or '',
                    'replacement_word':  rec.get('replacement_word') or '',
                    'concordant':        rec.get('concordant') if rec.get('concordant') is not None else '',
                    'rationale_changed': rec.get('rationale_changed') if rec.get('rationale_changed') is not None else '',
                    'mentions_change':   rec.get('mentions_change') if rec.get('mentions_change') is not None else '',
                    'rfc_score':         rec.get('rfc_score') if rec.get('rfc_score') is not None else '',
                    'top_attributed_token': '',
                    'attribution_score':    '',
                })

        # LIG rows (one row per event, first top token)
        for rec in lig_results:
            top = rec.get('top_tokens_original', [])
            top_token = top[0]['token']     if top else ''
            top_score = top[0]['attribution'] if top else ''
            rows.append({
                'source':            'lig',
                'model':             rec.get('model') or 'conflibert',
                'event_id':          rec.get('event_id') or '',
                'ambiguity_tier':    rec.get('ambiguity_tier') or '',
                'perturbation_type': '',
                'original_word':     '',
                'replacement_word':  '',
                'concordant':        '',
                'rationale_changed': '',
                'mentions_change':   '',
                'rfc_score':         '',
                'top_attributed_token': top_token if top_token else '',
                'attribution_score':    top_score if top_score else '',
            })

        if not rows:
            print(f"[DEBUG] No rows to write in summary CSV. RFC models: {list(rfc_results.keys())}, LIG count: {len(lig_results)}")
            return

        out_path = self.results_base / 'error_trace_summary.csv'
        fieldnames = [
            'source', 'model', 'event_id', 'ambiguity_tier',
            'perturbation_type', 'original_word', 'replacement_word',
            'concordant', 'rationale_changed', 'mentions_change', 'rfc_score',
            'top_attributed_token', 'attribution_score',
        ]
        
        # Get list of models being updated (from rfc_results)
        new_models = set(rfc_results.keys())
        
        # Smart merge: replace model-specific rows, keep others
        all_rows = []
        if out_path.exists():
            # Read with dtype=str to prevent pandas from converting empty strings to nan
            existing_df = pd.read_csv(out_path, dtype=str)
            # Keep rows for models NOT being re-run
            all_rows = existing_df[~existing_df['model'].isin(new_models)].to_dict('records')
        
        # Add new rows
        all_rows.extend(rows)
        
        # Write all rows
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"✓ Summary CSV saved: {out_path}")


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_rfc_stats(records: List[Dict]) -> Dict[str, Any]:
    """Compute concordance rates overall and stratified by perturbation type and EAS tier."""
    resolvable = [r for r in records if r.get('concordant') is not None]
    if not resolvable:
        return {'n_resolvable': 0, 'concordant_rate': None,
                'discordant_rate': None, 'by_perturbation_type': {},
                'by_ambiguity_tier': {}}

    n = len(resolvable)
    n_concordant = sum(1 for r in resolvable if r['concordant'])

    # By perturbation type
    by_type: Dict[str, Dict] = {}
    for r in resolvable:
        pt = r.get('perturbation_type', 'unknown')
        by_type.setdefault(pt, {'n': 0, 'n_concordant': 0})
        by_type[pt]['n'] += 1
        if r['concordant']:
            by_type[pt]['n_concordant'] += 1
    by_type_stats = {
        pt: {
            'n':               v['n'],
            'concordant_rate': round(v['n_concordant'] / v['n'], 4),
            'discordant_rate': round(1.0 - v['n_concordant'] / v['n'], 4),
        }
        for pt, v in by_type.items()
    }

    # By EAS ambiguity tier
    by_tier: Dict[str, Dict] = {}
    for r in resolvable:
        tier = r.get('ambiguity_tier', 'Low')
        by_tier.setdefault(tier, {'n': 0, 'n_concordant': 0})
        by_tier[tier]['n'] += 1
        if r['concordant']:
            by_tier[tier]['n_concordant'] += 1
    by_tier_stats = {
        tier: {
            'n':               v['n'],
            'concordant_rate': round(v['n_concordant'] / v['n'], 4),
            'discordant_rate': round(1.0 - v['n_concordant'] / v['n'], 4),
        }
        for tier, v in by_tier.items()
    }

    return {
        'n_resolvable':      n,
        'concordant_rate':   round(n_concordant / n, 4),
        'discordant_rate':   round(1.0 - n_concordant / n, 4),
        'by_perturbation_type': by_type_stats,
        'by_ambiguity_tier':    by_tier_stats,
    }


def _print_rfc_summary(model: str, agg: Dict) -> None:
    """Print a human-readable RFC summary to stdout."""
    n = agg.get('n_resolvable', 0)
    if not n:
        print(f"    {model}: no resolvable flips.")
        return
    concordant_rate = agg.get('concordant_rate', 0.0)
    discordant_rate = agg.get('discordant_rate', 0.0)
    print(f"    {model}: {n} flips analysed  |  "
          f"concordant {concordant_rate:.0%}  |  discordant {discordant_rate:.0%}")
    tier_stats = agg.get('by_ambiguity_tier', {})
    for tier in ('Low', 'Medium', 'High'):
        if tier in tier_stats:
            ts = tier_stats[tier]
            print(f"      EAS {tier:6s}: n={ts['n']:3d}  "
                  f"discordant {ts['discordant_rate']:.0%}")


def _json_safe(obj: Any) -> Any:
    """JSON serialiser for numpy and boolean types not in standard json."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run error trace analysis driven by environment variables.

    Required env vars:
        COUNTRY, STRATEGY, SAMPLE_SIZE

    Optional env vars:
        NUM_EXAMPLES          (for few_shot strategy)
        LIG_N_STEPS           (int, default 50; increase to 100 for publication figures)
        COUNTERFACTUAL_JSON   (path to specific counterfactual JSON; if set, only process that file)
        CF_MODELS             (comma-separated models; used to filter/validate discovery)
    """
    parser = argparse.ArgumentParser(description='Error trace analysis (RFC + LIG)')
    parser.add_argument('--counterfactual-json', default=None, help='Path to specific counterfactual JSON to process')
    args = parser.parse_args()
    
    country      = os.environ.get('COUNTRY', 'cmr')
    strategy     = os.environ.get('STRATEGY', 'zero_shot')
    sample_size  = os.environ.get('SAMPLE_SIZE', '1000')
    num_examples = os.environ.get('NUM_EXAMPLES')
    lig_n_steps  = int(os.environ.get('LIG_N_STEPS', '50'))
    cf_models_env = os.environ.get('CF_MODELS', '').strip()

    analyser = ErrorTraceAnalyzer(country, strategy, sample_size, num_examples)

    # Priority: explicit --counterfactual-json argument or COUNTERFACTUAL_JSON env var
    explicit_json = args.counterfactual_json or os.environ.get('COUNTERFACTUAL_JSON', '').strip()
    
    if explicit_json:
        # Process only the explicitly specified JSON
        json_path = Path(explicit_json)
        if not json_path.exists():
            print(f"Error: counterfactual JSON not found: {json_path}")
            sys.exit(1)
        unique_jsons = [json_path]
        print(f"Processing explicit counterfactual JSON: {json_path}")
    else:
        # Auto-discover counterfactual JSONs in the results dir
        pattern      = str(analyser.results_base / '*' / 'counterfactual_analysis_*.json')
        json_files   = sorted(glob.glob(pattern))

        # Also check the top-level results dir (multi-model runs write there)
        pattern_top  = str(analyser.results_base / 'counterfactual_analysis_*.json')
        json_files  += sorted(glob.glob(pattern_top))

        # De-duplicate paths while preserving order
        seen = set()
        unique_jsons: List[Path] = []
        for p in json_files:
            rp = str(Path(p).resolve())
            if rp in seen:
                continue
            seen.add(rp)
            unique_jsons.append(Path(p))
        
        # If CF_MODELS is set, filter discovered JSONs to only those matching
        # the current model set (to avoid reprocessing previous runs).
        if cf_models_env and unique_jsons:
            cf_models_set = set(m.strip() for m in cf_models_env.split(',') if m.strip())
            filtered = []
            for json_path in unique_jsons:
                try:
                    with open(json_path) as f:
                        cf_data = json.load(f)
                    json_models_set = set(cf_data.get('metadata', {}).get('models', []))
                    # Include JSON if its model set matches current CF_MODELS
                    if json_models_set == cf_models_set:
                        filtered.append(json_path)
                except Exception as e:
                    print(f"Warning: could not read {json_path}: {e}")
            
            if filtered:
                unique_jsons = filtered
            # If filtering resulted in no matches, proceed with all discovered (fallback)

    if not unique_jsons:
        disagreements_csv = analyser.results_base / 'top_disagreements.csv'
        if not disagreements_csv.exists():
            print(f"No counterfactual JSON found under {analyser.results_base}")
            print("No top_disagreements.csv found for standalone LIG mode.")
            print("Run per-class metrics first, or run counterfactual analysis.")
            sys.exit(0)

        # Determine ConfliBERT tokens to analyse in standalone mode.
        # Priority: explicit CONFLIBERT_MODELS -> CF_MODELS subset -> default token.
        conflibert_models_env = os.environ.get('CONFLIBERT_MODELS', '').strip()
        if conflibert_models_env:
            conflibert_models = [
                m.strip() for m in conflibert_models_env.split(',') if m.strip()
            ]
        else:
            conflibert_models = [
                m.strip() for m in cf_models_env.split(',')
                if m.strip().lower().startswith('conflibert')
            ] if cf_models_env else []
            if not conflibert_models:
                conflibert_models = ['conflibert']

        print("No counterfactual JSON found; running standalone LIG from top_disagreements.csv")
        analyser.run_lig_from_disagreements(
            disagreements_csv_path=disagreements_csv,
            conflibert_models=conflibert_models,
            lig_n_steps=lig_n_steps,
        )
        return

    print(f"Discovered {len(unique_jsons)} counterfactual JSON file(s) for tracing.")
    analyser.run_many(unique_jsons, lig_n_steps=lig_n_steps)


if __name__ == '__main__':
    main()
