#!/usr/bin/env python3
"""Event Ambiguity Scoring for targeted error analysis.

Implements a composite Event Ambiguity Score (EAS) that separates
task-inherent uncertainty (aleatoric) from model-induced bias (epistemic)
in inter-model disagreement events.

The EAS combines four independent dimensions:

  1. Label entropy across models
     Shannon entropy H({y_m}) / log2(N_classes), normalised to [0,1].
     Grounded in Item Response Theory (Passonneau & Carpenter 2014) and the
     LLM-as-annotator paradigm (Plank 2022; Gilardi et al. 2023): systematic
     cross-model label disagreement is the strongest behavioural signal of
     genuine item difficulty.

  2. Mean model confidence (inverted)
     1 - mean(pred_conf_temp across models). Low mean confidence indicates
     that no model "owns" the event with certainty, consistent with an
     inherently underspecified description. (Kendall & Gal 2017.)

  3. Confidence dispersion across models
     Std dev of calibrated confidence scores. High dispersion indicates some
     models are pulled strongly toward different regions of the decision space,
     consistent with boundary-case descriptions. (Epistemic uncertainty proxy.)

  4. Text-based ambiguity features (lower weight)
     Draws on existing auto_annotate.py detectors for:
       - Unidentified/unknown actors (V/B boundary signal; ACLED methodology)
       - Passive/agent-obscuring voice
       - Hedging epistemic markers
       - V/B boundary-specific lexical cues

Dimensions 1–3 are behavioural (model-derived) and receive higher weight;
dimension 4 is a text heuristic and receives lower weight. This ordering
follows the principle that behavioural evidence from multiple annotators is
more reliable than surface-level text cues (Aroyo & Welty 2015).

Tier assignment (for stratification):
    High:   EAS >= TIER_HIGH_THRESHOLD  (0.60)
    Medium: EAS >= TIER_MED_THRESHOLD   (0.30)
    Low:    EAS <  TIER_MED_THRESHOLD

Usage (standalone):
    from lib.analysis.event_ambiguity import compute_event_ambiguity_score, assign_ambiguity_tier

Usage (via dataframe):
    from lib.analysis.event_ambiguity import annotate_disagreements_with_ambiguity
    annotated_df = annotate_disagreements_with_ambiguity(wide_df)
"""
from __future__ import annotations

import re
import math
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Weights for the four EAS dimensions (must sum to 1.0)
W_LABEL_ENTROPY      = 0.35   # Strongest signal: behavioural label disagreement
W_CONF_UNCERTAINTY   = 0.25   # Inverted mean confidence
W_CONF_DISPERSION    = 0.20   # Confidence variance across models
W_TEXT_AMBIGUITY     = 0.20   # Text-based heuristics (lower weight)

# Ambiguity tier thresholds
TIER_HIGH_THRESHOLD  = 0.60
TIER_MED_THRESHOLD   = 0.30

# Number of event-type classes in ACLED coding scheme used here (V,B,E,P,R,S)
N_CLASSES = 6

# ---------------------------------------------------------------------------
# V/B boundary-specific text patterns
# These cues are specifically associated with the Violence-against-civilians /
# Battles coding boundary in ACLED (see Raleigh et al. 2010 and ACLED codebook).
# Unidentified actor language is the single strongest boundary signal.
# ---------------------------------------------------------------------------

VB_BOUNDARY_PATTERNS = [
    # Unidentified or attribution-ambiguous actors
    r'\bunknown\s+(assailants?|gunmen|attackers?|perpetrators?|actors?|militants?|armed\s+men)',
    r'\bunidentified\s+(assailants?|gunmen|attackers?|perpetrators?|actors?|militants?|armed\s+men)',
    r'\bsuspected\s+(militants?|gunmen|attackers?|terrorists?|jihadists?)',
    r'\balleged(ly)?\s+(militants?|gunmen|attackers?|terrorists?)',
    r'\b(armed\s+men|armed\s+group)\b',
    r'\bno\s+(group|one)\s+(has\s+)?claimed',
    r'\b(disputed|conflicting)\s+(reports?|accounts?)',
    # Ambiguous target language (civilian vs. combatant)
    r'\b(?:civilian\s+casualt|civilian\s+victim)',     # explicit civilian framing
    r'\b(?:killed\s+and\s+injur|injur\w+\s+and\s+kill)',  # mixed casualty descriptions
    # Dual-status actors (state actor who could be V or B perpetrator)
    r'\b(self-defense|self\s+defense|exchange\s+of\s+fire|crossfire)',
    r'\b(counter[- ]?attack|retali)',
]

# Hedging/epistemic uncertainty markers that reduce classification confidence
HEDGING_PATTERNS = [
    r'\ballegedly?\b',
    r'\breportedly?\b',
    r'\bapparently?\b',
    r'\bseemingly\b',
    r'\bpossibly?\b',
    r'\bsupposedly?\b',
    r'\baccording\s+to\s+(?:unconfirmed|unverified)',
    r'\bcould\s+not\s+be\s+(confirmed|verified|independently)',
    r'\b(unconfirmed|unverified)\b',
]


# ---------------------------------------------------------------------------
# Text-based ambiguity scoring
# ---------------------------------------------------------------------------

def _score_text_ambiguity(notes: Optional[str], actor_norm: Optional[str]) -> float:
    """Return a text-based ambiguity score in [0, 1].

    Draws on V/B boundary patterns, hedging markers, passive voice, and
    actor identification signals.  All components are binary presence tests;
    the final score is a normalised count, capped at 1.0.
    """
    if not notes:
        return 0.0

    text = str(notes).lower()
    actor = str(actor_norm or '').lower()
    score = 0.0

    # --- Component 1: V/B boundary patterns (weight 0.5 of text score) -----
    vb_hits = sum(1 for p in VB_BOUNDARY_PATTERNS if re.search(p, text, re.IGNORECASE))
    vb_sub = min(1.0, vb_hits / 3.0)   # normalise: 3+ hits = max score
    score += 0.50 * vb_sub

    # --- Component 2: Hedging patterns (weight 0.25 of text score) ----------
    hedge_hits = sum(1 for p in HEDGING_PATTERNS if re.search(p, text, re.IGNORECASE))
    hedge_sub = min(1.0, hedge_hits / 2.0)   # 2+ hedging markers = max
    score += 0.25 * hedge_sub

    # --- Component 3: Actor ambiguity via actor_norm column (weight 0.25) ---
    # Uses the same actor_norm signal as auto_annotate.detect_ambiguous_actor
    actor_ambiguous = any(term in actor for term in [
        'unknown', 'unidentified', 'suspected', 'generic', 'armed group'
    ])
    score += 0.25 * (1.0 if actor_ambiguous else 0.0)

    return min(1.0, score)


# ---------------------------------------------------------------------------
# Behavioural (model-derived) ambiguity components
# ---------------------------------------------------------------------------

def _compute_label_entropy(labels: List[str]) -> float:
    """Normalised Shannon entropy of a list of predicted labels.

    Returns a value in [0, 1].  Maximum entropy (1.0) when all N_CLASSES
    labels appear with equal frequency.
    """
    if not labels:
        return 0.0
    counts: Dict[str, int] = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    n = len(labels)
    h = -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)
    max_entropy = math.log2(N_CLASSES)
    return h / max_entropy if max_entropy > 0 else 0.0


def _compute_confidence_components(probs: List[float]):
    """Return (mean_uncertainty, dispersion) from a list of confidence scores.

    mean_uncertainty = 1 - mean(probs), in [0, 1]
    dispersion       = std(probs), in [0, 0.5] (normalised to [0, 1])
    """
    if not probs:
        return 0.0, 0.0
    arr = np.array(probs, dtype=float)
    mean_unc = float(1.0 - np.clip(np.mean(arr), 0.0, 1.0))
    # Std of a [0,1] variable is at most 0.5 (Bernoulli); normalise to [0,1]
    dispersion = float(np.clip(np.std(arr) / 0.5, 0.0, 1.0))
    return mean_unc, dispersion


# ---------------------------------------------------------------------------
# Composite EAS computation
# ---------------------------------------------------------------------------

def compute_event_ambiguity_score(
    notes: Optional[str],
    actor_norm: Optional[str],
    model_labels: List[str],
    model_probs: List[float],
) -> Dict[str, Any]:
    """Compute the composite Event Ambiguity Score (EAS) for a single event.

    Args:
        notes:        Raw event description text (ACLED NOTES field).
        actor_norm:   Normalised actor string from the dataset.
        model_labels: List of predicted labels, one per model.
        model_probs:  List of calibrated confidence scores, one per model.
                      Must align positionally with model_labels.

    Returns:
        Dict with keys:
            ambiguity_score       float  [0, 1]  composite EAS
            ambiguity_tier        str    'High' | 'Medium' | 'Low'
            eas_label_entropy     float  label-entropy component
            eas_conf_uncertainty  float  confidence-uncertainty component
            eas_conf_dispersion   float  confidence-dispersion component
            eas_text_ambiguity    float  text-based component
    """
    # Behavioural components
    label_entropy = _compute_label_entropy(model_labels)
    conf_uncertainty, conf_dispersion = _compute_confidence_components(model_probs)

    # Text component
    text_ambiguity = _score_text_ambiguity(notes, actor_norm)

    # Composite EAS (weighted sum)
    eas = (
        W_LABEL_ENTROPY    * label_entropy
        + W_CONF_UNCERTAINTY * conf_uncertainty
        + W_CONF_DISPERSION  * conf_dispersion
        + W_TEXT_AMBIGUITY   * text_ambiguity
    )
    eas = float(np.clip(eas, 0.0, 1.0))

    return {
        'ambiguity_score':       round(eas, 4),
        'ambiguity_tier':        assign_ambiguity_tier(eas),
        'eas_label_entropy':     round(label_entropy, 4),
        'eas_conf_uncertainty':  round(conf_uncertainty, 4),
        'eas_conf_dispersion':   round(conf_dispersion, 4),
        'eas_text_ambiguity':    round(text_ambiguity, 4),
    }


def assign_ambiguity_tier(score: float) -> str:
    """Map a continuous EAS to a categorical tier label.

    High   >= TIER_HIGH_THRESHOLD (0.60)
    Medium >= TIER_MED_THRESHOLD  (0.30)
    Low    <  TIER_MED_THRESHOLD
    """
    if score >= TIER_HIGH_THRESHOLD:
        return 'High'
    elif score >= TIER_MED_THRESHOLD:
        return 'Medium'
    return 'Low'


# ---------------------------------------------------------------------------
# DataFrame-level annotation helper (called from per_class_metrics.py)
# ---------------------------------------------------------------------------

def annotate_disagreements_with_ambiguity(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Annotate a wide-format disagreement dataframe with EAS columns.

    The input dataframe is expected to have:
      - 'notes'      (str):  event description
      - 'actor_norm' (str):  normalised actor name
      - Columns matching 'pred_label_*'   (one per model)
      - Columns matching 'pred_prob_*'    (one per model)

    Missing 'notes' or 'actor_norm' columns are handled gracefully (text
    component will be 0).

    Returns a copy of the dataframe with six new columns:
        ambiguity_score, ambiguity_tier,
        eas_label_entropy, eas_conf_uncertainty,
        eas_conf_dispersion, eas_text_ambiguity
    """
    df = wide_df.copy()

    label_cols = [c for c in df.columns if c.startswith('pred_label_')]
    prob_cols  = [c for c in df.columns if c.startswith('pred_prob_')]

    if not label_cols:
        # No model prediction columns present — fill with neutral defaults
        for col in ['ambiguity_score', 'ambiguity_tier', 'eas_label_entropy',
                    'eas_conf_uncertainty', 'eas_conf_dispersion', 'eas_text_ambiguity']:
            df[col] = None
        return df

    def _score_row(row: pd.Series) -> pd.Series:
        notes      = row.get('notes')
        actor_norm = row.get('actor_norm')

        model_labels = [row[c] for c in label_cols if pd.notna(row.get(c))]
        model_probs  = [float(row[c]) for c in prob_cols  if pd.notna(row.get(c))]

        result = compute_event_ambiguity_score(notes, actor_norm, model_labels, model_probs)
        return pd.Series(result)

    scores = df.apply(_score_row, axis=1)
    df = pd.concat([df, scores], axis=1)
    return df
