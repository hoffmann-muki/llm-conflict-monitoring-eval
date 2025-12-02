"""Analysis tools for metrics, calibration, and visualization."""

from .auto_annotate import (
    classify_provenance,
    classify_verb_intensity,
    detect_casualties,
    detect_passive_voice,
    detect_ambiguous_actor,
    auto_annotate_row,
    auto_annotate_dataframe,
)

__all__ = [
    'classify_provenance',
    'classify_verb_intensity',
    'detect_casualties',
    'detect_passive_voice',
    'detect_ambiguous_actor',
    'auto_annotate_row',
    'auto_annotate_dataframe',
]
