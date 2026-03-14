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

from .event_ambiguity import (
    compute_event_ambiguity_score,
    assign_ambiguity_tier,
    annotate_disagreements_with_ambiguity,
)

__all__ = [
    'classify_provenance',
    'classify_verb_intensity',
    'detect_casualties',
    'detect_passive_voice',
    'detect_ambiguous_actor',
    'auto_annotate_row',
    'auto_annotate_dataframe',
    'compute_event_ambiguity_score',
    'assign_ambiguity_tier',
    'annotate_disagreements_with_ambiguity',
]


def __getattr__(name):
    """Lazy-load heavy analysis modules on demand."""
    if name == 'ErrorTraceAnalyzer':
        from .error_trace import ErrorTraceAnalyzer
        return ErrorTraceAnalyzer
    raise AttributeError(name)


__all__.append('ErrorTraceAnalyzer')
