"""Helper utilities for prompting strategies.

This module centralizes strategy instantiation and the strategy registry.
It lives under `lib.core` so other modules can import it without
creating circular import issues.
"""
from typing import Dict


def _build_strategy_registry() -> Dict[str, type]:
    # Import strategy classes lazily to avoid import-time cycles
    from experiments.prompting_strategies import ZeroShotStrategy, FewShotStrategy, ExplainableStrategy
    return {
        'zero_shot': ZeroShotStrategy,
        'few_shot': FewShotStrategy,
        'explainable': ExplainableStrategy,
    }


STRATEGY_REGISTRY = _build_strategy_registry()


def get_strategy(strategy_name: str):
    """Instantiate a prompting strategy by name using repo defaults.

    For `few_shot`, this reads `EXAMPLES_PER_CATEGORY` from the environment
    (if present) to set the number of demonstration examples.
    """
    import os

    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    config = None
    if strategy_name == 'few_shot':
        examples_per_category = os.environ.get('EXAMPLES_PER_CATEGORY', '1')
        try:
            config = {'examples_per_category': int(examples_per_category)}
        except ValueError:
            config = {'examples_per_category': 1}

    cls = STRATEGY_REGISTRY[strategy_name]
    return cls(config=config if config else None)
