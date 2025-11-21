"""Strategy helpers.

This module exposes a small, centralized registry and factory for the
prompting strategies used across the repository. Strategy classes are
imported lazily inside the module to avoid import-time cycles during
package initialization.

API
- `STRATEGY_REGISTRY`: mapping from strategy name to strategy class
- `get_strategy(name)`: instantiate a strategy by name
"""
import os
from typing import Any, Dict

# Import strategy classes at module import time. These modules are lightweight
# and do not create an import cycle in the current package layout.
from experiments.prompting_strategies import ZeroShotStrategy, FewShotStrategy, ExplainableStrategy


def _build_strategy_registry() -> Dict[str, type]:
    """Return the mapping of canonical strategy names to classes."""
    return {
        'zero_shot': ZeroShotStrategy,
        'few_shot': FewShotStrategy,
        'explainable': ExplainableStrategy,
    }


STRATEGY_REGISTRY = _build_strategy_registry()


def get_strategy(strategy_name: str) -> Any:
    """Instantiate a prompting strategy by name.

    The `few_shot` strategy reads `EXAMPLES_PER_CATEGORY` from the environment
    (if present) to set the number of demonstration examples.

    Returns an instance of the selected strategy class.
    """
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
