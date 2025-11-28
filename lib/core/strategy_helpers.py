"""Strategy helpers.

This module exposes a small, centralized registry and factory for the
prompting strategies used across the repository. Strategy classes are
imported lazily inside the module to avoid import-time cycles during
package initialization.

API
- `STRATEGY_REGISTRY`: mapping from strategy name to strategy class
- `get_strategy(name, num_examples)`: instantiate a strategy by name
"""
import os
from typing import Any, Dict, Optional

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


def get_strategy(strategy_name: str, num_examples: Optional[int] = None) -> Any:
    """Instantiate a prompting strategy by name.

    Args:
        strategy_name: Name of the strategy ('zero_shot', 'few_shot', 'explainable')
        num_examples: Number of few-shot examples (1-5). Only used for 'few_shot' strategy.
                     If None, reads from NUM_EXAMPLES or EXAMPLES_PER_CATEGORY env vars.

    Returns an instance of the selected strategy class.
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    config = None
    if strategy_name == 'few_shot':
        # Priority: explicit argument -> NUM_EXAMPLES env -> EXAMPLES_PER_CATEGORY env -> default 1
        if num_examples is not None:
            examples_per_category = num_examples
        else:
            env_val = os.environ.get('NUM_EXAMPLES') or os.environ.get('EXAMPLES_PER_CATEGORY', '1')
            try:
                examples_per_category = int(env_val)
            except ValueError:
                examples_per_category = 1
        config = {'examples_per_category': examples_per_category}

    cls = STRATEGY_REGISTRY[strategy_name]
    return cls(config=config if config else None)
