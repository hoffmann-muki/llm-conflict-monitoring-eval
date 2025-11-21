"""Project-wide constants for labels and event classes."""
LABEL_MAP = {
    "Violence against civilians": "V",
    "Battles": "B",
    "Explosions/Remote violence": "E",
    "Protests": "P",
    "Riots": "R",
    "Strategic developments": "S"
}

EVENT_CLASSES_FULL = [
    "Violence against civilians",
    "Battles",
    "Explosions/Remote violence",
    "Protests",
    "Riots",
    "Strategic developments"
]

# Source CSV used by country pipelines
CSV_SRC = "datasets/Africa_lagged_data_up_to-2024-10-24.csv"

WORKING_MODELS = ["llama3.1:8b", "qwen3:8b", "mistral:7b", "gemma3:7b", "olmo2:7b"]

# Country name mapping used across pipelines
COUNTRY_NAMES = {
    'cmr': 'Cameroon',
    'nga': 'Nigeria',
}


# Strategy registry and helper: expose common prompting strategy names
# as a repository-wide constant. Import strategy classes lazily to avoid
# circular imports during module import time in certain environments.
def _build_strategy_registry():
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
