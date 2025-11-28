# Prompting Strategies

Modular prompting strategies for event classification experiments.

## Available Strategies

| Strategy | Description | Environment Variable | Use Case |
|----------|-------------|---------------------|----------|
| `zero_shot` | Direct classification without examples | `STRATEGY=zero_shot` | Default, baseline comparison |
| `few_shot` | Classification with 1-5 examples per category | `STRATEGY=few_shot NUM_EXAMPLES=3` | Improved accuracy |
| `explainable` | Chain-of-thought reasoning prompts | `STRATEGY=explainable` | Transparent decision-making |

## Usage

```python
from experiments.prompting_strategies import ZeroShotStrategy, FewShotStrategy

# Zero-shot (default)
strategy = ZeroShotStrategy()
prompt = strategy.make_prompt("Military forces attacked civilians in the village")
system_msg = strategy.get_system_message()

# Few-shot with configurable number of examples
strategy = FewShotStrategy(num_examples=3)  # 1-5 examples per category
prompt = strategy.make_prompt("Protesters gathered in the capital")
```

### Environment-Based Configuration

The number of few-shot examples can be configured via environment variable:

```bash
# Run with 3 examples per category
COUNTRY=cmr STRATEGY=few_shot SAMPLE_SIZE=500 NUM_EXAMPLES=3 \
  python experiments/pipelines/ollama/run_ollama_classification.py
```

Results are organized into subdirectories by number of examples:
```
results/cmr/few_shot/500/3/   # NUM_EXAMPLES=3
results/cmr/few_shot/500/5/   # NUM_EXAMPLES=5
```

## Creating Custom Strategies

All strategies must inherit from the `PromptingStrategy` base class:

```python
from experiments.prompting_strategies.base import PromptingStrategy
from typing import Dict, Any, Optional

class MyStrategy(PromptingStrategy):
    def make_prompt(self, note: str) -> str:
        """Generate classification prompt for the event note."""
        return f"Classify this event: {note}"
    
    def get_schema(self) -> Dict[str, Any]:
        """Define expected JSON response schema."""
        return {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "confidence": {"type": "number"}
            },
            "required": ["label", "confidence"]
        }
    
    def get_system_message(self) -> Optional[str]:
        """Optional system message for the model."""
        return "You are an expert conflict event classifier."
    
    def get_name(self) -> str:
        """Strategy name for results organization."""
        return "my_strategy"
```

## Base Class Interface

### Required Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `make_prompt(note)` | `str` | Generate prompt for classification |
| `get_schema()` | `Dict` | JSON schema for structured responses |
| `get_system_message()` | `Optional[str]` | System message (or None) |

### Optional Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_name()` | `str` | Strategy name (defaults to class name) |

## Event Categories

All strategies classify events into ACLED categories:

| Code | Category |
|------|----------|
| V | Violence against civilians |
| B | Battles |
| E | Explosions/Remote violence |
| P | Protests |
| R | Riots |
| S | Strategic developments |

## Registering New Strategies

1. Create your strategy file in this directory
2. Add import to `__init__.py`
3. Register in `lib/core/strategy_helpers.py`:

```python
# In STRATEGY_REGISTRY
STRATEGY_REGISTRY = {
    'zero_shot': ZeroShotStrategy,
    'few_shot': FewShotStrategy,
    'explainable': ExplainableStrategy,
    'my_strategy': MyStrategy,  # Add your strategy
}
```

4. Run experiments:
```bash
STRATEGY=my_strategy COUNTRY=cmr SAMPLE_SIZE=500 \
  ./experiments/scripts/run_ollama_full_analysis.sh
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Prompting Strategy                                               │
│   - Generates prompts from event text                           │
│   - Defines response schema                                      │
│   - Provides system message                                      │
│   - Configurable parameters (e.g., num_examples for few_shot)   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Strategy Registry (lib/core/strategy_helpers.py)                │
│   - Maps strategy names to classes                              │
│   - Factory function get_strategy()                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Inference Client (lib/inference/ollama_client.py)               │
│   - Handles API communication                                   │
│   - No hardcoded prompts - uses strategy output                 │
└─────────────────────────────────────────────────────────────────┘
```
