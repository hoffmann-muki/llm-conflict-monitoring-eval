# Prompting Strategies

Modular prompting strategies for event classification.

## Available Strategies

| Strategy | Description | Config |
|----------|-------------|--------|
| `zero_shot` | Direct classification without examples | Default |
| `few_shot` | Classification with examples per category | `NUM_EXAMPLES=1..5` |
| `explainable` | Chain-of-thought reasoning | - |

## Usage

```python
from experiments.prompting_strategies import ZeroShotStrategy, FewShotStrategy

# Zero-shot
strategy = ZeroShotStrategy()
prompt = strategy.make_prompt("Military forces attacked civilians")
system_msg = strategy.get_system_message()

# Few-shot with configurable examples
strategy = FewShotStrategy(num_examples=3)
prompt = strategy.make_prompt("Protesters gathered in the capital")
```

## Creating Custom Strategies

Inherit from `PromptingStrategy`:

```python
from experiments.prompting_strategies.base import PromptingStrategy
from typing import Dict, Any, Optional

class MyStrategy(PromptingStrategy):
    def make_prompt(self, note: str) -> str:
        """Generate classification prompt."""
        return f"Classify this event: {note}"
    
    def get_schema(self) -> Dict[str, Any]:
        """JSON schema for structured response."""
        return {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "confidence": {"type": "number"}
            },
            "required": ["label", "confidence"]
        }
    
    def get_system_message(self) -> Optional[str]:
        """Optional system message."""
        return "You are an expert conflict event classifier."
    
    def get_name(self) -> str:
        """Strategy name for results organization."""
        return "my_strategy"
```

## Base Class Interface

| Method | Returns | Required |
|--------|---------|----------|
| `make_prompt(note)` | `str` | Yes |
| `get_schema()` | `Dict` | Yes |
| `get_system_message()` | `Optional[str]` | Yes |
| `get_name()` | `str` | No (defaults to class name) |

## Registering Strategies

1. Create strategy file in this directory
2. Add import to `__init__.py`
3. Register in `lib/core/strategy_helpers.py`
4. Run: `STRATEGY=my_strategy ./experiments/scripts/run_ollama_full_analysis.sh`

## Architecture

```
Prompting Strategy → Strategy Registry → Inference Client
     (prompts)      (lib/core)          (lib/inference)
```

Strategies generate prompts; the inference client handles API communication with no hardcoded prompts.
