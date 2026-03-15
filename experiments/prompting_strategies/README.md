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
        """JSON schema for structured response.
        
        IMPORTANT: Use enum constraint on label field to ensure valid outputs.
        """
        return {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": ["V", "B", "E", "P", "R", "S"]},
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

### Prompt Best Practices

To ensure models output valid labels:
1. List categories with letter codes (V, B, E, P, R, S) - avoid numbered lists
2. Include explicit instruction: "CRITICAL: The label must be exactly one of: V, B, E, P, R, S"
3. Use enum constraint in JSON schema (enforced by Ollama structured output)

`explainable` and `few_shot` strategies also provide a `reasoning` property in the response schema. The downstream pipeline writes this field into every per-model CSV so you can trace how the model justified each prediction.

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
4. Run: `STRATEGY=my_strategy ./experiments/scripts/run_full_analysis.sh`

## Architecture

```
Prompting Strategy → Strategy Registry → Inference Client
     (prompts)      (lib/core)          (lib/inference)
```

Strategies generate prompts; the inference client handles API communication with no hardcoded prompts.
