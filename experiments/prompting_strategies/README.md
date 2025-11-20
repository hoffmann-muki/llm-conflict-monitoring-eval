# Prompting Strategies

Modular prompting strategies for classification experiments.

## Available Strategies

### Zero-Shot
Direct classification without examples. Default strategy for experiments.

```python
from experiments.prompting_strategies import ZeroShotStrategy

strategy = ZeroShotStrategy()
prompt = strategy.make_prompt("Military forces attacked civilians")
```

### Few-Shot
Classification with example demonstrations (1-5 examples per category).

### Explainable
Chain-of-thought reasoning prompts for transparent decision-making.

## Creating New Strategies

All strategies must inherit from `PromptingStrategy` base class:

```python
from experiments.prompting_strategies.base import PromptingStrategy
from typing import Dict, Any, Optional

class MyStrategy(PromptingStrategy):
    def make_prompt(self, note: str) -> str:
        """Generate prompt for the event note."""
        return f"Your prompt template: {note}"
    
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
        return "You are an expert classifier."
```

## Base Class Interface

### Required Methods

- `make_prompt(note: str) -> str`: Generate prompt for classification
- `get_schema() -> Dict`: Return JSON schema for responses
- `get_system_message() -> Optional[str]`: Return system message (or None)

### Optional Methods

- `get_name() -> str`: Override to customize strategy name (used in results organization)

## Event Categories

All strategies should support these ACLED event types:
- V = Violence against civilians
- B = Battles
- E = Explosions/Remote violence
- P = Protests
- R = Riots
- S = Strategic developments

## Integration

1. Create your strategy file in this directory
2. Add import to `__init__.py`
3. Register in `experiments/pipelines/run_classification.py` STRATEGY_REGISTRY
4. Run experiments:
   ```bash
   STRATEGY=my_strategy COUNTRY=cmr SAMPLE_SIZE=500 \
     ./experiments/scripts/run_experiment.sh
   ```

**Architecture Note:** The inference layer (`lib.inference.ollama_client`) requires explicit prompts - it contains no hardcoded prompts. This ensures complete separation: strategies generate prompts, the inference client handles API communication.
