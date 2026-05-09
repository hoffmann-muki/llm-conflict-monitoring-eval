# Prompting Strategies

Prompting strategies define the user prompt, optional system message, and JSON schema used by generative classifiers in our paper **Are LLMs Ready for Conflict Monitoring? Empirical Evidence from West Africa** ([arXiv:2605.04177](https://arxiv.org/abs/2605.04177)). The active registry lives in `lib/core/strategy_helpers.py`.

Our primary comparisons use controlled prompting so that differences in False Legitimization, False Illegitimization, actor bias, and lexical robustness are attributable to model behavior rather than ad hoc prompt variation.

## Available Strategies

| Strategy name | Class | Response fields | Notes |
| --- | --- | --- | --- |
| `zero_shot` | `ZeroShotStrategy` | `label`, `confidence`, optional `logits` | Default direct classification prompt. |
| `few_shot` | `FewShotStrategy` | `label`, `confidence`, `logits` | Uses 1 to 5 fixed demonstrations per class. |
| `explainable` | `ExplainableStrategy` | `reasoning`, `label`, `confidence`, `logits` | Requires exactly three short reasoning observations before the final JSON. |

All strategies classify into the same label set: `V`, `B`, `E`, `P`, `R`, and `S`.

## Usage

```python
from experiments.prompting_strategies import ZeroShotStrategy, FewShotStrategy, ExplainableStrategy

strategy = ZeroShotStrategy()
prompt = strategy.make_prompt("Military forces attacked civilians in the village.")
schema = strategy.get_schema()
system_msg = strategy.get_system_message()

few_shot = FewShotStrategy(config={"examples_per_category": 3})
few_shot_prompt = few_shot.make_prompt("Protesters gathered in the capital.")

explainable = ExplainableStrategy()
explainable_schema = explainable.get_schema()
```

Most callers should use the central factory:

```python
from lib.core.strategy_helpers import get_strategy

strategy = get_strategy("few_shot", num_examples=3)
```

## Base Interface

Each strategy subclasses `PromptingStrategy` from `base.py`.

| Method | Required | Purpose |
| --- | --- | --- |
| `make_prompt(event_note)` | yes | Return the text prompt for one event. |
| `get_schema()` | yes | Return the JSON schema passed to structured generation. |
| `get_system_message()` | yes | Return a system message string or `None`. |
| `get_name()` | no | Defaults to the class name without `Strategy`, lowercased. |

## JSON Output Contract

Generative inference expects JSON with a valid `label` enum:

```json
{
  "label": "V",
  "confidence": 0.87,
  "logits": {
    "V": 0.87,
    "B": 0.04,
    "E": 0.02,
    "P": 0.01,
    "R": 0.03,
    "S": 0.03
  }
}
```

The `explainable` strategy also requires:

```json
{
  "reasoning": [
    "Actors: government soldiers and civilians.",
    "Actions: soldiers fired on civilians.",
    "Rationale: direct harm to civilians fits V."
  ],
  "label": "V",
  "confidence": 0.91
}
```

Downstream CSVs include a `reasoning` column when the model returns reasoning; non-explainable runs leave it empty.

## Few-Shot Configuration

The few-shot strategy accepts `examples_per_category` from 1 to 5:

```python
strategy = FewShotStrategy(config={"examples_per_category": 5})
```

CLI and shell paths use:

```bash
STRATEGY=few_shot NUM_EXAMPLES=3 ./experiments/scripts/run_full_analysis.sh

python experiments/pipelines/ollama/run_ollama_classification.py cmr \
  --strategy few_shot \
  --num-examples 3
```

Few-shot results are written under:

```text
results/{country}/few_shot/{sample_size}/{num_examples}/
```

## Adding A Strategy

1. Create a new file in this directory, for example `my_strategy.py`.
2. Subclass `PromptingStrategy`.
3. Add the class import to `experiments/prompting_strategies/__init__.py`.
4. Register the strategy name in `lib/core/strategy_helpers.py`.
5. If the strategy needs special CLI options, update the relevant pipeline argument parsing.
6. Run the smoke test and a tiny inference run.

Example:

```python
from typing import Any, Dict, Optional

from experiments.prompting_strategies.base import PromptingStrategy


class MyStrategy(PromptingStrategy):
    def make_prompt(self, event_note: str) -> str:
        return f"""Classify this ACLED event.

Event: {event_note}

Return JSON with label and confidence."""

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": ["V", "B", "E", "P", "R", "S"]},
                "confidence": {"type": "number"},
            },
            "required": ["label", "confidence"],
        }

    def get_system_message(self) -> Optional[str]:
        return "You are an expert conflict event classifier."
```

## Prompt Guidelines

Use these conventions so the analysis pipeline stays robust:

- Keep the label field constrained to the enum `["V", "B", "E", "P", "R", "S"]`.
- Ask for JSON only; avoid prose after the object.
- Include `confidence` as a number between 0 and 1.
- Prefer `logits` as an object keyed by the six labels so calibration can use stable ordering.
- If asking for reasoning, keep it bounded and structured; do not request long free-form chain-of-thought.

For experiments aligned with our paper, keep prompts stable across countries and models unless the experimental condition explicitly studies prompt variation.
