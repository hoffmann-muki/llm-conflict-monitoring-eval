# Experimental prompt conditions

A strategy defines the instructions, system message, and response schema presented to a generative classifier. Keeping these components explicit makes prompt changes traceable across models, countries, and SFT runs.

## What varies

| Condition | Source | Information supplied to the model |
| --- | --- | --- |
| `zero_shot` | [zero_shot.py](zero_shot.py) | The six event categories and a structured classification request. |
| `few_shot` | [few_shot.py](few_shot.py) | Labeled demonstrations followed by the event to classify. |
| `explainable` | [explainable.py](explainable.py) | A request for three brief observations about actors, actions, and the assigned category. |

Few-shot examples are embedded in the source. For each class, the strategy takes the first *k* entries from a fixed pool, with `k` between 1 and 5. Thus `k=3` supplies 18 demonstrations. It does not draw examples from the current evaluation sample.

## Construct a prompt

```python
from lib.core.strategy_helpers import get_strategy

condition = get_strategy("few_shot", num_examples=3)
messages = {
    "system": condition.get_system_message(),
    "user": condition.make_prompt("Residents demonstrated outside the town hall."),
}
response_schema = condition.get_schema()
```

The factory defaults to one example per class; the shell experiment runner defaults to three. Set the count explicitly when recording an experimental condition.

## Read the response

Every condition requests `label` and `confidence`. Labels are constrained to `V/B/E/P/R/S`, and the prompt asks for confidence between zero and one. All prompts also request a six-class probability object named `logits`; the name is historical. It is optional in the zero-shot schema and required in the other two.

The explainable condition adds a `reasoning` array. Its instructions limit the response to three observations of at most 20 words each, although the schema does not enforce those lengths. Treat these observations as model-generated rationales for analysis.

[prepare_sft_data.py](../pipelines/ollama/prepare_sft_data.py) reuses the selected strategy to format training examples. It constructs targets from gold labels, using unit confidence, one-hot class scores, and templated reasoning where those fields are present. These targets are generated supervision, not human rationale annotations.

## Introduce a new condition

Implement the three abstract methods in [PromptingStrategy](base.py), export the class in [__init__.py](__init__.py), and add its name to the [strategy registry](../../lib/core/strategy_helpers.py) and relevant pipeline CLI choices. Verify prompt construction and a small inference run before comparing conditions. Retain the label vocabulary and record the exact prompt source, example count, and example order with the results.
