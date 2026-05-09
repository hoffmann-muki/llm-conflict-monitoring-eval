# Tests

The test suite is a lightweight smoke-test harness for the companion repository to our paper **Are LLMs Ready for Conflict Monitoring? Empirical Evidence from West Africa** ([arXiv:2605.04177](https://arxiv.org/abs/2605.04177)). It verifies that core helpers, strategy imports, and counterfactual scaffolding still load and behave enough for development sanity checks.

These tests are meant to protect the research pipeline from breakage; they are not statistical validation of our findings.

## Running Tests

```bash
source .venv/bin/activate

PYTHONPATH=. python tests/test_generic_pipeline.py

# Optional, if pytest is installed in your environment.
python -m pytest tests/ -v
```

The counterfactual test expects a Nigeria sample file. If no file exists under `datasets/nga/state_actor_sample_nga_*.csv`, run a small pipeline first:

```bash
COUNTRY=nga SAMPLE_SIZE=5 STRATEGY=zero_shot INFERENCE_MODELS="mistral:7b" \
  ./experiments/scripts/run_full_analysis.sh
```

That command requires Ollama and the requested model; for import-only validation, use `test_generic_pipeline.py`.

## Test Files

| File | What it checks |
| --- | --- |
| `test_generic_pipeline.py` | Imports and basic behavior for path setup and `ZeroShotStrategy`. |
| `test_counterfactual.py` | Counterfactual analyzer construction, perturbation generation, and output-directory setup using an existing sample. |

## Current Caveats

- These tests are not a full regression suite for model quality.
- They do not mock Ollama, Hugging Face, or ConfliBERT backends.
- `test_counterfactual.py` still prints older suggested command names in its terminal output; the real current modules are `lib.analysis.counterfactual` and `lib.analysis.visualize_counterfactual`.
- `python -m pytest tests/ -v` may create `test_counterfactual_output/` when the counterfactual smoke test runs.

## Useful Validation Commands

```bash
# Import-level smoke test.
PYTHONPATH=. python tests/test_generic_pipeline.py

# List discovered per-model result files for a run.
python -m lib.core.result_aggregator \
  --country cmr --strategy zero_shot --sample-size 1000 --list-only

# Recompute analysis from existing per-model predictions.
COUNTRY=cmr STRATEGY=zero_shot SAMPLE_SIZE=1000 SKIP_INFERENCE=true \
SKIP_COUNTERFACTUAL=true \
  ./experiments/scripts/run_full_analysis.sh
```

## Writing New Tests

Prefer normal `pytest` tests with assertions:

```python
def test_strategy_schema_has_label_enum():
    from experiments.prompting_strategies import ZeroShotStrategy

    schema = ZeroShotStrategy().get_schema()
    assert schema["properties"]["label"]["enum"] == ["V", "B", "E", "P", "R", "S"]
```

For code that touches model backends, keep tests small and isolate network or local-model dependencies behind explicit fixtures or environment gates.
