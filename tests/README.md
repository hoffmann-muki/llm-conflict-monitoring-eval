# Tests

Test suite for validating repository components.

## Running Tests

```bash
source .venv/bin/activate

# Core tests
python tests/test_generic_pipeline.py

# Counterfactual tests (requires sample data)
python tests/test_counterfactual.py

# With pytest
python -m pytest tests/ -v
```

## Test Files

### test_generic_pipeline.py

| Test | Description |
|------|-------------|
| Data helpers | Environment setup, path resolution |
| Pipeline imports | Experiment pipeline modules |
| Prompting strategies | Strategy creation and prompts |
| Analysis modules | Metrics, calibration, harm |

### test_counterfactual.py

| Test | Description |
|------|-------------|
| Perturbation generation | Actor, intensity, action substitutions |
| Counterfactual analyzer | Event analysis and flip detection |
| Visualization | Report generation |

## Writing Tests

```python
def test_my_feature():
    """Test description."""
    try:
        from lib.my_module import my_function
        result = my_function(test_input)
        assert result is not None
        print("PASS: My feature works")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False

if __name__ == "__main__":
    exit(0 if test_my_feature() else 1)
```

**Conventions:**
- Files named `test_*.py`
- Print PASS/FAIL status
- Return boolean for success
- Include docstrings

## Test Data

Tests use standard paths from the main codebase:
- `datasets/{country}/` - ACLED data
- `datasets/{country}/state_actor_sample_{country}_{sample_size}.csv` - Sample files
- `results/{country}/{strategy}/{sample_size}/` - Output files

Tests respect environment variables: `COUNTRY`, `STRATEGY`, `SAMPLE_SIZE`, `NUM_EXAMPLES`

## Pre-Commit Validation

```bash
python tests/test_generic_pipeline.py && python -m pytest tests/ -v --tb=short
```
