# Tests

Test suite for validating repository components.

## Running Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run all basic tests
python tests/test_generic_pipeline.py

# Run counterfactual tests (requires sample data)
python tests/test_counterfactual.py

# Run specific test file
python -m pytest tests/test_generic_pipeline.py -v
```

## Test Files

### test_generic_pipeline.py

Tests core components and imports:

| Test | Description |
|------|-------------|
| Data helpers | Country setup and path resolution |
| Pipeline imports | Experiment pipeline modules |
| Prompting strategies | Strategy creation and prompt generation |
| Analysis modules | Metrics, calibration, harm analysis |

### test_counterfactual.py

Tests counterfactual analysis framework:

| Test | Description |
|------|-------------|
| Perturbation generation | Actor, intensity, action substitutions |
| Counterfactual analyzer | Event analysis and flip detection |
| Visualization | Report generation |

## Writing New Tests

Tests should validate:

1. **Module imports** work correctly
2. **Core functions** execute without errors
3. **Expected output structures** are generated
4. **Integration** between components

### Example Test

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
        print(f"FAIL: My feature failed: {e}")
        return False

if __name__ == "__main__":
    success = test_my_feature()
    exit(0 if success else 1)
```

### Test Conventions

- Test files should be named `test_*.py`
- Each test function should print PASS/FAIL status
- Return boolean indicating success
- Include docstrings describing what is tested

## Test Data

Tests may require sample data in the following locations:

| Path | Description |
|------|-------------|
| `datasets/{country}/` | Country-specific ACLED data |
| `results/{country}/{strategy}/{sample_size}/` | Analysis output files |
| `datasets/{country}/state_actor_sample_{country}.csv` | Sample event files |

## Continuous Integration

Run the full test suite before committing:

```bash
# Quick validation
python tests/test_generic_pipeline.py

# Full test with counterfactual (requires data)
python tests/test_counterfactual.py
```
