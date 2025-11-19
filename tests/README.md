# Tests

Test suite for validating repository components.

## Running Tests

```bash
# Run all basic tests
.venv/bin/python tests/test_generic_pipeline.py

# Run counterfactual tests (requires sample data)
.venv/bin/python tests/test_counterfactual.py
```

## Test Coverage

### test_generic_pipeline.py

Tests core components:
- Data helpers and country setup
- Experiments pipeline imports
- Prompting strategies
- Library analysis modules

### test_counterfactual.py

Tests counterfactual analysis framework:
- Perturbation generation
- Counterfactual analyzer
- Visualization pipeline

## Adding New Tests

Tests should validate:
1. Module imports work correctly
2. Core functions execute without errors
3. Expected output structures are generated
4. Integration between components

Example:
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
```
