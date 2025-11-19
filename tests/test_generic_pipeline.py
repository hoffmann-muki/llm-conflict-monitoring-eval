#!/usr/bin/env python3
"""Test the generic pipeline with a dry run."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_generic_pipeline():
    """Test that the generic pipeline can be imported and basic functions work."""
    
    print("Testing Generic Pipeline Components")
    print("=" * 50)
    
    # Test the data helpers
    try:
        from lib.core.data_helpers import setup_country_environment
        country, results_dir = setup_country_environment('cmr')
        print(f"PASS: Setup function works: {country}, {results_dir}")
    except Exception as e:
        print(f"FAIL: Setup function failed: {e}")
        return False
    
    # Test import of the experiments pipeline
    try:
        from experiments.pipelines import run_classification
        print("PASS: Experiments pipeline imports successfully")
    except Exception as e:
        print(f"FAIL: Experiments pipeline import failed: {e}")
        return False
    
    # Test country name mapping
    try:
        country_names = run_classification.COUNTRY_NAMES
        print(f"PASS: Country mapping: {country_names}")
    except Exception as e:
        print(f"FAIL: Country mapping failed: {e}")
        return False
    
    # Test prompting strategies
    try:
        from experiments.prompting_strategies import ZeroShotStrategy
        strategy = ZeroShotStrategy()
        prompt = strategy.make_prompt("Test event")
        print(f"PASS: Zero-shot strategy works: {len(prompt)} chars")
    except Exception as e:
        print(f"FAIL: Prompting strategy failed: {e}")
        return False
    
    # Test lib analysis import with new setup
    try:
        from lib.analysis import metrics
        print("PASS: lib.analysis.metrics imports with new setup")
    except Exception as e:
        print(f"FAIL: lib.analysis.metrics import failed: {e}")
        return False
    
    print("\nAll basic tests passed!")
    print("\nTo run a full pipeline test:")
    print("  STRATEGY=zero_shot COUNTRY=cmr SAMPLE_SIZE=5 python experiments/pipelines/run_classification.py")
    return True

if __name__ == "__main__":
    success = test_generic_pipeline()
    sys.exit(0 if success else 1)
