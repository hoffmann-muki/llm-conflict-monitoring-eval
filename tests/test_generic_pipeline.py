#!/usr/bin/env python3
"""Test the generic pipeline with a dry run."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Add tools directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tools'))

def test_generic_pipeline():
    """Test that the generic pipeline can be imported and basic functions work."""
    
    print("Testing Generic Pipeline Components")
    print("=" * 50)
    
    # Test the data helpers
    try:
        from tools.data_helpers import setup_country_environment
        country, results_dir = setup_country_environment('cmr')
        print(f"PASS: Setup function works: {country}, {results_dir}")
    except Exception as e:
        print(f"FAIL: Setup function failed: {e}")
        return False
    
    # Test import of the generic pipeline
    try:
        import political_bias_of_llms_generic as generic_pipeline
        print("PASS: Generic pipeline imports successfully")
    except Exception as e:
        print(f"FAIL: Generic pipeline import failed: {e}")
        return False
    
    # Test country name mapping
    try:
        country_names = generic_pipeline.COUNTRY_NAMES
        print(f"PASS: Country mapping: {country_names}")
    except Exception as e:
        print(f"FAIL: Country mapping failed: {e}")
        return False
    
    # Test tools import with new setup
    try:
        from tools.compute_metrics import main as compute_metrics_main
        print("PASS: compute_metrics imports with new setup")
    except Exception as e:
        print(f"FAIL: compute_metrics import failed: {e}")
        return False
    
    print("\nAll basic tests passed!")
    print("\nTo run a full pipeline test:")
    print("  COUNTRY=cmr SAMPLE_SIZE=5 python political_bias_of_llms_generic.py")
    return True

if __name__ == "__main__":
    success = test_generic_pipeline()
    sys.exit(0 if success else 1)
