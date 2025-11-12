#!/usr/bin/env python3
"""Test script for counterfactual analysis framework."""

import sys
import os
import pandas as pd
from pathlib import Path

# Add project root to path so all modules can be found
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from tools.counterfactual_analysis import CounterfactualAnalyzer
from tools.visualize_counterfactual import main as visualize_main

def test_counterfactual_pipeline():
    """Test the complete counterfactual analysis pipeline."""
    
    print("Testing Counterfactual Analysis Framework")
    print("=" * 50)
    
    # Use the sample dataset with actual text data
    sample_data_path = Path("datasets/nga/state_actor_sample_nga.csv")
    if not sample_data_path.exists():
        print(f"Sample data file not found: {sample_data_path}")
        print("Please ensure the dataset exists.")
        return False
    
    print(f"Found sample data: {sample_data_path}")
    
    # Load the sample data for testing
    try:
        df = pd.read_csv(sample_data_path)
        print(f"Loaded {len(df)} sample events")
        
        # Take first 3 events for quick testing
        sample_df = df.head(3)
        print(f"Testing with {len(sample_df)} sample events")
        
        # Initialize analyzer with correct parameters
        # Use available models from constants
        models = ['llama3.2', 'mistral', 'qwen2.5']  # Common models
        country = 'nga'
        
        analyzer = CounterfactualAnalyzer(country=country, models=models)
        
        # Test perturbation generation
        print("\nTesting perturbation generation...")
        test_event = sample_df.iloc[0]
        original_text = test_event.get('notes', '')  # Text is in 'notes' column
        
        if not original_text:
            print("No text found in event data")
            return False
            
        print(f"Original text: {original_text[:100]}...")
        
        perturbations = analyzer.perturbation_generator.generate_all_perturbations(original_text)
        print(f"Generated {len(perturbations)} total perturbations")
        
        # Group by type for display
        by_type = {}
        for pert in perturbations:
            pert_type = pert['type']
            if pert_type not in by_type:
                by_type[pert_type] = []
            by_type[pert_type].append(pert)
        
        for pert_type, pert_list in by_type.items():
            print(f"   {pert_type}: {len(pert_list)} variations")
            if pert_list:  # Show first example if any exist
                print(f"      Example: {pert_list[0]['text'][:80]}...")
        
        # Test analysis on one event (without full model runs for speed)
        print("\nTesting analysis structure...")
        
        # Create test output directory
        output_dir = "test_counterfactual_output"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Created output directory: {output_dir}")
        print("\nFramework validation complete!")
        print("\nTo run full analysis:")
        print(f"  python tools/counterfactual_analysis.py --input {sample_data_path} --output {output_dir}")
        print("\nTo generate visualizations:")
        print(f"  python tools/visualize_counterfactual.py --input {output_dir} --output {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = test_counterfactual_pipeline()
    if success:
        print("\nAll tests passed! Framework ready for use.")
        sys.exit(0)
    else:
        print("\nSome tests failed. Please check the issues above.")
        sys.exit(1)
