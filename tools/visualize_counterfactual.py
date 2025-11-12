#!/usr/bin/env python3
"""
Visualization script for counterfactual analysis results.

Usage:
    COUNTRY=nga python -m tools.visualize_counterfactual --input results/nga/counterfactual_analysis.json
"""

import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any
import numpy as np

def plot_flip_rates(flip_metrics: Dict[str, Any], output_dir: str):
    """Plot flip rates by perturbation type and model."""
    
    # Prepare data for plotting
    data = []
    for pert_type, model_data in flip_metrics.items():
        for model, metrics in model_data.items():
            data.append({
                'perturbation_type': pert_type.replace('_', ' ').title(),
                'model': model,
                'flip_rate': metrics['flip_rate'],
                'n_perturbations': metrics['n_perturbations']
            })
    
    if not data:
        print("No flip rate data to plot")
        return
    
    df = pd.DataFrame(data)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Flip rates by perturbation type
    pert_types = df['perturbation_type'].unique()
    models = df['model'].unique()
    x_pos = np.arange(len(pert_types))
    width = 0.35
    
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        flip_rates = [model_data[model_data['perturbation_type'] == pt]['flip_rate'].iloc[0] 
                     if len(model_data[model_data['perturbation_type'] == pt]) > 0 else 0 
                     for pt in pert_types]
        ax1.bar(x_pos + i*width, flip_rates, width, label=model)
    
    ax1.set_title('Model Sensitivity: Flip Rates by Perturbation Type')
    ax1.set_ylabel('Flip Rate')
    ax1.set_xticks(x_pos + width/2)
    ax1.set_xticklabels(pert_types, rotation=45)
    ax1.legend(title='Model')
    
    # Plot 2: Sample sizes
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        n_perturbations = [model_data[model_data['perturbation_type'] == pt]['n_perturbations'].iloc[0] 
                          if len(model_data[model_data['perturbation_type'] == pt]) > 0 else 0 
                          for pt in pert_types]
        ax2.bar(x_pos + i*width, n_perturbations, width, label=model)
    
    ax2.set_title('Number of Perturbations by Type')
    ax2.set_ylabel('Number of Perturbations')
    ax2.set_xticks(x_pos + width/2)
    ax2.set_xticklabels(pert_types, rotation=45)
    ax2.legend(title='Model')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'flip_rates_by_perturbation.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence_deltas(flip_metrics: Dict[str, Any], output_dir: str):
    """Plot confidence changes by perturbation type."""
    
    data = []
    for pert_type, model_data in flip_metrics.items():
        for model, metrics in model_data.items():
            data.append({
                'perturbation_type': pert_type.replace('_', ' ').title(),
                'model': model,
                'mean_confidence_delta': metrics['mean_confidence_delta'],
                'mean_abs_confidence_delta': metrics['mean_abs_confidence_delta']
            })
    
    if not data:
        print("No confidence delta data to plot")
        return
    
    df = pd.DataFrame(data)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Mean confidence delta (signed)
    pert_types = df['perturbation_type'].unique()
    models = df['model'].unique()
    x_pos = np.arange(len(pert_types))
    width = 0.35
    
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        deltas = [model_data[model_data['perturbation_type'] == pt]['mean_confidence_delta'].iloc[0] 
                 if len(model_data[model_data['perturbation_type'] == pt]) > 0 else 0 
                 for pt in pert_types]
        ax1.bar(x_pos + i*width, deltas, width, label=model)
    
    ax1.set_title('Mean Confidence Change by Perturbation')
    ax1.set_ylabel('Mean Confidence Delta')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xticks(x_pos + width/2)
    ax1.set_xticklabels(pert_types, rotation=45)
    ax1.legend(title='Model')
    
    # Plot 2: Absolute confidence delta
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        abs_deltas = [model_data[model_data['perturbation_type'] == pt]['mean_abs_confidence_delta'].iloc[0] 
                     if len(model_data[model_data['perturbation_type'] == pt]) > 0 else 0 
                     for pt in pert_types]
        ax2.bar(x_pos + i*width, abs_deltas, width, label=model)
    
    ax2.set_title('Mean Absolute Confidence Change')
    ax2.set_ylabel('Mean |Confidence Delta|')
    ax2.set_xticks(x_pos + width/2)
    ax2.set_xticklabels(pert_types, rotation=45)
    ax2.legend(title='Model')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_deltas.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_sensitivity_clusters(clusters: Dict[str, list], output_dir: str):
    """Plot distribution of events across sensitivity clusters."""
    
    cluster_sizes = {cluster: len(events) for cluster, events in clusters.items()}
    
    if not cluster_sizes:
        print("No cluster data to plot")
        return
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    
    labels = [label.replace('_', ' ').title() for label in cluster_sizes.keys()]
    sizes = list(cluster_sizes.values())
    
    pie_result = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    
    ax.set_title('Distribution of Events by Sensitivity Pattern')
    
    # Add legend with counts
    legend_labels = [f"{label} (n={size})" for label, size in zip(labels, sizes)]
    ax.legend(pie_result[0], legend_labels, title="Sensitivity Clusters", 
             loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.savefig(os.path.join(output_dir, 'sensitivity_clusters.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_statistical_tests(test_results: Dict[str, Any], output_dir: str):
    """Plot statistical test results."""
    
    valid_tests = {k: v for k, v in test_results.items() if 'error' not in v}
    
    if not valid_tests:
        print("No valid statistical test results to plot")
        return
    
    # Extract p-values and test statistics
    test_names = []
    p_values = []
    statistics = []
    
    for test_name, result in valid_tests.items():
        test_names.append(test_name.replace('_chi2', '').replace('_', ' ').title())
        p_values.append(result['p_value'])
        statistics.append(result['statistic'])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: P-values with significance threshold
    bars1 = ax1.bar(range(len(p_values)), p_values)
    ax1.axhline(y=0.05, color='red', linestyle='--', label='alpha = 0.05')
    ax1.set_xticks(range(len(test_names)))
    ax1.set_xticklabels(test_names, rotation=45)
    ax1.set_ylabel('P-value')
    ax1.set_title('Statistical Test Results: P-values')
    ax1.legend()
    
    # Color bars based on significance
    for i, (bar, p_val) in enumerate(zip(bars1, p_values)):
        if p_val < 0.05:
            bar.set_color('red')
        else:
            bar.set_color('blue')
    
    # Plot 2: Test statistics
    ax2.bar(range(len(statistics)), statistics)
    ax2.set_xticks(range(len(test_names)))
    ax2.set_xticklabels(test_names, rotation=45)
    ax2.set_ylabel('Test Statistic')
    ax2.set_title('Statistical Test Results: Test Statistics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'statistical_tests.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(flip_metrics: Dict[str, Any], output_dir: str):
    """Create a summary table of key findings."""
    
    # Find most sensitive perturbation types per model
    summary_data = []
    
    for pert_type, model_data in flip_metrics.items():
        for model, metrics in model_data.items():
            summary_data.append({
                'Perturbation Type': pert_type.replace('_', ' ').title(),
                'Model': model,
                'Flip Rate': f"{metrics['flip_rate']:.3f}",
                'Sample Size': metrics['n_perturbations'],
                'Mean Confidence Delta': f"{metrics['mean_confidence_delta']:+.3f}",
                'Mean |Confidence Delta|': f"{metrics['mean_abs_confidence_delta']:.3f}"
            })
    
    if not summary_data:
        print("No summary data available")
        return
    
    df = pd.DataFrame(summary_data)
    
    # Create figure for table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values.tolist(), colLabels=list(df.columns),
                     cellLoc='center', loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Counterfactual Analysis Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, 'summary_table.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def generate_text_report(data: Dict[str, Any], output_path: str):
    """Generate a text-based summary report."""
    
    with open(output_path, 'w') as f:
        f.write("COUNTERFACTUAL ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Metadata
        meta = data['metadata']
        f.write(f"Country: {meta['country'].upper()}\n")
        f.write(f"Models: {', '.join(meta['models'])}\n")
        f.write(f"Events analyzed: {meta['n_events']}\n")
        f.write(f"Total perturbations: {meta['n_perturbations_total']}\n\n")
        
        # Key findings
        f.write("KEY FINDINGS\n")
        f.write("-" * 20 + "\n\n")
        
        flip_metrics = data['flip_metrics']
        
        # Find most sensitive perturbation types
        max_flip_rates = {}
        for pert_type, model_data in flip_metrics.items():
            for model, metrics in model_data.items():
                key = f"{pert_type}_{model}"
                max_flip_rates[key] = metrics['flip_rate']
        
        if max_flip_rates:
            most_sensitive = max(max_flip_rates.keys(), key=lambda x: max_flip_rates[x])
            pert_type, model = most_sensitive.rsplit('_', 1)
            f.write(f"Most sensitive combination: {model} to {pert_type.replace('_', ' ')} ")
            f.write(f"(flip rate: {max_flip_rates[most_sensitive]:.3f})\n\n")
        
        # Cluster summary
        clusters = data['sensitivity_clusters']
        f.write("SENSITIVITY CLUSTERS\n")
        f.write("-" * 20 + "\n")
        for cluster, events in clusters.items():
            f.write(f"{cluster.replace('_', ' ').title()}: {len(events)} events\n")
        f.write("\n")
        
        # Statistical significance
        test_results = data['statistical_tests']
        f.write("STATISTICAL TESTS\n")
        f.write("-" * 17 + "\n")
        for test_name, result in test_results.items():
            if 'error' not in result:
                significance = "significant" if result['p_value'] < 0.05 else "not significant"
                f.write(f"{test_name}: p = {result['p_value']:.4f} ({significance})\n")
        f.write("\n")

def main():
    parser = argparse.ArgumentParser(description='Visualize counterfactual analysis results')
    parser.add_argument('--input', required=True, help='Input JSON file from counterfactual analysis')
    parser.add_argument('--output-dir', default=None, help='Output directory for plots')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Load results
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.input)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating visualizations in {args.output_dir}...")
    
    # Configure plotting  
    plt.style.use('default')
    # Set color cycle for consistent colors across plots
    from cycler import cycler
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
    
    # Generate plots
    plot_flip_rates(data['flip_metrics'], args.output_dir)
    plot_confidence_deltas(data['flip_metrics'], args.output_dir)
    plot_sensitivity_clusters(data['sensitivity_clusters'], args.output_dir)
    plot_statistical_tests(data['statistical_tests'], args.output_dir)
    create_summary_table(data['flip_metrics'], args.output_dir)
    
    # Generate text report
    text_report_path = os.path.join(args.output_dir, 'counterfactual_report.txt')
    generate_text_report(data, text_report_path)
    
    print("Visualizations complete!")
    print(f"Generated files in {args.output_dir}:")
    print("- flip_rates_by_perturbation.png")
    print("- confidence_deltas.png") 
    print("- sensitivity_clusters.png")
    print("- statistical_tests.png")
    print("- summary_table.png")
    print("- counterfactual_report.txt")

if __name__ == '__main__':
    main()
