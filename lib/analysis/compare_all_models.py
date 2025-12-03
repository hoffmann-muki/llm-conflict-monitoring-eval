#!/usr/bin/env python3
"""Cross-model comparison analysis: ConfliBERT vs Ollama models.

Discovers all model results (ConfliBERT + Ollama models) in a results directory
and produces unified comparison tables and visualizations.

Usage:
    COUNTRY=cmr STRATEGY=zero_shot SAMPLE_SIZE=1000 python -m lib.analysis.compare_all_models

Outputs:
    comparison/
    ├── all_models_metrics.csv          # Unified metrics table (accuracy, F1, etc.)
    ├── all_models_fairness.csv         # Fairness metrics by model
    ├── all_models_harm.csv             # Harm metrics (FL/FI rates) by model
    ├── model_comparison_accuracy.png   # Accuracy bar chart
    ├── model_comparison_f1.png         # F1 score comparison
    └── model_comparison_harm.png       # Harm metrics comparison
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

from lib.core.data_helpers import setup_country_environment
from lib.analysis.metrics import compute_metrics, compute_fairness_metrics

# Standard labels
LABELS = ['V', 'B', 'E', 'P', 'R', 'S']


def discover_result_files(results_dir: str, country: str) -> Dict[str, str]:
    """Discover all model result files in the results directory.
    
    Args:
        results_dir: Path to strategy/sample_size results directory
        country: Country code (e.g., 'cmr')
        
    Returns:
        Dict mapping model name to result file path
    """
    model_files = {}
    
    # Pattern 1: ConfliBERT results (conflibert_results_acled_{country}_actors.csv)
    conflibert_pattern = os.path.join(results_dir, f'conflibert_results_acled_{country}_actors.csv')
    if os.path.exists(conflibert_pattern):
        model_files['ConfliBERT'] = conflibert_pattern
    
    # Also check in conflibert/ subdirectory
    conflibert_subdir = os.path.join(results_dir, 'conflibert', f'conflibert_results_acled_{country}_actors.csv')
    if os.path.exists(conflibert_subdir):
        model_files['ConfliBERT'] = conflibert_subdir
    
    # Pattern 2: Ollama per-model results in subdirectories
    # e.g., results/{country}/{strategy}/{sample_size}/{model_slug}/ollama_results_{model}_acled_{country}_actors.csv
    for subdir in glob.glob(os.path.join(results_dir, '*/')):
        model_slug = os.path.basename(subdir.rstrip('/'))
        if model_slug in ('conflibert', 'comparison'):
            continue
        
        # Look for ollama results in this subdirectory
        pattern = os.path.join(subdir, f'ollama_results_*_acled_{country}_actors.csv')
        matches = glob.glob(pattern)
        if matches:
            # Use model slug as key, format nicely
            display_name = model_slug.replace('_', ':').replace(':', ':', 1)  # e.g., llama3_2_3b -> llama3.2:3b
            model_files[display_name] = matches[0]
    
    # Pattern 3: Combined Ollama results (extract individual models)
    combined_ollama = os.path.join(results_dir, f'ollama_results_acled_{country}_actors.csv')
    if os.path.exists(combined_ollama):
        df = pd.read_csv(combined_ollama)
        if 'model' in df.columns:
            for model in df['model'].unique():
                if model not in model_files:
                    model_files[model] = combined_ollama
    
    return model_files


def load_model_results(model_files: Dict[str, str]) -> pd.DataFrame:
    """Load and combine results from all discovered model files.
    
    Args:
        model_files: Dict mapping model name to file path
        
    Returns:
        Combined DataFrame with all model results, standardized 'model' column
    """
    dfs = []
    loaded_from_combined = set()
    
    for model_name, file_path in model_files.items():
        df = pd.read_csv(file_path)
        
        # For combined files, we need to filter to specific model
        is_combined = f'ollama_results_acled_' in os.path.basename(file_path) and 'model' in df.columns
        
        if is_combined:
            # Only process combined file once per unique file
            if file_path in loaded_from_combined:
                continue
            loaded_from_combined.add(file_path)
            # Keep all models from combined file
            dfs.append(df)
        else:
            # For per-model files, standardize model column
            if 'model' not in df.columns:
                df['model'] = model_name
            elif model_name == 'ConfliBERT':
                df['model'] = 'ConfliBERT'
            dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Deduplicate by event_id + model (keep first occurrence)
    if 'event_id' in combined.columns:
        combined = combined.drop_duplicates(subset=['event_id', 'model'], keep='first')
    
    return combined


def compute_all_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute classification, fairness, and harm metrics for all models.
    
    Args:
        df: Combined results DataFrame
        
    Returns:
        Tuple of (metrics_df, fairness_df, harm_df)
    """
    # Classification metrics
    metrics_list, _ = compute_metrics(df)
    metrics_df = pd.DataFrame(metrics_list)
    
    # Fairness metrics (if state_actor column available)
    fairness_df = pd.DataFrame()
    if 'actor_norm' in df.columns or 'state_actor' in df.columns:
        try:
            fairness_df = compute_fairness_metrics(df)
        except Exception as e:
            print(f"Warning: Could not compute fairness metrics: {e}")
    
    # Harm metrics (FL/FI rates)
    harm_rows = []
    for model in df['model'].unique():
        sub = df[df['model'] == model].copy()
        sub = sub[sub['true_label'].isin(LABELS) & sub['pred_label'].isin(LABELS)]
        
        if len(sub) == 0:
            continue
        
        # Violence = V, everything else = non-violence
        # False Legitimization: true=V, pred!=V (violence misclassified as non-violence)
        # False Illegitimization: true!=V, pred=V (non-violence misclassified as violence)
        
        violence_events = sub[sub['true_label'] == 'V']
        non_violence_events = sub[sub['true_label'] != 'V']
        
        fl_count = len(violence_events[violence_events['pred_label'] != 'V'])
        fi_count = len(non_violence_events[non_violence_events['pred_label'] == 'V'])
        
        fl_rate = fl_count / len(violence_events) if len(violence_events) > 0 else 0
        fi_rate = fi_count / len(non_violence_events) if len(non_violence_events) > 0 else 0
        
        harm_rows.append({
            'model': model,
            'n_violence': len(violence_events),
            'n_non_violence': len(non_violence_events),
            'false_legitimization_count': fl_count,
            'false_illegitimization_count': fi_count,
            'false_legitimization_rate': round(fl_rate, 4),
            'false_illegitimization_rate': round(fi_rate, 4)
        })
    
    harm_df = pd.DataFrame(harm_rows)
    
    return metrics_df, fairness_df, harm_df


def plot_model_comparison(metrics_df: pd.DataFrame, output_dir: str):
    """Generate comparison visualizations.
    
    Args:
        metrics_df: DataFrame with metrics per model
        output_dir: Directory to save plots
    """
    if len(metrics_df) == 0:
        print("No metrics to plot")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by accuracy for consistent ordering
    metrics_df = metrics_df.sort_values('accuracy', ascending=True)
    models = metrics_df['model'].tolist()
    
    # Color scheme: ConfliBERT in distinct color (green), Ollama models in blue
    colors = ['#27ae60' if 'conflibert' in m.lower() else '#3498db' for m in models]
    
    # 1. Accuracy comparison
    fig, ax = plt.subplots(figsize=(10, max(6, len(models) * 0.5)))
    y_pos = np.arange(len(models))
    ax.barh(y_pos, metrics_df['accuracy'].values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('Accuracy')
    ax.set_title('Model Comparison: Accuracy\n(ConfliBERT vs Ollama LLMs)')
    ax.set_xlim(0, 1)
    
    # Add value labels
    for i, v in enumerate(metrics_df['accuracy'].values):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'model_comparison_accuracy.png'), dpi=150)
    plt.close(fig)
    
    # 2. F1 Score comparison (macro)
    if 'macro_f1' in metrics_df.columns:
        fig, ax = plt.subplots(figsize=(10, max(6, len(models) * 0.5)))
        ax.barh(y_pos, metrics_df['macro_f1'].values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models)
        ax.set_xlabel('Macro F1 Score')
        ax.set_title('Model Comparison: Macro F1\n(ConfliBERT vs Ollama LLMs)')
        ax.set_xlim(0, 1)
        
        for i, v in enumerate(metrics_df['macro_f1'].values):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'model_comparison_f1.png'), dpi=150)
        plt.close(fig)
    
    # 3. Multi-metric grouped bar chart
    metric_cols = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
    available_cols = [c for c in metric_cols if c in metrics_df.columns]
    
    if len(available_cols) > 1:
        fig, ax = plt.subplots(figsize=(12, max(6, len(models) * 0.6)))
        
        x = np.arange(len(models))
        width = 0.8 / len(available_cols)
        
        for i, col in enumerate(available_cols):
            offset = (i - len(available_cols) / 2 + 0.5) * width
            bars = ax.barh(x + offset, metrics_df[col].values, width, 
                          label=col.replace('_', ' ').title())
        
        ax.set_yticks(x)
        ax.set_yticklabels(models)
        ax.set_xlabel('Score')
        ax.set_title('Model Comparison: All Metrics\n(ConfliBERT vs Ollama LLMs)')
        ax.set_xlim(0, 1)
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'model_comparison_all_metrics.png'), dpi=150)
        plt.close(fig)
    
    print(f"Saved comparison plots to {output_dir}/")


def plot_harm_comparison(harm_df: pd.DataFrame, output_dir: str):
    """Generate harm metrics comparison visualization.
    
    Args:
        harm_df: DataFrame with harm metrics per model
        output_dir: Directory to save plots
    """
    if len(harm_df) == 0:
        print("No harm metrics to plot")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    harm_df = harm_df.sort_values('false_legitimization_rate', ascending=True)
    models = harm_df['model'].tolist()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(models) * 0.5)))
    
    y_pos = np.arange(len(models))
    colors_fl = ['#c0392b' if 'conflibert' in m.lower() else '#e74c3c' for m in models]
    colors_fi = ['#8e44ad' if 'conflibert' in m.lower() else '#9b59b6' for m in models]
    
    # False Legitimization Rate
    axes[0].barh(y_pos, harm_df['false_legitimization_rate'].values, color=colors_fl)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(models)
    axes[0].set_xlabel('Rate')
    axes[0].set_title('False Legitimization Rate\n(Violence → Non-violence)')
    axes[0].set_xlim(0, max(0.5, harm_df['false_legitimization_rate'].max() * 1.2))
    
    for i, v in enumerate(harm_df['false_legitimization_rate'].values):
        axes[0].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    
    # False Illegitimization Rate
    harm_df_fi = harm_df.sort_values('false_illegitimization_rate', ascending=True)
    models_fi = harm_df_fi['model'].tolist()
    colors_fi_sorted = ['#8e44ad' if 'conflibert' in m.lower() else '#9b59b6' for m in models_fi]
    
    axes[1].barh(y_pos, harm_df_fi['false_illegitimization_rate'].values, color=colors_fi_sorted)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(models_fi)
    axes[1].set_xlabel('Rate')
    axes[1].set_title('False Illegitimization Rate\n(Non-violence → Violence)')
    axes[1].set_xlim(0, max(0.5, harm_df_fi['false_illegitimization_rate'].max() * 1.2))
    
    for i, v in enumerate(harm_df_fi['false_illegitimization_rate'].values):
        axes[1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.suptitle('Harm Metrics Comparison: ConfliBERT vs Ollama LLMs', fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'model_comparison_harm.png'), dpi=150)
    plt.close(fig)
    
    print(f"Saved harm comparison plot to {output_dir}/")


def main():
    """Main entry point for cross-model comparison."""
    COUNTRY, RESULTS_DIR = setup_country_environment()
    
    print(f"\n{'='*70}")
    print(f"Cross-Model Comparison Analysis: ConfliBERT vs Ollama LLMs")
    print(f"Country: {COUNTRY}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"{'='*70}\n")
    
    # Discover all model result files
    print("Discovering model result files...")
    model_files = discover_result_files(RESULTS_DIR, COUNTRY)
    
    if not model_files:
        print(f"No model result files found in {RESULTS_DIR}")
        print("Expected patterns:")
        print(f"  - conflibert_results_acled_{COUNTRY}_actors.csv")
        print(f"  - {{model_slug}}/ollama_results_*_acled_{COUNTRY}_actors.csv")
        print(f"  - ollama_results_acled_{COUNTRY}_actors.csv (combined)")
        return
    
    print(f"Found {len(model_files)} model(s):")
    for model, path in model_files.items():
        print(f"  - {model}: {os.path.basename(path)}")
    
    # Load and combine results
    print("\nLoading results...")
    df = load_model_results(model_files)
    print(f"Total records: {len(df):,}")
    print(f"Models: {sorted(df['model'].unique().tolist())}")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics_df, fairness_df, harm_df = compute_all_metrics(df)
    
    # Create comparison output directory
    comparison_dir = os.path.join(RESULTS_DIR, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Save metrics tables
    if len(metrics_df) > 0:
        metrics_out = os.path.join(comparison_dir, 'all_models_metrics.csv')
        metrics_df.to_csv(metrics_out, index=False)
        print(f"\nWrote metrics to {metrics_out}")
        print(metrics_df.to_string(index=False))
    
    if len(fairness_df) > 0:
        fairness_out = os.path.join(comparison_dir, 'all_models_fairness.csv')
        fairness_df.to_csv(fairness_out, index=False)
        print(f"\nWrote fairness metrics to {fairness_out}")
    
    if len(harm_df) > 0:
        harm_out = os.path.join(comparison_dir, 'all_models_harm.csv')
        harm_df.to_csv(harm_out, index=False)
        print(f"\nWrote harm metrics to {harm_out}")
        print(harm_df.to_string(index=False))
    
    # Generate visualizations
    print("\nGenerating comparison visualizations...")
    plot_model_comparison(metrics_df, comparison_dir)
    plot_harm_comparison(harm_df, comparison_dir)
    
    print(f"\n{'='*70}")
    print(f"Comparison analysis complete!")
    print(f"Outputs saved to: {comparison_dir}/")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
