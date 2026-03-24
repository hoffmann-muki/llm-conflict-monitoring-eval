#!/usr/bin/env python3
"""
Visualize word-level impacts grouped by original perturbation type.

Uses the actual 7 perturbation types from counterfactual analysis instead of
collapsing into 3 arbitrary categories (legitimizing/delegitimizing/neutral).
"""

import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Styling
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
})


def load_word_to_perturbation_mapping(json_path: Path) -> dict:
    """Extract word -> perturbation type mapping from counterfactual JSON."""
    with open(json_path) as f:
        data = json.load(f)

    word_to_type = {}

    for result in data['detailed_results']:
        for pert in result['perturbations']:
            ptype = pert['perturbation']['type']

            # Extract the word/phrase
            word = None
            if 'modifier' in pert['perturbation']:
                word = pert['perturbation']['modifier']
            elif 'source' in pert['perturbation']:
                word = pert['perturbation']['source']
            elif 'phrase' in pert['perturbation']:
                word = pert['perturbation']['phrase']
            elif 'replacement' in pert['perturbation']:
                word = pert['perturbation']['replacement']
            elif 'synonym' in pert['perturbation']:
                word = pert['perturbation']['synonym']

            if word:
                word_to_type[word] = ptype

    return word_to_type


def plot_scatter_by_perturbation(word_impacts_csv: Path,
                                   counterfactual_json: Path,
                                   output_path: Path,
                                   min_n: int = 20):
    """
    Scatter plot: Flip Rate vs Confidence Delta, colored by perturbation type.

    Args:
        word_impacts_csv: Path to word_impacts.csv
        counterfactual_json: Path to any counterfactual JSON for word mapping
        output_path: Where to save figure
        min_n: Minimum sample size for filled markers (hollow otherwise)
    """
    # Load data
    df = pd.read_csv(word_impacts_csv)
    word_to_type = load_word_to_perturbation_mapping(counterfactual_json)

    # Add perturbation type
    df['perturbation_type'] = df['word'].map(word_to_type)

    # Filter out negations (special case, not a word but a transformation)
    df = df[~df['word'].str.contains('did not', case=False, na=False)]

    # Map perturbation types to display names
    type_labels = {
        'legitimation_add': 'Legitimation Framing',
        'delegitimation_add': 'Delegitimation Framing',
        'provenance_add': 'Source Provenance',
        'intensity_high': 'Intensity Modifier',
        'actor_substitution': 'Actor Substitution',
        'action_substitution': 'Action Substitution',
        'decontextualization': 'Decontextualization'
    }
    df['perturbation_label'] = df['perturbation_type'].map(type_labels)

    # Color palette (7 colors)
    colors = {
        'Legitimation Framing': '#2E86AB',       # Blue
        'Delegitimation Framing': '#A23B72',     # Purple
        'Source Provenance': '#F18F01',          # Orange
        'Intensity Modifier': '#C73E1D',         # Red
        'Actor Substitution': '#6A994E',         # Green
        'Action Substitution': '#BC4B51',        # Rose
        'Decontextualization': '#8B8C89'         # Gray
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot by perturbation type
    for ptype in sorted(df['perturbation_label'].dropna().unique()):
        subset = df[df['perturbation_label'] == ptype]

        # Separate by sample size
        high_n = subset[subset['n_total'] >= min_n]
        low_n = subset[subset['n_total'] < min_n]

        # High-n: filled markers
        if len(high_n) > 0:
            ax.scatter(high_n['flip_rate'] * 100,
                      abs(high_n['delta_mean'] * 100),
                      s=high_n['n_total'] * 3,  # Size by sample
                      c=colors[ptype],
                      label=ptype if len(low_n) == 0 else None,
                      alpha=0.7,
                      edgecolors='black',
                      linewidths=1.5,
                      marker='o')

        # Low-n: hollow markers
        if len(low_n) > 0:
            ax.scatter(low_n['flip_rate'] * 100,
                      abs(low_n['delta_mean'] * 100),
                      s=low_n['n_total'] * 10,  # Larger to see them
                      facecolors='none',
                      edgecolors=colors[ptype],
                      linewidths=2,
                      label=f"{ptype} (n<{min_n})" if len(high_n) == 0 else f"n<{min_n}",
                      alpha=0.6,
                      marker='o')

    # Annotations for key words
    annotate_words = [
        'using excessive force',
        'unprovoked',
        'to restore order',
        'Official sources claimed that',
        'severely'
    ]

    for word in annotate_words:
        if word in df['word'].values:
            row = df[df['word'] == word].iloc[0]
            ax.annotate(word,
                       xy=(row['flip_rate'] * 100, abs(row['delta_mean'] * 100)),
                       xytext=(10, 5),
                       textcoords='offset points',
                       fontsize=8,
                       alpha=0.8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))

    # Labels and styling
    ax.set_xlabel('Classification Flip Rate (%)', fontweight='bold')
    ax.set_ylabel('|Confidence Delta| (Δφ, %)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right', frameon=True,
             framealpha=0.9, fontsize=9)

    # Add reference lines
    ax.axhline(y=3, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(x=10, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Saved scatter plot: {output_path.name}")


def plot_ranked_bars_by_perturbation(word_impacts_csv: Path,
                                       counterfactual_json: Path,
                                       output_path: Path,
                                       min_n: int = 20):
    """
    Ranked horizontal bar chart of flip rates, colored by perturbation type.
    Only includes words with n >= min_n OR p < 0.05.
    """
    # Load data
    df = pd.read_csv(word_impacts_csv)
    word_to_type = load_word_to_perturbation_mapping(counterfactual_json)

    # Add perturbation type
    df['perturbation_type'] = df['word'].map(word_to_type)

    # Filter: exclude negations, keep only well-powered OR significant
    df = df[~df['word'].str.contains('did not', case=False, na=False)]
    df = df[(df['n_total'] >= min_n) | (df['p_value'] < 0.05)]

    # Sort by flip rate
    df = df.sort_values('flip_rate')

    # Map to display labels
    type_labels = {
        'legitimation_add': 'Legitimation',
        'delegitimation_add': 'Delegitimation',
        'provenance_add': 'Provenance',
        'intensity_high': 'Intensity',
        'actor_substitution': 'Actor Sub.',
        'action_substitution': 'Action Sub.',
        'decontextualization': 'Decontext.'
    }
    df['perturbation_label'] = df['perturbation_type'].map(type_labels)

    # Colors (same as scatter plot)
    colors = {
        'Legitimation': '#2E86AB',
        'Delegitimation': '#A23B72',
        'Provenance': '#F18F01',
        'Intensity': '#C73E1D',
        'Actor Sub.': '#6A994E',
        'Action Sub.': '#BC4B51',
        'Decontext.': '#8B8C89'
    }

    bar_colors = [colors.get(pt, '#888888') for pt in df['perturbation_label']]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.4)))

    # Horizontal bars
    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df['flip_rate'] * 100,
                   color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['word'], fontsize=10)
    ax.set_xlabel('Classification Flip Rate (%)', fontweight='bold')

    # Add flip rate + significance markers
    for i, (_, row) in enumerate(df.iterrows()):
        # Flip rate value
        ax.text(row['flip_rate'] * 100 + 0.5, i, f"{row['flip_rate']*100:.1f}%",
               va='center', fontsize=9, fontweight='bold')

        # Significance markers
        if row['p_value'] < 0.001:
            sig_marker = '***'
        elif row['p_value'] < 0.01:
            sig_marker = '**'
        elif row['p_value'] < 0.05:
            sig_marker = '*'
        else:
            sig_marker = 'n.s.'

        ax.text(-0.5, i, sig_marker, va='center', ha='right', fontsize=8,
               color='gray', style='italic')

    # Legend (perturbation types)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=label, alpha=0.8, edgecolor='black')
                      for label, color in colors.items()
                      if label in df['perturbation_label'].values]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True,
             framealpha=0.95, fontsize=9, title='Perturbation Type')

    ax.set_xlim(0, max(df['flip_rate'] * 100) * 1.15)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Saved ranked bar chart: {output_path.name}")


def main():
    """Generate both word-impact visualizations.

    Reads COUNTRY, STRATEGY, SAMPLE_SIZE, NUM_EXAMPLES from environment to
    locate results directory, word_impacts.csv, and any counterfactual JSON.
    """
    country      = os.environ.get('COUNTRY',     'cmr')
    strategy     = os.environ.get('STRATEGY',    'zero_shot')
    sample_size  = os.environ.get('SAMPLE_SIZE', '1000')
    num_examples = os.environ.get('NUM_EXAMPLES')

    if strategy == 'few_shot' and num_examples:
        base_dir = Path(f"results/{country}/{strategy}/{sample_size}/{num_examples}")
    else:
        base_dir = Path(f"results/{country}/{strategy}/{sample_size}")

    word_impacts_csv = base_dir / 'word_impacts.csv'
    if not word_impacts_csv.exists():
        print(f"word_impacts.csv not found at {word_impacts_csv}")
        print("Run aggregate_word_impacts_from_counterfactuals first.")
        return

    # Find any counterfactual JSON for the scatter-plot reference colours
    json_candidates = sorted(glob.glob(str(base_dir / '*' / 'counterfactual_analysis_*.json')), 
                             key=lambda p: Path(p).stat().st_mtime, reverse=True)
    if not json_candidates:
        print(f"No counterfactual JSON found under {base_dir}; skipping visualizations.")
        return
    counterfactual_json = Path(json_candidates[0])

    output_scatter = base_dir / 'fig_word_impact_scatter_by_perturbation.png'
    output_bars    = base_dir / 'fig_word_impact_bars_by_perturbation.png'

    print("Generating word impact visualizations by perturbation type...\n")
    print(f"  Source:  {word_impacts_csv}")
    print(f"  JSON ref: {counterfactual_json}\n")

    # Scatter plot
    plot_scatter_by_perturbation(word_impacts_csv, counterfactual_json,
                                  output_scatter, min_n=5)

    # Bar chart
    plot_ranked_bars_by_perturbation(word_impacts_csv, counterfactual_json,
                                       output_bars, min_n=5)

    print("\n✓ Complete! Generated:")
    print(f"  - {output_scatter}")
    print(f"  - {output_bars}")


if __name__ == "__main__":
    main()
