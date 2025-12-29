"""
Reusable FL/FI (False Legitimacy/False Illegitimacy) analysis generator.

FL: V → B (illegitimate action is excused)
FI: B → V (legitimate action is condemned)

Usage:
    python generate_fl_fi_analysis.py --strategy zero_shot --actors Police Military
    python generate_fl_fi_analysis.py --strategy few_shot --shots 1 3 5
    python generate_fl_fi_analysis.py --strategy explainable
"""

import pandas as pd
import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint


def calculate_fl_fi_metrics(df: pd.DataFrame, actor_name: str) -> Dict:
    """
    Calculate FL/FI metrics for a specific actor.

    Args:
        df: DataFrame with model predictions
        actor_name: Name of the actor to analyze

    Returns:
        Dictionary with FL/FI metrics
    """
    # Filter data for this actor
    actor_df = df[df['actor_norm'].str.contains(actor_name, case=False, na=False)]

    # Count true labels
    true_v_count = len(actor_df[actor_df['true_label'] == 'V'])
    true_b_count = len(actor_df[actor_df['true_label'] == 'B'])

    # False Legitimacy (FL): True label is V, predicted as B
    fl_cases = actor_df[(actor_df['true_label'] == 'V') & (actor_df['pred_label'] == 'B')]
    fl_count = len(fl_cases)
    fl_pct = (fl_count / true_v_count * 100) if true_v_count > 0 else 0

    # False Illegitimacy (FI): True label is B, predicted as V
    fi_cases = actor_df[(actor_df['true_label'] == 'B') & (actor_df['pred_label'] == 'V')]
    fi_count = len(fi_cases)
    fi_pct = (fi_count / true_b_count * 100) if true_b_count > 0 else 0

    # Net Bias Score (NBS): (FI% - FL%) / (FI% + FL%)
    # Range: -1 (pure legitimization) to +1 (pure illegitimization)
    sum_pct = fi_pct + fl_pct
    nbs = (fi_pct - fl_pct) / sum_pct if sum_pct > 0 else 0

    return {
        'true_v': true_v_count,
        'true_b': true_b_count,
        'fl_count': fl_count,
        'fl_pct': fl_pct,
        'fi_count': fi_count,
        'fi_pct': fi_pct,
        'nbs': nbs
    }


def find_model_results(results_dir: Path) -> List[str]:
    """Find all model subdirectories with results."""
    model_dirs = []
    for item in results_dir.iterdir():
        if item.is_dir() and list(item.glob('ollama_results_*_acled_*_actors.csv')):
            model_dirs.append(item.name)
    return sorted(model_dirs)


def calculate_confidence_interval(count: int, total: int) -> Tuple[float, float]:
    """
    Calculate Wilson score 95% confidence interval for a proportion.

    Args:
        count: Number of events
        total: Total number of trials

    Returns:
        (lower, upper) bounds as percentages
    """
    if total == 0:
        return (0.0, 0.0)

    # Wilson score interval (better for small counts and extremes)
    ci_lower, ci_upper = proportion_confint(count, total, method='wilson', alpha=0.05)
    return (ci_lower * 100, ci_upper * 100)


def calculate_bias_significance(fl_count: int, fi_count: int) -> Tuple[float, float, str]:
    """
    Test statistical significance of bias using binomial test.

    Args:
        fl_count: Number of false legitimization errors
        fi_count: Number of false illegitimization errors

    Returns:
        (p_value, risk_ratio, interpretation)
    """
    total_errors = fl_count + fi_count

    if total_errors == 0:
        return (1.0, 1.0, "No errors")

    # Binomial test: H0: P(FL) = P(FI) = 0.5
    result = binomtest(fl_count, total_errors, p=0.5, alternative='two-sided')
    p_value = result.pvalue

    # Risk Ratio: εFI / εFL (requires rates, computed in calling function)
    # We'll return just the p-value here
    return p_value


def calculate_total_fl_fi_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate FL/FI metrics for entire dataset (all actors) with confidence intervals.

    Args:
        df: DataFrame with model predictions

    Returns:
        Dictionary with FL/FI metrics including CIs and statistical tests
    """
    # Count true labels
    true_v_count = len(df[df['true_label'] == 'V'])
    true_b_count = len(df[df['true_label'] == 'B'])

    # False Legitimacy (FL): True label is V, predicted as B
    fl_cases = df[(df['true_label'] == 'V') & (df['pred_label'] == 'B')]
    fl_count = len(fl_cases)
    fl_pct = (fl_count / true_v_count * 100) if true_v_count > 0 else 0

    # False Illegitimacy (FI): True label is B, predicted as V
    fi_cases = df[(df['true_label'] == 'B') & (df['pred_label'] == 'V')]
    fi_count = len(fi_cases)
    fi_pct = (fi_count / true_b_count * 100) if true_b_count > 0 else 0

    # Confidence intervals
    fl_ci = calculate_confidence_interval(fl_count, true_v_count)
    fi_ci = calculate_confidence_interval(fi_count, true_b_count)

    # Statistical significance
    p_value = calculate_bias_significance(fl_count, fi_count)

    # Risk ratio (εFI / εFL)
    if fl_pct > 0:
        risk_ratio = fi_pct / fl_pct
    elif fi_pct > 0:
        risk_ratio = float('inf')  # Infinite delegitimization bias
    else:
        risk_ratio = 1.0  # No errors

    # Interpretation
    if p_value >= 0.05:
        interpretation = "No bias"
    elif risk_ratio > 1.5:
        interpretation = "Strong Deleg."
    elif risk_ratio > 1.1:
        interpretation = "Weak Deleg."
    elif risk_ratio < 0.67:
        interpretation = "Strong Legit."
    elif risk_ratio < 0.91:
        interpretation = "Weak Legit."
    else:
        interpretation = "No bias"

    # Net Bias Score (NBS): (FI% - FL%) / (FI% + FL%)
    # Range: -1 (pure legitimization) to +1 (pure illegitimization)
    sum_pct = fi_pct + fl_pct
    nbs = (fi_pct - fl_pct) / sum_pct if sum_pct > 0 else 0

    return {
        'true_v': true_v_count,
        'true_b': true_b_count,
        'fl_count': fl_count,
        'fl_pct': fl_pct,
        'fl_ci_lower': fl_ci[0],
        'fl_ci_upper': fl_ci[1],
        'fi_count': fi_count,
        'fi_pct': fi_pct,
        'fi_ci_lower': fi_ci[0],
        'fi_ci_upper': fi_ci[1],
        'p_value': p_value,
        'risk_ratio': risk_ratio,
        'interpretation': interpretation,
        'nbs': nbs
    }


def generate_fl_fi_table(
    results_dir: Path,
    models: List[str],
    actors: List[str],
    output_file: Path
) -> pd.DataFrame:
    """
    Generate FL/FI analysis table for multiple models and actors.

    Args:
        results_dir: Path to results directory
        models: List of model names
        actors: List of actor names to analyze
        output_file: Path to save the output CSV

    Returns:
        DataFrame with results
    """
    results = []

    for model in models:
        # Find the CSV file for this model
        model_dir = results_dir / model
        csv_files = list(model_dir.glob('ollama_results_*_acled_*_actors.csv'))

        if not csv_files:
            print(f"Warning: No results file found for {model}")
            continue

        csv_file = csv_files[0]
        df = pd.read_csv(csv_file)

        # Add per-actor metrics
        for actor in actors:
            metrics = calculate_fl_fi_metrics(df, actor)

            results.append({
                'Model': model,
                'Actor': actor,
                'True V': metrics['true_v'],
                'True B': metrics['true_b'],
                'FL Count': metrics['fl_count'],
                'FL %': f"{metrics['fl_pct']:.2f}%",
                'FI Count': metrics['fi_count'],
                'FI %': f"{metrics['fi_pct']:.2f}%",
                'NBS': f"{metrics['nbs']:.3f}"
            })

        # Add total metrics for this model (with CIs and stats)
        total_metrics = calculate_total_fl_fi_metrics(df)
        results.append({
            'Model': model,
            'Actor': 'Total',
            'True V': total_metrics['true_v'],
            'True B': total_metrics['true_b'],
            'FL Count': total_metrics['fl_count'],
            'FL %': f"{total_metrics['fl_pct']:.2f}%",
            'FL 95% CI': f"({total_metrics['fl_ci_lower']:.1f}-{total_metrics['fl_ci_upper']:.1f})",
            'FI Count': total_metrics['fi_count'],
            'FI %': f"{total_metrics['fi_pct']:.2f}%",
            'FI 95% CI': f"({total_metrics['fi_ci_lower']:.1f}-{total_metrics['fi_ci_upper']:.1f})",
            'ΔLB': f"{total_metrics['fi_pct'] - total_metrics['fl_pct']:.2f}",
            'p-value': f"{total_metrics['p_value']:.3f}" if total_metrics['p_value'] < 0.999 else "<0.001",
            'RR': f"{total_metrics['risk_ratio']:.2f}" if total_metrics['risk_ratio'] != float('inf') else "∞",
            'Interpretation': total_metrics['interpretation'],
            'NBS': f"{total_metrics['nbs']:.3f}"
        })

    # Create DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Generate FL/FI analysis for LLM state actor bias experiments'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        choices=['zero_shot', 'few_shot', 'explainable'],
        help='Prompting strategy to analyze'
    )
    parser.add_argument(
        '--shots',
        type=int,
        nargs='+',
        help='Shot counts for few_shot strategy (e.g., 1 3 5)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=1000,
        help='Sample size (default: 1000)'
    )
    parser.add_argument(
        '--actors',
        type=str,
        nargs='+',
        default=['Police', 'Military'],
        help='Actors to analyze (default: Police Military)'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        help='Specific models to analyze (default: auto-detect all models)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory (default: results/analysis/fl_fi)'
    )
    parser.add_argument(
        '--country',
        type=str,
        default='cmr',
        choices=['cmr', 'nga'],
        help='Country to analyze (default: cmr)'
    )

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    output_dir = args.output_dir or (Path(__file__).parent / args.country)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle different strategies
    if args.strategy == 'few_shot':
        if not args.shots:
            print("Error: --shots required for few_shot strategy")
            return

        for shot_count in args.shots:
            results_dir = base_dir / args.country / args.strategy / str(args.sample_size) / str(shot_count)
            output_file = output_dir / f'fl_fi_analysis_few_shot_{shot_count}.csv'

            if not results_dir.exists():
                print(f"Warning: Directory not found: {results_dir}")
                continue

            print(f"\n{'='*80}")
            print(f"Analyzing Few-Shot ({shot_count} shots)")
            print(f"{'='*80}")

            models = args.models or find_model_results(results_dir)
            if not models:
                print(f"No models found in {results_dir}")
                continue

            print(f"Results directory: {results_dir}")
            print(f"Models: {', '.join(models)}")
            print(f"Actors: {', '.join(args.actors)}")

            results_df = generate_fl_fi_table(results_dir, models, args.actors, output_file)

            print(f"\nSaved to: {output_file}")
            print("\n" + results_df.to_string(index=False))

    else:
        # zero_shot or explainable
        results_dir = base_dir / args.country / args.strategy / str(args.sample_size)
        output_file = output_dir / f'fl_fi_analysis_{args.strategy}.csv'

        if not results_dir.exists():
            print(f"Error: Directory not found: {results_dir}")
            return

        print(f"\n{'='*80}")
        print(f"Analyzing {args.strategy.replace('_', ' ').title()}")
        print(f"{'='*80}")

        models = args.models or find_model_results(results_dir)
        if not models:
            print(f"No models found in {results_dir}")
            return

        print(f"Results directory: {results_dir}")
        print(f"Models: {', '.join(models)}")
        print(f"Actors: {', '.join(args.actors)}")

        results_df = generate_fl_fi_table(results_dir, models, args.actors, output_file)

        print(f"\nSaved to: {output_file}")
        print("\n" + results_df.to_string(index=False))


if __name__ == '__main__':
    main()
