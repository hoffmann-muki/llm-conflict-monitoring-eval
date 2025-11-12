#!/usr/bin/env python3
"""
Counterfactual Analysis for Model Disagreements

This script implements a systematic approach to understand why models make different 
classifications on the same events through hypothesis-driven perturbations.

Usage:
    COUNTRY=nga python -m tools.counterfactual_analysis --models llama3.2,mistral:7b --events 5
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
import re
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
import itertools

from tools.ollama_helpers import run_ollama_structured
from tools.data_helpers import paths_for_country
from tools.constants import LABEL_MAP, EVENT_CLASSES_FULL

class PerturbationGenerator:
    """Generate hypothesis-driven perturbations for counterfactual analysis."""
    
    def __init__(self):
        self.actor_replacements = {
            'military': ['security forces', 'armed forces', 'government troops', 'state forces'],
            'police': ['law enforcement', 'security personnel', 'officers', 'authorities'],
            'soldiers': ['troops', 'military personnel', 'armed personnel', 'government forces'],
            'forces': ['troops', 'personnel', 'units', 'operatives']
        }
        
        self.intensity_modifiers = {
            'high': ['severely', 'brutally', 'violently', 'aggressively'],
            'low': ['allegedly', 'reportedly', 'seemingly', 'apparently'],
            'hedging': ['may have', 'might have', 'possibly', 'supposedly']
        }
        
        self.action_synonyms = {
            'killed': ['shot', 'murdered', 'executed', 'eliminated'],
            'beat': ['assaulted', 'attacked', 'struck', 'hit'],
            'arrested': ['detained', 'captured', 'apprehended', 'seized'],
            'clashed': ['fought', 'engaged', 'confronted', 'battled']
        }

    def generate_actor_perturbations(self, text: str) -> List[Dict[str, Any]]:
        """Generate actor substitution perturbations."""
        perturbations = []
        
        for original, replacements in self.actor_replacements.items():
            if original.lower() in text.lower():
                for replacement in replacements:
                    perturbed = re.sub(
                        rf'\b{re.escape(original)}\b', 
                        replacement, 
                        text, 
                        flags=re.IGNORECASE
                    )
                    if perturbed != text:
                        perturbations.append({
                            'type': 'actor_substitution',
                            'original': original,
                            'replacement': replacement,
                            'text': perturbed,
                            'description': f'Replace "{original}" with "{replacement}"'
                        })
        
        return perturbations

    def generate_intensity_perturbations(self, text: str) -> List[Dict[str, Any]]:
        """Generate intensity/hedging perturbations."""
        perturbations = []
        
        # Add hedging words
        for category, modifiers in self.intensity_modifiers.items():
            for modifier in modifiers:
                # Insert before main action verbs
                for action in ['killed', 'beat', 'shot', 'arrested', 'attacked']:
                    if action in text.lower():
                        pattern = rf'\b({re.escape(action)})'
                        replacement = f'{modifier} {action}'
                        perturbed = re.sub(pattern, replacement, text, flags=re.IGNORECASE, count=1)
                        if perturbed != text:
                            perturbations.append({
                                'type': f'intensity_{category}',
                                'modifier': modifier,
                                'action': action,
                                'text': perturbed,
                                'description': f'Add "{modifier}" before "{action}"'
                            })
        
        return perturbations

    def generate_action_perturbations(self, text: str) -> List[Dict[str, Any]]:
        """Generate action synonym perturbations."""
        perturbations = []
        
        for original, synonyms in self.action_synonyms.items():
            if original.lower() in text.lower():
                for synonym in synonyms:
                    perturbed = re.sub(
                        rf'\b{re.escape(original)}\b',
                        synonym,
                        text,
                        flags=re.IGNORECASE
                    )
                    if perturbed != text:
                        perturbations.append({
                            'type': 'action_substitution',
                            'original': original,
                            'replacement': synonym,
                            'text': perturbed,
                            'description': f'Replace "{original}" with "{synonym}"'
                        })
        
        return perturbations

    def generate_negation_perturbations(self, text: str) -> List[Dict[str, Any]]:
        """Generate negation perturbations."""
        perturbations = []
        
        # Add negations
        negation_targets = ['killed', 'injured', 'arrested', 'beat', 'shot']
        for target in negation_targets:
            if target in text.lower():
                # Add "did not" before action
                pattern = rf'\b({re.escape(target)})'
                replacement = f'did not {target}'
                perturbed = re.sub(pattern, replacement, text, flags=re.IGNORECASE, count=1)
                if perturbed != text:
                    perturbations.append({
                        'type': 'negation',
                        'target': target,
                        'text': perturbed,
                        'description': f'Negate "{target}" action'
                    })
        
        return perturbations

    def generate_sufficiency_perturbations(self, text: str) -> List[Dict[str, Any]]:
        """Generate sufficiency test perturbations (keep only key tokens)."""
        perturbations = []
        
        # Keep only actor + action + object
        key_patterns = [
            r'\b(military|police|soldiers|forces|troops)\b.*?\b(killed|beat|shot|arrested|attacked)\b.*?\b(civilians?|fighters?|members?)\b',
            r'\b(security forces|armed forces)\b.*?\b(killed|beat|shot|arrested|attacked)\b.*?\b(civilians?|protesters?)\b'
        ]
        
        for i, pattern in enumerate(key_patterns):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                key_text = match.group(0)
                perturbations.append({
                    'type': 'sufficiency',
                    'pattern': f'pattern_{i}',
                    'text': key_text,
                    'description': f'Keep only key elements: "{key_text}"'
                })
        
        return perturbations

    def generate_all_perturbations(self, text: str, max_per_type: int = 3) -> List[Dict[str, Any]]:
        """Generate all types of perturbations for a given text."""
        all_perturbations = []
        
        # Generate each type
        generators = [
            self.generate_actor_perturbations,
            self.generate_intensity_perturbations,
            self.generate_action_perturbations,
            self.generate_negation_perturbations,
            self.generate_sufficiency_perturbations
        ]
        
        for generator in generators:
            perturbations = generator(text)
            # Limit per type to avoid explosion
            if len(perturbations) > max_per_type:
                perturbations = perturbations[:max_per_type]
            all_perturbations.extend(perturbations)
        
        return all_perturbations


class CounterfactualAnalyzer:
    """Main analyzer for counterfactual model behavior."""
    
    def __init__(self, country: str, models: List[str]):
        self.country = country
        self.models = models
        self.perturbation_generator = PerturbationGenerator()
        self.paths = paths_for_country(country)
        
        # Load original results
        self.original_results = self.load_original_results()
        
    def load_original_results(self) -> pd.DataFrame:
        """Load original model results for comparison."""
        calibrated_path = self.paths['calibrated_csv']
        if os.path.exists(calibrated_path):
            df = pd.read_csv(calibrated_path)
            # Filter to selected models
            return df[df['model'].isin(self.models)]
        else:
            raise FileNotFoundError(f"Calibrated results not found: {calibrated_path}")
    
    def run_model_on_perturbation(self, model: str, text: str) -> Dict[str, Any]:
        """Run a single model on perturbed text."""
        try:
            result = run_ollama_structured(model, text)
            return {
                'label': result.get('label', 'ERROR'),
                'confidence': result.get('confidence', 0.0),
                'success': True
            }
        except Exception as e:
            return {
                'label': 'ERROR',
                'confidence': 0.0,
                'success': False,
                'error': str(e)
            }
    
    def analyze_event(self, event_row: pd.Series) -> Dict[str, Any]:
        """Analyze a single event with all perturbations."""
        event_id = event_row['event_id']
        original_text = event_row['notes']
        true_label = event_row['gold_label']
        
        print(f"Analyzing event {event_id}...")
        
        # Get original predictions for this event
        original_preds = {}
        for model in self.models:
            model_results = self.original_results[
                (self.original_results['model'] == model) & 
                (self.original_results['event_id'] == event_id)
            ]
            if not model_results.empty:
                original_preds[model] = {
                    'label': model_results.iloc[0]['pred_label'],
                    'confidence': model_results.iloc[0]['pred_conf_temp']
                }
        
        # Generate perturbations
        perturbations = self.perturbation_generator.generate_all_perturbations(original_text)
        
        results = {
            'event_id': event_id,
            'original_text': original_text,
            'true_label': true_label,
            'original_predictions': original_preds,
            'perturbations': []
        }
        
        # Test each perturbation
        for pert in perturbations:
            pert_result = {
                'perturbation': pert,
                'model_results': {}
            }
            
            # Run all models on this perturbation
            for model in self.models:
                model_result = self.run_model_on_perturbation(model, pert['text'])
                pert_result['model_results'][model] = model_result
                
                # Compute metrics vs original
                if model in original_preds and model_result['success']:
                    orig = original_preds[model]
                    pert_result['model_results'][model].update({
                        'label_flipped': model_result['label'] != orig['label'],
                        'confidence_delta': model_result['confidence'] - orig['confidence'],
                        'flip_direction': f"{orig['label']} -> {model_result['label']}" if model_result['label'] != orig['label'] else None
                    })
            
            results['perturbations'].append(pert_result)
        
        return results
    
    def compute_flip_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregated flip metrics across all events."""
        metrics = defaultdict(lambda: defaultdict(list))
        
        for event_result in results:
            for pert_result in event_result['perturbations']:
                pert_type = pert_result['perturbation']['type']
                
                for model, model_result in pert_result['model_results'].items():
                    if model_result['success'] and 'label_flipped' in model_result:
                        metrics[pert_type][model].append({
                            'flipped': model_result['label_flipped'],
                            'confidence_delta': model_result.get('confidence_delta', 0),
                            'flip_direction': model_result.get('flip_direction')
                        })
        
        # Compute summary statistics
        summary = {}
        for pert_type, model_data in metrics.items():
            summary[pert_type] = {}
            for model, flips in model_data.items():
                if flips:
                    flip_rate = sum(1 for f in flips if f['flipped']) / len(flips)
                    conf_deltas = [f['confidence_delta'] for f in flips]
                    summary[pert_type][model] = {
                        'flip_rate': flip_rate,
                        'n_perturbations': len(flips),
                        'mean_confidence_delta': np.mean(conf_deltas),
                        'std_confidence_delta': np.std(conf_deltas),
                        'mean_abs_confidence_delta': np.mean(np.abs(conf_deltas))
                    }
        
        return summary
    
    def statistical_tests(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical tests on model differences."""
        test_results = {}
        
        # Collect paired data for McNemar tests
        if len(self.models) == 2:
            model1, model2 = self.models
            
            for pert_type in ['actor_substitution', 'intensity_high', 'action_substitution']:
                flips_m1, flips_m2 = [], []
                
                for event_result in results:
                    for pert_result in event_result['perturbations']:
                        if pert_result['perturbation']['type'] == pert_type:
                            m1_res = pert_result['model_results'].get(model1, {})
                            m2_res = pert_result['model_results'].get(model2, {})
                            
                            if (m1_res.get('success') and m2_res.get('success') and 
                                'label_flipped' in m1_res and 'label_flipped' in m2_res):
                                flips_m1.append(m1_res['label_flipped'])
                                flips_m2.append(m2_res['label_flipped'])
                
                if len(flips_m1) > 5:  # Minimum sample size
                    # McNemar test for label flips
                    contingency = [[0, 0], [0, 0]]
                    for f1, f2 in zip(flips_m1, flips_m2):
                        contingency[int(f1)][int(f2)] += 1
                    
                    try:
                        from scipy.stats import chi2_contingency
                        chi2, p_val, _, _ = chi2_contingency(contingency)
                        test_results[f'{pert_type}_chi2'] = {
                            'statistic': chi2,
                            'p_value': p_val,
                            'contingency': contingency
                        }
                    except Exception as e:
                        test_results[f'{pert_type}_chi2'] = {'error': str(e)}
        
        return test_results
    
    def cluster_sensitivity_patterns(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cluster events by sensitivity patterns."""
        event_profiles = {}
        
        for event_result in results:
            event_id = event_result['event_id']
            profile = defaultdict(float)  # Changed to float
            
            for pert_result in event_result['perturbations']:
                pert_type = pert_result['perturbation']['type']
                
                # Count flips across models for this perturbation type
                total_flips = 0
                total_models = 0
                
                for model, model_result in pert_result['model_results'].items():
                    if model_result['success'] and 'label_flipped' in model_result:
                        total_models += 1
                        if model_result['label_flipped']:
                            total_flips += 1
                
                if total_models > 0:
                    profile[pert_type] += total_flips / total_models
            
            event_profiles[event_id] = dict(profile)
        
        # Simple clustering by dominant sensitivity
        clusters = defaultdict(list)
        for event_id, profile in event_profiles.items():
            if profile:
                dominant_type = max(profile, key=profile.get)
                clusters[dominant_type].append(event_id)
            else:
                clusters['robust'].append(event_id)
        
        return dict(clusters)
    
    def generate_report(self, results: List[Dict[str, Any]], output_path: str):
        """Generate comprehensive analysis report."""
        flip_metrics = self.compute_flip_metrics(results)
        test_results = self.statistical_tests(results)
        clusters = self.cluster_sensitivity_patterns(results)
        
        report = {
            'metadata': {
                'country': self.country,
                'models': self.models,
                'n_events': len(results),
                'n_perturbations_total': sum(len(r['perturbations']) for r in results)
            },
            'flip_metrics': flip_metrics,
            'statistical_tests': test_results,
            'sensitivity_clusters': clusters,
            'detailed_results': results
        }
        
        # Save full report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate summary CSV
        summary_path = output_path.replace('.json', '_summary.csv')
        self.generate_summary_csv(flip_metrics, summary_path)
        
        print(f"Report saved to: {output_path}")
        print(f"Summary saved to: {summary_path}")
        
        return report
    
    def generate_summary_csv(self, flip_metrics: Dict, output_path: str):
        """Generate CSV summary of flip metrics."""
        rows = []
        for pert_type, model_data in flip_metrics.items():
            for model, metrics in model_data.items():
                rows.append({
                    'perturbation_type': pert_type,
                    'model': model,
                    'flip_rate': metrics['flip_rate'],
                    'n_perturbations': metrics['n_perturbations'],
                    'mean_confidence_delta': metrics['mean_confidence_delta'],
                    'mean_abs_confidence_delta': metrics['mean_abs_confidence_delta']
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description='Counterfactual analysis for model disagreements')
    parser.add_argument('--models', required=True, help='Comma-separated list of models to analyze')
    parser.add_argument('--events', type=int, default=None, help='Number of events to analyze (default: all)')
    parser.add_argument('--output', default=None, help='Output file path')
    
    args = parser.parse_args()
    
    country = os.environ.get('COUNTRY', 'cmr')
    models = [m.strip() for m in args.models.split(',')]
    
    # Load sample events
    paths = paths_for_country(country)
    sample_path = f"datasets/{country}/state_actor_sample_{country}.csv"
    
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Sample file not found: {sample_path}")
    
    events_df = pd.read_csv(sample_path)
    events_df = events_df.rename(columns={'event_id_cnty': 'event_id'})
    
    if args.events:
        events_df = events_df.head(args.events)
    
    print(f"Analyzing {len(events_df)} events with models: {models}")
    
    # Run analysis
    analyzer = CounterfactualAnalyzer(country, models)
    
    results = []
    for _, event_row in events_df.iterrows():
        try:
            event_result = analyzer.analyze_event(event_row)
            results.append(event_result)
        except Exception as e:
            print(f"Error analyzing event {event_row['event_id']}: {e}")
            continue
    
    # Generate report
    output_path = args.output or f"results/{country}/counterfactual_analysis_{'-'.join(models).replace(':', '_')}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    analyzer.generate_report(results, output_path)
    
    print(f"\nAnalysis complete! Analyzed {len(results)} events.")


if __name__ == '__main__':
    main()
