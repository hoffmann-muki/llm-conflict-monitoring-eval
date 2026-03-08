#!/usr/bin/env python3
"""Compute per-class metrics per model and extract top-N disagreement examples.

Includes qualitative error case sampling for annotation:
- Samples N=200 false legitimization errors (true=V, pred=B/S) 
- Samples N=200 false illegitimization errors (true=B/S, pred=V)
- Auto-generates annotations for provenance, verb intensity, casualties, passive voice

Reads: country-specific results/{country}/{strategy}/ollama_results_calibrated.csv
Writes: country-specific results/{country}/{strategy}/per_class_report.csv, top_disagreements.csv,
        error_cases_false_legitimization.csv, error_cases_false_illegitimization.csv
"""
from __future__ import annotations
import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from lib.core.data_helpers import setup_country_environment
from lib.analysis.auto_annotate import auto_annotate_dataframe
from lib.analysis.event_ambiguity import annotate_disagreements_with_ambiguity

# Module-level constants (no side effects)
# TOP_N can be overridden via environment variable for counterfactual analysis
TOP_N = int(os.environ.get('TOP_N_DISAGREEMENTS', '20'))
ERROR_SAMPLE_SIZE = 200  # N=200 as specified

def main():
    # Setup paths at runtime (not import time)
    COUNTRY, RESULTS_DIR = setup_country_environment()
    
    INPUT_CSV = os.path.join(RESULTS_DIR, 'ollama_results_calibrated.csv')
    OUT_PER_CLASS = os.path.join(RESULTS_DIR, 'per_class_report.csv')
    OUT_TOP = os.path.join(RESULTS_DIR, 'top_disagreements.csv')
    OUT_FL_ERRORS = os.path.join(RESULTS_DIR, 'error_cases_false_legitimization.csv')
    OUT_FI_ERRORS = os.path.join(RESULTS_DIR, 'error_cases_false_illegitimization.csv')
    
    os.makedirs("results", exist_ok=True)
    df = pd.read_csv(INPUT_CSV)

    # Normalize column names we expect
    required = {"model", "event_id", "true_label", "pred_label", "pred_conf_temp", "actor_norm"}
    if not required.issubset(set(df.columns)):
        raise SystemExit(f"Missing expected columns in {INPUT_CSV}. Found: {list(df.columns)}")

    # Per-class metrics per model
    labels = sorted(df["true_label"].dropna().unique())
    rows = []
    for model in sorted(df["model"].unique()):
        mdf = df[df["model"] == model]
        y_true = mdf["true_label"].astype(str)
        y_pred = mdf["pred_label"].astype(str)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        for lab, p, r, f, s in zip(labels, precision, recall, f1, support): # type: ignore
            rows.append({"model": model, "label": lab, "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)})

    per_class_df = pd.DataFrame(rows)
    per_class_df.to_csv(OUT_PER_CLASS, index=False)

    # Build wide table of predictions/probabilities per event
    # Use calibrated temperature-scaled probability column `pred_conf_temp`
    piv_label = df.pivot(index="event_id", columns="model", values="pred_label")
    piv_prob = df.pivot(index="event_id", columns="model", values="pred_conf_temp")

    # Retrieve canonical event-level columns (include notes for downstream counterfactual analysis)
    meta_cols = ["true_label", "actor_norm"]
    if "notes" in df.columns:
        meta_cols.append("notes")
    event_meta = df.drop_duplicates(subset=["event_id"]).set_index("event_id")[meta_cols]

    # Align
    wide = event_meta.join(piv_label.add_prefix("pred_label_")).join(piv_prob.add_prefix("pred_prob_"))
    # Drop rows with any missing model predictions
    wide = wide.dropna()

    # Determine disagreement rows (more than 1 unique predicted label across models)
    def disagreement_count(row):
        labels = [v for k, v in row.items() if k.startswith("pred_label_")]
        return len(set(labels))

    wide["disagreement_count"] = wide.apply(disagreement_count, axis=1)
    wide["max_confidence"] = wide[[c for c in wide.columns if c.startswith("pred_prob_")]].max(axis=1)

    disagreements = wide[wide["disagreement_count"] > 1].copy()
    disagreements = disagreements.sort_values(by=["max_confidence"], ascending=False)

    # Annotate every disagreement event with an Event Ambiguity Score (EAS).
    # This stratifies events by task-inherent uncertainty (aleatoric) vs.
    # model-induced instability (epistemic), enabling targeted error analysis
    # that distinguishes genuine boundary cases (e.g. unidentified armed groups
    # at the V/B boundary) from clear-cut events where model bias is operative.
    # See lib/analysis/event_ambiguity.py for the full methodology.
    try:
        disagreements = annotate_disagreements_with_ambiguity(disagreements)
    except Exception as e:
        print(f"Warning: ambiguity annotation failed ({e}) — proceeding without EAS columns.")

    topn = disagreements.head(TOP_N)
    topn.reset_index(inplace=True)
    topn.to_csv(OUT_TOP, index=False)

    print(f"Wrote per-class report: {OUT_PER_CLASS} ({len(per_class_df)} rows)")
    print(f"Wrote top-{TOP_N} disagreements: {OUT_TOP} ({len(topn)} rows)")
    
    # Sample error cases for qualitative annotation
    sample_error_cases(df, COUNTRY, OUT_FL_ERRORS, OUT_FI_ERRORS)

def sample_error_cases(df: pd.DataFrame, COUNTRY: str, OUT_FL_ERRORS: str, OUT_FI_ERRORS: str):
    """Sample N=200 errors of each harm type for qualitative annotation.
    
    Samples false legitimization and false illegitimization errors with metadata
    and auto-generated annotations for causal factors: provenance, ambiguous actor role,
    verb intensity, casualty counts, passive voice usage.
    """
    # Define legitimate and illegitimate labels
    LEGIT_LABELS = ['B', 'S']  # Battles, Strategic developments
    ILLEG_LABELS = ['V']  # Violence against civilians
    
    # False Legitimization: true=V (illegitimate), pred=B/S (legitimate)
    fl_errors = df[
        (df['true_label'].isin(ILLEG_LABELS)) & 
        (df['pred_label'].isin(LEGIT_LABELS))
    ].copy()
    
    # False Illegitimization: true=B/S (legitimate), pred=V (illegitimate)
    fi_errors = df[
        (df['true_label'].isin(LEGIT_LABELS)) & 
        (df['pred_label'].isin(ILLEG_LABELS))
    ].copy()
    
    # Sample up to N=200 for each type
    if len(fl_errors) > ERROR_SAMPLE_SIZE:
        # Stratified sampling by model if possible
        fl_sample = fl_errors.groupby('model', group_keys=False).apply(
            lambda x: x.sample(min(len(x), ERROR_SAMPLE_SIZE // len(df['model'].unique()) + 1))
        ).head(ERROR_SAMPLE_SIZE)
    else:
        fl_sample = fl_errors
    
    if len(fi_errors) > ERROR_SAMPLE_SIZE:
        fi_sample = fi_errors.groupby('model', group_keys=False).apply(
            lambda x: x.sample(min(len(x), ERROR_SAMPLE_SIZE // len(df['model'].unique()) + 1))
        ).head(ERROR_SAMPLE_SIZE)
    else:
        fi_sample = fi_errors
    
    # Try to load original ACLED data for additional metadata (NOTES, SOURCE, FATALITIES)
    original_data = None
    country_code = COUNTRY.lower()[:3]
    country_names = {
        'cmr': 'Cameroon',
        'nga': 'Nigeria',
    }
    country_name = country_names.get(country_code, COUNTRY.title())
    possible_paths = [
        f'datasets/{country_code}/{country_name}_lagged_data_up_to-2024-10-24.csv',
        f'datasets/{country_code.upper()}/{country_name}_lagged_data_up_to-2024-10-24.csv',
        f'datasets/2014-01-01-2024-12-31-{country_name}.csv',
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                original_data = pd.read_csv(path)
                print(f"  Loaded ACLED metadata from: {path}")
                break
            except Exception as e:
                print(f"  Warning: Could not load {path}: {e}")
    
    # Merge NOTES, SOURCE, FATALITIES from original data if available
    if original_data is not None:
        # Normalize column names
        orig_cols_map = {c.upper(): c for c in original_data.columns}
        event_id_col = orig_cols_map.get('EVENT_ID_CNTY') or orig_cols_map.get('EVENT_ID')
        
        if event_id_col:
            # Select columns to merge
            merge_cols = [event_id_col]
            for col_upper in ['NOTES', 'SOURCE', 'SOURCE_SCALE', 'FATALITIES']:
                if col_upper in orig_cols_map:
                    merge_cols.append(orig_cols_map[col_upper])
            
            orig_subset = original_data[merge_cols].drop_duplicates(subset=[event_id_col])
            
            # Merge into error samples
            if len(fl_sample) > 0:
                fl_sample = fl_sample.merge(orig_subset, left_on='event_id', right_on=event_id_col, how='left')
                # Normalize column names for auto_annotate
                if 'NOTES' in fl_sample.columns:
                    fl_sample['notes'] = fl_sample['NOTES']
            if len(fi_sample) > 0:
                fi_sample = fi_sample.merge(orig_subset, left_on='event_id', right_on=event_id_col, how='left')
                if 'NOTES' in fi_sample.columns:
                    fi_sample['notes'] = fi_sample['NOTES']
    
    # Apply automated annotations (data already merged above, so pass None)
    if len(fl_sample) > 0:
        fl_sample = auto_annotate_dataframe(fl_sample, None)
    if len(fi_sample) > 0:
        fi_sample = auto_annotate_dataframe(fi_sample, None)
    
    # Ensure all annotation columns exist
    annotation_columns = [
        'annotation_provenance',  # State media, independent, etc.
        'annotation_ambiguous_actor',  # Yes/No if actor role unclear
        'annotation_verb_intensity',  # High/Medium/Low
        'annotation_casualty_counts',  # Yes/No if casualties mentioned
        'annotation_passive_voice',  # Yes/No if passive voice used
        'annotation_notes'  # Free text for annotator notes
    ]
    
    for col in annotation_columns:
        if col not in fl_sample.columns:
            fl_sample[col] = ''
        if col not in fi_sample.columns:
            fi_sample[col] = ''
    
    # Select relevant columns for annotation
    output_cols = [
        'event_id', 'model', 'true_label', 'pred_label', 'pred_conf_temp',
        'notes', 'actor_norm'
    ] + annotation_columns
    
    # Keep only columns that exist
    fl_output_cols = [col for col in output_cols if col in fl_sample.columns]
    fi_output_cols = [col for col in output_cols if col in fi_sample.columns]
    
    fl_sample[fl_output_cols].to_csv(OUT_FL_ERRORS, index=False)
    fi_sample[fi_output_cols].to_csv(OUT_FI_ERRORS, index=False)
    
    print(f"\nQualitative Error Case Sampling:")
    print(f"  False Legitimization errors: {len(fl_sample)} cases -> {OUT_FL_ERRORS}")
    print(f"  False Illegitimization errors: {len(fi_sample)} cases -> {OUT_FI_ERRORS}")
    print(f"  Annotations auto-generated (review and refine as needed)")

if __name__ == "__main__":
    main()
