#!/usr/bin/env python3
"""Compare FL/FI across model sizes (e.g., gemma:2b vs gemma:7b).

Usage: 
  COUNTRY=cmr STRATEGY=zero_shot SAMPLE_SIZE=500 python -m lib.analysis.compare_models \
    --family gemma --sizes 2b,7b

What it does:
- Reads results from the appropriate strategy/sample_size directory
- Filters rows to models matching family:size (e.g., gemma:2b, gemma:7b).
- Computes FL/FI counts and rates per model, and per-family-size.
- Runs McNemar test for pairwise comparisons on the same events.
- Writes comparison CSV and prints results.

Note: Cross-model comparisons are only valid when comparing models run with:
  - Same country
  - Same strategy
  - Same sample_size
  - Same num_examples (for few_shot strategy)
"""
import os
import argparse
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from lib.core.data_helpers import setup_country_environment, get_strategy, get_sample_size, get_num_examples
from lib.core.metrics_helpers import aggregate_fl_fi, LEGIT, ILLEG
from lib.core.constants import LABEL_MAP
from lib.inference.ollama_client import run_model_on_rows

# Use setup_country_environment to get proper paths including strategy/sample_size
COUNTRY, RESULTS_DIR = setup_country_environment()
STRATEGY = get_strategy()
SAMPLE_SIZE = get_sample_size()
NUM_EXAMPLES = get_num_examples()

LABELS = list(LABEL_MAP.values())

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--family', required=True, help='Model family prefix (e.g., gemma)')
    p.add_argument('--sizes', required=True, help='Comma-separated sizes to compare, e.g. 2b,7b')
    p.add_argument('--run-missing', default='true', choices=['true','false'], help="Run inference for missing models on the sample and append to calibrated CSV (true/false). Defaults to 'true'.")
    p.add_argument('--out', default=None, help='Output CSV path')
    return p.parse_args()

def fl_fi_for_model(df):
    """Return dict with total, fl, fi.
    """
    total = len(df)
    fl = int(((df['true_label'].isin(ILLEG)) & (df['pred_label'].isin(LEGIT))).sum())
    fi = int(((df['true_label'].isin(LEGIT)) & (df['pred_label'].isin(ILLEG))).sum())
    return dict(total=total, fl=fl, fi=fi)

def mcnemar_test(cont_table):
    """
    Run McNemar test on 2x2 contingency table for paired binary classifications.
    
    McNemar tests whether two classifiers differ on the same items. 
    While it focuses on discordant pairs (b and c), it requires the full 2x2 table:
    - a: both models correct
    - b: model A correct, model B wrong
    - c: model A wrong, model B correct  
    - d: both models wrong
    
    If b and c are significantly different, p < 0.05 → models differ significantly.
    """
    try:
        table = [[cont_table.get('a',0), cont_table.get('b',0)], 
                 [cont_table.get('c',0), cont_table.get('d',0)]]
        res = mcnemar(table, exact=False, correction=True)
        return res.statistic, res.pvalue # type: ignore
    except Exception:
        return None, None

def main():
    args = parse_args()
    # normalize run_missing to boolean (accepts 'true'/'false' strings)
    run_missing = args.run_missing
    if isinstance(run_missing, str):
        run_missing = run_missing.lower() in ('true', '1', 'yes')
    family = args.family
    sizes = [s.strip() for s in args.sizes.split(',') if s.strip()]
    models = [f"{family}:{s}" for s in sizes]
    
    # Print comparison context for clarity
    print(f"\\n{'='*70}")
    print(f"Cross-Model Comparison: {family} family")
    print(f"Context: {COUNTRY}, {STRATEGY}, sample_size={SAMPLE_SIZE}" + 
          (f", num_examples={NUM_EXAMPLES}" if STRATEGY == 'few_shot' and NUM_EXAMPLES else ""))
    print(f"Models: {', '.join(models)}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"{'='*70}\\n")

    # Load calibrated CSV from the correct strategy/sample_size directory
    cal_csv_path = os.path.join(RESULTS_DIR, 'ollama_results_calibrated.csv')
    if os.path.exists(cal_csv_path):
        df = pd.read_csv(cal_csv_path)
    else:
        df = pd.DataFrame()

    # Detect missing models and run them on the sample when run_missing is true
    missing = [m for m in models if (df.empty or m not in df['model'].unique())]
    if missing and run_missing:
        # Use the correct sample path with sample_size suffix
        sample_path = os.path.join('datasets', COUNTRY, f'state_actor_sample_{COUNTRY}_{SAMPLE_SIZE}.csv')
        if not os.path.exists(sample_path):
            raise SystemExit(
                f"Missing sample file for inference: {sample_path}\\n"
                f"Run the main pipeline first to create the sample, or check SAMPLE_SIZE={SAMPLE_SIZE}"
            )
        sample = pd.read_csv(sample_path)
        print(f'Running inference for missing models on sample ({len(sample)} events):', missing)
        new_results = []
        for mrun in missing:
            print('Running model', mrun)
            model_results = run_model_on_rows(mrun, sample)
            new_results.extend(model_results)
        if new_results:
            new_df = pd.DataFrame(new_results)
            # write new inference-only file under the country results dir for traceability
            os.makedirs(RESULTS_DIR, exist_ok=True)
            inf_name = f'ollama_inference_{family}-{"-".join(sizes)}.csv'
            inf_path = os.path.join(RESULTS_DIR, inf_name)
            new_df.to_csv(inf_path, index=False)
            print('Wrote inference-only results to', inf_path)
            # use combined df for comparisons (do not overwrite calibrated CSV)
            if df.empty:
                df = new_df.copy()
            else:
                df = pd.concat([df, new_df], ignore_index=True)
    elif missing:
        print('Missing models detected but run_missing is false; skipping inference for:', missing)

    # per-model stats via shared helper — restrict to requested models only
    df_for_agg = df[df['model'].isin(models)] if not df.empty else pd.DataFrame()
    out_df = aggregate_fl_fi(df_for_agg, by='model')
    # ensure requested models are present in the output (add zeros for missing)
    for m in models:
        if m not in out_df['model'].values:
            out_df = pd.concat([out_df, pd.DataFrame([{'model': m, 'total': 0, 'fl': 0, 'fi': 0}])], ignore_index=True)
    out_df = out_df.sort_values('model')
    
    # Add comparison context metadata to output
    out_df.insert(0, 'country', COUNTRY)
    out_df.insert(1, 'strategy', STRATEGY)
    out_df.insert(2, 'sample_size', SAMPLE_SIZE)
    out_df.insert(3, 'num_examples', NUM_EXAMPLES if STRATEGY == 'few_shot' else None)

    # pairwise comparisons on same event ids (if event_id column exists)
    comparisons = []
    # Add comparison metadata for traceability
    comparison_metadata = {
        'country': COUNTRY,
        'strategy': STRATEGY,
        'sample_size': SAMPLE_SIZE,
        'num_examples': NUM_EXAMPLES if STRATEGY == 'few_shot' else None
    }
    
    if 'event_id' in df.columns:
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                a = models[i]
                b = models[j]
                left = df[df['model']==a][['event_id','true_label','pred_label']].rename(columns={'pred_label':'pred_a'})
                right = df[df['model']==b][['event_id','true_label','pred_label']].rename(columns={'pred_label':'pred_b'})
                merged = left.merge(right, on=['event_id','true_label'])
                if merged.empty:
                    comparisons.append({
                        'country': COUNTRY,
                        'strategy': STRATEGY,
                        'sample_size': SAMPLE_SIZE,
                        'num_examples': NUM_EXAMPLES if STRATEGY == 'few_shot' else None,
                        'model_a': a,
                        'model_b': b,
                        'n_common': 0,
                        'note': 'no common events'
                    })
                    continue
                # compute contingency for correctness on relevant labels
                merged['a_correct'] = merged.apply(lambda r: r['pred_a']==r['true_label'], axis=1)
                merged['b_correct'] = merged.apply(lambda r: r['pred_b']==r['true_label'], axis=1)
                # McNemar 2x2 contingency table:
                # a: both correct, b: a correct & b wrong, c: a wrong & b correct, d: both wrong
                a_count = int(((merged['a_correct']==True) & (merged['b_correct']==True)).sum())
                b_count = int(((merged['a_correct']==True) & (merged['b_correct']==False)).sum())
                c_count = int(((merged['a_correct']==False) & (merged['b_correct']==True)).sum())
                d_count = int(((merged['a_correct']==False) & (merged['b_correct']==False)).sum())
                stat, p = mcnemar_test({'a':a_count,'b':b_count,'c':c_count,'d':d_count})
                comparisons.append({
                    'country': COUNTRY,
                    'strategy': STRATEGY, 
                    'sample_size': SAMPLE_SIZE,
                    'num_examples': NUM_EXAMPLES if STRATEGY == 'few_shot' else None,
                    'model_a': a,
                    'model_b': b,
                    'n_common': len(merged),
                    'both_correct': a_count,
                    'a_correct_b_wrong': b_count,
                    'a_wrong_b_correct': c_count,
                    'both_wrong': d_count,
                    'mcnemar_stat': stat,
                    'mcnemar_p': p
                })
    else:
        comparisons.append({'note':'event_id not present; cannot do paired comparisons'})

    # Ensure results directory exists and write outputs
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = args.out or os.path.join(RESULTS_DIR, f'compare_{family}_sizes.csv')
    out_df.to_csv(out_path, index=False)
    # (inference/appending handled earlier when run_missing is True)
    print('Wrote', out_path)
    print(out_df.to_string(index=False))

    comp_path = os.path.join(RESULTS_DIR, f'compare_{family}_sizes_pairwise.csv')
    pd.DataFrame(comparisons).to_csv(comp_path, index=False)
    print('Wrote', comp_path)
    print(pd.DataFrame(comparisons).to_string(index=False))

if __name__ == '__main__':
    main()
