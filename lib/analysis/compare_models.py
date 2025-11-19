#!/usr/bin/env python3
"""Compare FL/FI across model sizes (e.g., gemma:2b vs gemma:7b).

Usage: COUNTRY=cmr python tools/compare_model_sizes.py --family gemma --sizes 2b,7b --out results/cmr/compare_gemma_sizes.csv

What it does:
- Reads `results/<COUNTRY>/ollama_results_calibrated.csv`.
- Filters rows to models matching family:size (e.g., gemma:2b, gemma:7b).
- Computes FL/FI counts and rates per model, and per-family-size.
- Runs McNemar test for pairwise comparisons on the same events.
- Writes a small CSV with comparisons and prints results.
"""
import os
import argparse
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from tools.data_helpers import paths_for_country
from tools.metrics_helpers import aggregate_fl_fi, LEGIT, ILLEG
from tools.constants import LABEL_MAP
from tools.ollama_helpers import run_model_on_rows

COUNTRY = os.environ.get('COUNTRY', 'cmr')
RESULTS_DIR = os.path.join('results', COUNTRY)

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
    Run McNemar test on contingency table.
    McNemar tests whether two classifiers differ on the same items by comparing only the discordant pairs (cases where A is right & B wrong vs A wrong & B right).
    If those counts are very different, p < 0.05 → significant difference.
    """
    # cont_table is dict {'b':count, 'c':count} where b = modelA correct & modelB incorrect, c = modelA incorrect & modelB correct
    try:
        table = [[cont_table.get('a',0), cont_table.get('b',0)], [cont_table.get('c',0), cont_table.get('d',0)]]
        res = mcnemar(table, exact=False)
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

    # Load calibrated CSV if present, otherwise start empty
    paths = paths_for_country(COUNTRY)
    cal_csv_path = paths['calibrated_csv']
    if os.path.exists(cal_csv_path):
        df = pd.read_csv(cal_csv_path)
    else:
        df = pd.DataFrame()

    # Detect missing models and run them on the sample when run_missing is true
    missing = [m for m in models if (df.empty or m not in df['model'].unique())]
    if missing and run_missing:
        sample_path = os.path.join('datasets', COUNTRY, f'state_actor_sample_{COUNTRY}.csv')
        if not os.path.exists(sample_path):
            raise SystemExit(f"Missing sample file for inference: {sample_path}")
        sample = pd.read_csv(sample_path)
        print('Running inference for missing models:', missing)
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

    # pairwise comparisons on same event ids (if event_id column exists)
    comparisons = []
    if 'event_id' in df.columns:
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                a = models[i]
                b = models[j]
                left = df[df['model']==a][['event_id','true_label','pred_label']].rename(columns={'pred_label':'pred_a'})
                right = df[df['model']==b][['event_id','true_label','pred_label']].rename(columns={'pred_label':'pred_b'})
                merged = left.merge(right, on=['event_id','true_label'])
                if merged.empty:
                    comparisons.append({'model_a':a,'model_b':b,'n_common':0,'note':'no common events'})
                    continue
                # compute contingency for correctness on relevant labels
                merged['a_correct'] = merged.apply(lambda r: r['pred_a']==r['true_label'], axis=1)
                merged['b_correct'] = merged.apply(lambda r: r['pred_b']==r['true_label'], axis=1)
                # b: a_correct & not b_correct ; c: not a_correct & b_correct
                b_count = int(((merged['a_correct']==True) & (merged['b_correct']==False)).sum())
                c_count = int(((merged['a_correct']==False) & (merged['b_correct']==True)).sum())
                stat, p = mcnemar_test({'a':0,'b':b_count,'c':c_count,'d':0})
                comparisons.append({'model_a':a,'model_b':b,'n_common':len(merged),'b':b_count,'c':c_count,'mcnemar_stat':stat,'mcnemar_p':p})
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
