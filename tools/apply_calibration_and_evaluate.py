import os
import json
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

RESULTS_CSV = 'results/ollama_results_acled_cameroon_state_actors.csv'
CAL_PARAMS = 'results/calibration_params_acled_cameroon_state_actors.json'
OUT_CAL_CSV = 'results/ollama_results_calibrated.csv'
OUT_METRICS_CSV = 'results/metrics_thresholds_calibrated.csv'
OUT_PLOT_REL = 'results/reliability_diagrams.png'
OUT_PLOT_ACC = 'results/accuracy_vs_coverage.png'
OUT_ISO_MAP = 'results/isotonic_mappings.json'

labels = ['V','B','E','P','R','S']
thresholds = [0.0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]

def temp_scaled(p, T):
    p = np.clip(p, 1e-12, 1-1e-12)
    logits = np.log(p/(1-p))
    scaled = 1/(1+np.exp(-logits / T))
    return scaled

def compute_threshold_metrics(df, prob_col):
    rows = []
    for m in df['model'].unique():
        sub = df[df['model']==m].copy()
        sub['pred_conf_num'] = pd.to_numeric(sub[prob_col], errors='coerce').fillna(0.0)
        total_valid = int((sub['true_label'].isin(labels)).sum())
        for t in thresholds:
            sel = sub[(sub['pred_conf_num']>=t) & sub['true_label'].isin(labels) & sub['pred_label'].isin(labels)]
            accepted = len(sel)
            coverage = accepted/total_valid if total_valid>0 else 0
            correct = (sel['pred_label']==sel['true_label']).sum()
            acc = correct/accepted if accepted>0 else None
            rows.append({'model':m, 'prob_col':prob_col, 'threshold':t, 'accepted':accepted, 'coverage':coverage, 'correct':int(correct), 'accuracy':(None if acc is None else round(acc,3)), 'total_valid':total_valid})
    return pd.DataFrame(rows)

def reliability_curve_plot(df, mappings, cal_params):
    # For each model, plot before and after calibration reliability diagram
    n_models = len(df['model'].unique())
    fig, axes = plt.subplots(n_models, 1, figsize=(6,4*n_models))
    axes = np.atleast_1d(axes).ravel()
    for ax, m in zip(axes, df['model'].unique()):
        sub = df[df['model']==m].copy()
        sub = sub[sub['true_label'].isin(labels) & sub['pred_label'].isin(labels)]
        if len(sub)==0:
            continue
        y_true = (sub['pred_label']==sub['true_label']).astype(int)
        probs = pd.to_numeric(sub['pred_conf'], errors='coerce').fillna(0.0).values
        # isotonic
        iso = mappings.get(m)
        if iso is not None:
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(iso['x'], iso['y'])
            iso_probs = ir.transform(probs)
        else:
            iso_probs = probs
        # temp
        T = cal_params.get(m, {}).get('T', 1.0)
        temp_probs = temp_scaled(probs, T)

        for name, pvals in [('raw',probs), ('isotonic', iso_probs), ('temp', temp_probs)]:
            frac_pos, mean_pred = calibration_curve(y_true, pvals, n_bins=10)
            ax.plot(mean_pred, frac_pos, marker='o', label=name) 
        ax.plot([0,1],[0,1],'k--',alpha=0.6)
        ax.set_title(m)
        ax.set_xlabel('Mean predicted prob')
        ax.set_ylabel('Fraction positive')
        ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_PLOT_REL)
    plt.close(fig)

def main():
    if not os.path.exists(RESULTS_CSV):
        print('Missing', RESULTS_CSV); return
    df = pd.read_csv(RESULTS_CSV)
    cal_params = {}
    if os.path.exists(CAL_PARAMS):
        with open(CAL_PARAMS) as f:
            cal_params = json.load(f)
    iso_mappings = {}
    # Fit isotonic on full data per model
    for m in df['model'].unique():
        sub = df[(df['model']==m) & df['true_label'].isin(labels) & df['pred_label'].isin(labels)].copy()
        if len(sub)==0:
            continue
        probs = pd.to_numeric(sub['pred_conf'], errors='coerce').fillna(0.0).values
        y = (sub['pred_label']==sub['true_label']).astype(int).values
        try:
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(probs, y)
            # store mapping as sampled points
            xs = np.linspace(0,1,101)
            ys = ir.transform(xs).tolist()
            iso_mappings[m] = {'x': xs.tolist(), 'y': ys}
        except Exception:
            iso_mappings[m] = None
    # apply calibrations
    df['pred_conf_iso'] = df['pred_conf']
    df['pred_conf_temp'] = df['pred_conf']
    for m, mapv in iso_mappings.items():
        mask = df['model']==m
        if mapv is not None:
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(mapv['x'], mapv['y'])
            # Ensure we operate on a pandas Series so .fillna is available,
            # then convert to a plain numpy float array for the transformer.
            numeric_vals = pd.to_numeric(df.loc[mask, 'pred_conf'], errors='coerce').fillna(0.0).to_numpy(dtype=float) # type: ignore
            df.loc[mask, 'pred_conf_iso'] = ir.transform(numeric_vals)
        T = cal_params.get(m, {}).get('T', 1.0)
        numeric_vals_temp = pd.to_numeric(df.loc[mask, 'pred_conf'], errors='coerce').fillna(0.0).to_numpy(dtype=float) # type: ignore
        df.loc[mask, 'pred_conf_temp'] = temp_scaled(numeric_vals_temp, T)
    # Save calibrated CSV
    os.makedirs('results', exist_ok=True)
    df.to_csv(OUT_CAL_CSV, index=False)
    print('Saved calibrated CSV to', OUT_CAL_CSV)
    # Save isotonic mappings
    with open(OUT_ISO_MAP,'w') as f:
        json.dump(iso_mappings, f)
    # Compute threshold metrics for raw, isotonic, temp
    df_raw = compute_threshold_metrics(df, 'pred_conf')
    df_iso = compute_threshold_metrics(df, 'pred_conf_iso')
    df_temp = compute_threshold_metrics(df, 'pred_conf_temp')
    all_metrics = pd.concat([df_raw, df_iso, df_temp], ignore_index=True)
    all_metrics.to_csv(OUT_METRICS_CSV, index=False)
    print('Saved threshold metrics to', OUT_METRICS_CSV)
    # Plots
    reliability_curve_plot(df, iso_mappings, cal_params)
    print('Saved reliability diagrams to', OUT_PLOT_REL)
    # Accuracy vs coverage plots
    fig, ax = plt.subplots(figsize=(8,6))
    for prob_col, label in [('pred_conf','raw'), ('pred_conf_iso','isotonic'), ('pred_conf_temp','temp')]:
        for m in df['model'].unique():
            sub = all_metrics[(all_metrics['model']==m)&(all_metrics['prob_col']==prob_col)]
            ax.plot(sub['coverage'], sub['accuracy'], marker='o', label=f"{m}-{label}")
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Accuracy')
    ax.legend()
    fig.savefig(OUT_PLOT_ACC)
    print('Saved accuracy vs coverage plot to', OUT_PLOT_ACC)

if __name__=='__main__':
    main()
