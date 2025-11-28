import os
import json
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from lib.core.data_helpers import setup_country_environment

COUNTRY, RESULTS_DIR = setup_country_environment()

# Input/raw predictions CSV (combined from aggregator)
RESULTS_CSV = os.path.join(RESULTS_DIR, f'ollama_results_acled_{COUNTRY}_state_actors.csv')
# Calibration params (written by calibrate_confidences)
CAL_PARAMS = os.path.join(RESULTS_DIR, f'calibration_params_acled_{COUNTRY}_state_actors.json')
# Combined outputs
OUT_CAL_CSV = os.path.join(RESULTS_DIR, 'ollama_results_calibrated.csv')
OUT_METRICS_CSV = os.path.join(RESULTS_DIR, 'metrics_thresholds_calibrated.csv')
OUT_BRIER_CSV = os.path.join(RESULTS_DIR, 'calibration_brier_scores.csv')
OUT_PLOT_REL = os.path.join(RESULTS_DIR, 'reliability_diagrams.png')
OUT_PLOT_ACC = os.path.join(RESULTS_DIR, 'accuracy_vs_coverage.png')
OUT_ISO_MAP = os.path.join(RESULTS_DIR, 'isotonic_mappings.json')

labels = ['V','B','E','P','R','S']
thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
RANDOM_SEED = 42  # For reproducible train/test splits


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

def compute_brier_scores(df, mappings, cal_params):
    """Compute Brier scores for raw and calibrated probabilities."""
    brier_results = []
    
    for model in df['model'].unique():
        sub = df[df['model'] == model].copy()
        sub = sub[sub['true_label'].isin(labels) & sub['pred_label'].isin(labels)]
        
        if len(sub) == 0:
            continue
        
        # Binary outcome: prediction matches ground truth
        y_true = (sub['pred_label'] == sub['true_label']).astype(int).values
        
        # Raw probabilities
        probs_raw = pd.to_numeric(sub['pred_conf'], errors='coerce').fillna(0.0).values
        
        # Isotonic calibrated
        iso = mappings.get(model)
        if iso is not None:
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(iso['x'], iso['y'])
            probs_iso = ir.transform(probs_raw)
        else:
            probs_iso = probs_raw
        
        # Temperature calibrated
        T = cal_params.get(model, {}).get('T', 1.0)
        probs_temp = temp_scaled(probs_raw, T)
        
        # Compute Brier scores
        try:
            brier_raw = brier_score_loss(y_true, probs_raw)
            brier_iso = brier_score_loss(y_true, probs_iso)
            brier_temp = brier_score_loss(y_true, probs_temp)
        except:
            brier_raw = brier_iso = brier_temp = None
        
        brier_results.append({
            'model': model,
            'n_samples': len(sub),
            'brier_score_raw': brier_raw,
            'brier_score_isotonic': brier_iso,
            'brier_score_temperature': brier_temp,
            'brier_improvement_iso': (brier_raw - brier_iso) if brier_raw and brier_iso else None,
            'brier_improvement_temp': (brier_raw - brier_temp) if brier_raw and brier_temp else None
        })
    
    return pd.DataFrame(brier_results)

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
    
    # Split data into train (for fitting isotonic) and test (for evaluation)
    # Filter to valid labels first
    df_valid = df[df['true_label'].isin(labels) & df['pred_label'].isin(labels)].copy()
    
    if len(df_valid) == 0:
        print("No valid data for calibration")
        return
    
    # Stratified split by model to ensure each model has train/test data
    train_indices = []
    test_indices = []
    for m in df_valid['model'].unique():
        model_mask = df_valid['model'] == m
        model_indices = df_valid[model_mask].index.tolist()
        if len(model_indices) < 2:
            # Too few samples, use all for both train and test
            train_indices.extend(model_indices)
            test_indices.extend(model_indices)
        else:
            m_train, m_test = train_test_split(
                model_indices, 
                test_size=0.2, 
                random_state=RANDOM_SEED
            )
            train_indices.extend(m_train)
            test_indices.extend(m_test)
    
    df_train = df_valid.loc[train_indices].copy()
    df_test = df_valid.loc[test_indices].copy()
    
    print(f"Train set: {len(df_train)} samples")
    print(f"Test set: {len(df_test)} samples")
    
    iso_mappings = {}
    # Fit isotonic on TRAIN set per model
    for m in df_train['model'].unique():
        sub = df_train[df_train['model']==m].copy()
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
    
    # Apply calibrations to TEST set only
    df_test['pred_conf_iso'] = df_test['pred_conf']
    df_test['pred_conf_temp'] = df_test['pred_conf']
    # initialize per-class prob columns (if logits absent these will remain NaN)
    for lab in labels:
        df_test[f'prob_{lab}'] = float('nan')
    for m, mapv in iso_mappings.items():
        mask = df_test['model']==m
        if mapv is not None:
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(mapv['x'], mapv['y'])
            # Ensure we operate on a pandas Series so .fillna is available,
            # then convert to a plain numpy float array for the transformer.
            numeric_vals = pd.to_numeric(df_test.loc[mask, 'pred_conf'], errors='coerce').fillna(0.0).to_numpy(dtype=float) # type: ignore
            df_test.loc[mask, 'pred_conf_iso'] = ir.transform(numeric_vals)
        T = cal_params.get(m, {}).get('T', 1.0)
        numeric_vals_temp = pd.to_numeric(df_test.loc[mask, 'pred_conf'], errors='coerce').fillna(0.0).to_numpy(dtype=float) # type: ignore
        df_test.loc[mask, 'pred_conf_temp'] = temp_scaled(numeric_vals_temp, T)
        # If multiclass temperature available and logits column exists, compute per-class probs
        T_multi = cal_params.get(m, {}).get('T_multiclass', 1.0)
        if 'logits' in df_test.columns:
            # parse JSON lists and compute softmax(logits / T_multi)
            def compute_probs_json(x):
                try:
                    arr = json.loads(x)
                    arr = np.asarray(arr, dtype=float)
                    # ensure shape matches labels
                    if arr.size != len(labels):
                        return [np.nan]*len(labels)
                    scaled = arr / float(T_multi)
                    z = scaled - np.max(scaled)
                    expz = np.exp(z)
                    probs = expz / (expz.sum() + 1e-12)
                    return probs.tolist()
                except Exception:
                    return [np.nan]*len(labels)
            probs_mat = df_test.loc[mask, 'logits'].apply(compute_probs_json).tolist() # type: ignore
            if len(probs_mat) > 0:
                probs_arr = np.vstack(probs_mat)
                for i, lab in enumerate(labels):
                    df_test.loc[mask, f'prob_{lab}'] = probs_arr[:, i]
    # Save calibrated CSV (test set only to avoid data leakage)
    os.makedirs('results', exist_ok=True)
    df_test.to_csv(OUT_CAL_CSV, index=False)
    print('Saved calibrated TEST set to', OUT_CAL_CSV)
    
    # Save isotonic mappings
    with open(OUT_ISO_MAP,'w') as f:
        json.dump(iso_mappings, f)
    # Compute threshold metrics for raw, isotonic, temp (on TEST set only)
    df_raw = compute_threshold_metrics(df_test, 'pred_conf')
    df_iso = compute_threshold_metrics(df_test, 'pred_conf_iso')
    df_temp = compute_threshold_metrics(df_test, 'pred_conf_temp')
    all_metrics = pd.concat([df_raw, df_iso, df_temp], ignore_index=True)
    all_metrics.to_csv(OUT_METRICS_CSV, index=False)
    print('Saved threshold metrics (test set) to', OUT_METRICS_CSV)
    
    # Compute Brier scores (on TEST set only)
    brier_df = compute_brier_scores(df_test, iso_mappings, cal_params)
    brier_df.to_csv(OUT_BRIER_CSV, index=False)
    print('\nBrier Scores on TEST set (lower is better):')
    print(brier_df.to_string(index=False))
    print(f'\nSaved Brier scores to {OUT_BRIER_CSV}')
    
    # Plots (on TEST set only)
    reliability_curve_plot(df_test, iso_mappings, cal_params)
    print('\nSaved reliability diagrams (test set) to', OUT_PLOT_REL)
    # Accuracy vs coverage plots
    fig, ax = plt.subplots(figsize=(8,6))
    for prob_col, label in [('pred_conf','raw'), ('pred_conf_iso','isotonic'), ('pred_conf_temp','temp')]:
        for m in df_test['model'].unique():
            sub = all_metrics[(all_metrics['model']==m)&(all_metrics['prob_col']==prob_col)]
            ax.plot(sub['coverage'], sub['accuracy'], marker='o', label=f"{m}-{label}")
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Accuracy')
    ax.legend()
    fig.savefig(OUT_PLOT_ACC)
    print('Saved accuracy vs coverage plot (test set) to', OUT_PLOT_ACC)

if __name__=='__main__':
    main()
