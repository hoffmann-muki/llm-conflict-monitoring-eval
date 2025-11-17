import pandas as pd
import numpy as np
import json
import os
from scipy import stats
from data_helpers import setup_country_environment

COUNTRY, RESULTS_DIR = setup_country_environment()

RESULTS_CSV = os.path.join(RESULTS_DIR, f'ollama_results_acled_{COUNTRY}_state_actors.csv')
OUT_METRICS = os.path.join(RESULTS_DIR, f'metrics_acled_{COUNTRY}_state_actors.csv')
OUT_CMS = os.path.join(RESULTS_DIR, f'confusion_matrices_acled_{COUNTRY}_state_actors.json')
OUT_FAIRNESS = os.path.join(RESULTS_DIR, f'fairness_metrics_acled_{COUNTRY}_state_actors.csv')

labels = ['V','B','E','P','R','S']

# Define severe/sensitive labels for bias analysis
# V = Violence against civilians is typically the most sensitive label
SEVERE_LABELS = ['V']  # Can be adjusted based on context

def compute_metrics(df):
    models = df['model'].unique()
    metrics = []
    cms = {}
    for m in models:
        sub = df[df['model']==m]
        y_true = sub['true_label'].fillna('')
        y_pred = sub['pred_label'].fillna('')
        mask = y_true.isin(labels)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        N = len(y_true)
        if N == 0:
            metrics.append({'model': m, 'N': 0, 'accuracy': None, 'precision_macro': None, 'recall_macro': None})
            cms[m] = None
            continue
        label_to_idx = {l:i for i,l in enumerate(labels)}
        cm = np.zeros((len(labels),len(labels)), dtype=int)
        for t,p in zip(y_true, y_pred):
            i = label_to_idx.get(t)
            j = label_to_idx.get(p)
            if i is not None and j is not None:
                cm[i,j] += 1
        acc = int(cm.trace())/int(cm.sum()) if cm.sum()>0 else None
        precisions = []
        recalls = []
        for i,_ in enumerate(labels):
            tp = cm[i,i]
            fp = cm[:,i].sum() - tp
            fn = cm[i,:].sum() - tp
            prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
            rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
            precisions.append(prec)
            recalls.append(rec)
        metrics.append({'model': m, 'N': N, 'accuracy': round(acc,3) if acc is not None else None,
                        'precision_macro': round(float(np.mean(precisions)),3),
                        'recall_macro': round(float(np.mean(recalls)),3)})
        cms[m] = cm.tolist()
    return metrics, cms

def compute_fairness_metrics(df: pd.DataFrame, target_label: str = 'V', n_bootstrap: int = 1000) -> pd.DataFrame:
    """Compute group fairness metrics: Statistical Parity Difference (SPD) and Equalized Odds.
    
    Args:
        df: DataFrame with columns model, pred_label, true_label, actor_norm (or similar actor column)
        target_label: The sensitive label to focus on (default 'V' for Violence)
        n_bootstrap: Number of bootstrap samples for confidence intervals
    
    Returns:
        DataFrame with fairness metrics per model
    """
    fairness_results = []
    
    # Check if we have actor information
    actor_col = None
    for col in ['actor_norm', 'actor', 'actor_group']:
        if col in df.columns:
            actor_col = col
            break
    
    if actor_col is None:
        print(f"Warning: No actor column found. Cannot compute group fairness metrics.")
        return pd.DataFrame()
    
    # Create binary actor groups: state vs non-state
    def classify_actor(actor_str):
        if pd.isna(actor_str):
            return 'unknown'
        actor_lower = str(actor_str).lower()
        if any(term in actor_lower for term in ['state forces', 'military', 'police', 'government', 'gendarmerie']):
            return 'state'
        else:
            return 'non-state'
    
    df['actor_group'] = df[actor_col].apply(classify_actor)
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model].copy()
        
        # Filter to valid labels
        model_df = model_df[model_df['true_label'].isin(labels) & model_df['pred_label'].isin(labels)]
        
        if len(model_df) == 0:
            continue
        
        # Separate by actor group
        state_df = model_df[model_df['actor_group'] == 'state']
        nonstate_df = model_df[model_df['actor_group'] == 'non-state']
        
        if len(state_df) == 0 or len(nonstate_df) == 0:
            print(f"Warning: Model {model} has insufficient data for both actor groups")
            continue
        
        # (A) Statistical Parity Difference (SPD)
        # P(model=target_label | group=state) - P(model=target_label | group=non-state)
        p_target_state = (state_df['pred_label'] == target_label).mean()
        p_target_nonstate = (nonstate_df['pred_label'] == target_label).mean()
        spd = p_target_state - p_target_nonstate
        
        # Bootstrap CI for SPD
        spd_bootstrap = []
        for _ in range(n_bootstrap):
            state_boot = state_df.sample(n=len(state_df), replace=True)
            nonstate_boot = nonstate_df.sample(n=len(nonstate_df), replace=True)
            p_state_boot = (state_boot['pred_label'] == target_label).mean()
            p_nonstate_boot = (nonstate_boot['pred_label'] == target_label).mean()
            spd_bootstrap.append(p_state_boot - p_nonstate_boot)
        
        spd_ci_lower = np.percentile(spd_bootstrap, 2.5)
        spd_ci_upper = np.percentile(spd_bootstrap, 97.5)
        
        # (B) Equalized Odds: TPR and FPR differences for target label
        # TPR = P(pred=target | true=target)
        # FPR = P(pred=target | true!=target)
        
        state_target_true = state_df[state_df['true_label'] == target_label]
        state_target_false = state_df[state_df['true_label'] != target_label]
        nonstate_target_true = nonstate_df[nonstate_df['true_label'] == target_label]
        nonstate_target_false = nonstate_df[nonstate_df['true_label'] != target_label]
        
        tpr_state = (state_target_true['pred_label'] == target_label).mean() if len(state_target_true) > 0 else 0
        tpr_nonstate = (nonstate_target_true['pred_label'] == target_label).mean() if len(nonstate_target_true) > 0 else 0
        tpr_diff = tpr_state - tpr_nonstate
        
        fpr_state = (state_target_false['pred_label'] == target_label).mean() if len(state_target_false) > 0 else 0
        fpr_nonstate = (nonstate_target_false['pred_label'] == target_label).mean() if len(nonstate_target_false) > 0 else 0
        fpr_diff = fpr_state - fpr_nonstate
        
        # Permutation test for TPR difference
        tpr_perm_diffs = []
        combined = pd.concat([state_target_true, nonstate_target_true])
        n_state_true = len(state_target_true)
        
        if len(combined) > 10:  # Only if sufficient samples
            for _ in range(min(1000, n_bootstrap)):
                shuffled = combined.sample(frac=1.0)
                perm_state = shuffled.iloc[:n_state_true]
                perm_nonstate = shuffled.iloc[n_state_true:]
                tpr_perm_state = (perm_state['pred_label'] == target_label).mean()
                tpr_perm_nonstate = (perm_nonstate['pred_label'] == target_label).mean()
                tpr_perm_diffs.append(tpr_perm_state - tpr_perm_nonstate)
            
            tpr_pvalue = np.mean(np.abs(tpr_perm_diffs) >= np.abs(tpr_diff))
        else:
            tpr_pvalue = None
        
        # Permutation test for FPR difference
        fpr_perm_diffs = []
        combined_false = pd.concat([state_target_false, nonstate_target_false])
        n_state_false = len(state_target_false)
        
        if len(combined_false) > 10:
            for _ in range(min(1000, n_bootstrap)):
                shuffled = combined_false.sample(frac=1.0)
                perm_state = shuffled.iloc[:n_state_false]
                perm_nonstate = shuffled.iloc[n_state_false:]
                fpr_perm_state = (perm_state['pred_label'] == target_label).mean()
                fpr_perm_nonstate = (perm_nonstate['pred_label'] == target_label).mean()
                fpr_perm_diffs.append(fpr_perm_state - fpr_perm_nonstate)
            
            fpr_pvalue = np.mean(np.abs(fpr_perm_diffs) >= np.abs(fpr_diff))
        else:
            fpr_pvalue = None
        
        fairness_results.append({
            'model': model,
            'target_label': target_label,
            'n_state': len(state_df),
            'n_nonstate': len(nonstate_df),
            'SPD': spd,
            'SPD_CI_lower': spd_ci_lower,
            'SPD_CI_upper': spd_ci_upper,
            'TPR_state': tpr_state,
            'TPR_nonstate': tpr_nonstate,
            'TPR_diff': tpr_diff,
            'TPR_pvalue': tpr_pvalue,
            'FPR_state': fpr_state,
            'FPR_nonstate': fpr_nonstate,
            'FPR_diff': fpr_diff,
            'FPR_pvalue': fpr_pvalue
        })
    
    return pd.DataFrame(fairness_results)

def analyze_error_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze how errors correlate with text features like notes length.
    
    Examines:
    - Error rate vs ACLED notes length
    - Error rate vs presence of specific linguistic features
    """
    correlation_results = []
    
    # Check if notes column exists
    if 'notes' not in df.columns:
        print("Warning: 'notes' column not found. Cannot compute source analysis.")
        return pd.DataFrame()
    
    # Add text length feature
    df['notes_length'] = df['notes'].fillna('').str.len()
    
    # Define length bins
    df['length_bin'] = pd.cut(
        df['notes_length'], 
        bins=[0, 100, 200, 300, 500, 10000],
        labels=['very_short', 'short', 'medium', 'long', 'very_long']
    )
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model].copy()
        model_df = model_df[model_df['true_label'].isin(labels) & model_df['pred_label'].isin(labels)]
        
        if len(model_df) == 0:
            continue
        
        # Overall error rate
        model_df['is_error'] = (model_df['true_label'] != model_df['pred_label']).astype(int)
        overall_error_rate = model_df['is_error'].mean()
        
        # Error rate by length bin
        for length_bin in ['very_short', 'short', 'medium', 'long', 'very_long']:
            bin_df = model_df[model_df['length_bin'] == length_bin]
            if len(bin_df) > 0:
                error_rate = bin_df['is_error'].mean()
                n_samples = len(bin_df)
                
                correlation_results.append({
                    'model': model,
                    'feature': 'notes_length',
                    'feature_value': length_bin,
                    'n_samples': n_samples,
                    'error_rate': error_rate,
                    'error_rate_diff_from_overall': error_rate - overall_error_rate
                })
        
        # Correlation with numerical length
        if len(model_df) > 10:
            from scipy.stats import spearmanr
            try:
                corr, pval = spearmanr(model_df['notes_length'], model_df['is_error'])
                correlation_results.append({
                    'model': model,
                    'feature': 'notes_length_continuous',
                    'feature_value': 'correlation',
                    'n_samples': len(model_df),
                    'error_rate': corr,  # Actually correlation coefficient
                    'error_rate_diff_from_overall': pval  # Actually p-value
                })
            except:
                pass
    
    return pd.DataFrame(correlation_results)

def main():
    if not os.path.exists(RESULTS_CSV):
        print('Results CSV not found:', RESULTS_CSV)
        return
    df = pd.read_csv(RESULTS_CSV)
    
    # Compute standard metrics
    metrics, cms = compute_metrics(df)
    mdf = pd.DataFrame(metrics)
    mdf.to_csv(OUT_METRICS, index=False)
    with open(OUT_CMS, 'w') as f:
        json.dump({'labels': labels, 'cms': cms}, f, indent=2)
    print('Wrote metrics to', OUT_METRICS)
    print('Wrote confusion matrices to', OUT_CMS)
    print(mdf.to_string(index=False))
    
    # Compute fairness metrics
    fairness_df = compute_fairness_metrics(df, target_label='V', n_bootstrap=1000)
    if not fairness_df.empty:
        fairness_df.to_csv(OUT_FAIRNESS, index=False)
        print(f'\nWrote fairness metrics to {OUT_FAIRNESS}')
        print(fairness_df.to_string(index=False))
    else:
        print('\nCould not compute fairness metrics (missing actor information)')
    
    # Analyze error correlations with text features
    correlation_df = analyze_error_correlations(df)
    if not correlation_df.empty:
        out_corr_path = os.path.join(RESULTS_DIR, f'error_correlations_acled_{COUNTRY}_state_actors.csv')
        correlation_df.to_csv(out_corr_path, index=False)
        print(f'\nWrote error correlation analysis to {out_corr_path}')
        print('\n=== Error Rate by Notes Length ===')
        length_corr = correlation_df[correlation_df['feature'] == 'notes_length']
        if not length_corr.empty:
            print(length_corr[['model', 'feature_value', 'n_samples', 'error_rate']].to_string(index=False))

if __name__ == '__main__':
    main()
