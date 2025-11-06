import json
import os
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss
from scipy.optimize import minimize

RESULTS_CSV = 'results/ollama_results_acled_cameroon_state_actors.csv'
OUT_PARAMS = 'results/calibration_params_acled_cameroon_state_actors.json'

labels = ['V','B','E','P','R','S']

# map labels to indices for logits ordering when available
LABEL_TO_IDX = {lab: i for i, lab in enumerate(labels)}

def temp_scale_logits(logits, T):
    return logits / T

def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum(axis=-1, keepdims=True)

def compute_brier(y_true_bin, prob_pos):
    return brier_score_loss(y_true_bin, prob_pos)

def fit_temperature(probs, y_true_bin):
    # probs: raw predicted probability for the positive class
    # transform via logit, scale temperature T>0, then sigmoid back
    eps=1e-12
    probs = np.clip(probs, eps, 1-eps)
    logits = np.log(probs/(1-probs))

    def loss_fn(log_T):
        T = np.exp(log_T)
        scaled = 1/(1+np.exp(-logits / T))
        return log_loss(y_true_bin, scaled, labels=[0,1])

    res = minimize(loss_fn, x0=0.0)
    T = float(np.exp(res.x[0])) if res.success else 1.0
    return T

def fit_multiclass_temperature(logits_array, y_true_idx):
    """Fit a single temperature T>0 to scale logits (divide by T) minimizing multiclass log-loss.
    logits_array: shape (n_examples, n_classes)
    y_true_idx: integer class indices shape (n_examples,)
    """
    eps = 1e-12

    if logits_array is None or len(logits_array) == 0:
        return 1.0

    logits_array = np.asarray(logits_array, dtype=float)
    def loss_fn(log_T):
        T = float(np.exp(log_T))
        scaled = logits_array / T
        # stable softmax
        z = scaled - np.max(scaled, axis=1, keepdims=True)
        expz = np.exp(z)
        probs = expz / (expz.sum(axis=1, keepdims=True) + eps)
        # compute negative log-likelihood
        p_true = probs[np.arange(len(y_true_idx)), y_true_idx]
        p_true = np.clip(p_true, eps, 1.0)
        return -np.mean(np.log(p_true))

    res = minimize(loss_fn, x0=0.0)
    T = float(np.exp(res.x[0])) if res.success else 1.0
    return T

def main():
    if not os.path.exists(RESULTS_CSV):
        print('Results CSV not found:', RESULTS_CSV)
        return
    df = pd.read_csv(RESULTS_CSV)
    models = df['model'].unique()
    params = {}

    for m in models:
        sub = df[df['model']==m].copy()
        # Only use rows with valid labels
        sub = sub[sub['true_label'].isin(labels) & sub['pred_label'].isin(labels)].reset_index(drop=True)
        if len(sub)==0:
            continue
        # we'll do one-vs-rest calibration per predicted label probability
        # but we only have a single pred_conf (confidence for predicted label)
        # so we treat pred_conf as the model's probability for its predicted class.
        # For calibration we create binary tasks: is prediction correct?
        y_true = (sub['pred_label'] == sub['true_label']).astype(int)
        probs = sub['pred_conf'].astype(float).values

        # train/val split
        n = len(probs)
        idx = np.arange(n)
        np.random.seed(0)
        np.random.shuffle(idx)
        split = int(n*0.7)
        train_idx = idx[:split]
        val_idx = idx[split:]

        ir = IsotonicRegression(out_of_bounds='clip')
        try:
            ir.fit(probs[train_idx], y_true.values[train_idx])
        except Exception:
            ir = None
        val_probs = probs[val_idx]
        val_true = y_true.values[val_idx]

        # Fit temperature scaling on validation: treat probs as probabilities for positive class
        try:
            T = fit_temperature(probs[train_idx], y_true.values[train_idx])
        except Exception:
            T = 1.0

        # Try to fit multiclass temperature scaling if logits are available
        multiclass_T = 1.0
        logits_col = None
        if 'logits' in sub.columns:
            logits_col = sub['logits'].values
        elif 'log_probs' in sub.columns:
            logits_col = sub['log_probs'].values
        if logits_col is not None:
            # parse JSON lists to numpy array where possible
            parsed = []
            y_idx = []
            for i, val in enumerate(logits_col):
                try:
                    arr = np.array(json.loads(val)) if pd.notna(val) else None
                except Exception:
                    arr = None
                if arr is None:
                    continue
                # require correct shape
                if arr.shape[-1] != len(labels):
                    continue
                parsed.append(arr)
                # use true label index
                true_lab = sub.iloc[i]['true_label']
                if true_lab in LABEL_TO_IDX:
                    y_idx.append(LABEL_TO_IDX[true_lab])
                else:
                    y_idx.append(-1)
            if len(parsed) > 0 and all([yi >= 0 for yi in y_idx]):
                parsed = np.vstack(parsed)
                y_idx = np.array(y_idx, dtype=int)
                try:
                    multiclass_T = fit_multiclass_temperature(parsed[train_idx], y_idx[train_idx])
                except Exception:
                    multiclass_T = 1.0

        # Evaluate before/after
        def apply_temp(p):
            p = np.clip(p, 1e-12, 1-1e-12)
            logit = np.log(p/(1-p))
            scaled = 1/(1+np.exp(-logit / T))
            return scaled

        if ir is not None:
            cal_val = ir.transform(val_probs)
        else:
            cal_val = val_probs
        temp_val = apply_temp(val_probs)

        ll_before = log_loss(val_true, val_probs, labels=[0,1]) if len(val_true)>0 else None
        ll_isotonic = log_loss(val_true, cal_val, labels=[0,1]) if len(val_true)>0 else None
        ll_temp = log_loss(val_true, temp_val, labels=[0,1]) if len(val_true)>0 else None

        b_before = brier_score_loss(val_true, val_probs) if len(val_true)>0 else None
        b_iso = brier_score_loss(val_true, cal_val) if len(val_true)>0 else None
        b_temp = brier_score_loss(val_true, temp_val) if len(val_true)>0 else None

        params[m] = {
            'isotonic_trained': ir is not None,
            'T': float(T),
            'T_multiclass': float(multiclass_T),
            'val_logloss_before': None if ll_before is None else float(ll_before),
            'val_logloss_isotonic': None if ll_isotonic is None else float(ll_isotonic),
            'val_logloss_temp': None if ll_temp is None else float(ll_temp),
            'val_brier_before': None if b_before is None else float(b_before),
            'val_brier_isotonic': None if b_iso is None else float(b_iso),
            'val_brier_temp': None if b_temp is None else float(b_temp),
            'n_train': int(len(train_idx)),
            'n_val': int(len(val_idx))
        }

        print('Model', m)
        print('  T=', params[m]['T'])
        print('  LogLoss before/iso/temp =', params[m]['val_logloss_before'], params[m]['val_logloss_isotonic'], params[m]['val_logloss_temp'])
        print('  Brier before/iso/temp =', params[m]['val_brier_before'], params[m]['val_brier_isotonic'], params[m]['val_brier_temp'])

    with open(OUT_PARAMS, 'w') as f:
        json.dump(params, f, indent=2)
    print('Wrote calibration params to', OUT_PARAMS)

if __name__ == '__main__':
    main()
