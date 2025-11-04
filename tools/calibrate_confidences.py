import json
import os
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, brier_score_loss
from scipy.optimize import minimize

RESULTS_CSV = 'results/ollama_results_acled_cameroon_state_actors.csv'
OUT_PARAMS = 'results/calibration_params_acled_cameroon_state_actors.json'

labels = ['V','B','E','P','R','S']


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
