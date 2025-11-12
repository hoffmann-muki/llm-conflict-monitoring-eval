import pandas as pd
import numpy as np
import json
import os
from data_helpers import setup_country_environment

COUNTRY, RESULTS_DIR = setup_country_environment()

RESULTS_CSV = os.path.join(RESULTS_DIR, f'ollama_results_acled_{COUNTRY}_state_actors.csv')
OUT_METRICS = os.path.join(RESULTS_DIR, f'metrics_acled_{COUNTRY}_state_actors.csv')
OUT_CMS = os.path.join(RESULTS_DIR, f'confusion_matrices_acled_{COUNTRY}_state_actors.json')

labels = ['V','B','E','P','R','S']

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

def main():
    if not os.path.exists(RESULTS_CSV):
        print('Results CSV not found:', RESULTS_CSV)
        return
    df = pd.read_csv(RESULTS_CSV)
    metrics, cms = compute_metrics(df)
    # Save metrics
    mdf = pd.DataFrame(metrics)
    mdf.to_csv(OUT_METRICS, index=False)
    with open(OUT_CMS, 'w') as f:
        json.dump({'labels': labels, 'cms': cms}, f, indent=2)
    print('Wrote metrics to', OUT_METRICS)
    print('Wrote confusion matrices to', OUT_CMS)
    print(mdf.to_string(index=False))

if __name__ == '__main__':
    main()
