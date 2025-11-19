import numpy as np
import pandas as pd

LEGIT = set(['B','S'])
ILLEG = set(['V'])

def fl_fi_for_df(df: pd.DataFrame) -> dict:
    """Compute counts total, fl, fi for the provided dataframe slice.

    Returns dict(total=int, fl=int, fi=int).
    """
    total = len(df)
    fl = int(((df['true_label'].isin(ILLEG)) & (df['pred_label'].isin(LEGIT))).sum())
    fi = int(((df['true_label'].isin(LEGIT)) & (df['pred_label'].isin(ILLEG))).sum())
    return {'total': total, 'fl': fl, 'fi': fi}

def aggregate_fl_fi(df: pd.DataFrame, by: str = 'model') -> pd.DataFrame:
    rows = []
    for key in df[by].unique():
        sub = df[df[by] == key]
        stats = fl_fi_for_df(sub)
        stats[by] = key
        rows.append(stats)
    return pd.DataFrame(rows)
