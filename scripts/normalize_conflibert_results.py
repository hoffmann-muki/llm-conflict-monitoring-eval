#!/usr/bin/env python3
"""Normalize legacy ConflibERT model labels in CSV result files.

This script searches the `results/` tree for CSV files and:
- makes a safe backup (`<file>.bak`) for any file it will modify
- normalizes any cell in the `model` column that starts with `conflibert` (case-insensitive)
  to the canonical string `conflibert`
- renames any CSV column headers that include the legacy token `conflibert_conflibert`
  (e.g., `pred_label_conflibert_conflibert`) to use `conflibert` instead

Usage:
  python scripts/normalize_conflibert_results.py [--root results/]

This is destructive to the CSVs (it overwrites), but creates `.bak` copies first.
"""

import argparse
import shutil
from pathlib import Path
import pandas as pd


def normalize_csv(path: Path) -> bool:
    """Return True if file was modified."""
    df = pd.read_csv(path)
    modified = False

    # Normalize model column values
    if 'model' in df.columns:
        model_col = df['model'].astype(str)
        mask = model_col.str.lower().str.startswith('conflibert')
        if mask.any():
            df.loc[mask, 'model'] = 'conflibert'
            modified = True

    # Rename any header columns containing the legacy token
    new_cols = []
    renamed = False
    for c in df.columns:
        if 'conflibert_conflibert' in str(c):
            new_c = str(c).replace('conflibert_conflibert', 'conflibert')
            new_cols.append(new_c)
            renamed = True
        else:
            new_cols.append(c)

    if renamed:
        df.columns = new_cols
        modified = True

    if modified:
        bak = path.with_suffix(path.suffix + '.bak')
        shutil.copy2(path, bak)
        df.to_csv(path, index=False)
        print(f"Updated: {path} (backup: {bak})")
    return modified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='results', help='Root results directory to scan')
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Root not found: {root}")
        return

    csv_files = list(root.rglob('*.csv'))
    print(f"Found {len(csv_files)} CSV files under {root}")

    changed = 0
    for p in csv_files:
        try:
            if normalize_csv(p):
                changed += 1
        except Exception as e:
            print(f"Error processing {p}: {e}")

    print(f"Normalization complete. Files changed: {changed}")


if __name__ == '__main__':
    main()
