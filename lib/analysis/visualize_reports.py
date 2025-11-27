#!/usr/bin/env python3
"""Visualize per-class metrics and top disagreements.

Outputs:
- results/{country}/{strategy}/per_class_metrics.png
- results/{country}/{strategy}/top_disagreements_table.png
"""
from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
from lib.core.data_helpers import setup_country_environment

COUNTRY, OUT_DIR = setup_country_environment()
os.makedirs(OUT_DIR, exist_ok=True)
PER_CLASS_CSV = os.path.join(OUT_DIR, "per_class_report.csv")
TOP_CSV = os.path.join(OUT_DIR, "top_disagreements.csv")

def plot_per_class():
    df = pd.read_csv(PER_CLASS_CSV)
    # pivot to have models as columns for f1
    pivot = df.pivot(index="label", columns="model", values="f1")
    ax = pivot.plot(kind="bar", rot=0, figsize=(10, 6))
    ax.set_ylabel("F1 score")
    ax.set_title(f"Per-class F1 by model ({COUNTRY})")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "per_class_metrics.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Wrote {out}")

def render_top_table():
    df = pd.read_csv(TOP_CSV)
    
    # Handle empty dataframe
    if df.empty:
        print(f"No disagreements to visualize (only one model or perfect agreement)")
        return
    
    # Select columns to display (keep event_id, true_label, actor_norm and preds/probs)
    cols = [c for c in df.columns if c in ("event_id", "true_label", "actor_norm") or c.startswith("pred_label_") or c.startswith("pred_prob_")]
    tab = df[cols].copy()
    # Shorten column names for readability
    tab.columns = [c.replace("pred_label_", "lbl:") .replace("pred_prob_", "pr:") for c in tab.columns]

    fig, ax = plt.subplots(figsize=(12, max(2, 0.3 * len(tab))))
    ax.axis("off")
    table = ax.table(cellText=tab.values, colLabels=tab.columns, cellLoc="left", loc="center") # type: ignore
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    out = os.path.join(OUT_DIR, "top_disagreements_table.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Wrote {out}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(PER_CLASS_CSV) or not os.path.exists(TOP_CSV):
        raise SystemExit("Missing required CSVs in results/. Run the reports generator first.")
    plot_per_class()
    render_top_table()

if __name__ == "__main__":
    main()
