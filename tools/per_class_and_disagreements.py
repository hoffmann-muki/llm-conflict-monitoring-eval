#!/usr/bin/env python3
"""Compute per-class metrics per model and extract top-N disagreement examples.

Reads: results/ollama_results_calibrated.csv
Writes: results/per_class_report.csv, results/top_disagreements.csv
"""
from __future__ import annotations
import os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

INPUT_CSV = os.path.join("results", "ollama_results_calibrated.csv")
OUT_PER_CLASS = os.path.join("results", "per_class_report.csv")
OUT_TOP = os.path.join("results", "top_disagreements.csv")
TOP_N = 20

def main():
    os.makedirs("results", exist_ok=True)
    df = pd.read_csv(INPUT_CSV)

    # Normalize column names we expect
    required = {"model", "event_id", "true_label", "pred_label", "pred_conf_temp", "actor_norm"}
    if not required.issubset(set(df.columns)):
        raise SystemExit(f"Missing expected columns in {INPUT_CSV}. Found: {list(df.columns)}")

    # Per-class metrics per model
    labels = sorted(df["true_label"].dropna().unique())
    rows = []
    for model in sorted(df["model"].unique()):
        mdf = df[df["model"] == model]
        y_true = mdf["true_label"].astype(str)
        y_pred = mdf["pred_label"].astype(str)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        for lab, p, r, f, s in zip(labels, precision, recall, f1, support): # type: ignore
            rows.append({"model": model, "label": lab, "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)})

    per_class_df = pd.DataFrame(rows)
    per_class_df.to_csv(OUT_PER_CLASS, index=False)

    # Build wide table of predictions/probabilities per event
    # Use calibrated temperature-scaled probability column `pred_conf_temp`
    piv_label = df.pivot(index="event_id", columns="model", values="pred_label")
    piv_prob = df.pivot(index="event_id", columns="model", values="pred_conf_temp")

    # Retrieve canonical event-level columns
    event_meta = df.drop_duplicates(subset=["event_id"]).set_index("event_id")[ ["true_label", "actor_norm"] ]

    # Align
    wide = event_meta.join(piv_label.add_prefix("pred_label_")).join(piv_prob.add_prefix("pred_prob_"))
    # Drop rows with any missing model predictions
    wide = wide.dropna()

    # Determine disagreement rows (more than 1 unique predicted label across models)
    def disagreement_count(row):
        labels = [v for k, v in row.items() if k.startswith("pred_label_")]
        return len(set(labels))

    wide["disagreement_count"] = wide.apply(disagreement_count, axis=1)
    wide["max_confidence"] = wide[[c for c in wide.columns if c.startswith("pred_prob_")]].max(axis=1)

    disagreements = wide[wide["disagreement_count"] > 1].copy()
    disagreements = disagreements.sort_values(by=["max_confidence"], ascending=False)

    topn = disagreements.head(TOP_N)
    topn.reset_index(inplace=True)
    topn.to_csv(OUT_TOP, index=False)

    print(f"Wrote per-class report: {OUT_PER_CLASS} ({len(per_class_df)} rows)")
    print(f"Wrote top-{TOP_N} disagreements: {OUT_TOP} ({len(topn)} rows)")

if __name__ == "__main__":
    main()
