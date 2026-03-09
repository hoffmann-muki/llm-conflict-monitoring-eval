#!/usr/bin/env python3
"""Generate reviewer-ready comparison tables for fine-tuned baselines.

This script aggregates prediction CSVs from:
  results/baselines/{split_version}/conflibert/
  results/baselines/{split_version}/small_llm/

and produces publication-facing summary tables with:
- accuracy
- macro F1
- per-class F1 (V/B/E/P/R/S)
- fairness metrics (SPD, TPR/FPR gaps for label V)
- harm metrics (FLR/FIR)

If counterfactual/error-trace artifacts are present in model directories,
optional columns for CFR and RFC discordance are also populated.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from lib.analysis.harm import compute_harm_rates
from lib.analysis.metrics import compute_fairness_metrics


LABELS = ["V", "B", "E", "P", "R", "S"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fine-tuned baseline models")
    parser.add_argument("--split-version", required=True, help="Split version (e.g., acled_v1)")
    parser.add_argument("--results-root", default="results/baselines", help="Baselines results root")
    parser.add_argument("--out-dir", default="", help="Optional output directory override")
    return parser.parse_args()


def _collect_prediction_files(base_dir: Path) -> List[Path]:
    patterns = [
        str(base_dir / "conflibert" / "*.csv"),
        str(base_dir / "small_llm" / "*.csv"),
    ]
    files: List[Path] = []
    for pat in patterns:
        files.extend(Path(p) for p in glob.glob(pat))
    return sorted([p for p in files if p.is_file()])


def _infer_country_from_name(path: Path) -> str:
    name = path.name.lower()
    if "cmr" in name or "cameroon" in name:
        return "cmr"
    if "nga" in name or "nigeria" in name:
        return "nga"
    return "unknown"


def _load_and_tag(files: List[Path]) -> pd.DataFrame:
    frames = []
    for f in files:
        df = pd.read_csv(f)
        if df.empty:
            continue
        if "model" not in df.columns:
            continue
        df = df.copy()
        df["source_file"] = str(f)
        df["country"] = _infer_country_from_name(f)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    return merged


def _per_class_f1(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for lab in LABELS:
        # Binary one-vs-rest F1 for class-specific interpretability
        yt = (y_true == lab).astype(int)
        yp = (y_pred == lab).astype(int)
        out[f"f1_{lab}"] = float(f1_score(yt, yp, zero_division=0))
    return out


def _extract_optional_cfr_rfc(source_file: str) -> Tuple[float | None, float | None]:
    """Try to read optional CFR and RFC discordance metrics near prediction file.

    Expected optional artifacts:
      - counterfactual_analysis_*.json
      - error_trace_report.json

    Returns:
      (cfr_mean, rfc_low_ambiguity_discordance)
    """
    parent = Path(source_file).parent

    cfr_mean = None
    cf_candidates = sorted(parent.glob("counterfactual_analysis_*.json"))
    if cf_candidates:
        try:
            with open(cf_candidates[0]) as f:
                cf = json.load(f)
            fm = cf.get("counterfactual_flip_rate_CFR", {})
            rates = []
            for _, model_data in fm.items():
                for _, metrics in model_data.items():
                    if isinstance(metrics, dict):
                        rate = metrics.get("counterfactual_flip_rate_CFR")
                        if rate is not None:
                            rates.append(float(rate))
            if rates:
                cfr_mean = float(np.mean(rates))
        except Exception:
            cfr_mean = None

    rfc_low_discord = None
    et_path = parent / "error_trace_report.json"
    if et_path.exists():
        try:
            with open(et_path) as f:
                et = json.load(f)
            by_model = et.get("ollama_rationale_analysis", {}).get("by_model", {})
            low_rates = []
            for _, info in by_model.items():
                low = info.get("aggregate", {}).get("by_ambiguity_tier", {}).get("Low", {})
                d = low.get("discordant_rate")
                if d is not None:
                    low_rates.append(float(d))
            if low_rates:
                rfc_low_discord = float(np.mean(low_rates))
        except Exception:
            rfc_low_discord = None

    return cfr_mean, rfc_low_discord


def _compute_core_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, country), sub in df.groupby(["model", "country"], dropna=False):
        y_true = sub["true_label"].fillna("")
        y_pred = sub["pred_label"].fillna("")
        mask = y_true.isin(LABELS) & y_pred.isin(LABELS)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            continue

        macro_f1 = float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))
        acc = float(accuracy_score(y_true, y_pred))
        cls = _per_class_f1(y_true, y_pred)

        sample_file = sub["source_file"].iloc[0]
        cfr_mean, rfc_low_discord = _extract_optional_cfr_rfc(sample_file)

        rows.append({
            "model": model,
            "country": country,
            "n": int(len(y_true)),
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            **{k: round(v, 4) for k, v in cls.items()},
            "cfr_mean_optional": None if cfr_mean is None else round(cfr_mean, 4),
            "rfc_low_discord_optional": None if rfc_low_discord is None else round(rfc_low_discord, 4),
        })

    return pd.DataFrame(rows)


def _compute_fairness_table(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for country, sub in df.groupby("country", dropna=False):
        fair = compute_fairness_metrics(sub.copy(), target_label="V", n_bootstrap=300)
        if fair.empty:
            continue
        fair = fair.copy()
        fair["country"] = country
        parts.append(fair)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    keep = [
        "model", "country", "SPD", "SPD_CI_lower", "SPD_CI_upper",
        "TPR_diff", "TPR_pvalue", "FPR_diff", "FPR_pvalue",
        "n_state", "n_nonstate",
    ]
    return out[[c for c in keep if c in out.columns]]


def _compute_harm_table(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for country, sub in df.groupby("country", dropna=False):
        harm = compute_harm_rates(sub.copy())
        if harm.empty:
            continue
        harm = harm.copy()
        harm["country"] = country
        parts.append(harm)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    keep = [
        "model", "country",
        "false_legitimization_rate_FLR", "false_illegitimization_rate_FIR",
        "false_legitimization_count_FL", "false_illegitimization_count_FI",
        "harm_ratio_FL_to_FI", "n_illegit_events", "n_legit_events",
    ]
    return out[[c for c in keep if c in out.columns]]


def _merge_publication_table(core: pd.DataFrame, fairness: pd.DataFrame, harm: pd.DataFrame) -> pd.DataFrame:
    out = core.copy()
    if not fairness.empty:
        out = out.merge(
            fairness,
            on=["model", "country"],
            how="left",
            suffixes=("", "_fair"),
        )
    if not harm.empty:
        out = out.merge(
            harm,
            on=["model", "country"],
            how="left",
            suffixes=("", "_harm"),
        )
    return out


def main() -> None:
    args = parse_args()

    base_dir = Path(args.results_root) / args.split_version
    if not base_dir.exists():
        raise SystemExit(f"Baselines directory not found: {base_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else base_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _collect_prediction_files(base_dir)
    if not files:
        raise SystemExit(f"No baseline prediction CSVs found under {base_dir}")

    df = _load_and_tag(files)
    if df.empty:
        raise SystemExit("Prediction files were found but no usable rows/columns were loaded")

    core = _compute_core_table(df)
    fairness = _compute_fairness_table(df)
    harm = _compute_harm_table(df)
    publication = _merge_publication_table(core, fairness, harm)

    core_path = out_dir / "baseline_core_metrics.csv"
    fair_path = out_dir / "baseline_fairness_metrics.csv"
    harm_path = out_dir / "baseline_harm_metrics.csv"
    pub_path = out_dir / "baseline_publication_table.csv"

    core.to_csv(core_path, index=False)
    if not fairness.empty:
        fairness.to_csv(fair_path, index=False)
    if not harm.empty:
        harm.to_csv(harm_path, index=False)
    publication.to_csv(pub_path, index=False)

    print("=" * 80)
    print("FINE-TUNED BASELINE COMPARISON COMPLETE")
    print("=" * 80)
    print(f"Loaded prediction files: {len(files)}")
    print(f"Core metrics table:      {core_path}")
    if not fairness.empty:
        print(f"Fairness table:          {fair_path}")
    if not harm.empty:
        print(f"Harm table:              {harm_path}")
    print(f"Publication table:       {pub_path}")


if __name__ == "__main__":
    main()
