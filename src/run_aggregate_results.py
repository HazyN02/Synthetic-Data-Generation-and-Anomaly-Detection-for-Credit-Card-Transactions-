#!/usr/bin/env python3
"""
Aggregate all experiment results into a unified summary.
Run from project root: python -m src.run_aggregate_results

Reads from:
  - results/protocol/results.csv   (baseline, CTGAN, TabDDPM)
  - results/smote/results.csv      (baseline, SMOTE)
  - results/sliding_window/results.csv (static vs sliding)
  - Legacy: results/results.csv, results/smote_baseline_results.csv

Writes: results/SUMMARY.md and results/aggregate_summary.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

RESULTS_DIR = _ROOT / "results"
SUMMARY_MD = RESULTS_DIR / "SUMMARY.md"
AGGREGATE_CSV = RESULTS_DIR / "aggregate_summary.csv"


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def _safe_agg(df: pd.DataFrame, group_cols: list, metric_cols: list) -> pd.DataFrame:
    if df.empty or not group_cols:
        return pd.DataFrame()
    agg_dict = {c: ["mean", "std", "count"] for c in metric_cols if c in df.columns}
    if not agg_dict:
        return pd.DataFrame()
    return df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 1. Protocol (baseline, CTGAN, TabDDPM) ---
    protocol_paths = [
        RESULTS_DIR / "protocol" / "results.csv",
        RESULTS_DIR / "results.csv",  # legacy
    ]
    protocol_df = None
    for p in protocol_paths:
        protocol_df = _load_csv(p)
        if protocol_df is not None:
            break

    # --- 2. SMOTE ---
    smote_paths = [
        RESULTS_DIR / "smote" / "results.csv",
        RESULTS_DIR / "smote_baseline_results.csv",  # legacy
    ]
    smote_df = None
    for p in smote_paths:
        smote_df = _load_csv(p)
        if smote_df is not None:
            break

    # --- 3. Sliding window ---
    sliding_paths = [
        RESULTS_DIR / "sliding_window" / "results.csv",
        RESULTS_DIR / "sliding_window_results.csv",  # legacy
    ]
    sliding_df = None
    for p in sliding_paths:
        sliding_df = _load_csv(p)
        if sliding_df is not None:
            break

    lines = []
    lines.append("# Fraud Detection Results — Summary")
    lines.append("")
    lines.append("This document summarizes all model experiments for the fraud-synth project.")
    lines.append("")

    # --- Protocol summary ---
    if protocol_df is not None and not protocol_df.empty:
        pr_col = "pr_auc" if "pr_auc" in protocol_df.columns else None
        rec_col = "recall_at_1pct_fpr" if "recall_at_1pct_fpr" in protocol_df.columns else None
        metric_cols = [c for c in [pr_col, rec_col] if c]

        if metric_cols and "method" in protocol_df.columns:
            grp = protocol_df.groupby(["method", protocol_df["target_pos_rate"].fillna("")], dropna=False)
            summ = grp[metric_cols].agg(["mean", "std"]).round(4)
            lines.append("## 1. Main Protocol (Baseline vs CTGAN vs TabDDPM)")
            lines.append("")
            lines.append("Oversampling methods: real data + synthetic fraud from generative models.")
            lines.append("")
            lines.append("| Method | Target Pos Rate | PR-AUC (mean ± std) | Recall@1%FPR (mean ± std) |")
            lines.append("|--------|-----------------|----------------------|----------------------------|")
            for (method, rate), row in summ.iterrows():
                rate_str = f"{rate:.0%}" if isinstance(rate, (int, float)) and rate else "—"
                prauc = row.get(("pr_auc", "mean"), row.get("pr_auc", np.nan))
                prauc_std = row.get(("pr_auc", "std"), np.nan)
                rec = row.get(("recall_at_1pct_fpr", "mean"), row.get("recall_at_1pct_fpr", np.nan))
                rec_std = row.get(("recall_at_1pct_fpr", "std"), np.nan)
                if np.isnan(prauc):
                    prauc, prauc_std = row.get(("pr_auc", "mean"), np.nan), row.get(("pr_auc", "std"), np.nan)
                lines.append(f"| {method} | {rate_str} | {prauc:.4f} ± {prauc_std:.4f} | {rec:.4f} ± {rec_std:.4f} |")
            lines.append("")

    # --- SMOTE summary ---
    if smote_df is not None and not smote_df.empty:
        pr_col = "pr_auc" if "pr_auc" in smote_df.columns else None
        rec_col = "recall_at_1pct_fpr" if "recall_at_1pct_fpr" in smote_df.columns else None
        metric_cols = [c for c in [pr_col, rec_col] if c]

        if metric_cols and "method" in smote_df.columns:
            grp = smote_df.groupby(["method", smote_df["target_pos_rate"].fillna("")], dropna=False)
            summ = grp[metric_cols].agg(["mean", "std"]).round(4)
            lines.append("## 2. SMOTE Baseline")
            lines.append("")
            lines.append("Non-generative oversampling: SMOTE (k-NN interpolation of minority class).")
            lines.append("")
            lines.append("| Method | Target Pos Rate | PR-AUC (mean ± std) | Recall@1%FPR (mean ± std) |")
            lines.append("|--------|-----------------|----------------------|----------------------------|")
            for (method, rate), row in summ.iterrows():
                rate_str = f"{rate:.0%}" if isinstance(rate, (int, float)) and rate else "—"
                prauc = row.get(("pr_auc", "mean"), np.nan)
                prauc_std = row.get(("pr_auc", "std"), np.nan)
                rec = row.get(("recall_at_1pct_fpr", "mean"), np.nan)
                rec_std = row.get(("recall_at_1pct_fpr", "std"), np.nan)
                lines.append(f"| {method} | {rate_str} | {prauc:.4f} ± {prauc_std:.4f} | {rec:.4f} ± {rec_std:.4f} |")
            lines.append("")

    # --- Sliding window ---
    if sliding_df is not None and not sliding_df.empty and "strategy" in sliding_df.columns:
        grp = sliding_df.groupby("strategy")
        summ = grp[["pr_auc", "recall_at_1pct_fpr"]].agg(["mean", "std"]).round(4)
        lines.append("## 3. Sliding Window vs Static Retraining")
        lines.append("")
        lines.append("- **Static**: train on all past data for each fold.")
        lines.append("- **Sliding**: train on a fixed recent window.")
        lines.append("")
        lines.append("| Strategy | PR-AUC (mean ± std) | Recall@1%FPR (mean ± std) |")
        lines.append("|----------|----------------------|----------------------------|")
        for strategy, row in summ.iterrows():
            prauc = row[("pr_auc", "mean")]
            prauc_std = row[("pr_auc", "std")]
            rec = row[("recall_at_1pct_fpr", "mean")]
            rec_std = row[("recall_at_1pct_fpr", "std")]
            lines.append(f"| {strategy} | {prauc:.4f} ± {prauc_std:.4f} | {rec:.4f} ± {rec_std:.4f} |")
        lines.append("")

    # --- How This Serves the Project ---
    lines.append("## How These Results Serve the Project")
    lines.append("")
    lines.append("| Component | Purpose |")
    lines.append("|-----------|---------|")
    lines.append("| **Baseline** | Honest baseline (real data only) — reviewers expect this. |")
    lines.append("| **SMOTE** | Standard non-generative oversampling baseline — must beat this. |")
    lines.append("| **CTGAN** | GAN-based synthetic fraud — compares deep generative vs interpolation. |")
    lines.append("| **TabDDPM** | Diffusion-based synthetic fraud — state-of-the-art tabular generation. |")
    lines.append("| **Sliding vs Static** | Temporal robustness: does retraining on recent data help? |")
    lines.append("")
    lines.append("**Metrics** (fraud detection standard):")
    lines.append("- **PR-AUC**: Precision-Recall AUC (handles imbalance well)")
    lines.append("- **Recall@1%FPR**: Recall when false positive rate = 1%")
    lines.append("")
    lines.append("**Project goal**: Establish that generative oversampling (CTGAN, TabDDPM) improves fraud detection vs baseline and SMOTE, under temporal evaluation.")
    lines.append("")

    # Write summary
    content = "\n".join(lines)
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote {SUMMARY_MD}")

    # Build aggregate CSV for programmatic use
    rows = []
    for name, df in [("protocol", protocol_df), ("smote", smote_df), ("sliding_window", sliding_df)]:
        if df is None or df.empty:
            continue
        df = df.copy()
        df["experiment"] = name
        rows.append(df)

    if rows:
        agg = pd.concat(rows, ignore_index=True)
        # Harmonize column names
        if "recall@1%fpr" in agg.columns and "recall_at_1pct_fpr" not in agg.columns:
            agg["recall_at_1pct_fpr"] = agg["recall@1%fpr"]
        agg.to_csv(AGGREGATE_CSV, index=False)
        print(f"Wrote {AGGREGATE_CSV}")

    print("\nDone. Open results/SUMMARY.md for the human-readable summary.")


if __name__ == "__main__":
    main()
