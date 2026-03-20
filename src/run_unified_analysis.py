#!/usr/bin/env python3
"""
Unified analysis: merge all results, drift-harm analysis, when-it-helps table.
Run from project root: python -m src.run_unified_analysis

Expects (creates if missing):
- results/results.csv (run_protocol: baseline, ctgan, tabddpm)
- results/smote_baseline_results.csv
- results/sliding_window_results.csv
- experiments/results/drift_report.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

RESULTS_DIR = os.path.join(_ROOT, "results")
PAPER_DIR = os.path.join(_ROOT, "paper")

try:
    from src.statistical_tests import run_comparisons as _run_stat_comparisons
except ImportError:
    _run_stat_comparisons = None
TABLES_DIR = os.path.join(PAPER_DIR, "tables")
FIGURES_DIR = os.path.join(PAPER_DIR, "figures")
DRIFT_PATH = os.path.join(_ROOT, "experiments", "results", "drift_report.csv")


def _load_dedup(df: pd.DataFrame, key_cols: list) -> pd.DataFrame:
    """Keep last row per key (handles reruns)."""
    if df.empty or not key_cols:
        return df
    valid = [c for c in key_cols if c in df.columns]
    if not valid:
        return df
    return df.drop_duplicates(subset=valid, keep="last")


def load_protocol_results(delay_days: int | None = 0) -> pd.DataFrame:
    """Load run_protocol results (baseline, ctgan, tabddpm). If delay_days column exists, filter to that value (default 0 = no delay)."""
    paths = [
        os.path.join(RESULTS_DIR, "protocol", "results.csv"),
        os.path.join(RESULTS_DIR, "results.csv"),  # legacy
    ]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            rename = {"recall_at_1pct_fpr": "recall_1fpr"}
            df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
            if "delay_days" in df.columns and delay_days is not None:
                # Include legacy rows (NaN) when delay_days=0
                mask = (df["delay_days"] == delay_days) | (
                    df["delay_days"].isna() & (delay_days == 0)
                )
                df = df[mask]
            dedup_cols = ["fold", "method", "target_pos_rate"]
            if "delay_days" in df.columns:
                dedup_cols = ["fold", "method", "target_pos_rate", "delay_days"]
            df = _load_dedup(df, dedup_cols)
            df["method"] = df["method"].str.lower()
            return df
    return pd.DataFrame()


def load_smote_results() -> pd.DataFrame:
    """Load SMOTE results (protocol now includes SMOTE; fallback to legacy paths)."""
    paths = [
        os.path.join(RESULTS_DIR, "smote", "results.csv"),
        os.path.join(RESULTS_DIR, "smote_baseline_results.csv"),
    ]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break
    else:
        return pd.DataFrame()
    rename = {"recall_at_1pct_fpr": "recall_1fpr"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df = _load_dedup(df, ["fold", "method", "target_pos_rate"])
    return df


def load_sliding_results() -> pd.DataFrame:
    """Load sliding window results."""
    paths = [
        os.path.join(RESULTS_DIR, "sliding_window", "results.csv"),
        os.path.join(RESULTS_DIR, "sliding_window_results.csv"),  # legacy
    ]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            rename = {"recall_at_1pct_fpr": "recall_1fpr"}
            df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
            return df
    return pd.DataFrame()


def load_drift() -> pd.DataFrame:
    """Load drift report."""
    if not os.path.exists(DRIFT_PATH):
        return pd.DataFrame()
    return pd.read_csv(DRIFT_PATH)


def _get_recall_col(df: pd.DataFrame) -> str:
    for c in ["recall_1fpr", "recall_at_1pct_fpr"]:
        if c in df.columns:
            return c
    return "recall_1fpr"


def build_unified_table(
    protocol: pd.DataFrame,
    smote: pd.DataFrame,
) -> pd.DataFrame:
    """Merge protocol + SMOTE into unified comparison (best per method per fold)."""
    rows = []
    all_folds = set()
    if not protocol.empty:
        all_folds.update(protocol["fold"].unique())
    if not smote.empty:
        all_folds.update(smote["fold"].unique())

    for fold in sorted(all_folds):
        # Baseline: prefer protocol if it has this fold, else smote
        base = None
        if not protocol.empty:
            base = protocol[(protocol["fold"] == fold) & (protocol["method"] == "baseline")]
        if (base is None or base.empty) and not smote.empty:
            base = smote[(smote["fold"] == fold) & (smote["method"] == "baseline")]
        if base is not None and not base.empty:
            rc = _get_recall_col(base)
            rows.append({
                "fold": fold, "method": "baseline", "target_pos_rate": None,
                "pr_auc": float(base["pr_auc"].iloc[-1]),
                "recall_1fpr": float(base[rc].iloc[-1]),
            })

        # CTGAN, TabDDPM, SMOTE, recency variants: prefer protocol
        if not protocol.empty:
            methods = ["ctgan", "tabddpm", "smote", "ctgan_recency03", "tabddpm_recency03", "smote_recency03"]
            for method in methods:
                m = protocol[(protocol["fold"] == fold) & (protocol["method"] == method)]
                if not m.empty:
                    best = m.loc[m["pr_auc"].idxmax()]
                    rc = _get_recall_col(m)
                    rows.append({
                        "fold": fold, "method": method, "target_pos_rate": best["target_pos_rate"],
                        "pr_auc": float(best["pr_auc"]),
                        "recall_1fpr": float(best[rc]),
                    })

        # SMOTE from smote results only if not in protocol (legacy)
        if not protocol.empty and (protocol["method"] == "smote").any():
            pass  # already got SMOTE from protocol
        elif not smote.empty:
            m = smote[(smote["fold"] == fold) & (smote["method"] == "smote")]
            if not m.empty:
                best = m.loc[m["pr_auc"].idxmax()]
                rc = _get_recall_col(m)
                rows.append({
                    "fold": fold, "method": "smote", "target_pos_rate": best["target_pos_rate"],
                    "pr_auc": float(best["pr_auc"]),
                    "recall_1fpr": float(best[rc]),
                })

    return pd.DataFrame(rows)


def drift_harm_analysis(unified: pd.DataFrame, drift: pd.DataFrame) -> pd.DataFrame:
    """Merge drift metrics with unified results, compute harm = baseline - method."""
    if unified.empty or drift.empty or "domain_auc_holdout_no_time" not in drift.columns:
        return pd.DataFrame()

    base = unified[unified["method"] == "baseline"][["fold", "pr_auc"]].rename(columns={"pr_auc": "baseline_pr_auc"})
    merged = unified[unified["method"] != "baseline"].merge(
        base, on="fold", how="left"
    ).merge(
        drift[["fold", "domain_auc_holdout_no_time"]], on="fold", how="left"
    )
    merged["pr_auc_delta"] = merged["baseline_pr_auc"] - merged["pr_auc"]
    merged["domain_auc"] = merged["domain_auc_holdout_no_time"]
    return merged


def when_it_helps_table(drift_harm: pd.DataFrame) -> pd.DataFrame:
    """Summarize: when does each method help or hurt?"""
    if drift_harm.empty:
        return pd.DataFrame()

    def classify(delta):
        if pd.isna(delta): return "—"
        if delta > 0.01: return "hurts"
        if delta < -0.01: return "helps"
        return "neutral"

    out = []
    for method in drift_harm["method"].unique():
        m = drift_harm[drift_harm["method"] == method]
        out.append({
            "method": method,
            "mean_delta": m["pr_auc_delta"].mean(),
            "folds_helps": (m["pr_auc_delta"] < -0.01).sum(),
            "folds_neutral": ((m["pr_auc_delta"] >= -0.01) & (m["pr_auc_delta"] <= 0.01)).sum(),
            "folds_hurts": (m["pr_auc_delta"] > 0.01).sum(),
            "verdict": classify(m["pr_auc_delta"].mean()),
        })
    return pd.DataFrame(out)


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading results...")
    protocol = load_protocol_results()
    smote = load_smote_results()
    drift = load_drift()

    print(f"  Protocol: {len(protocol)} rows")
    print(f"  SMOTE: {len(smote)} rows")
    print(f"  Drift: {len(drift)} rows")

    # 1. Unified table
    unified = build_unified_table(protocol, smote)
    if not unified.empty:
        out_path = os.path.join(TABLES_DIR, "unified_comparison.csv")
        unified.to_csv(out_path, index=False)
        print(f"\nSaved unified table -> {out_path}")
        print(unified.pivot_table(index="fold", columns="method", values="pr_auc").round(4).to_string())

    # 2. Drift-harm analysis
    drift_harm = drift_harm_analysis(unified, drift)
    if not drift_harm.empty:
        out_path = os.path.join(TABLES_DIR, "drift_harm_analysis.csv")
        drift_harm.to_csv(out_path, index=False)
        print(f"\nSaved drift-harm -> {out_path}")
        # Correlation: domain_auc vs pr_auc_delta
        for method in drift_harm["method"].unique():
            m = drift_harm[drift_harm["method"] == method]
            if len(m) >= 2:
                r = np.corrcoef(m["domain_auc"], m["pr_auc_delta"])[0, 1]
                print(f"  Correlation domain_auc vs harm ({method}): {r:.3f}")

    # 3. When it helps/hurts
    when = when_it_helps_table(drift_harm)
    if not when.empty:
        out_path = os.path.join(TABLES_DIR, "when_it_helps_hurts.csv")
        when.to_csv(out_path, index=False)
        print(f"\nSaved when-it-helps -> {out_path}")
        print(when.to_string(index=False))

    # 4. Summary stats
    if not unified.empty:
        summary = unified.groupby("method").agg(
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            recall_mean=("recall_1fpr", "mean"),
        ).round(4)
        out_path = os.path.join(TABLES_DIR, "method_summary.csv")
        summary.to_csv(out_path)
        print(f"\nSaved summary -> {out_path}")

    # 5. Statistical tests (paired permutation tests across folds)
    if not unified.empty and _run_stat_comparisons is not None:
        stat_df = _run_stat_comparisons(unified, metric="pr_auc", use_permutation=True)
        if not stat_df.empty:
            out_path = os.path.join(TABLES_DIR, "statistical_comparisons.csv")
            stat_df.to_csv(out_path, index=False)
            print(f"\nSaved statistical comparisons -> {out_path}")
            print(stat_df.to_string(index=False))

    print("\nDone. Run python -m src.paper_figures to generate figures.")


if __name__ == "__main__":
    main()
