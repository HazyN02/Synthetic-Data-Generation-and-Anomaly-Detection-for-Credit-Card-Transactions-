#!/usr/bin/env python3
"""
Generate paper-ready figures.
Run after run_unified_analysis: python -m src.paper_figures
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

PAPER_DIR = os.path.join(_ROOT, "paper")
TABLES_DIR = os.path.join(PAPER_DIR, "tables")
FIGURES_DIR = os.path.join(PAPER_DIR, "figures")


def _ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def fig_method_comparison():
    """Bar chart: PR-AUC by method and fold."""
    p = os.path.join(TABLES_DIR, "unified_comparison.csv")
    if not os.path.exists(p):
        print("Skip fig_method_comparison: no unified_comparison.csv")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Skip figures: matplotlib not installed")
        return

    df = pd.read_csv(p)
    if df.empty:
        return

    pivot = df.pivot_table(index="fold", columns="method", values="pr_auc")
    cols = [c for c in ["baseline", "smote", "ctgan", "tabddpm"] if c in pivot.columns]
    pivot = pivot[cols] if cols else pivot

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(pivot.index))
    width = 0.2
    n_cols = len(pivot.columns)
    for i, col in enumerate(pivot.columns):
        offset = (i - n_cols / 2 + 0.5) * width
        ax.bar(x + offset, pivot[col], width, label=col)

    ax.set_xlabel("Temporal fold")
    ax.set_ylabel("PR-AUC")
    ax.set_title("PR-AUC by method across temporal folds")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in pivot.index])
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "method_comparison_pr_auc.pdf"), dpi=150)
    fig.savefig(os.path.join(FIGURES_DIR, "method_comparison_pr_auc.png"), dpi=150)
    plt.close()
    print(f"Saved method_comparison_pr_auc.*")


def fig_drift_vs_harm():
    """Scatter: domain AUC vs synthetic harm (delta PR-AUC)."""
    p = os.path.join(TABLES_DIR, "drift_harm_analysis.csv")
    if not os.path.exists(p):
        print("Skip fig_drift_vs_harm: no drift_harm_analysis.csv")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    df = pd.read_csv(p)
    if df.empty or "domain_auc" not in df.columns or "pr_auc_delta" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"ctgan": "C1", "tabddpm": "C2", "smote": "C3"}
    for method in df["method"].unique():
        m = df[df["method"] == method]
        ax.scatter(
            m["domain_auc"],
            m["pr_auc_delta"],
            label=method,
            color=colors.get(method, "gray"),
            s=80,
        )

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Domain AUC (train vs val separation)")
    ax.set_ylabel("PR-AUC delta (baseline − method)")
    ax.set_title("Drift vs oversampling effect (negative = benefit)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "drift_vs_harm.pdf"), dpi=150)
    fig.savefig(os.path.join(FIGURES_DIR, "drift_vs_harm.png"), dpi=150)
    plt.close()
    print(f"Saved drift_vs_harm.*")


def fig_sliding_window():
    """Static vs sliding comparison."""
    for p in [
        os.path.join(_ROOT, "results", "sliding_window", "results.csv"),
        os.path.join(_ROOT, "results", "sliding_window_results.csv"),
    ]:
        if os.path.exists(p):
            break
    else:
        p = None
    if p is None:
        print("Skip fig_sliding_window: no sliding_window results.csv found")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    df = pd.read_csv(p)
    if df.empty:
        return

    pivot = df.pivot_table(index="fold", columns="strategy", values="pr_auc")
    strat_cols = {c.lower(): c for c in pivot.columns}
    static_col = strat_cols.get("static") or strat_cols.get("Static")
    sliding_col = strat_cols.get("sliding") or strat_cols.get("Sliding")
    if static_col is None or sliding_col is None:
        print("Skip fig_sliding_window: need 'static' and 'sliding' strategy columns")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(pivot.index))
    width = 0.35
    ax.bar(x - width / 2, pivot[static_col], width, label="Static (all past)")
    ax.bar(x + width / 2, pivot[sliding_col], width, label="Sliding (recent)")

    ax.set_xlabel("Fold")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Static vs sliding window retraining")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in pivot.index])
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "sliding_window_comparison.pdf"), dpi=150)
    fig.savefig(os.path.join(FIGURES_DIR, "sliding_window_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved sliding_window_comparison.*")


def main():
    _ensure_figures_dir()
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        pass
    fig_method_comparison()
    fig_drift_vs_harm()
    fig_sliding_window()
    print(f"\nFigures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
