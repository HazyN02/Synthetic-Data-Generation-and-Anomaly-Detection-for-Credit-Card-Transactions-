#!/usr/bin/env python3
"""
Label delay ablation: run protocol with delay_days in [0, 3, 7, 14] and plot PR-AUC vs delay.

Usage:
  python -m src.run_label_delay_ablation [--quick]

Runs:
  python -m src.run_protocol --medium --delay-days 0
  python -m src.run_protocol --medium --delay-days 3
  python -m src.run_protocol --medium --delay-days 7
  python -m src.run_protocol --medium --delay-days 14

Then merges results and produces paper/figures/label_delay_ablation.pdf.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
os.chdir(_ROOT)

RESULTS_DIR = _ROOT / "results"
PROTOCOL_DIR = RESULTS_DIR / "protocol"
PAPER_DIR = _ROOT / "paper"
FIGURES_DIR = PAPER_DIR / "figures"
TABLES_DIR = PAPER_DIR / "tables"

DELAY_DAYS = [0, 3, 7, 14]


def main():
    parser = argparse.ArgumentParser(description="Label delay ablation: run 0, 3, 7, 14 days")
    parser.add_argument("--quick", action="store_true", help="2 folds, faster (skips 3-day)")
    parser.add_argument("--plot-only", action="store_true", help="Skip runs, only merge and plot existing results")
    args = parser.parse_args()

    delay_days_list = [0, 7, 14] if args.quick else DELAY_DAYS

    if not args.plot_only:
        for d in delay_days_list:
            cmd = [
                sys.executable, "-m", "src.run_protocol",
                "--quick" if args.quick else "--medium",
                "--delay-days", str(d),
            ]
            print(f"\n=== Running delay_days={d} ===")
            subprocess.run(cmd, check=True)

    # Merge results from all run dirs that have delay_days
    dfs = []
    for p in sorted(PROTOCOL_DIR.iterdir()):
        if not p.is_dir() or not p.name.startswith("run_"):
            continue
        rp = p / "results.csv"
        cp = p / "config.json"
        if not rp.exists():
            continue
        try:
            import json
            with open(cp) as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
        dd = cfg.get("delay_days", 0)
        df = pd.read_csv(rp)
        df["delay_days"] = dd
        df["_source"] = p.name
        dfs.append(df)

    # Also main results.csv if it has delay_days
    main_csv = PROTOCOL_DIR / "results.csv"
    if main_csv.exists():
        df = pd.read_csv(main_csv)
        if "delay_days" in df.columns:
            dfs.append(df)

    if not dfs:
        print("No protocol results with delay_days found. Run protocol with --delay-days first.")
        return

    merged = pd.concat(dfs, ignore_index=True)
    merged["method"] = merged["method"].str.lower()

    agg = merged.groupby(["delay_days", "method"])["pr_auc"].agg(["mean", "std", "count"]).reset_index()

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    out_csv = TABLES_DIR / "canonical_by_delay.csv"
    agg.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    if agg["delay_days"].nunique() < 2:
        print("Need at least 2 delay_days for plot. Run with multiple --delay-days.")
        return

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for method in agg["method"].unique():
            sub = agg[agg["method"] == method].sort_values("delay_days")
            ax.errorbar(
                sub["delay_days"],
                sub["mean"],
                yerr=sub["std"],
                label=method,
                marker="o",
                capsize=3,
            )
        ax.set_xlabel("Label delay (days)")
        ax.set_ylabel("PR-AUC (mean ± std)")
        ax.set_title("Label delay ablation: PR-AUC vs delay")
        ax.legend()
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "label_delay_ablation.pdf", dpi=150)
        fig.savefig(FIGURES_DIR / "label_delay_ablation.png", dpi=150)
        plt.close()
        print(f"Saved {FIGURES_DIR / 'label_delay_ablation.pdf'}")
    except ImportError:
        print("matplotlib not installed, skipping plot")


if __name__ == "__main__":
    main()
