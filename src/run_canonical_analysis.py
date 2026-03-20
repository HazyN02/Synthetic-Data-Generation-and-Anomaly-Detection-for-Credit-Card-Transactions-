#!/usr/bin/env python3
"""
Canonical analysis: merge all protocol runs into one pipeline.
Produces canonical tables/figures and documents which run_id / delay_days corresponds to each.

Run from project root: python -m src.run_canonical_analysis

Sources:
- results/protocol/results.csv (main merged results)
- results/protocol/run_*/results.csv (timestamped runs)
- results/protocol/FROZEN/frozen_results.csv (frozen snapshot)
- results/smote/results.csv, results/sliding_window/results.csv

Outputs:
- paper/tables/canonical_main.csv (no-delay, best per method per fold)
- paper/tables/canonical_run_mapping.csv (run_id -> table, config)
- paper/tables/canonical_by_delay.csv (label-delay ablation: PR-AUC vs delay_days)
- paper/figures/label_delay_ablation.pdf (if delay data exists)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

RESULTS_DIR = _ROOT / "results"
PAPER_DIR = _ROOT / "paper"
TABLES_DIR = PAPER_DIR / "tables"
FIGURES_DIR = PAPER_DIR / "figures"
PROTOCOL_DIR = RESULTS_DIR / "protocol"


def _load_csv(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df if not df.empty else None
    except Exception:
        return None


def _rename_recall(df: pd.DataFrame) -> pd.DataFrame:
    if "recall_at_1pct_fpr" in df.columns and "recall_1fpr" not in df.columns:
        df = df.rename(columns={"recall_at_1pct_fpr": "recall_1fpr"})
    return df


def collect_protocol_sources() -> list[tuple[str, Path, dict]]:
    """Return [(source_name, path, config_dict), ...] for all protocol result sources."""
    sources = []
    # Main merged
    p = PROTOCOL_DIR / "results.csv"
    if p.exists():
        sources.append(("main", p, {}))

    # Run dirs
    for d in sorted(PROTOCOL_DIR.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("run_"):
            rp = d / "results.csv"
            cp = d / "config.json"
            cfg = {}
            if cp.exists():
                try:
                    with open(cp) as f:
                        cfg = json.load(f)
                except Exception:
                    pass
            if rp.exists():
                sources.append((d.name, rp, cfg))

    # FROZEN
    fp = PROTOCOL_DIR / "FROZEN" / "frozen_results.csv"
    fcfg = PROTOCOL_DIR / "FROZEN" / "frozen_config.json"
    cfg = {}
    if fcfg.exists():
        try:
            with open(fcfg) as f:
                cfg = json.load(f)
        except Exception:
            pass
    if fp.exists():
        sources.append(("FROZEN", fp, cfg))

    return sources


def load_all_protocol() -> pd.DataFrame:
    """Load and concatenate all protocol sources with source tags."""
    rows = []
    for name, path, cfg in collect_protocol_sources():
        df = _load_csv(path)
        if df is None:
            continue
        df = _rename_recall(df)
        df["_source"] = name
        df["_run_id"] = cfg.get("run_id", name)
        df["_delay_days"] = cfg.get("delay_days", 0)
        df["_recency_ablation"] = cfg.get("recency_ablation", False)
        if "delay_days" not in df.columns:
            df["delay_days"] = cfg.get("delay_days", 0)
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_run_mapping(all_df: pd.DataFrame) -> pd.DataFrame:
    """Build table mapping run_id / delay_days / recency -> which tables use it."""
    if all_df.empty:
        return pd.DataFrame()

    sub = all_df[["_source", "_run_id", "_delay_days", "_recency_ablation"]].drop_duplicates()
    sub = sub.rename(columns={
        "_source": "source",
        "_run_id": "run_id",
        "_delay_days": "delay_days",
        "_recency_ablation": "recency_ablation",
    })
    sub["used_in"] = sub.apply(
        lambda r: "canonical_main" if r["delay_days"] == 0 and not r["recency_ablation"] else "ablation",
        axis=1,
    )
    return sub


def build_canonical_main(all_df: pd.DataFrame) -> pd.DataFrame:
    """Best per method per fold for no-delay runs (exclude recency variant methods)."""
    df = all_df[all_df["delay_days"].fillna(0) == 0].copy()
    # Exclude recency variant methods (ctgan_recency03, etc.)
    df = df[~df["method"].str.contains("_recency", na=False)]
    if df.empty:
        return pd.DataFrame()

    df["method"] = df["method"].str.lower()
    rc = "recall_1fpr" if "recall_1fpr" in df.columns else "recall_at_1pct_fpr"
    methods = ["baseline", "smote", "ctgan", "tabddpm"]
    rows = []
    for fold in df["fold"].unique():
        for method in methods:
            sub = df[(df["fold"] == fold) & (df["method"] == method)]
            if sub.empty:
                continue
            best = sub.loc[sub["pr_auc"].idxmax()]
            rows.append({
                "fold": fold,
                "method": method,
                "target_pos_rate": best.get("target_pos_rate"),
                "pr_auc": float(best["pr_auc"]),
                "recall_1fpr": float(best[rc]) if rc in best else None,
            })

    return pd.DataFrame(rows)


def build_canonical_by_delay(all_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PR-AUC by delay_days for label-delay ablation."""
    if "delay_days" not in all_df.columns:
        return pd.DataFrame()

    df = all_df.copy()
    df["delay_days"] = df["delay_days"].fillna(0).astype(int)
    df["method"] = df["method"].str.lower()
    # Exclude recency variants for clean ablation
    df = df[~df["_recency_ablation"].fillna(False)]

    return df.groupby(["delay_days", "method"])["pr_auc"].agg(["mean", "std", "count"]).reset_index()


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Collecting protocol sources...")
    all_protocol = load_all_protocol()
    if all_protocol.empty:
        print("No protocol results found.")
        return

    print(f"  Loaded {len(all_protocol)} rows from {all_protocol['_source'].nunique()} sources")

    # Run mapping
    mapping = build_run_mapping(all_protocol)
    if not mapping.empty:
        out = TABLES_DIR / "canonical_run_mapping.csv"
        mapping.to_csv(out, index=False)
        print(f"\nSaved run mapping -> {out}")
        print(mapping.to_string(index=False))

    # Canonical main table (no delay, no recency)
    canonical = build_canonical_main(all_protocol)
    if not canonical.empty:
        out = TABLES_DIR / "canonical_main.csv"
        canonical.to_csv(out, index=False)
        print(f"\nSaved canonical_main -> {out}")
        pivot = canonical.pivot_table(index="fold", columns="method", values="pr_auc")
        print(pivot.round(4).to_string())

    # Label-delay ablation
    by_delay = build_canonical_by_delay(all_protocol)
    if not by_delay.empty and by_delay["delay_days"].nunique() > 1:
        out = TABLES_DIR / "canonical_by_delay.csv"
        by_delay.to_csv(out, index=False)
        print(f"\nSaved canonical_by_delay -> {out}")

        # Plot PR-AUC vs delay
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 4))
            for method in by_delay["method"].unique():
                sub = by_delay[by_delay["method"] == method].sort_values("delay_days")
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
            print(f"Saved label_delay_ablation.*")
        except ImportError:
            print("matplotlib not available, skipping label_delay_ablation plot")

    print("\nDone. Canonical tables in paper/tables/")
    print("Run python -m src.run_unified_analysis for drift-harm and statistical tests.")


if __name__ == "__main__":
    main()
