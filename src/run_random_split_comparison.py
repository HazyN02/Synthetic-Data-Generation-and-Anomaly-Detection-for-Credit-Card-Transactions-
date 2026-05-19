"""
Random split vs expanding window comparison — multi-seed distributional version.

Runs 10 independent 80/20 stratified splits (seeds 0-9) and for each computes:
  - Baseline LightGBM PR-AUC
  - CTGAN-150ep LightGBM PR-AUC
  - Delta (CTGAN - baseline)

Then reports mean / std / range of delta across seeds and counts how many splits
show positive vs negative CTGAN benefit.

This converts a single-draw observation into a distributional statement:
  "Across 10 random splits, CTGAN delta ranges from X to Y (mean Z +/- std);
   under temporal evaluation the mean is +0.0037."

Outputs
-------
  results/random_split_per_seed.csv    -- 10 rows (one per seed), appended as completed
  results/random_split_summary.csv     -- single summary row per condition
  results/random_split_comparison.csv  -- overwritten with first-seed row for backward compat

Usage
-----
  python -m src.run_random_split_comparison               # full run: seeds 0-9, CTGAN 150ep
  python -m src.run_random_split_comparison --seeds 0,1  # test two seeds only
  python -m src.run_random_split_comparison --ctgan-epochs 10  # quick smoke-test
  python -m src.run_random_split_comparison --summary-only    # reprint summary from saved CSV
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.preprocess_synth import (
    DROP_COLS, HASH_COLS, MAX_COLS, PRIORITY_COLS,
    TARGET_COL, TIME_COL, get_cat_cols_for_synth, preprocess_fold,
)
from src.synth_ctgan import make_synthetic_positives
from src.train import train_and_eval

# Canonical CTGAN config (identical to run_protocol.py)
CTGAN_EPOCHS_DEFAULT = 150
CTGAN_BATCH_SIZE = 500
CTGAN_DISC_STEPS = 5
CTGAN_PAC = 1
CTGAN_SEED = 0        # generator seed — held fixed across splits; we vary DATA split seed
TARGET_POS_RATE = 0.05
DEFAULT_SEEDS = list(range(10))  # seeds 0-9

# Canonical expanding-window results for comparison
CANONICAL_RESULTS_CSV = os.path.join(
    _ROOT, "results", "protocol", "run_20260330_180216", "results.csv"
)

# Output paths
PER_SEED_CSV = os.path.join(_ROOT, "results", "random_split_per_seed.csv")
SUMMARY_CSV  = os.path.join(_ROOT, "results", "random_split_summary.csv")
COMPAT_CSV   = os.path.join(_ROOT, "results", "random_split_comparison.csv")


# ---------------------------------------------------------------------------
# Column-projected parquet load (identical to other scripts in this project)
# ---------------------------------------------------------------------------

def _select_columns_to_load(all_cols: List[str]) -> List[str]:
    drop = set(DROP_COLS)
    present = [c for c in all_cols if c not in drop]

    def ok(c: str) -> bool:
        if c.startswith("id_"):
            return c in set(PRIORITY_COLS)
        if c.startswith("V"):
            try:
                return int(c[1:]) <= 50
            except ValueError:
                return True
        return True

    priority = [c for c in PRIORITY_COLS if c in present]
    rest = [c for c in present if c not in set(priority) and ok(c)]
    selected = priority + rest[: MAX_COLS - len(priority)]
    must_have = {TARGET_COL, TIME_COL, *HASH_COLS}
    for c in must_have:
        if c in all_cols and c not in selected:
            selected.append(c)
    return [c for c in selected if c in all_cols]


def _load_train_data(data_dir: str) -> pd.DataFrame:
    for name in ["train_merged.parquet", "train_transaction.csv"]:
        p = os.path.join(data_dir, name)
        if not os.path.exists(p):
            continue
        print(f"[RSPLIT] Loading: {p}", flush=True)
        if p.endswith(".parquet"):
            import pyarrow.parquet as _pq
            schema_cols = _pq.ParquetFile(p).schema_arrow.names
            cols = _select_columns_to_load(schema_cols)
            print(f"[RSPLIT] Reading {len(cols)} / {len(schema_cols)} columns", flush=True)
            df = pd.read_parquet(p, columns=cols)
        else:
            df = pd.read_csv(p)
        for c in df.select_dtypes(include=["float64"]).columns:
            df[c] = df[c].astype("float32")
        for c in df.select_dtypes(include=["int64"]).columns:
            try:
                df[c] = pd.to_numeric(df[c], downcast="integer")
            except Exception:
                pass
        gc.collect()
        print(f"[RSPLIT] df shape={df.shape}", flush=True)
        return df
    raise FileNotFoundError("No train data found in data/")


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _completed_seeds(csv_path: str) -> set:
    """Return set of seeds already written to the per-seed CSV."""
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["seed"])
        return set(int(s) for s in df["seed"].dropna().unique())
    except Exception:
        return set()


def _append_row(row: dict, path: str) -> None:
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(path)
    df_row.to_csv(path, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# Load canonical temporal results (called once)
# ---------------------------------------------------------------------------

def _load_canonical() -> tuple[float, float]:
    """Return (canonical_baseline_mean, canonical_ctgan_mean) at rate=0.05."""
    if not os.path.exists(CANONICAL_RESULTS_CSV):
        return float("nan"), float("nan")
    can = pd.read_csv(CANONICAL_RESULTS_CSV)
    bl = can[can["method"].str.lower().str.contains("baseline", na=False)]["pr_auc"]
    if len(bl) == 0:
        bl = can[can["target_pos_rate"].isna()]["pr_auc"]
    canonical_bl = float(bl.mean()) if len(bl) > 0 else float("nan")
    ctg = can[
        can["method"].str.lower().str.contains("ctgan", na=False) &
        (can["target_pos_rate"].round(3) == 0.05)
    ]["pr_auc"]
    canonical_ctg = float(ctg.mean()) if len(ctg) > 0 else float("nan")
    return canonical_bl, canonical_ctg


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------

def _run_one_seed(
    df_raw: pd.DataFrame,
    seed: int,
    ctgan_epochs: int,
) -> dict:
    """Run one 80/20 split for the given seed. Returns result dict."""
    n_total = len(df_raw)
    y_all = df_raw[TARGET_COL].values

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(n_total), y_all))
    train_df = df_raw.iloc[train_idx].reset_index(drop=True)
    val_df   = df_raw.iloc[val_idx].reset_index(drop=True)

    n_fraud_train = int((train_df[TARGET_COL] == 1).sum())
    n_fraud_val   = int((val_df[TARGET_COL] == 1).sum())
    print(f"  [seed={seed}] train={len(train_df)} (fraud={n_fraud_train}), "
          f"val={len(val_df)} (fraud={n_fraud_val})", flush=True)

    train_prep, val_prep, used_cols = preprocess_fold(train_df, val_df)
    cat_cols = get_cat_cols_for_synth(train_prep, used_cols)
    del train_df, val_df
    gc.collect()

    # Baseline
    print(f"  [seed={seed}] baseline ...", flush=True)
    m_bl = train_and_eval(train_prep, val_prep)
    pr_bl = float(m_bl["pr_auc"])
    print(f"  [seed={seed}] baseline PR-AUC={pr_bl:.6f}", flush=True)

    # CTGAN
    print(f"  [seed={seed}] CTGAN (epochs={ctgan_epochs}) ...", flush=True)
    synth = make_synthetic_positives(
        train_df=train_prep,
        cat_cols=cat_cols,
        used_cols=used_cols,
        target_pos_rate=TARGET_POS_RATE,
        max_synth=50000,
        epochs=ctgan_epochs,
        batch_size=CTGAN_BATCH_SIZE,
        pac=CTGAN_PAC,
        seed=CTGAN_SEED,
        discriminator_steps=CTGAN_DISC_STEPS,
        verbose=False,
    )
    train_ctgan = pd.concat([train_prep, synth], axis=0, ignore_index=True)
    del synth
    gc.collect()
    m_ctg = train_and_eval(train_ctgan, val_prep)
    pr_ctg = float(m_ctg["pr_auc"])
    del train_ctgan
    gc.collect()
    print(f"  [seed={seed}] CTGAN PR-AUC={pr_ctg:.6f}  delta={pr_ctg - pr_bl:+.6f}", flush=True)

    return {
        "seed": seed,
        "n_train": len(train_prep),
        "n_val": len(val_prep),
        "n_fraud_train": n_fraud_train,
        "n_fraud_val": n_fraud_val,
        "ctgan_epochs": ctgan_epochs,
        "target_pos_rate": TARGET_POS_RATE,
        "pr_auc_baseline": pr_bl,
        "pr_auc_ctgan": pr_ctg,
        "delta": pr_ctg - pr_bl,
        "recall_baseline": float(m_bl.get("recall@1%fpr", float("nan"))),
        "recall_ctgan": float(m_ctg.get("recall@1%fpr", float("nan"))),
    }


# ---------------------------------------------------------------------------
# Summary printer / saver
# ---------------------------------------------------------------------------

def _print_and_save_summary(per_seed_csv: str, canonical_bl: float, canonical_ctg: float) -> None:
    if not os.path.exists(per_seed_csv):
        print("[RSPLIT] No per-seed CSV found; nothing to summarise.", flush=True)
        return

    df = pd.read_csv(per_seed_csv)
    n = len(df)
    if n == 0:
        print("[RSPLIT] Per-seed CSV is empty.", flush=True)
        return

    bl_vals   = df["pr_auc_baseline"].values
    ctg_vals  = df["pr_auc_ctgan"].values
    delta_vals = df["delta"].values

    n_pos = int((delta_vals > 0).sum())
    n_neg = int((delta_vals < 0).sum())
    n_zero = n - n_pos - n_neg

    print("\n" + "=" * 70, flush=True)
    print(f"RANDOM SPLIT DISTRIBUTIONAL SUMMARY  (n={n} seeds, 80/20 stratified)", flush=True)
    print("=" * 70, flush=True)

    # Per-seed table
    print(f"\n{'Seed':>4}  {'Baseline':>8}  {'CTGAN':>8}  {'Delta':>8}  {'Sign':>4}", flush=True)
    print("-" * 44, flush=True)
    for _, row in df.sort_values("seed").iterrows():
        sign = "+" if row["delta"] > 0 else ("-" if row["delta"] < 0 else "0")
        print(f"{int(row['seed']):>4}  {row['pr_auc_baseline']:>8.4f}  "
              f"{row['pr_auc_ctgan']:>8.4f}  {row['delta']:>+8.4f}  {sign:>4}", flush=True)
    print("-" * 44, flush=True)
    print(f"{'Mean':>4}  {bl_vals.mean():>8.4f}  {ctg_vals.mean():>8.4f}  "
          f"{delta_vals.mean():>+8.4f}", flush=True)
    print(f"{'Std':>4}  {bl_vals.std():>8.4f}  {ctg_vals.std():>8.4f}  "
          f"{delta_vals.std():>8.4f}", flush=True)
    print(f"{'Min':>4}  {bl_vals.min():>8.4f}  {ctg_vals.min():>8.4f}  "
          f"{delta_vals.min():>+8.4f}", flush=True)
    print(f"{'Max':>4}  {bl_vals.max():>8.4f}  {ctg_vals.max():>8.4f}  "
          f"{delta_vals.max():>+8.4f}", flush=True)

    print(f"\nCTGAN delta: {n_pos}/{n} splits positive, "
          f"{n_neg}/{n} negative, {n_zero}/{n} zero", flush=True)

    if not np.isnan(canonical_bl) and not np.isnan(canonical_ctg):
        print(f"\nComparison with temporal 8-fold expanding window:", flush=True)
        print(f"  Temporal baseline mean PR-AUC : {canonical_bl:.4f}", flush=True)
        print(f"  Temporal CTGAN mean PR-AUC    : {canonical_ctg:.4f}", flush=True)
        print(f"  Temporal mean delta            : {canonical_ctg - canonical_bl:+.4f}", flush=True)
        print(f"  Random split mean baseline     : {bl_vals.mean():.4f}  "
              f"(+{bl_vals.mean() - canonical_bl:.4f} vs temporal)", flush=True)
        print(f"  Random split mean CTGAN delta  : {delta_vals.mean():+.4f}  "
              f"vs temporal {canonical_ctg - canonical_bl:+.4f}", flush=True)

    print("=" * 70, flush=True)

    # Save summary CSV
    summary = {
        "n_seeds": n,
        "ctgan_epochs": int(df["ctgan_epochs"].iloc[0]) if "ctgan_epochs" in df.columns else float("nan"),
        "target_pos_rate": TARGET_POS_RATE,
        "baseline_mean": round(float(bl_vals.mean()), 6),
        "baseline_std":  round(float(bl_vals.std()),  6),
        "baseline_min":  round(float(bl_vals.min()),  6),
        "baseline_max":  round(float(bl_vals.max()),  6),
        "ctgan_mean": round(float(ctg_vals.mean()), 6),
        "ctgan_std":  round(float(ctg_vals.std()),  6),
        "ctgan_min":  round(float(ctg_vals.min()),  6),
        "ctgan_max":  round(float(ctg_vals.max()),  6),
        "delta_mean": round(float(delta_vals.mean()), 6),
        "delta_std":  round(float(delta_vals.std()),  6),
        "delta_min":  round(float(delta_vals.min()),  6),
        "delta_max":  round(float(delta_vals.max()),  6),
        "n_positive_delta": n_pos,
        "n_negative_delta": n_neg,
        "temporal_baseline_mean": round(canonical_bl, 6) if not np.isnan(canonical_bl) else float("nan"),
        "temporal_ctgan_mean": round(canonical_ctg, 6) if not np.isnan(canonical_ctg) else float("nan"),
        "temporal_delta": round(canonical_ctg - canonical_bl, 6) if not np.isnan(canonical_ctg) else float("nan"),
        "random_vs_temporal_baseline_gap": round(float(bl_vals.mean()) - canonical_bl, 6) if not np.isnan(canonical_bl) else float("nan"),
    }
    pd.DataFrame([summary]).to_csv(SUMMARY_CSV, index=False)
    print(f"\n[RSPLIT] Summary saved to {SUMMARY_CSV}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-seed random split comparison (10 seeds x 80/20 stratified)."
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated list of seeds to run (default: 0,1,...,9).",
    )
    parser.add_argument(
        "--ctgan-epochs",
        type=int,
        default=CTGAN_EPOCHS_DEFAULT,
        help=f"CTGAN epochs per split (default: {CTGAN_EPOCHS_DEFAULT}). "
             "Use 10 for a quick smoke-test (~15 min vs ~15 hours for 150).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip all training; reprint summary from existing per-seed CSV.",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    ctgan_epochs = args.ctgan_epochs

    os.makedirs(os.path.join(_ROOT, "results"), exist_ok=True)

    canonical_bl, canonical_ctg = _load_canonical()

    if args.summary_only:
        _print_and_save_summary(PER_SEED_CSV, canonical_bl, canonical_ctg)
        return

    # Load data once (expensive; ~2 GB RAM for parquet)
    df_raw = _load_train_data(os.path.join(_ROOT, "data"))
    n_total = len(df_raw)
    n_fraud = int((df_raw[TARGET_COL] == 1).sum())
    print(f"[RSPLIT] Full dataset: {n_total} rows, fraud={n_fraud} ({100*n_fraud/n_total:.2f}%)", flush=True)
    print(f"[RSPLIT] Running {len(seeds)} seeds: {seeds}, CTGAN epochs={ctgan_epochs}", flush=True)

    completed = _completed_seeds(PER_SEED_CSV)
    if completed:
        print(f"[RSPLIT] Resuming — already completed seeds: {sorted(completed)}", flush=True)

    for i, seed in enumerate(seeds):
        if seed in completed:
            print(f"[RSPLIT] Seed {seed} already done — skipping.", flush=True)
            continue

        print(f"\n[RSPLIT] ===== Seed {seed} ({i+1}/{len(seeds)}) =====", flush=True)
        result = _run_one_seed(df_raw, seed=seed, ctgan_epochs=ctgan_epochs)
        _append_row(result, PER_SEED_CSV)
        print(f"[RSPLIT] Seed {seed} written to {PER_SEED_CSV}", flush=True)

        # Keep backward-compat CSV updated with most recent completed seed=0 result
        if seed == 0:
            compat_rows = [
                {
                    "split_type": "random_80_20",
                    "method": "baseline",
                    "pr_auc": result["pr_auc_baseline"],
                    "recall_at_1pct_fpr": result["recall_baseline"],
                    "n_train": result["n_train"],
                    "n_val": result["n_val"],
                    "n_fraud_train": result["n_fraud_train"],
                    "n_fraud_val": result["n_fraud_val"],
                    "ctgan_epochs": float("nan"),
                    "target_pos_rate": float("nan"),
                    "canonical_8fold_mean_pr_auc": canonical_bl,
                    "delta_vs_canonical": result["pr_auc_baseline"] - canonical_bl,
                },
                {
                    "split_type": "random_80_20",
                    "method": "ctgan",
                    "pr_auc": result["pr_auc_ctgan"],
                    "recall_at_1pct_fpr": result["recall_ctgan"],
                    "n_train": result["n_train"],
                    "n_val": result["n_val"],
                    "n_fraud_train": result["n_fraud_train"],
                    "n_fraud_val": result["n_fraud_val"],
                    "ctgan_epochs": ctgan_epochs,
                    "target_pos_rate": TARGET_POS_RATE,
                    "canonical_8fold_mean_pr_auc": canonical_ctg,
                    "delta_vs_canonical": result["pr_auc_ctgan"] - canonical_ctg,
                },
            ]
            pd.DataFrame(compat_rows).to_csv(COMPAT_CSV, index=False)

    del df_raw
    gc.collect()

    _print_and_save_summary(PER_SEED_CSV, canonical_bl, canonical_ctg)


if __name__ == "__main__":
    main()
