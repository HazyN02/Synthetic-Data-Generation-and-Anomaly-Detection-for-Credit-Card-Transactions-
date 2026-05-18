"""
Random split vs expanding window comparison.
Runs one 80/20 stratified split (random_state=42) with:
  - Baseline LightGBM (no synthesis)
  - CTGAN-150 + LightGBM (same config as canonical)

Compares PR-AUC against canonical 8-fold expanding window results
loaded from results/protocol/run_20260330_180216/results.csv.

Outputs: results/random_split_comparison.csv
"""
from __future__ import annotations

import gc
import os
import sys
from typing import List

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
CTGAN_EPOCHS = 150
CTGAN_BATCH_SIZE = 500
CTGAN_DISC_STEPS = 5
CTGAN_PAC = 1
CTGAN_SEED = 0
TARGET_POS_RATE = 0.05

# Canonical expanding-window results for comparison
CANONICAL_RESULTS_CSV = os.path.join(
    _ROOT, "results", "protocol", "run_20260330_180216", "results.csv"
)


# ---------------------------------------------------------------------------
# Column-projected parquet load
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    data_dir = os.path.join(_ROOT, "data")
    out_csv = os.path.join(_ROOT, "results", "random_split_comparison.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Load full dataset
    df_raw = _load_train_data(data_dir)
    n_total = len(df_raw)
    n_fraud = int((df_raw[TARGET_COL] == 1).sum())
    print(f"[RSPLIT] Full dataset: {n_total} rows, fraud={n_fraud} ({100*n_fraud/n_total:.2f}%)", flush=True)

    # 80/20 stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    y_all = df_raw[TARGET_COL].values
    train_idx, val_idx = next(sss.split(np.zeros(n_total), y_all))
    train_df = df_raw.iloc[train_idx].reset_index(drop=True)
    val_df = df_raw.iloc[val_idx].reset_index(drop=True)
    del df_raw
    gc.collect()

    print(f"[RSPLIT] Random split — train: {len(train_df)} rows "
          f"(fraud={int((train_df[TARGET_COL]==1).sum())}), "
          f"val: {len(val_df)} rows "
          f"(fraud={int((val_df[TARGET_COL]==1).sum())})", flush=True)

    # Preprocess (fit on train, transform val)
    train_prep, val_prep, used_cols = preprocess_fold(train_df, val_df)
    cat_cols = get_cat_cols_for_synth(train_prep, used_cols)
    del train_df, val_df
    gc.collect()

    print(f"[RSPLIT] Preprocessed: train={len(train_prep)}, val={len(val_prep)}, cols={len(used_cols)}", flush=True)

    # --- Baseline ---
    print("\n[RSPLIT] Running baseline ...", flush=True)
    metrics_baseline = train_and_eval(train_prep, val_prep)
    pr_baseline = metrics_baseline["pr_auc"]
    print(f"[RSPLIT] Baseline PR-AUC: {pr_baseline:.6f}", flush=True)

    # --- CTGAN-150 ---
    print(f"\n[RSPLIT] Running CTGAN (epochs={CTGAN_EPOCHS}) ...", flush=True)
    synth = make_synthetic_positives(
        train_df=train_prep,
        cat_cols=cat_cols,
        used_cols=used_cols,
        target_pos_rate=TARGET_POS_RATE,
        max_synth=50000,
        epochs=CTGAN_EPOCHS,
        batch_size=CTGAN_BATCH_SIZE,
        pac=CTGAN_PAC,
        seed=CTGAN_SEED,
        discriminator_steps=CTGAN_DISC_STEPS,
        verbose=True,
    )
    print(f"[RSPLIT] Synthetic rows generated: {len(synth)}", flush=True)
    train_ctgan = pd.concat([train_prep, synth], axis=0, ignore_index=True)
    del synth
    gc.collect()

    metrics_ctgan = train_and_eval(train_ctgan, val_prep)
    pr_ctgan = metrics_ctgan["pr_auc"]
    print(f"[RSPLIT] CTGAN-150 PR-AUC: {pr_ctgan:.6f}", flush=True)
    del train_ctgan
    gc.collect()

    # --- Load canonical expanding-window results for comparison ---
    canonical_baseline_mean = float("nan")
    canonical_ctgan_mean = float("nan")
    if os.path.exists(CANONICAL_RESULTS_CSV):
        can = pd.read_csv(CANONICAL_RESULTS_CSV)
        # Baseline: method==baseline
        bl_rows = can[can["method"].str.lower().str.contains("baseline", na=False)]["pr_auc"]
        if len(bl_rows) == 0:
            # Try numeric fallback: rows where target_pos_rate is NaN
            bl_rows = can[can["target_pos_rate"].isna()]["pr_auc"]
        if len(bl_rows) > 0:
            canonical_baseline_mean = float(bl_rows.mean())

        # CTGAN @ 0.05 rate
        ctg_rows = can[
            can["method"].str.lower().str.contains("ctgan", na=False) &
            (can["target_pos_rate"].round(3) == 0.05)
        ]["pr_auc"]
        if len(ctg_rows) > 0:
            canonical_ctgan_mean = float(ctg_rows.mean())

        print(f"\n[RSPLIT] Canonical expanding-window (8-fold):", flush=True)
        print(f"  baseline mean PR-AUC: {canonical_baseline_mean:.6f}", flush=True)
        print(f"  CTGAN@0.05 mean PR-AUC: {canonical_ctgan_mean:.6f}", flush=True)

    # --- Save results ---
    rows = [
        {
            "split_type": "random_80_20",
            "method": "baseline",
            "pr_auc": pr_baseline,
            "recall_at_1pct_fpr": float(metrics_baseline.get("recall@1%fpr", float("nan"))),
            "n_train": len(train_prep),
            "n_val": len(val_prep),
            "n_fraud_train": int((train_prep[TARGET_COL] == 1).sum()),
            "n_fraud_val": int((val_prep[TARGET_COL] == 1).sum()),
            "ctgan_epochs": float("nan"),
            "target_pos_rate": float("nan"),
            "canonical_8fold_mean_pr_auc": canonical_baseline_mean,
            "delta_vs_canonical": pr_baseline - canonical_baseline_mean,
        },
        {
            "split_type": "random_80_20",
            "method": "ctgan",
            "pr_auc": pr_ctgan,
            "recall_at_1pct_fpr": float(metrics_ctgan.get("recall@1%fpr", float("nan"))),
            "n_train": len(train_prep),
            "n_val": len(val_prep),
            "n_fraud_train": int((train_prep[TARGET_COL] == 1).sum()),
            "n_fraud_val": int((val_prep[TARGET_COL] == 1).sum()),
            "ctgan_epochs": CTGAN_EPOCHS,
            "target_pos_rate": TARGET_POS_RATE,
            "canonical_8fold_mean_pr_auc": canonical_ctgan_mean,
            "delta_vs_canonical": pr_ctgan - canonical_ctgan_mean,
        },
    ]

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)
    print(f"\n[RSPLIT] Results saved to {out_csv}", flush=True)
    print(df_out[["split_type", "method", "pr_auc", "canonical_8fold_mean_pr_auc",
                   "delta_vs_canonical"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
