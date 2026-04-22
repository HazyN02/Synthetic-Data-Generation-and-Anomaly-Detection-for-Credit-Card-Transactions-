# src/run_tvae_protocol.py
"""
Standalone protocol track: TVAE (SDV TVAESynthesizer, 300 epochs).

Runs 8 temporal expanding-window folds, three oversampling rates
(5% / 10% / 20%), with per-class DCR/NNDR fidelity diagnostics on
fraud-only synthetic rows (same hooks as run_protocol.py).

Writes (does NOT overwrite existing files):
  results/protocol/run_<run_id>/results_tvae.csv
  results/protocol/run_<run_id>/fidelity_tvae.csv

Statistical significance (Wilcoxon signed-rank) against baseline is
printed at the end and also written to:
  results/protocol/run_<run_id>/significance_tvae.csv

Usage:
  python -m src.run_tvae_protocol [--run-id 20260330_180216] [--n-folds 8]

The default run-id matches the canonical 8-fold run so TVAE results
land alongside the existing baseline/CTGAN/SMOTE/TabDDPM CSVs.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Allow running as both `python -m src.run_tvae_protocol` and direct exec
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fidelity_eval import fidelity_summary, filter_by_dcr
from src.folds import get_temporal_folds
from src.preprocess_synth import (
    DROP_COLS,
    HASH_COLS,
    MAX_COLS,
    PRIORITY_COLS,
    get_cat_cols_for_synth,
    preprocess_fold,
)
from src.synth_tvae import make_synthetic_positives_tvae
from src.train import train_and_eval

TARGET_COL = "isFraud"
TIME_COL = "TransactionDT"
N_FOLDS = 8
TARGET_POS_RATES = [0.05, 0.10, 0.20]
MAX_SYNTH_POS = 50_000

TVAE_EPOCHS = 300
TVAE_BATCH_SIZE = 500
TVAE_EMBEDDING_DIM = 128
TVAE_COMPRESS_DIMS = (128, 128)
TVAE_DECOMPRESS_DIMS = (128, 128)
TVAE_L2SCALE = 1e-5
TVAE_LOSS_FACTOR = 2
TVAE_SEED = 0


# ---------------------------------------------------------------------------
# Helpers (same pattern as run_smote_tabddpm_protocol.py)
# ---------------------------------------------------------------------------

def _now_str() -> str:
    return datetime.now().strftime("%d-%m-%Y %H:%M")


def _get_metric(res: Dict[str, Any], candidates: List[str]) -> Optional[float]:
    for k in candidates:
        if k in res:
            try:
                return float(res[k])
            except Exception:
                return None
    return None


def _append_row_csv(row: Dict[str, Any], path: str) -> None:
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(path)
    df_row.to_csv(path, mode="a", header=write_header, index=False)


def _load_completed_pairs(path: str) -> set:
    """
    Read results_tvae.csv (if it exists) and return the set of
    (fold, round(target_pos_rate, 4)) pairs already written.
    Used for idempotent resume: skip any (fold, rate) already on disk.
    """
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path, usecols=["fold", "target_pos_rate"])
    except Exception:
        return set()
    pairs = set()
    for _, r in df.iterrows():
        try:
            pairs.add((int(r["fold"]), round(float(r["target_pos_rate"]), 4)))
        except Exception:
            continue
    return pairs


def _select_columns_to_load(all_cols: List[str]) -> List[str]:
    """
    Mirror preprocess_for_synth()'s column-selection logic on the parquet
    schema so we can load only the ~100 columns we'd keep anyway.
    Saves ~3GB of resident memory on this 16GB-total / ~4GB-free box.
    """
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

    # Always include target + time + hash cols (used by preprocessing / folding)
    must_have = {TARGET_COL, TIME_COL, *HASH_COLS}
    for c in must_have:
        if c in all_cols and c not in selected:
            selected.append(c)
    return [c for c in selected if c in all_cols]


def _wilcoxon_vs_baseline(
    results_tvae_csv: str,
    baseline_csv: str,
    target_pos_rate: float = 0.05,
    metric: str = "pr_auc",
) -> Dict[str, Any]:
    """
    Wilcoxon signed-rank test: TVAE(target_rate) vs baseline across folds.
    Returns dict with mean_delta, p_value, significant_005, n_folds.
    """
    from scipy.stats import wilcoxon

    tvae_df = pd.read_csv(results_tvae_csv)
    base_df = pd.read_csv(baseline_csv)

    tvae_sub = tvae_df[tvae_df["target_pos_rate"].round(4) == round(target_pos_rate, 4)]
    base_sub = base_df[base_df["method"] == "baseline"]

    piv_t = tvae_sub.groupby("fold")[metric].mean()
    piv_b = base_sub.groupby("fold")[metric].mean()
    shared = piv_t.index.intersection(piv_b.index)

    if len(shared) < 2:
        return {"mean_delta": np.nan, "p_value": np.nan, "significant_005": False, "n_folds": len(shared)}

    t_vals = piv_t.loc[shared].values
    b_vals = piv_b.loc[shared].values
    deltas = t_vals - b_vals
    mean_delta = float(np.mean(deltas))

    try:
        stat, p = wilcoxon(t_vals, b_vals, alternative="two-sided")
        p = float(p)
    except Exception:
        p = 1.0

    return {
        "mean_delta": mean_delta,
        "p_value": p,
        "significant_005": p < 0.05,
        "n_folds": len(shared),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TVAE standalone protocol — SDV TVAESynthesizer 300 epochs, 8 temporal folds."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="20260330_180216",
        help="Run directory id (results/protocol/run_<id>/...); default=canonical 8-fold run.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=N_FOLDS,
        help="Number of temporal folds (default 8).",
    )
    parser.add_argument(
        "--start-fold",
        type=int,
        default=0,
        help="Resume from this fold index.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TVAE_EPOCHS,
        help="TVAE training epochs (default 300).",
    )
    args = parser.parse_args()

    run_id = args.run_id
    n_folds = int(args.n_folds)
    start_fold = int(args.start_fold)
    epochs = int(args.epochs)

    root = _ROOT
    run_dir = os.path.join(root, "results", "protocol", f"run_{run_id}")

    # Allow running against the canonical run dir OR creating a fresh one
    if not os.path.isdir(run_dir):
        print(f"[TVAE] Run dir not found, creating: {run_dir}")
        os.makedirs(run_dir, exist_ok=True)

    results_csv = os.path.join(run_dir, "results_tvae.csv")
    fidelity_csv = os.path.join(run_dir, "fidelity_tvae.csv")
    significance_csv = os.path.join(run_dir, "significance_tvae.csv")
    baseline_csv = os.path.join(run_dir, "results.csv")  # canonical baseline lives here

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    data_dir = os.path.join(root, "data")
    df = None
    for name in ["train_merged.parquet", "train_transaction.csv"]:
        p = os.path.join(data_dir, name)
        if not os.path.exists(p):
            continue
        print(f"[TVAE] Loading: {p}")
        if p.endswith(".parquet"):
            # Project to only the ~100 columns preprocess_synth would keep.
            # Avoids materializing all 436 columns in memory.
            import pyarrow.parquet as _pq
            schema_cols = _pq.ParquetFile(p).schema_arrow.names
            cols = _select_columns_to_load(schema_cols)
            print(f"[TVAE] Reading {len(cols)} / {len(schema_cols)} columns from parquet")
            df = pd.read_parquet(p, columns=cols)
        else:
            df = pd.read_csv(p)
        break
    if df is None:
        raise FileNotFoundError("No train data found in data/ (need train_merged.parquet or train_transaction.csv)")

    # Downcast numerics to halve memory footprint before folding
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes(include=["int64"]).columns:
        try:
            df[c] = pd.to_numeric(df[c], downcast="integer")
        except Exception:
            pass
    gc.collect()
    try:
        import psutil as _ps
        print(f"[TVAE] df shape={df.shape}  RSS={_ps.Process().memory_info().rss / 1024**2:.0f} MB")
    except Exception:
        print(f"[TVAE] df shape={df.shape}")

    assert TARGET_COL in df.columns, f"Missing target column: {TARGET_COL}"
    assert TIME_COL in df.columns, f"Missing time column: {TIME_COL}"

    folds_raw = get_temporal_folds(df, n_folds=n_folds, time_col=TIME_COL)

    # Free the raw dataframe — each fold dict holds its own copy of train/val
    # (iloc + reset_index(drop=True) materializes copies). Keeping df alive
    # wastes ~4GB of the ~4GB headroom this machine has.
    del df
    gc.collect()

    print(f"\n[TVAE] run_id={run_id} | folds={n_folds} | rates={TARGET_POS_RATES} | epochs={epochs}")
    print(f"[TVAE] results  -> {results_csv}")
    print(f"[TVAE] fidelity -> {fidelity_csv}")
    print(f"[TVAE] SDV TVAESynthesizer config: batch={TVAE_BATCH_SIZE}, "
          f"embed={TVAE_EMBEDDING_DIM}, compress={TVAE_COMPRESS_DIMS}, seed={TVAE_SEED}\n")

    fidelity_rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    # Resume support: skip (fold, rate) pairs already on disk
    completed_pairs = _load_completed_pairs(results_csv)
    if completed_pairs:
        print(f"[TVAE] Resume: {len(completed_pairs)} (fold, rate) pairs already in CSV — will be skipped.")

    # ------------------------------------------------------------------
    # Fold loop
    # ------------------------------------------------------------------
    for fold_info in folds_raw:
        fold = fold_info["fold"]
        if fold < start_fold:
            continue

        train_df = fold_info["train_df"]
        val_df = fold_info["val_df"]

        if len(val_df) == 0:
            print(f"[WARN] Fold {fold} has empty val set — skipping.")
            continue

        # Skip the whole fold if every rate for it is already done
        remaining_rates_for_fold = [
            r for r in TARGET_POS_RATES
            if (int(fold), round(float(r), 4)) not in completed_pairs
        ]
        if not remaining_rates_for_fold:
            print(f"[TVAE] Fold {fold}: all rates already completed — skipping.")
            continue

        print("\n" + "=" * 60)
        print(f"===== FOLD {fold} (tvae) =====")

        train_df, val_df, used_cols = preprocess_fold(train_df, val_df)
        cat_cols = get_cat_cols_for_synth(train_df, used_cols)
        n_train_pos = int((train_df[TARGET_COL] == 1).sum())
        n_train_neg = int((train_df[TARGET_COL] == 0).sum())
        print(f"[INFO] train={len(train_df)}, val={len(val_df)}, features={len(used_cols)}, "
              f"train_pos={n_train_pos}")

        if n_train_pos < 50:
            print(f"[WARN] Fold {fold}: too few positives ({n_train_pos}) — skipping.")
            continue

        # ----------------------------------------------------------
        # Per-rate loop
        # ----------------------------------------------------------
        for target_rate in TARGET_POS_RATES:
            if (int(fold), round(float(target_rate), 4)) in completed_pairs:
                print(f"\n[TVAE] Fold {fold}, rate {target_rate}: already done — skipping.")
                continue

            print("\n" + "-" * 50)
            print(f"[TVAE] target_pos_rate={target_rate}")

            synth_pos = make_synthetic_positives_tvae(
                train_df=train_df,
                cat_cols=cat_cols,
                used_cols=used_cols,
                target_pos_rate=target_rate,
                max_synth=MAX_SYNTH_POS,
                epochs=epochs,
                batch_size=TVAE_BATCH_SIZE,
                embedding_dim=TVAE_EMBEDDING_DIM,
                compress_dims=TVAE_COMPRESS_DIMS,
                decompress_dims=TVAE_DECOMPRESS_DIMS,
                l2scale=TVAE_L2SCALE,
                loss_factor=TVAE_LOSS_FACTOR,
                seed=TVAE_SEED,
                verbose=True,
            )

            n_synth = int(len(synth_pos))

            # ----------------------------------------------------------
            # Fidelity diagnostics (fraud samples only, same as CTGAN path)
            # ----------------------------------------------------------
            if n_synth > 0:
                real_fraud = train_df[train_df[TARGET_COL] == 1]
                real_legit = train_df[train_df[TARGET_COL] == 0]

                fsum = fidelity_summary(
                    synthetic_fraud=synth_pos,
                    real_fraud=real_fraud,
                    real_legit=real_legit,
                    method="tvae",
                    fold=fold,
                )

                # p90 DCR filter (diagnostic only — ungated run uses full synth_pos)
                synth_filtered = filter_by_dcr(synth_pos, real_fraud, percentile=90)
                n_after = int(len(synth_filtered))
                print(
                    f"[FIDELITY][tvae][fold={fold}] "
                    f"survived={n_after}, discarded={n_synth - n_after}"
                )

                fidelity_rows.append(
                    {
                        "method": "tvae",
                        "fold": int(fold),
                        "target_pos_rate": float(target_rate),
                        "mean_dcr": fsum.get("dcr_mean"),
                        "median_dcr": fsum.get("dcr_median"),
                        "p95_dcr": fsum.get("dcr_p95"),
                        "mean_nndr": fsum.get("nndr_mean"),
                        "median_nndr": fsum.get("nndr_median"),
                        "ks_mean": fsum.get("ks_mean"),
                        "ks_max": fsum.get("ks_max"),
                        "n_synthetic": n_synth,
                        "n_after_dcr_filter": n_after,
                    }
                )

            # ----------------------------------------------------------
            # Train + evaluate (ungated: full synth_pos mixed with real)
            # ----------------------------------------------------------
            mixed_train = pd.concat([train_df, synth_pos], axis=0, ignore_index=True)
            res = train_and_eval(mixed_train, val_df)

            pr_auc_m = _get_metric(res, ["pr_auc", "prauc", "prAUC"])
            recall_m = _get_metric(res, ["recall_at_1pct_fpr", "recall@1%fpr", "recall_at_1fpr", "recall_at_1_fpr"])

            print(f"TVAE+REAL PR-AUC: {pr_auc_m:.4f}, Recall@1%FPR: {recall_m:.4f}")

            _append_row_csv(
                {
                    "timestamp": _now_str(),
                    "fold": fold,
                    "delay_days": 0,
                    "run_id": run_id,
                    "method": "tvae",
                    "target_pos_rate": float(target_rate),
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "train_pos": n_train_pos,
                    "train_neg": n_train_neg,
                    "synth_rows": n_synth,
                    "final_train_rows": int(len(mixed_train)),
                    "final_pos_rate": float((mixed_train[TARGET_COL] == 1).mean()),
                    "pr_auc": pr_auc_m,
                    "recall_at_1pct_fpr": recall_m,
                    "notes": f"tvae_epochs={epochs}",
                },
                results_csv,
            )

            # Free per-rate temporaries
            del synth_pos, mixed_train, res
            gc.collect()

        # Free this fold's dataframes before moving to next fold
        del train_df, val_df
        fold_info["train_df"] = None
        fold_info["val_df"] = None
        gc.collect()

    # ------------------------------------------------------------------
    # Flush fidelity rows
    # ------------------------------------------------------------------
    if fidelity_rows:
        fid_df = pd.DataFrame(fidelity_rows)
        write_header = not os.path.exists(fidelity_csv)
        fid_df.to_csv(fidelity_csv, mode="a", header=write_header, index=False)
        print(f"\n[FIDELITY] wrote -> {fidelity_csv} ({len(fidelity_rows)} rows)")

    # ------------------------------------------------------------------
    # Statistical significance: Wilcoxon vs baseline (per rate)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[SIGNIFICANCE] Wilcoxon signed-rank test vs baseline")
    sig_rows = []

    if os.path.exists(results_csv) and os.path.exists(baseline_csv):
        tvae_df = pd.read_csv(results_csv)
        base_df = pd.read_csv(baseline_csv)

        base_sub = base_df[base_df["method"] == "baseline"]
        piv_b = base_sub.groupby("fold")["pr_auc"].mean()

        for rate in TARGET_POS_RATES:
            tvae_sub = tvae_df[tvae_df["target_pos_rate"].round(4) == round(rate, 4)]
            piv_t = tvae_sub.groupby("fold")["pr_auc"].mean()
            shared = piv_t.index.intersection(piv_b.index)

            if len(shared) < 2:
                print(f"  rate={rate}: not enough folds for test (n={len(shared)})")
                continue

            t_vals = piv_t.loc[shared].values
            b_vals = piv_b.loc[shared].values
            deltas = t_vals - b_vals
            mean_delta = float(np.mean(deltas))

            try:
                from scipy.stats import wilcoxon as _wil
                _, p = _wil(t_vals, b_vals, alternative="two-sided")
                p = float(p)
            except Exception:
                p = 1.0

            sig = p < 0.05
            fold_deltas_str = ", ".join([f"{d:+.4f}" for d in deltas])
            print(
                f"  rate={rate:.2f} | mean_delta={mean_delta:+.4f} | "
                f"p={p:.4f} | sig={'YES' if sig else 'no'} | "
                f"n_folds={len(shared)}"
            )
            print(f"    fold deltas: [{fold_deltas_str}]")

            sig_rows.append(
                {
                    "comparison": f"tvae_vs_baseline",
                    "target_pos_rate": float(rate),
                    "mean_delta_pr_auc": mean_delta,
                    "p_value": p,
                    "significant_at_0.05": sig,
                    "n_folds": len(shared),
                    "fold_deltas": fold_deltas_str,
                }
            )
    else:
        print(f"  [WARN] Missing results files for significance test.")
        if not os.path.exists(baseline_csv):
            print(f"  Baseline CSV not found: {baseline_csv}")
            print(f"  Run the canonical protocol first or point --run-id to a dir with results.csv")

    if sig_rows:
        sig_df = pd.DataFrame(sig_rows)
        sig_df.to_csv(significance_csv, index=False)
        print(f"\n[SIGNIFICANCE] wrote -> {significance_csv}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[RESULTS SUMMARY] Mean PR-AUC by target_pos_rate")
    if os.path.exists(results_csv):
        tvae_df = pd.read_csv(results_csv)
        summary = (
            tvae_df.groupby("target_pos_rate")["pr_auc"]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "mean_pr_auc", "std": "std_pr_auc", "count": "n_folds"})
            .round(4)
        )
        print(summary.to_string())
        best_rate = summary["mean_pr_auc"].idxmax()
        best_pr_auc = summary.loc[best_rate, "mean_pr_auc"]
        print(f"\n  Best rate: {best_rate} -> mean PR-AUC = {best_pr_auc:.4f}")

    elapsed = int(time.perf_counter() - t0)
    print(f"\n[DONE] tvae elapsed: {elapsed // 60}m {elapsed % 60}s")


if __name__ == "__main__":
    main()
