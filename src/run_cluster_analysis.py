# src/run_cluster_analysis.py
"""
Standalone runner for the fraud cluster analysis (KMeans on fraud feature space,
per-cluster baseline-vs-CTGAN PR-AUC delta, per-cluster DCR for CTGAN synth).

Why this exists separately from fraud_cluster_analysis.py:
  - That script does pd.read_parquet(...) of the full 590k x 436 col table and
    OOMs on this 16GB box (same failure mode we hit on the TVAE protocol).
  - This runner re-uses every helper in fraud_cluster_analysis.py but loads the
    parquet with column projection (~100 cols) + float32 downcast (same pattern
    as src/run_tvae_protocol.py), and drops the SMOTE leg (the "why" question
    is about CTGAN vs baseline, per the paper's failure-mode analysis).

Outputs (next to the canonical run dir):
  results/protocol/run_<run_id>/cluster_per_fold.csv     long-form rows
  results/protocol/run_<run_id>/cluster_summary.csv      per-cluster aggregate

Usage:
  python -m src.run_cluster_analysis [--run-id 20260330_180216] [--n-folds 8]
                                     [--ctgan-epochs 150] [--n-clusters 5]
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# --- helpers from the original cluster analysis script ---------------------
from fraud_cluster_analysis import (
    align_df_to_used_cols,
    fit_kmeans_global,
    fold_val_index_range,
    load_best_target_rates_per_fold,
    pr_auc_cluster_slice,
    transform_for_cluster_model,
    _lgbm_val_predictions,
    CTGAN_EPOCHS,
    CTGAN_BATCH_SIZE,
    CTGAN_DISCRIMINATOR_STEPS,
    CTGAN_PAC,
    CTGAN_SEED,
    MAX_SYNTH_POS,
    N_CLUSTERS,
)
from fidelity_eval import compute_dcr
from src.folds import get_temporal_folds
from src.preprocess_synth import (
    DROP_COLS,
    HASH_COLS,
    MAX_COLS,
    PRIORITY_COLS,
    TARGET_COL,
    TIME_COL,
    get_cat_cols_for_synth,
    preprocess_fold,
    preprocess_for_synth,
)
from src.synth_ctgan import make_synthetic_positives


# ---------- column-projected load (same logic as run_tvae_protocol) -------

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
        print(f"[CLUSTER] Loading: {p}")
        if p.endswith(".parquet"):
            import pyarrow.parquet as _pq
            schema_cols = _pq.ParquetFile(p).schema_arrow.names
            cols = _select_columns_to_load(schema_cols)
            print(f"[CLUSTER] Reading {len(cols)} / {len(schema_cols)} columns from parquet")
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
        try:
            import psutil as _ps
            print(f"[CLUSTER] df shape={df.shape}  RSS={_ps.Process().memory_info().rss / 1024**2:.0f} MB")
        except Exception:
            print(f"[CLUSTER] df shape={df.shape}")
        return df
    raise FileNotFoundError("No train data found in data/")


# -------------------------------------------------------------------------

def _make_summary(rows: List[Dict[str, Any]], n_clusters: int) -> pd.DataFrame:
    """
    Roll up per-fold-per-cluster rows into one row per cluster:
      - fraud_count_global  : how many real fraud rows landed in cluster (from the full-data fit)
      - mean_dcr_ctgan      : mean (over folds) of mean DCR of CTGAN synth points routed to cluster
      - mean_baseline_pr_auc: mean across folds of cluster-slice PR-AUC for baseline LightGBM
      - mean_ctgan_pr_auc   : same for CTGAN-augmented LightGBM
      - mean_delta_pr_auc   : mean_ctgan - mean_baseline (per-cluster mechanistic effect)
      - n_synth_per_cluster : mean number of CTGAN synth rows routed to each cluster
    """
    df = pd.DataFrame(rows)
    out = []

    # 1. fraud-count per cluster (from fraud_cluster_assignment rows)
    assn = df[df["record_type"] == "fraud_cluster_assignment"]
    pra = df[df["record_type"] == "pr_auc_per_cluster"]
    dcr = df[df["record_type"] == "dcr_ctgan_synthetic"]

    for c in range(n_clusters):
        n_fraud_global = int((assn["cluster_id"] == c).sum())

        base_sub = pra[(pra["cluster_id"] == c) & (pra["method"] == "baseline")]
        ctg_sub = pra[(pra["cluster_id"] == c) & (pra["method"] == "ctgan")]
        # per-fold delta
        delta_per_fold = []
        for f in sorted(base_sub["fold"].unique()):
            b = base_sub[base_sub["fold"] == f]["pr_auc"].mean()
            t = ctg_sub[ctg_sub["fold"] == f]["pr_auc"].mean()
            if pd.notna(b) and pd.notna(t):
                delta_per_fold.append(t - b)

        dcr_sub = dcr[dcr["cluster_id"] == c]

        out.append({
            "cluster_id": c,
            "fraud_count_global": n_fraud_global,
            "fraud_share_global": (
                n_fraud_global / max(1, len(assn))
            ),
            "n_folds_evaluated": int(base_sub["fold"].nunique()),
            "mean_baseline_pr_auc": float(base_sub["pr_auc"].mean(skipna=True)),
            "mean_ctgan_pr_auc": float(ctg_sub["pr_auc"].mean(skipna=True)),
            "mean_delta_pr_auc": float(np.mean(delta_per_fold)) if delta_per_fold else float("nan"),
            "std_delta_pr_auc": float(np.std(delta_per_fold, ddof=1)) if len(delta_per_fold) > 1 else float("nan"),
            "mean_n_synth_routed": float(dcr_sub["n_synthetic"].mean(skipna=True))
            if not dcr_sub.empty else float("nan"),
            "mean_dcr_synth": float(dcr_sub["dcr_mean"].mean(skipna=True))
            if not dcr_sub.empty else float("nan"),
            "median_dcr_synth": float(dcr_sub["dcr_median"].mean(skipna=True))
            if not dcr_sub.empty else float("nan"),
        })

    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-cluster fraud diagnostics: baseline vs CTGAN PR-AUC + DCR."
    )
    ap.add_argument("--run-id", default="20260330_180216")
    ap.add_argument("--n-folds", type=int, default=8)
    ap.add_argument("--n-clusters", type=int, default=N_CLUSTERS)
    ap.add_argument("--ctgan-epochs", type=int, default=CTGAN_EPOCHS)
    args = ap.parse_args()

    run_dir = os.path.join(_ROOT, "results", "protocol", f"run_{args.run_id}")
    os.makedirs(run_dir, exist_ok=True)
    long_csv = os.path.join(run_dir, "cluster_per_fold.csv")
    summary_csv = os.path.join(run_dir, "cluster_summary.csv")
    results_csv = os.path.join(run_dir, "results.csv")

    if not os.path.isfile(results_csv):
        raise FileNotFoundError(f"Need {results_csv} for best CTGAN rates")

    best_rates = load_best_target_rates_per_fold(results_csv, ("ctgan",))

    # Load raw data
    df_raw = _load_train_data(os.path.join(_ROOT, "data"))
    df_sorted = df_raw.sort_values(TIME_COL).reset_index(drop=True)
    del df_raw
    gc.collect()
    n_total = len(df_sorted)

    full_prep, used_cols_global = preprocess_for_synth(df_sorted)
    n_fraud = int((full_prep[TARGET_COL] == 1).sum())
    print(f"[CLUSTER] preprocessed: rows={len(full_prep)}, cols={len(used_cols_global)}, fraud={n_fraud}")

    print(f"[CLUSTER] Fitting KMeans(k={args.n_clusters}) on fraud feature space ...")
    km, scaler, enc, obj_cols, num_cols, clusters_global = fit_kmeans_global(
        full_prep, used_cols_global, args.n_clusters
    )

    rows_out: List[Dict[str, Any]] = []

    # global cluster assignment (one row per real fraud)
    fraud_idx = np.where(full_prep[TARGET_COL].to_numpy() == 1)[0]
    for gix in fraud_idx:
        rows_out.append({
            "record_type": "fraud_cluster_assignment",
            "fold": np.nan,
            "cluster_id": int(clusters_global[gix]),
            "global_row_idx": int(gix),
            "method": "",
            "target_pos_rate": np.nan,
            "pr_auc": np.nan,
            "n_pos_in_slice": np.nan,
            "n_neg_in_slice": np.nan,
            "dcr_mean": np.nan,
            "dcr_median": np.nan,
            "n_synthetic": np.nan,
            "notes": "KMeans on full-data preprocessed fraud",
        })

    # cluster size summary up front (so we can read it even if folds blow up)
    sizes = pd.Series(clusters_global[clusters_global >= 0]).value_counts().sort_index()
    print("\n[CLUSTER] Global fraud per cluster:")
    for cid, n in sizes.items():
        print(f"   cluster {cid}: {n} ({100.0 * n / n_fraud:.1f}%)")

    folds = get_temporal_folds(df_sorted, n_folds=args.n_folds, time_col=TIME_COL)

    for fold_info in folds:
        fold = int(fold_info["fold"])
        train_df = fold_info["train_df"]
        val_df = fold_info["val_df"]
        if len(val_df) == 0:
            continue

        t0 = time.perf_counter()
        print("\n" + "=" * 60)
        print(f"===== FOLD {fold} =====")

        train_prep, val_prep, used_cols_fold = preprocess_fold(train_df, val_df)
        cat_cols = get_cat_cols_for_synth(train_prep, used_cols_fold)

        va_start, va_end = fold_val_index_range(n_total, args.n_folds, fold)
        if va_end - va_start != len(val_prep):
            print(f"[WARN] Fold {fold}: val slice mismatch "
                  f"({va_end - va_start} vs {len(val_prep)}) — skipping fold.")
            continue

        val_cluster = np.full(len(val_prep), -1, dtype=int)
        target_arr = full_prep[TARGET_COL].to_numpy()
        for local_i in range(len(val_prep)):
            g = va_start + local_i
            if int(target_arr[g]) == 1:
                val_cluster[local_i] = int(clusters_global[g])

        rate = best_rates.get("ctgan", {}).get(fold, 0.10)

        # baseline
        p0, y0 = _lgbm_val_predictions(train_prep, val_prep)

        # CTGAN
        synth = make_synthetic_positives(
            train_df=train_prep,
            cat_cols=cat_cols,
            used_cols=used_cols_fold,
            target_pos_rate=float(rate),
            max_synth=MAX_SYNTH_POS,
            epochs=args.ctgan_epochs,
            batch_size=CTGAN_BATCH_SIZE,
            pac=CTGAN_PAC,
            seed=CTGAN_SEED,
            discriminator_steps=CTGAN_DISCRIMINATOR_STEPS,
            verbose=False,
        )
        mixed = pd.concat([train_prep, synth], axis=0, ignore_index=True)
        pc, yc = _lgbm_val_predictions(mixed, val_prep)

        for method, pred, y_va, trate in [
            ("baseline", p0, y0, None),
            ("ctgan", pc, yc, float(rate)),
        ]:
            for c in range(args.n_clusters):
                pr_c, np_c, nn_c = pr_auc_cluster_slice(y_va, pred, val_cluster, c)
                rows_out.append({
                    "record_type": "pr_auc_per_cluster",
                    "fold": fold,
                    "cluster_id": c,
                    "global_row_idx": np.nan,
                    "method": method,
                    "target_pos_rate": trate if trate is not None else np.nan,
                    "pr_auc": pr_c,
                    "n_pos_in_slice": np_c,
                    "n_neg_in_slice": nn_c,
                    "dcr_mean": np.nan,
                    "dcr_median": np.nan,
                    "n_synthetic": np.nan,
                    "notes": "subset = all val negatives U val fraud in cluster",
                })

        # per-cluster DCR for CTGAN synth
        if len(synth) > 0:
            real_fraud = train_prep[train_prep[TARGET_COL] == 1]
            dcr = compute_dcr(synth, real_fraud)
            synth_aligned = align_df_to_used_cols(synth, used_cols_global, full_prep)
            try:
                synth_clusters = transform_for_cluster_model(
                    synth_aligned, used_cols_global, km, scaler, enc, obj_cols, num_cols
                )
            except Exception as e:
                print(f"[WARN] fold {fold}: synth cluster transform failed ({e}); using -1 for all")
                synth_clusters = np.full(len(synth), -1, dtype=int)

            for c in range(args.n_clusters):
                m = synth_clusters == c
                n_s = int(m.sum())
                if n_s == 0:
                    d_mean, d_med = float("nan"), float("nan")
                else:
                    d_mean = float(np.mean(dcr[m]))
                    d_med = float(np.median(dcr[m]))
                rows_out.append({
                    "record_type": "dcr_ctgan_synthetic",
                    "fold": fold,
                    "cluster_id": c,
                    "global_row_idx": np.nan,
                    "method": "ctgan",
                    "target_pos_rate": float(rate),
                    "pr_auc": np.nan,
                    "n_pos_in_slice": np.nan,
                    "n_neg_in_slice": np.nan,
                    "dcr_mean": d_mean,
                    "dcr_median": d_med,
                    "n_synthetic": n_s,
                    "notes": "DCR vs fold-train real fraud; KMeans.predict on synth features",
                })

        # checkpoint
        pd.DataFrame(rows_out).to_csv(long_csv, index=False)
        summary = _make_summary(rows_out, args.n_clusters)
        summary.to_csv(summary_csv, index=False)
        print(f"[CLUSTER] fold {fold} done in {time.perf_counter() - t0:.1f}s "
              f"-> wrote {len(rows_out)} rows ({long_csv})")

        # free
        del train_prep, val_prep, train_df, val_df, synth, mixed
        gc.collect()

    # final summary
    summary = _make_summary(rows_out, args.n_clusters)
    summary.to_csv(summary_csv, index=False)
    print("\n[CLUSTER] === per-cluster summary ===")
    print(summary.to_string(index=False))
    print(f"\n[CLUSTER] wrote {summary_csv}")
    print(f"[CLUSTER] wrote {long_csv}")


if __name__ == "__main__":
    main()
