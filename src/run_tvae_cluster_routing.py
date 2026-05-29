# src/run_tvae_cluster_routing.py
"""
TVAE cluster routing analysis — equivalent of Table 10 (CTGAN) for TVAE.

For each of the 8 temporal folds:
  1. Loads fold training data
  2. Trains TVAE (300 epochs, same canonical config as run_tvae_protocol.py)
  3. Generates synthetic fraud at r=0.05
  4. Routes each synthetic sample to its nearest real fraud neighbour's cluster
     (using the global HDBSCAN partition from cluster_per_fold_hdbscan.csv)
  5. Computes per-cluster DCR (mean L2 distance to nearest real fraud)
  6. Records aggregate fold PR-AUC delta from results_tvae.csv

Outputs:
  results/tvae_cluster_routing.csv  -- per-fold × per-cluster routing stats
  results/tvae_cluster_routing_summary.csv -- averaged across folds

Usage:
  python -m src.run_tvae_cluster_routing [--run-dir PATH] [--fold N]
"""
from __future__ import annotations

import argparse
import os
import sys
import csv
import gc
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.folds import get_temporal_folds
from src.preprocess_synth import (
    TARGET_COL, TIME_COL, DROP_COLS, HASH_COLS,
    PRIORITY_COLS, MAX_COLS,
    get_cat_cols_for_synth, preprocess_for_synth,
)
from src.synth_tvae import fit_tvae, sample_tvae

# ── Config ────────────────────────────────────────────────────────────────────
PARQUET_PATH  = os.path.join(_ROOT, "data", "train_merged.parquet")
RUN_DIR       = os.path.join(_ROOT, "results", "protocol", "run_20260330_180216")
OUT_CSV       = os.path.join(_ROOT, "results", "tvae_cluster_routing.csv")
SUMMARY_CSV   = os.path.join(_ROOT, "results", "tvae_cluster_routing_summary.csv")

N_FOLDS       = 8
TVAE_EPOCHS   = 300
TVAE_SEED     = 0
TARGET_RATE   = 0.05
MAX_SYNTH     = 50000

COLS_TO_READ  = 100   # column-projected parquet (avoid OOM)


# ── Column-projected parquet load ─────────────────────────────────────────────
def _load_parquet_projected(path: str, n_cols: int = COLS_TO_READ) -> pd.DataFrame:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    all_cols = pf.schema_arrow.names
    priority = [c for c in PRIORITY_COLS if c in all_cols]
    remaining = [c for c in all_cols if c not in priority]
    selected = priority + remaining
    selected = selected[:n_cols]
    if TIME_COL not in selected:
        selected = [TIME_COL] + selected
    if TARGET_COL not in selected:
        selected = [TARGET_COL] + selected
    selected = list(dict.fromkeys(selected))
    print(f"[TVAE-ROUTING] Reading {len(selected)} / {len(all_cols)} columns")
    return pf.read(columns=selected).to_pandas()


# ── Load global HDBSCAN cluster assignments ───────────────────────────────────
def _load_cluster_assignments(run_dir: str):
    """
    Returns:
      global_row_idx : np.ndarray of shape (20663,) — row indices in original df
      cluster_labels : np.ndarray of shape (20663,) — HDBSCAN label (-1 = noise)
    """
    path = os.path.join(run_dir, "cluster_per_fold_hdbscan.csv")
    df = pd.read_csv(path)
    fraud_assign = df[df["record_type"] == "fraud_cluster_assignment"].copy()
    fraud_assign = fraud_assign.dropna(subset=["global_row_idx", "cluster_id"])
    global_row_idx = fraud_assign["global_row_idx"].astype(int).values
    cluster_labels = fraud_assign["cluster_id"].astype(int).values
    print(f"[TVAE-ROUTING] Loaded {len(global_row_idx)} cluster assignments, "
          f"clusters={sorted(set(cluster_labels))}")
    return global_row_idx, cluster_labels


# ── Load per-fold TVAE PR-AUC from results_tvae.csv ──────────────────────────
def _load_tvae_results(run_dir: str) -> Dict[int, float]:
    path = os.path.join(run_dir, "results_tvae.csv")
    df = pd.read_csv(path)
    baseline_path = os.path.join(run_dir, "results.csv")
    base_df = pd.read_csv(baseline_path)
    baseline = base_df[base_df["method"] == "baseline"].set_index("fold")["pr_auc"]
    tvae_005 = df[(df["method"] == "tvae") & (df["target_pos_rate"] == 0.05)].set_index("fold")["pr_auc"]
    deltas = {}
    for fold in range(N_FOLDS):
        if fold in tvae_005.index and fold in baseline.index:
            deltas[fold] = float(tvae_005[fold] - baseline[fold])
    return deltas


# ── Resume: check which folds are already done ────────────────────────────────
def _completed_folds(out_csv: str):
    if not os.path.exists(out_csv):
        return set()
    df = pd.read_csv(out_csv)
    return set(df["fold"].unique().tolist())


# ── Main routing loop ─────────────────────────────────────────────────────────
def main(run_dir: str = RUN_DIR, only_fold: int = -1):
    print(f"[TVAE-ROUTING] Loading data from {PARQUET_PATH}")
    df_raw = _load_parquet_projected(PARQUET_PATH)
    print(f"[TVAE-ROUTING] df shape={df_raw.shape}")

    # Global cluster assignments (index into time-sorted full df)
    global_row_idx, cluster_labels = _load_cluster_assignments(run_dir)
    cluster_ids = sorted(set(cluster_labels))

    # Per-fold TVAE PR-AUC deltas
    tvae_deltas = _load_tvae_results(run_dir)
    print(f"[TVAE-ROUTING] TVAE deltas available for folds: {sorted(tvae_deltas)}")

    # Temporal folds (time-sorted, same split as main protocol)
    folds = get_temporal_folds(df_raw, n_folds=N_FOLDS, time_col=TIME_COL)

    done = _completed_folds(OUT_CSV)
    print(f"[TVAE-ROUTING] Already done folds: {sorted(done)}")

    # Output CSV writer
    fieldnames = [
        "fold", "cluster_id", "n_real_fraud_in_cluster",
        "pct_real_fraud", "n_synth_routed", "pct_synth_routed",
        "mean_dcr", "median_dcr", "delta_pr_auc_fold",
    ]
    write_header = not os.path.exists(OUT_CSV)
    out_f = open(OUT_CSV, "a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    # Global fraud counts per cluster
    global_fraud_counts = {c: int((cluster_labels == c).sum()) for c in cluster_ids}
    total_fraud = len(cluster_labels)

    for fold_info in folds:
        fold = fold_info["fold"]
        if only_fold >= 0 and fold != only_fold:
            continue
        if fold in done:
            print(f"[TVAE-ROUTING] Fold {fold} already done — skipping.")
            continue

        print(f"\n[TVAE-ROUTING] ===== Fold {fold} ({fold+1}/{N_FOLDS}) =====")
        train_df = fold_info["train_df"]

        # Preprocess for synthesis
        train_proc, used_cols = preprocess_for_synth(train_df)
        cat_cols = get_cat_cols_for_synth(train_proc, used_cols)
        fraud_train = train_proc[train_proc[TARGET_COL] == 1]
        print(f"  fraud_train rows: {len(fraud_train)}")

        n_neg = int((train_proc[TARGET_COL] == 0).sum())
        n_pos = len(fraud_train)
        synth_add = int(n_neg * TARGET_RATE / (1 - TARGET_RATE)) - n_pos
        synth_add = max(0, min(synth_add, MAX_SYNTH))
        print(f"  synth_add={synth_add}")

        if synth_add == 0 or n_pos < 50:
            print(f"  Skipping fold {fold}: synth_add={synth_add}, n_pos={n_pos}")
            continue

        # ── Train TVAE ────────────────────────────────────────────────────
        print(f"  Training TVAE ({TVAE_EPOCHS} epochs)...")
        synth_cols = [c for c in used_cols if c != TARGET_COL]
        synth_input = fraud_train[synth_cols].copy()
        # Add target col back (TVAE expects it in _prep_for_tvae)
        synth_input_with_target = fraud_train[used_cols].copy()

        synthesizer, artifacts = fit_tvae(
            synth_input_with_target,
            cat_cols=cat_cols,
            used_cols=used_cols,
            epochs=TVAE_EPOCHS,
            batch_size=500,
            seed=TVAE_SEED,
            verbose=True,
        )

        # ── Generate synthetic fraud ──────────────────────────────────────
        print(f"  Generating {synth_add} synthetic fraud rows...")
        synth_df = sample_tvae(synthesizer, n=synth_add, artifacts=artifacts, verbose=True)
        del synthesizer
        gc.collect()

        # ── Get real fraud in training set + their cluster labels ─────────
        # Map global_row_idx to the rows in the time-sorted df_raw that fall in train_df
        # train_df is df_raw rows [0 : tr_end], same time-sort order
        tr_end = len(train_df)
        # global_row_idx are absolute indices into the time-sorted full df
        mask_in_train = global_row_idx < tr_end
        train_fraud_global_idx = global_row_idx[mask_in_train]
        train_fraud_labels = cluster_labels[mask_in_train]

        # Get feature matrix for train fraud (preprocessed, cont cols only for DCR)
        # Use the same preprocessing
        real_fraud_full_proc = train_proc[train_proc[TARGET_COL] == 1].copy()
        cont_cols = [c for c in artifacts.cont_cols if c in real_fraud_full_proc.columns]

        if len(cont_cols) == 0:
            print(f"  WARNING: no continuous cols for DCR — using all synth cols")
            cont_cols = [c for c in artifacts.used_cols
                         if c != TARGET_COL and c in real_fraud_full_proc.columns]

        # Scale for distance computation
        scaler = StandardScaler()
        real_matrix = scaler.fit_transform(
            real_fraud_full_proc[cont_cols].fillna(0).values
        )
        synth_cont = synth_df[[c for c in cont_cols if c in synth_df.columns]].fillna(0)
        # Align columns
        missing = [c for c in cont_cols if c not in synth_cont.columns]
        for c in missing:
            synth_cont[c] = 0.0
        synth_matrix = scaler.transform(synth_cont[cont_cols].values)

        # ── Nearest-neighbour routing ─────────────────────────────────────
        print(f"  Running nearest-neighbour routing ({len(synth_matrix)} synth -> {len(real_matrix)} real)...")
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean", algorithm="auto", n_jobs=1)
        nn.fit(real_matrix)
        distances, nn_indices = nn.kneighbors(synth_matrix)
        distances = distances.flatten()
        nn_indices = nn_indices.flatten()

        # Route each synthetic sample to its nearest real fraud's cluster
        routed_labels = train_fraud_labels[nn_indices]

        # ── Per-cluster stats ─────────────────────────────────────────────
        fold_delta = tvae_deltas.get(fold, float("nan"))
        n_synth_total = len(routed_labels)

        for c in cluster_ids:
            mask_c = routed_labels == c
            n_synth_c = int(mask_c.sum())
            dcr_vals = distances[mask_c]
            mean_dcr = float(np.mean(dcr_vals)) if n_synth_c > 0 else float("nan")
            median_dcr = float(np.median(dcr_vals)) if n_synth_c > 0 else float("nan")
            pct_synth = n_synth_c / n_synth_total * 100 if n_synth_total > 0 else 0.0

            writer.writerow({
                "fold": fold,
                "cluster_id": c,
                "n_real_fraud_in_cluster": global_fraud_counts[c],
                "pct_real_fraud": round(global_fraud_counts[c] / total_fraud * 100, 1),
                "n_synth_routed": n_synth_c,
                "pct_synth_routed": round(pct_synth, 1),
                "mean_dcr": round(mean_dcr, 1) if not np.isnan(mean_dcr) else "",
                "median_dcr": round(median_dcr, 1) if not np.isnan(median_dcr) else "",
                "delta_pr_auc_fold": round(fold_delta, 4) if not np.isnan(fold_delta) else "",
            })
        out_f.flush()
        print(f"  Fold {fold} written.")
        del synth_df, real_matrix, synth_matrix
        gc.collect()

    out_f.close()
    print(f"\n[TVAE-ROUTING] Done. Saved: {OUT_CSV}")
    _print_summary()


def _print_summary():
    if not os.path.exists(OUT_CSV):
        return
    df = pd.read_csv(OUT_CSV)
    cluster_ids = sorted(df["cluster_id"].unique())
    total_fraud = df[["cluster_id", "n_real_fraud_in_cluster"]].drop_duplicates()
    total_fraud_n = int(total_fraud["n_real_fraud_in_cluster"].sum())

    print("\n=== TVAE CLUSTER ROUTING SUMMARY (mean across folds) ===")
    print(f"{'Cluster':>10}  {'Fraud n':>8}  {'Fraud%':>7}  "
          f"{'Synth/fold':>10}  {'Synth%':>7}  {'Mean DCR':>10}")
    print("-" * 65)

    rows = []
    for c in cluster_ids:
        sub = df[df["cluster_id"] == c]
        n_real = int(sub["n_real_fraud_in_cluster"].iloc[0])
        pct_real = round(n_real / total_fraud_n * 100, 1)
        mean_synth = sub["n_synth_routed"].mean()
        total_synth = df.groupby("fold")["n_synth_routed"].sum()
        pct_synth_vals = []
        for fold in sub["fold"].values:
            total_f = total_synth.get(fold, 0)
            n_f = sub[sub["fold"] == fold]["n_synth_routed"].values
            if len(n_f) and total_f > 0:
                pct_synth_vals.append(n_f[0] / total_f * 100)
        mean_pct_synth = np.mean(pct_synth_vals) if pct_synth_vals else 0.0
        dcr_vals = pd.to_numeric(sub["mean_dcr"], errors="coerce").dropna()
        mean_dcr = dcr_vals.mean() if len(dcr_vals) else float("nan")
        label = "noise" if c == -1 else str(c)
        print(f"  {label:>8}  {n_real:>8,}  {pct_real:>6.1f}%  "
              f"{mean_synth:>10.0f}  {mean_pct_synth:>6.1f}%  {mean_dcr:>10.1f}")
        rows.append({
            "cluster_id": c,
            "label": label,
            "n_real_fraud": n_real,
            "pct_real_fraud": pct_real,
            "mean_synth_per_fold": round(mean_synth, 1),
            "mean_pct_synth": round(mean_pct_synth, 1),
            "mean_dcr": round(mean_dcr, 1) if not np.isnan(mean_dcr) else "",
        })

    # Compare vs CTGAN routing (from cluster_per_fold_hdbscan.csv)
    _compare_with_ctgan(df)

    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved summary: {SUMMARY_CSV}")


def _compare_with_ctgan(tvae_df: pd.DataFrame):
    ctgan_path = os.path.join(RUN_DIR, "cluster_per_fold_hdbscan.csv")
    if not os.path.exists(ctgan_path):
        return
    ctgan_df = pd.read_csv(ctgan_path)
    ctgan_dcr = ctgan_df[ctgan_df["record_type"] == "dcr_ctgan_synthetic"]
    cluster_ids = sorted(tvae_df["cluster_id"].unique())

    print("\n=== TVAE vs CTGAN ROUTING COMPARISON ===")
    print(f"{'Cluster':>10}  {'TVAE synth%':>12}  {'CTGAN synth%':>13}  "
          f"{'TVAE DCR':>10}  {'CTGAN DCR':>10}")
    print("-" * 65)

    total_synth_by_fold = tvae_df.groupby("fold")["n_synth_routed"].sum()

    for c in cluster_ids:
        tvae_sub = tvae_df[tvae_df["cluster_id"] == c]
        tvae_pct_vals = []
        for fold in tvae_sub["fold"].values:
            total_f = total_synth_by_fold.get(fold, 0)
            n_f = tvae_sub[tvae_sub["fold"] == fold]["n_synth_routed"].values
            if len(n_f) and total_f > 0:
                tvae_pct_vals.append(n_f[0] / total_f * 100)
        tvae_pct = np.mean(tvae_pct_vals) if tvae_pct_vals else 0.0
        tvae_dcr_vals = pd.to_numeric(tvae_sub["mean_dcr"], errors="coerce").dropna()
        tvae_dcr = tvae_dcr_vals.mean() if len(tvae_dcr_vals) else float("nan")

        ctgan_sub = ctgan_dcr[ctgan_dcr["cluster_id"] == c]
        ctgan_pct = ctgan_sub["n_synthetic"].sum() / ctgan_dcr["n_synthetic"].sum() * 100 if len(ctgan_sub) else 0.0
        ctgan_dcr_val = ctgan_sub["dcr_mean"].mean() if len(ctgan_sub) else float("nan")

        label = "noise" if c == -1 else str(c)
        print(f"  {label:>8}  {tvae_pct:>11.1f}%  {ctgan_pct:>12.1f}%  "
              f"{tvae_dcr:>10.1f}  {ctgan_dcr_val:>10.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=RUN_DIR)
    parser.add_argument("--fold", type=int, default=-1,
                        help="Run only this fold (-1 = all)")
    args = parser.parse_args()
    main(run_dir=args.run_dir, only_fold=args.fold)
