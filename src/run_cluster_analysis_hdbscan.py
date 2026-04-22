# src/run_cluster_analysis_hdbscan.py
"""
HDBSCAN-first fraud-cluster analysis (falls back to KMeans k=5 if HDBSCAN
produces degenerate clusters). Per-fold, per-cluster PR-AUC delta
(CTGAN vs baseline) and per-cluster DCR for CTGAN synth.

Same infra as src/run_cluster_analysis.py:
  - Column-projected parquet load (avoids the OOM the original script hit)
  - Skips SMOTE (paper only needs CTGAN vs baseline mechanism)
  - Checkpoints after each fold
  - Fold-level resume: if a fold is already in the output CSV, skip it

Degeneracy check for HDBSCAN:
  - n_clusters (excluding noise label -1) <= 1      -> fallback
  - noise fraction > 0.5                             -> fallback
  - largest cluster holds > 95% of fraud rows        -> fallback

Outputs (separate from the KMeans run so nothing on disk is clobbered):
  results/protocol/run_<run_id>/cluster_per_fold_hdbscan.csv
  results/protocol/run_<run_id>/cluster_summary_hdbscan.csv
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sklearn.cluster import HDBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

from fraud_cluster_analysis import (
    _float_matrix_aligned,
    _infer_obj_num_cols,
    _lgbm_val_predictions,
    align_df_to_used_cols,
    fold_val_index_range,
    load_best_target_rates_per_fold,
    pr_auc_cluster_slice,
    transform_for_cluster_model,
    CTGAN_BATCH_SIZE,
    CTGAN_DISCRIMINATOR_STEPS,
    CTGAN_EPOCHS,
    CTGAN_PAC,
    CTGAN_SEED,
    MAX_SYNTH_POS,
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


# ---------------------------------------------------------------------------
# Column-projected parquet load (same pattern as run_tvae_protocol.py)
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
        print(f"[CLUSTER] Loading: {p}", flush=True)
        if p.endswith(".parquet"):
            import pyarrow.parquet as _pq
            schema_cols = _pq.ParquetFile(p).schema_arrow.names
            cols = _select_columns_to_load(schema_cols)
            print(f"[CLUSTER] Reading {len(cols)} / {len(schema_cols)} columns from parquet", flush=True)
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
        print(f"[CLUSTER] df shape={df.shape}", flush=True)
        return df
    raise FileNotFoundError("No train data found in data/")


# ---------------------------------------------------------------------------
# Clustering: HDBSCAN first, KMeans fallback
# ---------------------------------------------------------------------------

def fit_clusterer(
    full_prep: pd.DataFrame,
    used_cols: List[str],
    kmeans_k: int,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: Optional[int],
    seed: int,
) -> Tuple[str, object, StandardScaler, object, List[str], List[str], np.ndarray]:
    """
    Returns (algo_name, estimator, scaler, ord_encoder, obj_cols, num_cols, clusters_global).
    - algo_name in {"hdbscan", "kmeans"}
    - clusters_global has shape (len(full_prep),), int; -1 for non-fraud; HDBSCAN
      noise rows are also -1 (merged into "no cluster").
    - estimator is either an HDBSCAN (no predict) or KMeans (has predict)
    """
    fraud = full_prep[full_prep[TARGET_COL] == 1].copy()
    obj_cols, num_cols = _infer_obj_num_cols(fraud, used_cols)
    X, enc = _float_matrix_aligned(fraud, used_cols, None, obj_cols, num_cols, fit_enc=True)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n_fraud = Xs.shape[0]

    print(f"[CLUSTER] Fitting HDBSCAN on {n_fraud} fraud rows "
          f"(min_cluster_size={hdbscan_min_cluster_size}, min_samples={hdbscan_min_samples}) ...",
          flush=True)
    hdb = HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        metric="euclidean",
        n_jobs=1,
    )
    hdb_labels = hdb.fit_predict(Xs)

    uniq = np.unique(hdb_labels)
    real_clusters = uniq[uniq >= 0]
    n_clusters_found = int(len(real_clusters))
    noise_frac = float(np.mean(hdb_labels == -1))
    if n_clusters_found >= 1:
        sizes = np.array([int((hdb_labels == c).sum()) for c in real_clusters])
        biggest_frac = float(sizes.max() / max(1, sizes.sum()))
    else:
        biggest_frac = 1.0

    print(f"[CLUSTER] HDBSCAN: clusters={n_clusters_found}, noise_frac={noise_frac:.3f}, "
          f"biggest_cluster_frac_of_signal={biggest_frac:.3f}", flush=True)

    degenerate = (
        n_clusters_found <= 1
        or noise_frac > 0.5
        or biggest_frac > 0.95
    )
    if not degenerate:
        algo = "hdbscan"
        estimator = hdb
        labels = hdb_labels
    else:
        print(f"[CLUSTER] HDBSCAN degenerate -> falling back to KMeans(k={kmeans_k})",
              flush=True)
        km = KMeans(n_clusters=kmeans_k, n_init=10, random_state=seed)
        km.fit(Xs)
        labels = km.labels_.astype(int)
        algo = "kmeans"
        estimator = km

    clusters_global = np.full(len(full_prep), -1, dtype=int)
    fraud_idx = np.where(full_prep[TARGET_COL].to_numpy() == 1)[0]
    for j, gix in enumerate(fraud_idx):
        clusters_global[gix] = int(labels[j])

    return algo, estimator, scaler, enc, obj_cols, num_cols, clusters_global


def predict_clusters_for_synth(
    algo: str,
    estimator: object,
    scaler: StandardScaler,
    enc: object,
    obj_cols: List[str],
    num_cols: List[str],
    used_cols_global: List[str],
    synth: pd.DataFrame,
    full_prep_template: pd.DataFrame,
    all_cluster_ids: List[int],
) -> np.ndarray:
    """
    Assign synth rows to clusters.
      - KMeans: use .predict
      - HDBSCAN (no predict): nearest-centroid over per-cluster real-fraud centroid
    """
    synth_aligned = align_df_to_used_cols(synth, used_cols_global, full_prep_template)
    X, _ = _float_matrix_aligned(synth_aligned, used_cols_global, enc, obj_cols, num_cols, fit_enc=False)
    Xs = scaler.transform(X)

    if algo == "kmeans":
        return estimator.predict(Xs).astype(int)

    # HDBSCAN: use nearest-centroid against per-cluster centroids in Xs space
    # Build centroids from the training fraud rows (we only have labels_, not centroids)
    # We reuse the estimator's labels_ which matches the fraud rows used at fit time.
    # But those are in the fraud-only index, not the full index. So recompute centroids
    # from the estimator using its internal attribute if available.
    # Simplest: recompute fraud feature matrix inside the caller is heavier;
    # instead we stash fraud centroids outside this fn. Not available here.
    # Alternate path: use approximate_predict if available (hdbscan pkg has it,
    # sklearn's HDBSCAN has .fit_predict but exposes .centers_? No — but we have
    # .labels_ and the original X we fit on).
    # We'll carry the fraud X and labels in a closure on the estimator at call site.
    centroids = getattr(estimator, "_fraud_cluster_centroids", None)
    if centroids is None:
        raise RuntimeError("HDBSCAN centroids not precomputed; caller must attach "
                           "estimator._fraud_cluster_centroids and ._fraud_cluster_ids")
    ids = getattr(estimator, "_fraud_cluster_ids")
    # Nearest-centroid in Xs space
    from sklearn.metrics import pairwise_distances_argmin
    idx = pairwise_distances_argmin(Xs, centroids)
    return np.array([ids[i] for i in idx], dtype=int)


# ---------------------------------------------------------------------------
# Resume helper
# ---------------------------------------------------------------------------

def _completed_folds(long_csv: str) -> set:
    if not os.path.exists(long_csv):
        return set()
    try:
        df = pd.read_csv(long_csv, usecols=["record_type", "fold"])
    except Exception:
        return set()
    done = df[df["record_type"] == "pr_auc_per_cluster"]["fold"].dropna().unique()
    return {int(f) for f in done}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _make_summary(rows: List[Dict[str, Any]], cluster_ids: List[int]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=[
            "cluster_id", "fraud_count_global", "fraud_share_pct",
            "n_folds_evaluated", "mean_val_fraud_per_fold",
            "mean_baseline_pr_auc", "mean_ctgan_pr_auc",
            "mean_delta_pr_auc", "std_delta_pr_auc",
            "mean_n_synth_routed", "synth_share_pct", "mean_dcr_synth",
        ])
    assn = df[df["record_type"] == "fraud_cluster_assignment"]
    pra = df[df["record_type"] == "pr_auc_per_cluster"]
    dcr = df[df["record_type"] == "dcr_ctgan_synthetic"]
    n_fraud_total = max(1, len(assn))

    out = []
    for c in cluster_ids:
        n_fraud_global = int((assn["cluster_id"] == c).sum())
        base = pra[(pra["cluster_id"] == c) & (pra["method"] == "baseline")]
        ctg = pra[(pra["cluster_id"] == c) & (pra["method"] == "ctgan")]
        deltas = []
        for f in sorted(set(base["fold"].dropna().unique()).intersection(ctg["fold"].dropna().unique())):
            b = base[base["fold"] == f]["pr_auc"].mean()
            t = ctg[ctg["fold"] == f]["pr_auc"].mean()
            if pd.notna(b) and pd.notna(t):
                deltas.append(t - b)
        dc = dcr[dcr["cluster_id"] == c]
        out.append({
            "cluster_id": c,
            "fraud_count_global": n_fraud_global,
            "fraud_share_pct": round(100.0 * n_fraud_global / n_fraud_total, 2),
            "n_folds_evaluated": len(deltas),
            "mean_val_fraud_per_fold": float(base["n_pos_in_slice"].mean(skipna=True))
                if not base.empty else float("nan"),
            "mean_baseline_pr_auc": float(base["pr_auc"].mean(skipna=True))
                if not base.empty else float("nan"),
            "mean_ctgan_pr_auc": float(ctg["pr_auc"].mean(skipna=True))
                if not ctg.empty else float("nan"),
            "mean_delta_pr_auc": float(np.mean(deltas)) if deltas else float("nan"),
            "std_delta_pr_auc": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else float("nan"),
            "mean_n_synth_routed": float(dc["n_synthetic"].mean(skipna=True))
                if not dc.empty else float("nan"),
            "synth_share_pct": float(100.0 * dc["n_synthetic"].mean(skipna=True) / MAX_SYNTH_POS)
                if not dc.empty else float("nan"),
            "mean_dcr_synth": float(dc["dcr_mean"].mean(skipna=True))
                if not dc.empty else float("nan"),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="HDBSCAN-first fraud cluster analysis (KMeans fallback)."
    )
    ap.add_argument("--run-id", default="20260330_180216")
    ap.add_argument("--n-folds", type=int, default=8)
    ap.add_argument("--ctgan-epochs", type=int, default=CTGAN_EPOCHS)
    ap.add_argument("--kmeans-k", type=int, default=5,
                    help="Fallback k for KMeans if HDBSCAN degenerates.")
    ap.add_argument("--hdbscan-min-cluster-size", type=int, default=500,
                    help="Minimum cluster size for HDBSCAN "
                         "(~2-3%% of ~20k fraud rows gives 4-8 clusters).")
    ap.add_argument("--hdbscan-min-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_dir = os.path.join(_ROOT, "results", "protocol", f"run_{args.run_id}")
    os.makedirs(run_dir, exist_ok=True)
    long_csv = os.path.join(run_dir, "cluster_per_fold_hdbscan.csv")
    summary_csv = os.path.join(run_dir, "cluster_summary_hdbscan.csv")
    results_csv = os.path.join(run_dir, "results.csv")
    if not os.path.isfile(results_csv):
        raise FileNotFoundError(f"Need {results_csv} for best CTGAN rates")

    best_rates = load_best_target_rates_per_fold(results_csv, ("ctgan",))

    df_raw = _load_train_data(os.path.join(_ROOT, "data"))
    df_sorted = df_raw.sort_values(TIME_COL).reset_index(drop=True)
    del df_raw
    gc.collect()
    n_total = len(df_sorted)

    full_prep, used_cols_global = preprocess_for_synth(df_sorted)
    n_fraud = int((full_prep[TARGET_COL] == 1).sum())
    print(f"[CLUSTER] preprocessed: rows={len(full_prep)}, cols={len(used_cols_global)}, fraud={n_fraud}",
          flush=True)

    algo, estimator, scaler, enc, obj_cols, num_cols, clusters_global = fit_clusterer(
        full_prep=full_prep,
        used_cols=used_cols_global,
        kmeans_k=args.kmeans_k,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        seed=args.seed,
    )

    # Enumerate cluster ids actually present among fraud rows (>=0)
    cluster_ids = sorted(int(c) for c in np.unique(clusters_global) if c >= 0)
    # Noise bucket = -1 (only present under HDBSCAN); include if any fraud landed there
    fraud_mask = full_prep[TARGET_COL].to_numpy() == 1
    n_noise_fraud = int(((clusters_global == -1) & fraud_mask).sum())
    if algo == "hdbscan" and n_noise_fraud > 0:
        cluster_ids = [-1] + cluster_ids  # keep noise as its own bucket for reporting

    # For HDBSCAN: precompute per-cluster centroid from the fitted fraud matrix
    if algo == "hdbscan":
        fraud = full_prep[full_prep[TARGET_COL] == 1].copy()
        X, _ = _float_matrix_aligned(fraud, used_cols_global, enc, obj_cols, num_cols, fit_enc=False)
        Xs = scaler.transform(X)
        fraud_labels = np.array([clusters_global[gix] for gix in np.where(full_prep[TARGET_COL].to_numpy() == 1)[0]])
        ids = [c for c in cluster_ids if c >= 0]
        centroids = np.vstack([Xs[fraud_labels == c].mean(axis=0) for c in ids])
        setattr(estimator, "_fraud_cluster_centroids", centroids)
        setattr(estimator, "_fraud_cluster_ids", ids)

    rows_out: List[Dict[str, Any]] = []
    completed = _completed_folds(long_csv)

    # Fresh assignment rows only if CSV doesn't already have them
    if not os.path.exists(long_csv):
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
                "notes": f"{algo} on full-data preprocessed fraud (-1=noise if hdbscan)",
            })
    else:
        # Resume: load previous content
        prev = pd.read_csv(long_csv)
        rows_out = prev.to_dict("records")
        print(f"[CLUSTER] Resume: loaded {len(rows_out)} prior rows, "
              f"completed folds={sorted(completed)}", flush=True)

    # Count per-cluster fraud size (restrict to fraud rows to avoid conflating
    # the non-fraud rows that are also labeled -1 in clusters_global).
    fraud_cluster_labels = clusters_global[fraud_mask]
    sizes = pd.Series(fraud_cluster_labels[fraud_cluster_labels >= 0]).value_counts().sort_index()
    print(f"\n[CLUSTER] Global fraud per cluster (algo={algo}):", flush=True)
    for cid, n in sizes.items():
        print(f"   cluster {cid}: {n} ({100.0 * n / n_fraud:.1f}%)", flush=True)
    if algo == "hdbscan":
        print(f"   noise (-1): {n_noise_fraud} ({100.0 * n_noise_fraud / n_fraud:.1f}%)", flush=True)

    folds = get_temporal_folds(df_sorted, n_folds=args.n_folds, time_col=TIME_COL)

    for fold_info in folds:
        fold = int(fold_info["fold"])
        if fold in completed:
            print(f"[CLUSTER] fold {fold} already complete — skipping", flush=True)
            continue

        train_df = fold_info["train_df"]
        val_df = fold_info["val_df"]
        if len(val_df) == 0:
            continue

        t0 = time.perf_counter()
        print("\n" + "=" * 60, flush=True)
        print(f"===== FOLD {fold} (algo={algo}) =====", flush=True)

        train_prep, val_prep, used_cols_fold = preprocess_fold(train_df, val_df)
        cat_cols = get_cat_cols_for_synth(train_prep, used_cols_fold)

        va_start, va_end = fold_val_index_range(n_total, args.n_folds, fold)
        if va_end - va_start != len(val_prep):
            print(f"[WARN] Fold {fold}: val slice mismatch — skipping", flush=True)
            continue

        val_cluster = np.full(len(val_prep), -2, dtype=int)  # -2 = non-fraud (ignore)
        target_arr = full_prep[TARGET_COL].to_numpy()
        for local_i in range(len(val_prep)):
            g = va_start + local_i
            if int(target_arr[g]) == 1:
                val_cluster[local_i] = int(clusters_global[g])

        rate = best_rates.get("ctgan", {}).get(fold, 0.10)

        p0, y0 = _lgbm_val_predictions(train_prep, val_prep)

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
            for c in cluster_ids:
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
                    "notes": f"subset=all val neg U val fraud in cluster (algo={algo})",
                })

        if len(synth) > 0:
            real_fraud = train_prep[train_prep[TARGET_COL] == 1]
            dcr = compute_dcr(synth, real_fraud)
            try:
                synth_clusters = predict_clusters_for_synth(
                    algo, estimator, scaler, enc, obj_cols, num_cols,
                    used_cols_global, synth, full_prep, cluster_ids,
                )
            except Exception as e:
                print(f"[WARN] fold {fold}: synth cluster predict failed ({e}); using -1", flush=True)
                synth_clusters = np.full(len(synth), -1, dtype=int)

            for c in cluster_ids:
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
                    "notes": f"DCR vs fold-train real fraud; synth routed via {'nearest-centroid' if algo=='hdbscan' else 'KMeans.predict'}",
                })

        pd.DataFrame(rows_out).to_csv(long_csv, index=False)
        summary = _make_summary(rows_out, cluster_ids)
        summary.to_csv(summary_csv, index=False)
        print(f"[CLUSTER] fold {fold} done in {time.perf_counter() - t0:.1f}s "
              f"-> {len(rows_out)} rows ({long_csv})", flush=True)

        del train_prep, val_prep, train_df, val_df, synth, mixed
        gc.collect()

    summary = _make_summary(rows_out, cluster_ids)
    summary.to_csv(summary_csv, index=False)
    print("\n[CLUSTER] === per-cluster summary (algo={}) ===".format(algo), flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\n[CLUSTER] wrote {summary_csv}", flush=True)
    print(f"[CLUSTER] wrote {long_csv}", flush=True)


if __name__ == "__main__":
    main()
