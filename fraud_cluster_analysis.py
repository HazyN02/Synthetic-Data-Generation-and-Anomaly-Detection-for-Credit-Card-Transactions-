#!/usr/bin/env python3
"""
Fraud cluster analysis: KMeans(5) on preprocessed feature space (full dataset),
per-fold per-cluster PR-AUC (baseline vs best CTGAN vs best SMOTE from protocol results),
and per-cluster DCR for CTGAN synthetic samples.
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from fidelity_eval import compute_dcr
from src.eval import pr_auc as eval_pr_auc
from src.features_aligned import TARGET_COL, prepare_features_aligned
from src.folds import get_temporal_folds
from src.preprocess_synth import (
    TIME_COL,
    get_cat_cols_for_synth,
    preprocess_fold,
    preprocess_for_synth,
)
from src.synth_ctgan import make_synthetic_positives
from src.synth_smote import _smote_oversample

N_FOLDS = 8
N_CLUSTERS = 5
KMEANS_SEED = 42
MAX_SYNTH_POS = 50000
CTGAN_EPOCHS = 150
CTGAN_BATCH_SIZE = 500
CTGAN_DISCRIMINATOR_STEPS = 5
CTGAN_PAC = 1
CTGAN_SEED = 0


def _float_matrix_aligned(
    df: pd.DataFrame,
    used_cols: List[str],
    enc: Optional[OrdinalEncoder],
    obj_cols: List[str],
    num_cols: List[str],
    fit_enc: bool,
) -> Tuple[np.ndarray, Optional[OrdinalEncoder]]:
    feature_cols = [c for c in used_cols if c != TARGET_COL]
    Xdf = df[feature_cols].copy()
    for c in num_cols:
        Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce").fillna(-1)
    if obj_cols:
        tr_cat = Xdf[obj_cols].astype(str).fillna("__MISSING__")
        if fit_enc:
            enc = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            Xdf[obj_cols] = enc.fit_transform(tr_cat)
        else:
            assert enc is not None
            Xdf[obj_cols] = enc.transform(tr_cat)
    X = Xdf[feature_cols].fillna(-1).astype(np.float32).values
    return X, enc


def _infer_obj_num_cols(df: pd.DataFrame, used_cols: List[str]) -> Tuple[List[str], List[str]]:
    feature_cols = [c for c in used_cols if c != TARGET_COL]
    obj_cols = [
        c
        for c in feature_cols
        if df[c].dtype == "object" or df[c].dtype.name == "category"
    ]
    num_cols = [c for c in feature_cols if c not in obj_cols]
    return obj_cols, num_cols


def fit_kmeans_global(
    full_prep: pd.DataFrame,
    used_cols: List[str],
    n_clusters: int,
) -> Tuple[KMeans, StandardScaler, Optional[OrdinalEncoder], List[str], List[str], np.ndarray]:
    """Fit KMeans on all fraud rows in full_prep; returns cluster label per global row index."""
    fraud = full_prep[full_prep[TARGET_COL] == 1].copy()
    obj_cols, num_cols = _infer_obj_num_cols(fraud, used_cols)
    X, enc = _float_matrix_aligned(fraud, used_cols, None, obj_cols, num_cols, fit_enc=True)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=KMEANS_SEED,
    )
    km.fit(Xs)
    labels = km.labels_.astype(int)

    clusters_global = np.full(len(full_prep), -1, dtype=int)
    fraud_idx = np.where(full_prep[TARGET_COL].to_numpy() == 1)[0]
    for j, gix in enumerate(fraud_idx):
        clusters_global[gix] = labels[j]
    return km, scaler, enc, obj_cols, num_cols, clusters_global


def transform_for_cluster_model(
    df: pd.DataFrame,
    used_cols: List[str],
    km: KMeans,
    scaler: StandardScaler,
    enc: Optional[OrdinalEncoder],
    obj_cols: List[str],
    num_cols: List[str],
) -> np.ndarray:
    X, _ = _float_matrix_aligned(df, used_cols, enc, obj_cols, num_cols, fit_enc=False)
    Xs = scaler.transform(X)
    return km.predict(Xs)


def align_df_to_used_cols(df: pd.DataFrame, used_cols: List[str], template: pd.DataFrame) -> pd.DataFrame:
    """Reindex to used_cols; fill from template dtypes/medians."""
    out = pd.DataFrame(index=df.index)
    for c in used_cols:
        if c == TARGET_COL:
            out[c] = df[c] if c in df.columns else 0
            continue
        if c in df.columns:
            out[c] = df[c]
        else:
            if c in template.columns:
                if pd.api.types.is_numeric_dtype(template[c]):
                    med = float(pd.to_numeric(template[c], errors="coerce").median() or 0.0)
                    out[c] = med
                else:
                    out[c] = "__MISSING__"
            else:
                out[c] = -1.0
    for c in used_cols:
        if c == TARGET_COL:
            continue
        if out[c].dtype == "object" or out[c].dtype.name == "category":
            out[c] = out[c].astype("string").fillna("__MISSING__")
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(-1.0)
    return out


def load_best_target_rates_per_fold(
    results_path: str,
    methods: Tuple[str, ...] = ("ctgan", "smote"),
) -> Dict[str, Dict[int, float]]:
    """Per fold, target_pos_rate with highest pr_auc for each method."""
    df = pd.read_csv(results_path)
    out: Dict[str, Dict[int, float]] = {m: {} for m in methods}
    for m in methods:
        sub = df[(df["method"] == m) & (df["target_pos_rate"].notna()) & (df["pr_auc"].notna())].copy()
        if sub.empty:
            continue
        idx = sub.groupby("fold")["pr_auc"].idxmax()
        picked = sub.loc[idx, ["fold", "target_pos_rate"]]
        for _, row in picked.iterrows():
            out[m][int(row["fold"])] = float(row["target_pos_rate"])
    return out


def _lgbm_val_predictions(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X_tr, y_tr, X_va, y_va = prepare_features_aligned(train_df, val_df)
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        min_child_samples=200,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_tr, y_tr)
    pred = model.predict_proba(X_va)[:, 1]
    return pred, y_va


def _lgbm_val_predictions_smote(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_pos_rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    X_tr, y_tr, X_va, y_va = prepare_features_aligned(train_df, val_df)
    n_pos = int((y_tr == 1).sum())
    if n_pos < 2:
        return np.full(len(y_va), np.nan), y_va
    X_res, y_res = _smote_oversample(
        X_tr,
        y_tr,
        target_pos_rate=target_pos_rate,
        k_neighbors=5,
        random_state=42,
        max_synth=None,
    )
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        min_child_samples=200,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_res, y_res)
    pred = model.predict_proba(X_va)[:, 1]
    return pred, y_va


def pr_auc_cluster_slice(
    y_true: np.ndarray,
    y_score: np.ndarray,
    val_cluster: np.ndarray,
    cluster_id: int,
) -> Tuple[float, int, int]:
    """PR-AUC on (all negatives) ∪ (positives in cluster_id)."""
    mask = (y_true == 0) | ((y_true == 1) & (val_cluster == cluster_id))
    y_sub = y_true[mask]
    s_sub = y_score[mask]
    n_pos = int((y_sub == 1).sum())
    n_neg = int((y_sub == 0).sum())
    if n_pos == 0 or n_neg == 0 or not np.isfinite(s_sub).all():
        return float("nan"), n_pos, n_neg
    return float(eval_pr_auc(y_sub, s_sub)), n_pos, n_neg


def fold_val_index_range(n_rows: int, n_folds: int, fold_i: int) -> Tuple[int, int]:
    n_chunks = n_folds + 1
    edges = np.linspace(0, n_rows, num=n_chunks + 1, dtype=int)
    return int(edges[fold_i + 1]), int(edges[fold_i + 2])


def main() -> None:
    parser = argparse.ArgumentParser(description="Fraud KMeans + per-cluster PR-AUC / DCR.")
    parser.add_argument(
        "--results-csv",
        type=str,
        default=None,
        help="Protocol results CSV for best ctgan/smote rates (default: run_20260330_180216/results.csv)",
    )
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    default_results = os.path.join(
        root, "results", "protocol", "run_20260330_180216", "results.csv"
    )
    results_csv = args.results_csv or default_results
    if not os.path.isfile(results_csv):
        raise FileNotFoundError(f"Need protocol results for best rates: {results_csv}")

    out_path = os.path.join(root, "results", "protocol", "fraud_cluster_results.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    best_rates = load_best_target_rates_per_fold(results_csv, ("ctgan", "smote"))

    data_dir = os.path.join(root, "data")
    df_raw = None
    for name in ["train_merged.parquet", "train_transaction.csv"]:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            print(f"Loading {p}")
            df_raw = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
            break
    if df_raw is None:
        raise FileNotFoundError("No train data in data/")

    df_sorted = df_raw.sort_values(TIME_COL).reset_index(drop=True)
    n_total = len(df_sorted)
    full_prep, used_cols_global = preprocess_for_synth(df_sorted)
    assert TARGET_COL in full_prep.columns

    print(f"[cluster] Fitting KMeans(k={N_CLUSTERS}) on {int((full_prep[TARGET_COL]==1).sum())} fraud rows...")
    km, scaler, enc, obj_cols, num_cols, clusters_global = fit_kmeans_global(
        full_prep, used_cols_global, N_CLUSTERS
    )

    rows_out: List[Dict[str, Any]] = []

    for gix in range(n_total):
        if full_prep[TARGET_COL].iloc[gix] != 1:
            continue
        rows_out.append(
            {
                "record_type": "fraud_cluster_assignment",
                "fold": np.nan,
                "cluster_id": int(clusters_global[gix]),
                "global_row_idx": gix,
                "method": "",
                "target_pos_rate": np.nan,
                "pr_auc": np.nan,
                "n_pos_in_slice": np.nan,
                "n_neg_in_slice": np.nan,
                "dcr_mean": np.nan,
                "dcr_median": np.nan,
                "n_synthetic": np.nan,
                "notes": "KMeans on full-data preprocessed fraud; global_row_idx aligns with time-sorted train data",
            }
        )

    folds = get_temporal_folds(df_sorted, n_folds=args.n_folds, time_col=TIME_COL)

    for fold_info in folds:
        fold = int(fold_info["fold"])
        train_df = fold_info["train_df"]
        val_df = fold_info["val_df"]
        if len(val_df) == 0:
            continue

        train_prep, val_prep, used_cols_fold = preprocess_fold(train_df, val_df)
        cat_cols = get_cat_cols_for_synth(train_prep, used_cols_fold)

        va_start, va_end = fold_val_index_range(n_total, args.n_folds, fold)
        if va_end - va_start != len(val_prep):
            raise RuntimeError(f"Fold {fold}: val slice length mismatch")

        val_cluster = np.full(len(val_prep), -1, dtype=int)
        for local_i in range(len(val_prep)):
            g = va_start + local_i
            if int(full_prep[TARGET_COL].iloc[g]) == 1:
                val_cluster[local_i] = int(clusters_global[g])

        rate_ctgan = best_rates.get("ctgan", {}).get(fold)
        rate_smote = best_rates.get("smote", {}).get(fold)
        if rate_ctgan is None:
            rate_ctgan = 0.10
            print(f"[warn] fold={fold}: no best ctgan in results — using 0.10")
        if rate_smote is None:
            rate_smote = 0.10
            print(f"[warn] fold={fold}: no best smote in results — using 0.10")

        preds: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[float]]] = {}

        p0, y0 = _lgbm_val_predictions(train_prep, val_prep)
        preds["baseline"] = (p0, y0, None)

        synth = make_synthetic_positives(
            train_df=train_prep,
            cat_cols=cat_cols,
            used_cols=used_cols_fold,
            target_pos_rate=float(rate_ctgan),
            max_synth=MAX_SYNTH_POS,
            epochs=CTGAN_EPOCHS,
            batch_size=CTGAN_BATCH_SIZE,
            pac=CTGAN_PAC,
            seed=CTGAN_SEED,
            discriminator_steps=CTGAN_DISCRIMINATOR_STEPS,
            verbose=False,
        )
        mixed_ctgan = pd.concat([train_prep, synth], axis=0, ignore_index=True)
        pc, yc = _lgbm_val_predictions(mixed_ctgan, val_prep)
        preds["ctgan"] = (pc, yc, float(rate_ctgan))

        ps, ys = _lgbm_val_predictions_smote(train_prep, val_prep, float(rate_smote))
        preds["smote"] = (ps, ys, float(rate_smote))

        for method, (pred, y_va, trate) in preds.items():
            for c in range(N_CLUSTERS):
                pr_c, np_c, nn_c = pr_auc_cluster_slice(y_va, pred, val_cluster, c)
                rows_out.append(
                    {
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
                        "notes": "subset = all val negatives ∪ val fraud in cluster",
                    }
                )

        if len(synth) > 0:
            real_fraud = train_prep[train_prep[TARGET_COL] == 1]
            dcr = compute_dcr(synth, real_fraud)
            synth_aligned = align_df_to_used_cols(synth, used_cols_global, full_prep)
            synth_clusters = transform_for_cluster_model(
                synth_aligned,
                used_cols_global,
                km,
                scaler,
                enc,
                obj_cols,
                num_cols,
            )
            for c in range(N_CLUSTERS):
                m = synth_clusters == c
                n_s = int(m.sum())
                if n_s == 0:
                    d_mean, d_med = float("nan"), float("nan")
                else:
                    d_mean = float(np.mean(dcr[m]))
                    d_med = float(np.median(dcr[m]))
                rows_out.append(
                    {
                        "record_type": "dcr_ctgan_synthetic",
                        "fold": fold,
                        "cluster_id": c,
                        "global_row_idx": np.nan,
                        "method": "ctgan",
                        "target_pos_rate": float(rate_ctgan),
                        "pr_auc": np.nan,
                        "n_pos_in_slice": np.nan,
                        "n_neg_in_slice": np.nan,
                        "dcr_mean": d_mean,
                        "dcr_median": d_med,
                        "n_synthetic": n_s,
                        "notes": "DCR vs fold train real fraud; cluster via KMeans.predict on synth features",
                    }
                )

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(out_path, index=False)
    print(f"[done] wrote {len(out_df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
