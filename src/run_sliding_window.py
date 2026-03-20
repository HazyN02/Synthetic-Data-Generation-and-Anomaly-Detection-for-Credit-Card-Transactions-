#!/usr/bin/env python3
"""
Sliding window retraining: static vs sliding comparison.
Run from project root: python -m src.run_sliding_window [--quick]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _USE_LGB = True
except (OSError, ImportError) as e:
    from sklearn.ensemble import GradientBoostingClassifier
    _USE_LGB = False

# Ensure project root in path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from src.features_aligned import prepare_features_aligned
from src.eval import pr_auc, recall_at_fpr

# Config (overridden by --quick)
TARGET_COL = "isFraud"
TIME_COL = "TransactionDT"
N_FOLDS = 5
WINDOW_CHUNKS = 1

LGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "min_data_in_leaf": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary",
    "n_jobs": -1,
    "random_state": 42,
}
LGB_QUICK = {"n_estimators": 100, "min_data_in_leaf": 100}
GBC_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_samples_leaf": 200,
    "subsample": 0.8,
    "max_features": "sqrt",
    "random_state": 42,
}

RESULTS_DIR = os.path.join(_ROOT, "results")
SLIDING_DIR = os.path.join(RESULTS_DIR, "sliding_window")
RESULTS_CSV = os.path.join(SLIDING_DIR, "results.csv")


def _load_data():
    paths = [
        os.path.join(_ROOT, "data", "train_merged.parquet"),
        os.path.join(_ROOT, "data", "train_transaction.csv"),
    ]
    for p in paths:
        if os.path.exists(p):
            print(f"Loading: {p}")
            return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
    raise FileNotFoundError("No train data found in data/")


def train_and_eval_aligned(train_df: pd.DataFrame, val_df: pd.DataFrame, quick: bool = False) -> dict:
    X_train, y_train, X_val, y_val = prepare_features_aligned(train_df, val_df)
    n_pos = int((y_train == 1).sum())
    if n_pos < 10:
        return {"pr_auc": np.nan, "recall@1%fpr": np.nan}

    if _USE_LGB:
        params = {**LGB_PARAMS, **(LGB_QUICK if quick else {})}
        model = lgb.LGBMClassifier(**params)
    else:
        model = GradientBoostingClassifier(**GBC_PARAMS)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)[:, 1]
    return {
        "pr_auc": float(pr_auc(y_val, y_pred)),
        "recall@1%fpr": float(recall_at_fpr(y_val, y_pred, 0.01)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fewer folds (2) and trees (100) for fast debug")
    args = parser.parse_args()

    quick = args.quick
    n_folds = 2 if quick else N_FOLDS

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(SLIDING_DIR, exist_ok=True)
    print(f"Using: {'LightGBM' if _USE_LGB else 'GradientBoosting (LightGBM unavailable)'}")
    if quick:
        print("QUICK MODE: 2 folds, 100 trees")

    df = _load_data()
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    assert TARGET_COL in df.columns, f"Missing {TARGET_COL}"
    assert TIME_COL in df.columns, f"Missing {TIME_COL}"

    n = len(df)
    n_chunks = n_folds + 1
    edges = np.linspace(0, n, num=n_chunks + 1, dtype=int)
    chunk_size = int(edges[1] - edges[0])
    window_rows = chunk_size * WINDOW_CHUNKS

    print(f"Rows: {n}, Chunk size: {chunk_size}, Window: {window_rows}, Folds: {n_folds}")

    rows = []
    for fold in range(n_folds):
        va_start, va_end = int(edges[fold + 1]), int(edges[fold + 2])
        val_df = df.iloc[va_start:va_end].reset_index(drop=True)
        if len(val_df) == 0:
            continue

        sw_end = va_start
        sw_start = max(0, sw_end - window_rows)
        train_sliding = df.iloc[sw_start:sw_end].reset_index(drop=True)
        train_static = df.iloc[0:va_start].reset_index(drop=True)
        if len(train_static) == 0:
            continue

        print(f"\nFold {fold}: static={len(train_static)}, sliding={len(train_sliding)}, val={len(val_df)}", flush=True)

        res_static = train_and_eval_aligned(train_static, val_df, quick=quick)
        res_sliding = train_and_eval_aligned(train_sliding, val_df, quick=quick)

        def _fmt(v):
            return f"{v:.4f}" if not np.isnan(v) else "nan"
        print(f"  Static  PR-AUC: {_fmt(res_static['pr_auc'])}  Recall@1%FPR: {_fmt(res_static['recall@1%fpr'])}", flush=True)
        print(f"  Sliding PR-AUC: {_fmt(res_sliding['pr_auc'])}  Recall@1%FPR: {_fmt(res_sliding['recall@1%fpr'])}", flush=True)

        rows.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "fold": fold,
            "strategy": "static",
            "train_rows": len(train_static),
            "val_rows": len(val_df),
            "pr_auc": res_static["pr_auc"],
            "recall_at_1pct_fpr": res_static["recall@1%fpr"],
        })
        rows.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "fold": fold,
            "strategy": "sliding",
            "train_rows": len(train_sliding),
            "val_rows": len(val_df),
            "pr_auc": res_sliding["pr_auc"],
            "recall_at_1pct_fpr": res_sliding["recall@1%fpr"],
        })

        # Incremental save after each fold
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
        print(f"  [saved {len(rows)} rows]", flush=True)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved to {RESULTS_CSV}")
    print(results_df.groupby("strategy")[["pr_auc", "recall_at_1pct_fpr"]].agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()
