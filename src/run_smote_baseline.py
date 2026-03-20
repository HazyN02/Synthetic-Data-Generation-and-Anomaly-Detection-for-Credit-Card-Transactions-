#!/usr/bin/env python3
"""
SMOTE baseline: compare baseline vs SMOTE at target positive rates.
Run from project root: python -m src.run_smote_baseline [--quick]
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime

import numpy as np

# Suppress sklearn feature names warning (LightGBM with numpy arrays)
warnings.filterwarnings("ignore", message="X does not have valid feature names")
import pandas as pd
import lightgbm as lgb

# Project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from src.features_aligned import prepare_features_aligned
from src.eval import pr_auc, recall_at_fpr
from src.synth_smote import train_and_eval_smote

TARGET_COL = "isFraud"
TIME_COL = "TransactionDT"
TARGET_POS_RATES = [0.05, 0.10, 0.20]

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
    "verbosity": -1,
}
LGB_QUICK = {"n_estimators": 100, "min_data_in_leaf": 100}

RESULTS_DIR = os.path.join(_ROOT, "results")
SMOTE_DIR = os.path.join(RESULTS_DIR, "smote")
RESULTS_CSV = os.path.join(SMOTE_DIR, "results.csv")


def _load_data():
    for name in ["train_merged.parquet", "train_transaction.csv"]:
        p = os.path.join(_ROOT, "data", name)
        if os.path.exists(p):
            print(f"Loading: {p}")
            return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
    raise FileNotFoundError("No train data in data/")


def _train_baseline(train_df, val_df, quick=False):
    """Baseline with aligned encoding."""
    X_train, y_train, X_val, y_val = prepare_features_aligned(train_df, val_df)
    n_pos = int((y_train == 1).sum())
    if n_pos < 2:
        return {"pr_auc": np.nan, "recall@1%fpr": np.nan}

    params = {**LGB_PARAMS, **(LGB_QUICK if quick else {})}
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)[:, 1]
    return {
        "pr_auc": float(pr_auc(y_val, y_pred)),
        "recall@1%fpr": float(recall_at_fpr(y_val, y_pred, 0.01)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="2 folds, 100 trees")
    parser.add_argument("--fast", action="store_true", help="Subsample train to 40k, 1 SMOTE rate, k=3, max 3k synth")
    args = parser.parse_args()
    quick = args.quick
    fast = args.fast
    n_folds = 2 if quick else 4
    smote_rates = [0.05] if fast else TARGET_POS_RATES

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(SMOTE_DIR, exist_ok=True)
    if quick:
        print("QUICK MODE: 2 folds, 100 trees")
    if fast:
        print("FAST MODE: subsample 40k, SMOTE k=3, max_synth=3000, 1 rate (5%)")

    df = _load_data()
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    assert TARGET_COL in df.columns and TIME_COL in df.columns

    n = len(df)
    edges = np.linspace(0, n, num=n_folds + 2, dtype=int)
    print(f"Rows: {n}, Folds: {n_folds}")

    rows = []
    for fold in range(n_folds):
        va_start, va_end = int(edges[fold + 1]), int(edges[fold + 2])
        train_df = df.iloc[:va_start].reset_index(drop=True)
        val_df = df.iloc[va_start:va_end].reset_index(drop=True)
        if len(train_df) == 0 or len(val_df) == 0:
            continue

        print(f"\n===== Fold {fold} =====")
        if fast and len(train_df) > 40_000:
            # Stratified subsample: keep all positives, sample negatives
            pos = train_df[train_df[TARGET_COL] == 1]
            neg = train_df[train_df[TARGET_COL] == 0]
            n_keep = min(40_000 - len(pos), len(neg))
            neg_sub = neg.sample(n=n_keep, random_state=42) if n_keep > 0 else neg.iloc[:0]
            train_sub = pd.concat([pos, neg_sub], ignore_index=True).sample(frac=1, random_state=42)
            print(f"Train: {len(train_sub)} (subsampled from {len(train_df)}), Val: {len(val_df)}")
        else:
            train_sub = train_df
            print(f"Train: {len(train_df)}, Val: {len(val_df)}")

        # Baseline
        res_base = _train_baseline(train_sub, val_df, quick=quick)
        print(f"  Baseline PR-AUC: {res_base['pr_auc']:.4f}  Recall@1%FPR: {res_base['recall@1%fpr']:.4f}")

        rows.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "fold": fold,
            "method": "baseline",
            "target_pos_rate": "",
            "train_rows": len(train_sub),
            "val_rows": len(val_df),
            "pr_auc": res_base["pr_auc"],
            "recall_at_1pct_fpr": res_base["recall@1%fpr"],
        })

        # SMOTE
        for rate in smote_rates:
            res = train_and_eval_smote(
                train_sub, val_df,
                target_pos_rate=rate,
                k_neighbors=3 if fast else 5,
                lgb_params=LGB_QUICK if (quick or fast) else None,
                max_synth=3000 if fast else None,
            )
            print(f"  SMOTE {rate:.0%}  PR-AUC: {res['pr_auc']:.4f}  Recall@1%FPR: {res['recall@1%fpr']:.4f}")

            rows.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "fold": fold,
                "method": "smote",
                "target_pos_rate": rate,
                "train_rows": len(train_sub),
                "val_rows": len(val_df),
                "pr_auc": res["pr_auc"],
                "recall_at_1pct_fpr": res["recall@1%fpr"],
            })

        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
        print(f"  [saved {len(rows)} rows]")

    print(f"\nSaved to {RESULTS_CSV}")
    df_res = pd.DataFrame(rows)
    print(df_res.groupby(["method", "target_pos_rate"])[["pr_auc", "recall_at_1pct_fpr"]].agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()
