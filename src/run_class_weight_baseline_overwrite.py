#!/usr/bin/env python3
"""
Compute a class-weighted LightGBM baseline (scale_pos_weight) and overwrite
the existing "baseline" rows in results/protocol/results.csv.

This is intentionally implemented as a standalone script so we do not modify
the existing training code (src/train.py) while still enabling Step B.

Run:
  python -m src.run_class_weight_baseline_overwrite
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

import lightgbm as lgb

from src.eval import pr_auc, recall_at_fpr
from src.folds import get_temporal_folds
from src.features_aligned import prepare_features_aligned
from src.preprocess_synth import preprocess_fold


TARGET_COL = "isFraud"
TIME_COL = "TransactionDT"

N_FOLDS_DEFAULT = 4
DELAYS_DEFAULT = [0, 7, 14]


@dataclass(frozen=True)
class Key:
    fold: int
    delay_days: int


def _load_data() -> pd.DataFrame:
    root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(root, "data")
    for name in ["train_merged.parquet", "train_transaction.csv"]:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
    raise FileNotFoundError("No train data in data/ (need train_merged.parquet or train_transaction.csv)")


def _lgbm_params() -> Dict:
    # Mirror src/train.py defaults for fairness; only add scale_pos_weight below.
    return dict(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        min_child_samples=200,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        n_jobs=-1,
        random_state=42,
    )


def _train_eval_class_weighted(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, float]:
    X_train, y_train, X_val, y_val = prepare_features_aligned(train_df, val_df)

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos < 2 or n_neg < 1:
        return {"pr_auc": float("nan"), "recall@1%FPR": float("nan")}

    # scale_pos_weight is the multiplier for the positive class.
    scale_pos_weight = n_neg / n_pos
    params = {**_lgbm_params(), "scale_pos_weight": float(scale_pos_weight)}
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    y_val_pred = model.predict_proba(X_val)[:, 1]
    return {
        "pr_auc": float(pr_auc(y_val, y_val_pred)),
        "recall@1%FPR": float(recall_at_fpr(y_val, y_val_pred, 0.01)),
    }


def _compute_baseline_metrics(n_folds: int, delays_days: List[int]) -> Dict[Key, Dict[str, float]]:
    df = _load_data()
    folds_raw = get_temporal_folds(df, n_folds=n_folds, time_col=TIME_COL)

    out: Dict[Key, Dict[str, float]] = {}
    for fold_info in folds_raw:
        fold = int(fold_info["fold"])
        base_train_df = fold_info["train_df"]
        base_val_df = fold_info["val_df"]

        for delay_days in delays_days:
            train_df = base_train_df.copy()
            val_df = base_val_df.copy()

            if delay_days > 0:
                t_val_start = float(val_df[TIME_COL].min())
                cutoff = t_val_start - delay_days * 86400
                train_df = train_df[train_df[TIME_COL] <= cutoff].reset_index(drop=True)
                train_pos = int((train_df[TARGET_COL] == 1).sum())
                if len(train_df) == 0 or train_pos < 50:
                    continue

            # Per-fold preprocessing (fit on train, transform val) to avoid leakage.
            train_prep, val_prep, _used_cols = preprocess_fold(train_df, val_df)
            res = _train_eval_class_weighted(train_prep, val_prep)
            out[Key(fold=fold, delay_days=int(delay_days))] = res

    return out


def _overwrite_protocol_baseline(
    metrics: Dict[Key, Dict[str, float]],
    results_csv_path: str,
    delays_days: List[int],
    n_folds: int,
) -> None:
    df = pd.read_csv(results_csv_path)

    # Backup so we can revert if needed.
    backup_path = results_csv_path + ".backup_class_weight_baseline"
    if not os.path.exists(backup_path):
        shutil.copy2(results_csv_path, backup_path)

    # Column names in current protocol results.csv
    metric_cols = ("pr_auc", "recall_at_1pct_fpr")
    for c in metric_cols:
        if c not in df.columns:
            raise KeyError(f"Expected column {c} in {results_csv_path}")

    # Overwrite rows for method == baseline and delay in {0,7,14}.
    # For delay=0, we overwrite both delay_days==0.0 and delay_days isna().
    for delay_days in delays_days:
        if delay_days == 0:
            delay_mask = df["delay_days"].isna() | (df["delay_days"] == 0.0) | (df["delay_days"] == 0)
        else:
            delay_mask = df["delay_days"] == float(delay_days)

        for fold in range(n_folds):
            k = Key(fold=fold, delay_days=delay_days)
            if k not in metrics:
                continue
            res = metrics[k]
            pr = res["pr_auc"]
            rec = res["recall@1%FPR"]
            if np.isnan(pr) or np.isnan(rec):
                continue

            mask = (
                (df["method"] == "baseline")
                & (df["fold"] == fold)
                & delay_mask
            )
            df.loc[mask, "pr_auc"] = pr
            df.loc[mask, "recall_at_1pct_fpr"] = rec

    df.to_csv(results_csv_path, index=False)


def main():
    root = os.path.dirname(os.path.dirname(__file__))
    results_csv_path = os.path.join(root, "results", "protocol", "results.csv")
    if not os.path.exists(results_csv_path):
        raise FileNotFoundError(f"Missing {results_csv_path}")

    n_folds = N_FOLDS_DEFAULT
    delays_days = DELAYS_DEFAULT

    print(f"Computing class-weighted baseline: n_folds={n_folds}, delays={delays_days}")
    metrics = _compute_baseline_metrics(n_folds=n_folds, delays_days=delays_days)

    # Sanity: expect at least some fold results per delay.
    for delay in delays_days:
        got = sum(1 for k in metrics.keys() if k.delay_days == delay)
        print(f"  delay={delay}: computed {got} fold(s)")

    print(f"Overwriting baseline metrics in: {results_csv_path}")
    _overwrite_protocol_baseline(
        metrics=metrics,
        results_csv_path=results_csv_path,
        delays_days=delays_days,
        n_folds=n_folds,
    )

    print("Done. Next: run `python -m src.run_unified_analysis` and `python -m src.run_canonical_analysis`.")


if __name__ == "__main__":
    main()

