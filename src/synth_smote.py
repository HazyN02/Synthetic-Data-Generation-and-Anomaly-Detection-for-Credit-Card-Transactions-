"""
SMOTE oversampling for fraud detection.
Oversamples minority (fraud) class to reach target positive rate.
"""
from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np

# Suppress sklearn feature names warning
warnings.filterwarnings("ignore", message="X does not have valid feature names")
import pandas as pd

from src.features_aligned import prepare_features_aligned
from src.eval import pr_auc, recall_at_fpr

TARGET_COL = "isFraud"


def _smote_oversample(
    X: np.ndarray,
    y: np.ndarray,
    target_pos_rate: float,
    k_neighbors: int = 5,
    random_state: int = 42,
    max_synth: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to reach target_pos_rate.
    target_pos_rate = n_pos / (n_pos + n_neg) after resampling.
    max_synth: cap synthetic samples (faster when set).
    Returns (X_resampled, y_resampled).
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        raise ImportError("imbalanced-learn required. Install: pip install imbalanced-learn")

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos < 2:
        return X, y

    # Target: n_pos_new / (n_pos_new + n_neg) = target_pos_rate
    # => n_pos_new = target_pos_rate * n_neg / (1 - target_pos_rate)
    n_pos_target = int(np.ceil(target_pos_rate * n_neg / (1.0 - target_pos_rate)))
    if max_synth is not None:
        n_pos_target = min(n_pos_target, n_pos + max_synth)
    n_to_generate = max(0, n_pos_target - n_pos)

    if n_to_generate == 0:
        return X, y

    # sampling_strategy: ratio of minority to majority after resampling
    # We want n_pos_new / n_neg = ratio => ratio = n_pos_target / n_neg
    ratio = n_pos_target / n_neg
    if ratio >= 1.0:
        return X, y

    k = min(k_neighbors, n_pos - 1)
    if k < 1:
        return X, y

    try:
        smote = SMOTE(
            sampling_strategy=ratio,
            k_neighbors=k,
            random_state=random_state,
        )
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res
    except ValueError as e:
        # Fallback: dict format for older imblearn or edge cases
        try:
            smote = SMOTE(
                sampling_strategy={1: n_pos_target},
                k_neighbors=k,
                random_state=random_state,
            )
            X_res, y_res = smote.fit_resample(X, y)
            return X_res, y_res
        except Exception:
            return X, y


def _apply_recency_smote(
    train_df: pd.DataFrame,
    recency_frac: float | None,
    time_col: str,
    min_pos: int,
) -> pd.DataFrame:
    """Restrict to pos_recent + neg for SMOTE. Fallback to full if too few positives."""
    if recency_frac is None or recency_frac >= 1.0:
        return train_df
    if time_col not in train_df.columns:
        raise ValueError(f"[SMOTE recency] time_col={time_col} not in dataframe")
    pos_df = train_df[train_df[TARGET_COL] == 1].copy()
    neg_df = train_df[train_df[TARGET_COL] == 0]
    n_pos = len(pos_df)
    n_recent = max(min_pos, int(np.ceil(recency_frac * n_pos)))
    if n_recent >= n_pos:
        return train_df
    pos_sorted = pos_df.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    pos_recent = pos_sorted.tail(n_recent)
    return pd.concat([pos_recent, neg_df], ignore_index=True)


def train_and_eval_smote(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_pos_rate: float = 0.05,
    k_neighbors: int = 5,
    random_state: int = 42,
    lgb_params: dict | None = None,
    max_synth: int | None = None,
    recency_frac: float | None = None,
    time_col: str = "TransactionDT",
    min_pos_for_recency: int = 50,
) -> dict:
    """
    Prepare features, apply SMOTE to reach target_pos_rate, train LightGBM, evaluate.
    If recency_frac in (0,1), restrict to last recency_frac of positives + all negs before SMOTE.
    Returns dict with pr_auc and recall@1%fpr.
    """
    import lightgbm as lgb

    train_sub = _apply_recency_smote(train_df, recency_frac, time_col, min_pos_for_recency)
    X_train, y_train, X_val, y_val = prepare_features_aligned(train_sub, val_df)
    n_pos = int((y_train == 1).sum())
    if n_pos < 2:
        return {"pr_auc": np.nan, "recall@1%fpr": np.nan}

    X_res, y_res = _smote_oversample(
        X_train, y_train,
        target_pos_rate=target_pos_rate,
        k_neighbors=k_neighbors,
        random_state=random_state,
        max_synth=max_synth,
    )

    default_params = {
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
    params = {**default_params, **(lgb_params or {})}

    model = lgb.LGBMClassifier(**params)
    model.fit(X_res, y_res)
    y_pred = model.predict_proba(X_val)[:, 1]

    return {
        "pr_auc": float(pr_auc(y_val, y_pred)),
        "recall@1%fpr": float(recall_at_fpr(y_val, y_pred, 0.01)),
    }
