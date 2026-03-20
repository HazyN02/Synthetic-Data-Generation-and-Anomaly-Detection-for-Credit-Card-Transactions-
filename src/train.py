# src/train.py

import numpy as np
import lightgbm as lgb

from src.features_aligned import prepare_features_aligned
from src.eval import pr_auc, recall_at_fpr


def train_and_eval(train_df, val_df):
    """
    Trains LightGBM on train_df, evaluates on val_df.
    Uses aligned encoding (fit on train, transform both) to avoid train/val leakage.
    Returns a fixed metric dict.
    """

    # -----------------------
    # FEATURES / TARGET (aligned encoding for temporal CV)
    # -----------------------
    X_train, y_train, X_val, y_val = prepare_features_aligned(train_df, val_df)

    # -----------------------
    # MODEL
    # -----------------------
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        min_child_samples=200,  # conservative for imbalanced; silences min_data_in_leaf warning
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # -----------------------
    # PREDICTIONS
    # -----------------------
    y_val_pred = model.predict_proba(X_val)[:, 1]

    # -----------------------
    # METRICS (CANONICAL)
    # -----------------------
    pr = pr_auc(y_val, y_val_pred)
    rec = recall_at_fpr(y_val, y_val_pred, 0.01)

    return {
        "pr_auc": float(pr),
        "recall@1%fpr": float(rec),
    }


