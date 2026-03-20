"""
Aligned feature preparation for sliding-window / temporal CV.
Ensures train and val use consistent categorical encoding (no train/val leakage).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

TARGET_COL = "isFraud"


def prepare_features_aligned(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    Prepares (X_train, y_train, X_val, y_val) with consistent encoding.
    - Categoricals: OrdinalEncoder fit on train, transform both (unseen -> -1)
    - Numerics: fillna(-1)
    - Same column order for train and val
    """
    feature_cols = [c for c in train_df.columns if c != TARGET_COL]

    # Use only columns present in both
    shared = [c for c in feature_cols if c in val_df.columns]
    train_df = train_df[shared + [TARGET_COL]].copy()
    val_df = val_df[shared + [TARGET_COL]].copy()

    obj_cols = [c for c in shared if train_df[c].dtype == "object" or train_df[c].dtype.name == "category"]
    num_cols = [c for c in shared if c not in obj_cols]

    # Encode categoricals with consistent mapping
    X_train_encoded = train_df[shared].copy()
    X_val_encoded = val_df[shared].copy()

    if obj_cols:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        tr_cat = train_df[obj_cols].astype(str).fillna("__MISSING__")
        va_cat = val_df[obj_cols].astype(str).fillna("__MISSING__")
        X_train_encoded[obj_cols] = enc.fit_transform(tr_cat)
        X_val_encoded[obj_cols] = enc.transform(va_cat)

    # Numerics: fillna
    for c in num_cols:
        X_train_encoded[c] = pd.to_numeric(X_train_encoded[c], errors="coerce").fillna(-1)
        X_val_encoded[c] = pd.to_numeric(X_val_encoded[c], errors="coerce").fillna(-1)

    X_train = X_train_encoded[shared].fillna(-1).astype(np.float32).values
    X_val = X_val_encoded[shared].fillna(-1).astype(np.float32).values
    y_train = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values

    return X_train, y_train, X_val, y_val
