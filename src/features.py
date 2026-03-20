import pandas as pd
import numpy as np

TARGET_COL = "isFraud"


# ------------------------------------------------------------------
# Core feature preparation (used by LightGBM everywhere)
# ------------------------------------------------------------------

def prepare_features(df):
    """
    Converts a dataframe into (X, y) for LightGBM.
    - Encodes categoricals
    - Fills NaNs
    - Returns numpy arrays
    """

    df = df.copy()

    if TARGET_COL not in df.columns:
        raise ValueError("Target column 'isFraud' not found")

    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL])

    # Encode categoricals
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category").cat.codes

    # LightGBM-safe NaNs
    X = X.fillna(-1)

    return X.values, y


# ------------------------------------------------------------------
# Schema compression for CTGAN
# ------------------------------------------------------------------

def compress_schema(df, hash_bins=20):
    """
    Reduce CTGAN blow-up:
    - Drop ultra-high-cardinality junk
    - Hash email domains
    - Return (compressed_df, categorical_cols, dropped_cols)
    """

    df = df.copy()
    dropped = []

    DROP_COLS = ["DeviceInfo", "id_30", "id_31", "id_33"]
    for c in DROP_COLS:
        if c in df.columns:
            dropped.append(c)
            df = df.drop(columns=c)

    # Hash email domains
    HASH_COLS = ["P_emaildomain", "R_emaildomain"]
    for c in HASH_COLS:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .apply(lambda x: hash(x) % hash_bins if x != "nan" else -1)
            )

    categorical_cols = [
        c for c in df.columns
        if df[c].dtype == "object" or df[c].nunique() < 20
    ]

    return df, categorical_cols, dropped
