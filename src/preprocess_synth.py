"""
Preprocess dataset for CTGAN and TabDDPM (per rubric: reduce cardinality, compress schema).
Ensures fast, stable runs without OOM.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

TARGET_COL = "isFraud"
TIME_COL = "TransactionDT"

# Ultra-high-cardinality columns to drop (per rubric compress_schema)
DROP_COLS = ["DeviceInfo", "id_30", "id_31", "id_33", "TransactionID"]

# Columns to hash (reduce cardinality per rubric)
HASH_COLS = ["P_emaildomain", "R_emaildomain"]
HASH_BINS = 20

# Priority columns to keep (transaction + key fraud signals)
PRIORITY_COLS = [
    "TransactionAmt",
    "TransactionDT",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "DeviceType",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]
# C and D features (fraud signals)
for i in range(1, 15):
    PRIORITY_COLS.append(f"C{i}")
for i in range(1, 16):
    PRIORITY_COLS.append(f"D{i}")
# id_* with low cardinality
for i in [12, 15, 16, 23, 27, 28, 29, 34, 35, 36, 37, 38]:
    PRIORITY_COLS.append(f"id_{i}")

# Max cols for CTGAN/TabDDPM (per rubric: reduce to avoid blow-up)
MAX_COLS = 100


def _deterministic_hash(s: str, bins: int) -> int:
    """Deterministic hash for reproducible encoding."""
    h = 0
    for c in str(s):
        h = (31 * h + ord(c)) % (2**31)
    return int(h % bins)


def preprocess_for_synth(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Preprocess for CTGAN/TabDDPM (per rubric):
    - Drop ultra-high-cardinality cols
    - Hash email domains to reduce cardinality
    - Limit to ~MAX_COLS key columns
    Returns (df_reduced, used_cols). used_cols excludes TARGET_COL.
    """
    df = df.copy()

    # Drop
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=c)

    # Hash email domains (per rubric compress_schema)
    for c in HASH_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("__MISSING__").apply(
                lambda x: _deterministic_hash(x, HASH_BINS) if x != "__MISSING__" else -1
            )

    # Build used_cols: priority first, then fill up to MAX_COLS
    exclude = {TARGET_COL}
    available = [c for c in df.columns if c not in exclude]
    priority = [c for c in PRIORITY_COLS if c in available]
    rest = [c for c in available if c not in set(priority)]

    # Drop high-cardinality id_* and V* (keep V1-V50 if present)
    def ok_col(c: str) -> bool:
        if c.startswith("id_"):
            return c in set(PRIORITY_COLS)  # only low-card id_* we listed
        if c.startswith("V"):
            try:
                v = int(c[1:])
                return v <= 50  # keep first 50 V cols
            except ValueError:
                return True
        return True

    rest = [c for c in rest if ok_col(c)]
    used_cols = priority + rest[: MAX_COLS - len(priority)]
    used_cols = [c for c in used_cols if c in df.columns][:MAX_COLS]

    df_out = df[[c for c in used_cols] + ([TARGET_COL] if TARGET_COL in df.columns else [])].copy()

    # Coerce high-cardinality object cols to hash (CTGAN struggles with >50 cats)
    for c in used_cols:
        if df_out[c].dtype == "object" and df_out[c].nunique() > 50:
            df_out[c] = df_out[c].astype(str).fillna("__MISSING__").apply(
                lambda x, b=30: _deterministic_hash(x, b) if x != "__MISSING__" else -1
            )

    return df_out, used_cols


def preprocess_fold(
    train_df: pd.DataFrame, val_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Preprocess train and val per fold (fit on train, transform val).
    Avoids leakage from using full-dataset preprocessing before folding.
    Returns (train_prep, val_prep, used_cols).
    """
    train_prep, used_cols = preprocess_for_synth(train_df)
    val_df = val_df.copy()
    # Apply same column selection and transforms to val (no fitting)
    for c in DROP_COLS:
        if c in val_df.columns:
            val_df = val_df.drop(columns=c)
    for c in HASH_COLS:
        if c in val_df.columns:
            val_df[c] = (
                val_df[c]
                .astype(str)
                .fillna("__MISSING__")
                .apply(lambda x: _deterministic_hash(x, HASH_BINS) if x != "__MISSING__" else -1)
            )
    # Select same columns as train; add TARGET_COL if present
    cols_to_use = [c for c in used_cols if c in val_df.columns]
    if TARGET_COL in val_df.columns:
        cols_to_use = cols_to_use + [TARGET_COL]
    val_prep = val_df[[c for c in cols_to_use if c in val_df.columns]].copy()
    for c in used_cols:
        if c in val_prep.columns and val_prep[c].dtype == "object" and val_prep[c].nunique() > 50:
            val_prep[c] = (
                val_prep[c]
                .astype(str)
                .fillna("__MISSING__")
                .apply(lambda x, b=30: _deterministic_hash(x, b) if x != "__MISSING__" else -1)
            )
    # Ensure same schema: fill missing cols in val with -1
    for c in used_cols:
        if c not in val_prep.columns and c != TARGET_COL:
            val_prep[c] = -1
    val_prep = val_prep[[c for c in used_cols if c in val_prep.columns] + ([TARGET_COL] if TARGET_COL in val_prep.columns else [])]
    return train_prep, val_prep, used_cols


def get_cat_cols_for_synth(df: pd.DataFrame, used_cols: List[str]) -> List[str]:
    """Return cols that are categorical (object) for CTGAN discrete_columns."""
    return [c for c in used_cols if c in df.columns and (df[c].dtype == "object" or df[c].dtype.name == "category")]
