# src/folds.py
"""
Canonical temporal fold split used by protocol, drift, sliding, SMOTE.
Ensures fold alignment for valid drift-harm correlation.
"""
from __future__ import annotations

from typing import List, Dict, Any

import numpy as np
import pandas as pd


def get_temporal_folds(
    df: pd.DataFrame,
    n_folds: int,
    time_col: str = "TransactionDT",
) -> List[Dict[str, Any]]:
    """
    Make K temporal folds by sorting by time and slicing into K+1 chunks.
    Fold i trains on chunks [:i+1] and validates on chunk [i+1].

    Returns list of {"fold", "train_df", "val_df", "train_rows", "val_rows"}.
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    n_chunks = n_folds + 1
    n = len(df)
    edges = np.linspace(0, n, num=n_chunks + 1, dtype=int)

    folds = []
    for i in range(n_folds):
        tr_start, tr_end = int(edges[0]), int(edges[i + 1])
        va_start, va_end = int(edges[i + 1]), int(edges[i + 2])

        train_df = df.iloc[tr_start:tr_end].reset_index(drop=True)
        val_df = df.iloc[va_start:va_end].reset_index(drop=True)

        folds.append({
            "fold": i,
            "train_df": train_df,
            "val_df": val_df,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
        })
    return folds
