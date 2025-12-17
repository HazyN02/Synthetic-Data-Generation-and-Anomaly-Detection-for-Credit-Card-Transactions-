import numpy as np
from src.config import TIME_COL, N_SPLITS


def time_based_cv(df):
    """
    Leakage-safe time-based cross-validation.
    Progressive training, contiguous validation.
    """
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    n = len(df)

    fold_sizes = np.full(N_SPLITS, n // N_SPLITS, dtype=int)
    fold_sizes[: n % N_SPLITS] += 1

    indices = np.arange(n)
    current = 0

    splits = []
    for i in range(N_SPLITS):
        start = current
        stop = current + fold_sizes[i]

        val_idx = indices[start:stop]
        train_idx = indices[:start]

        # Skip empty training fold (first fold)
        if len(train_idx) == 0:
            current = stop
            continue

        splits.append((train_idx, val_idx))
        current = stop

    return splits
