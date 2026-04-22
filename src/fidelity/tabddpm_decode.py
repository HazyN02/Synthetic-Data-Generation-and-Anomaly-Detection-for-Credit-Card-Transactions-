from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.synth_tabddpm import TabDDPMArtifacts


def decode_tabddpm_samples(
    raw_samples: np.ndarray,
    artifacts: TabDDPMArtifacts,
) -> pd.DataFrame:
    """
    Decode mixed-type samples from TabDDPM latent output.
    Layout: [continuous_z ... | onehot(cat1) | onehot(cat2) | ...].
    """
    n_cont = len(artifacts.cont_cols)
    out = pd.DataFrame(index=np.arange(raw_samples.shape[0]))

    # Decode continuous part (inverse z-score)
    if n_cont > 0:
        cont = raw_samples[:, :n_cont]
        for i, c in enumerate(artifacts.cont_cols):
            out[c] = cont[:, i] * artifacts.cont_stds[c] + artifacts.cont_means[c]

    # Decode one-hot blocks per categorical feature
    offset = n_cont
    for c in artifacts.cat_cols:
        cats = artifacts.cat_maps[c]
        width = len(cats)
        if width <= 0:
            out[c] = "__MISSING__"
            continue
        block = raw_samples[:, offset : offset + width]
        idx = np.argmax(block, axis=1)
        vals = [cats[int(i)] for i in idx]
        out[c] = pd.Series(vals, dtype="string")
        offset += width

    # Keep schema order
    for c in artifacts.used_cols:
        if c not in out.columns:
            out[c] = 0.0
    out = out[artifacts.used_cols].copy()
    return out


def sanitize_tabddpm_decoded(
    decoded_df: pd.DataFrame,
    artifacts: TabDDPMArtifacts,
) -> pd.DataFrame:
    """
    Ensure decoded output respects basic schema constraints.
    """
    df = decoded_df.copy()
    for c in artifacts.cont_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            med = float(df[c].median()) if df[c].notna().any() else 0.0
            df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(med)
    for c in artifacts.cat_cols:
        if c in df.columns:
            allowed = set(artifacts.cat_maps.get(c, []))
            x = df[c].astype("string").fillna("__MISSING__")
            if allowed:
                fallback = next(iter(allowed))
                x = x.where(x.isin(allowed), fallback)
            df[c] = x
    return df
