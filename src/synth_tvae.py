# src/synth_tvae.py
"""
TVAE synthesizer wrapper using SDV >= 1.0 API.

SDV 1.x requires a SingleTableMetadata object. We build it from the
fraud-only training rows, overriding column sdtypes using the same
cat_cols list used by CTGAN/TabDDPM — so categorical detection stays
consistent across methods.

Interface mirrors synth_ctgan.py: fit_tvae / sample_tvae /
make_synthetic_positives_tvae.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


TARGET_COL = "isFraud"


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

@dataclass
class TVAEArtifacts:
    used_cols: List[str]
    cat_cols: List[str]
    cont_cols: List[str]
    medians: Dict[str, float]
    seed: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_metadata(df: pd.DataFrame, cat_cols: List[str]):
    """
    Build SDV Metadata from a dataframe (SDV >= 1.0 new-style API).
    - infer_keys=None prevents SDV from auto-detecting primary keys
      (avoids spurious 'id' type assignments on integer-like columns).
    - Columns listed in cat_cols are forced to sdtype='categorical'.
    - All other columns are sdtype='numerical'.
    """
    try:
        from sdv.metadata import Metadata as _Metadata
        meta = _Metadata.detect_from_dataframe(df, infer_keys=None)
    except ImportError:
        # Fallback for older SDV 1.x that still uses SingleTableMetadata
        from sdv.metadata import SingleTableMetadata as _Metadata  # type: ignore
        meta = _Metadata()
        meta.detect_from_dataframe(df)

    # Override sdtype for known categoricals
    cat_set = set(cat_cols)
    for col in df.columns:
        sdtype = "categorical" if col in cat_set else "numerical"
        try:
            meta.update_column(col, sdtype=sdtype)
        except Exception:
            # update_column signature varies between SDV sub-versions
            try:
                meta.update_column(column_name=col, sdtype=sdtype)
            except Exception:
                pass

    return meta


def _prep_for_tvae(
    df: pd.DataFrame,
    used_cols: List[str],
    cat_cols: List[str],
    seed: int,
) -> Tuple[pd.DataFrame, TVAEArtifacts]:
    """
    Prepare a features-only DataFrame for TVAESynthesizer.
    - Categorical cols: cast to string, fill NaN with sentinel
    - Continuous cols: coerce numeric, median-impute, replace inf
    """
    proc = df[used_cols].copy()
    cat_cols_present = [c for c in cat_cols if c in proc.columns]
    cont_cols = [c for c in used_cols if c not in set(cat_cols_present)]

    for c in cat_cols_present:
        proc[c] = proc[c].astype("string").fillna("__MISSING__")

    medians: Dict[str, float] = {}
    for c in cont_cols:
        if not pd.api.types.is_numeric_dtype(proc[c]):
            proc[c] = pd.to_numeric(proc[c], errors="coerce")
        proc[c] = proc[c].replace([np.inf, -np.inf], np.nan)
        med = float(proc[c].median(skipna=True)) if proc[c].notna().any() else 0.0
        medians[c] = med
        proc[c] = proc[c].fillna(med)

    artifacts = TVAEArtifacts(
        used_cols=used_cols,
        cat_cols=cat_cols_present,
        cont_cols=cont_cols,
        medians=medians,
        seed=seed,
    )
    return proc, artifacts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_tvae(
    train_df: pd.DataFrame,
    cat_cols: List[str],
    used_cols: List[str],
    epochs: int = 300,
    batch_size: int = 500,
    embedding_dim: int = 128,
    compress_dims: Tuple[int, ...] = (128, 128),
    decompress_dims: Tuple[int, ...] = (128, 128),
    l2scale: float = 1e-5,
    loss_factor: int = 2,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[Any, TVAEArtifacts]:
    """
    Fit TVAESynthesizer (SDV >= 1.0) on train_df (expected: fraud-only rows).
    Returns (synthesizer, artifacts).

    enable_gpu is always False to guarantee CPU reproducibility;
    GPU support can be added later via the cuda param.
    """
    from sdv.single_table import TVAESynthesizer

    proc, artifacts = _prep_for_tvae(train_df, used_cols, cat_cols, seed)

    if verbose:
        print(
            f"[TVAE] Training rows: {len(proc)} | cols: {proc.shape[1]} | "
            f"cat={len(artifacts.cat_cols)} | cont={len(artifacts.cont_cols)} | "
            f"epochs={epochs}"
        )

    meta = _build_metadata(proc, artifacts.cat_cols)

    synthesizer = TVAESynthesizer(
        metadata=meta,
        enforce_min_max_values=True,
        enforce_rounding=False,          # avoid over-rounding continuous cols
        embedding_dim=embedding_dim,
        compress_dims=compress_dims,
        decompress_dims=decompress_dims,
        l2scale=l2scale,
        batch_size=batch_size,
        verbose=verbose,
        epochs=epochs,
        loss_factor=loss_factor,
        enable_gpu=False,                # CPU-safe; override if GPU available
    )
    synthesizer.fit(proc)
    return synthesizer, artifacts


def sample_tvae(
    synthesizer: Any,
    n: int,
    artifacts: TVAEArtifacts,
    chunk: int = 5000,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Sample n rows from a fitted TVAESynthesizer.
    Samples in chunks to stay memory-safe on large n.
    Returns a DataFrame with columns == artifacts.used_cols.
    """
    remaining = int(n)
    out = []
    i = 0
    while remaining > 0:
        take = min(chunk, remaining)
        i += 1
        if verbose:
            print(f"[TVAE] Sampling chunk {i}: {take} rows (remaining after: {remaining - take})")
        samp = synthesizer.sample(num_rows=take)
        out.append(samp)
        remaining -= take

    synth = pd.concat(out, axis=0, ignore_index=True)

    # Align columns to used_cols (SDV may reorder or add extra cols)
    for c in artifacts.used_cols:
        if c not in synth.columns:
            synth[c] = artifacts.medians.get(c, 0.0)
    synth = synth[artifacts.used_cols].copy()

    # Post-process continuous: coerce to numeric + fill
    for c in artifacts.cont_cols:
        synth[c] = pd.to_numeric(synth[c], errors="coerce").fillna(artifacts.medians.get(c, 0.0))
        synth[c] = synth[c].replace([np.inf, -np.inf], artifacts.medians.get(c, 0.0))

    # Post-process categorical: keep as string
    for c in artifacts.cat_cols:
        synth[c] = synth[c].astype("string").fillna("__MISSING__")

    return synth


def make_synthetic_positives_tvae(
    train_df: pd.DataFrame,
    cat_cols: List[str],
    used_cols: List[str],
    target_pos_rate: float,
    max_synth: int = 50000,
    epochs: int = 300,
    batch_size: int = 500,
    embedding_dim: int = 128,
    compress_dims: Tuple[int, ...] = (128, 128),
    decompress_dims: Tuple[int, ...] = (128, 128),
    l2scale: float = 1e-5,
    loss_factor: int = 2,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fit TVAE on real fraud positives from train_df, then sample enough
    synthetic positives to reach target_pos_rate when mixed with train_df.

    Matches the interface of make_synthetic_positives() (CTGAN) and
    make_synthetic_positives_tabddpm() so it slots into run_protocol.py
    without changes.

    Returns a DataFrame with columns used_cols + [TARGET_COL], label=1.
    Returns an empty DataFrame if synth_add == 0 or n_pos < 50.
    """
    assert TARGET_COL in train_df.columns, f"Missing {TARGET_COL}"

    pos_df = train_df[train_df[TARGET_COL] == 1].copy()
    neg_df = train_df[train_df[TARGET_COL] == 0]

    n_pos = len(pos_df)
    n_neg = len(neg_df)

    if n_pos < 50:
        if verbose:
            print(f"[TVAE] Too few positives to train reliably: n_pos={n_pos}. Returning empty.")
        return pd.DataFrame(columns=used_cols + [TARGET_COL])

    # Compute how many synthetic positives to add
    total = n_pos + n_neg
    target = float(target_pos_rate)
    target_pos = int(math.ceil(target * total / max(1e-9, 1.0 - target)))
    synth_add = min(max_synth, max(0, target_pos - n_pos))

    if verbose:
        print(
            f"[TVAE] n_pos={n_pos}, n_neg={n_neg}, "
            f"target_pos={n_pos + synth_add}, synth_add={synth_add}"
        )

    if synth_add == 0:
        return pd.DataFrame(columns=used_cols + [TARGET_COL])

    synthesizer, artifacts = fit_tvae(
        train_df=pos_df,
        cat_cols=cat_cols,
        used_cols=used_cols,
        epochs=epochs,
        batch_size=batch_size,
        embedding_dim=embedding_dim,
        compress_dims=compress_dims,
        decompress_dims=decompress_dims,
        l2scale=l2scale,
        loss_factor=loss_factor,
        seed=seed,
        verbose=verbose,
    )

    synth_x = sample_tvae(synthesizer, n=synth_add, artifacts=artifacts, verbose=verbose)
    synth_x[TARGET_COL] = 1

    return synth_x[used_cols + [TARGET_COL]].copy()
