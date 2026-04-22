# src/synth_ctgan.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

from ctgan import CTGAN


TARGET_COL = "isFraud"


@dataclass
class CTGANArtifacts:
    used_cols: List[str]
    cat_cols: List[str]
    cont_cols: List[str]
    medians: Dict[str, float]
    seed: int


def _split_cols(df: pd.DataFrame, cat_cols: List[str]) -> Tuple[List[str], List[str]]:
    cat_cols = [c for c in cat_cols if c in df.columns]
    cont_cols = [c for c in df.columns if c not in cat_cols]
    return cat_cols, cont_cols


def _impute_continuous(df: pd.DataFrame, cont_cols: List[str]) -> Dict[str, float]:
    medians: Dict[str, float] = {}
    for c in cont_cols:
        # ensure numeric if possible
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        med = float(df[c].median(skipna=True)) if df[c].notna().any() else 0.0
        medians[c] = med
        df[c] = df[c].fillna(med)
    return medians


def _prep_for_ctgan(df: pd.DataFrame, used_cols: List[str], cat_cols: List[str], seed: int) -> Tuple[pd.DataFrame, CTGANArtifacts]:
    proc = df[used_cols].copy()

    # categorical as strings, fill NaN with sentinel
    for c in cat_cols:
        if c in proc.columns:
            proc[c] = proc[c].astype("string").fillna("__MISSING__")

    # continuous: numeric + median impute
    cat_cols_present, cont_cols = _split_cols(proc, cat_cols)
    medians = _impute_continuous(proc, cont_cols)

    # final safety: no NaNs
    proc = proc.replace([np.inf, -np.inf], np.nan)
    for c in cont_cols:
        proc[c] = proc[c].fillna(medians.get(c, 0.0))
    for c in cat_cols_present:
        proc[c] = proc[c].fillna("__MISSING__")

    artifacts = CTGANArtifacts(
        used_cols=used_cols,
        cat_cols=cat_cols_present,
        cont_cols=cont_cols,
        medians=medians,
        seed=seed,
    )
    return proc, artifacts


def fit_ctgan(
    train_df: pd.DataFrame,
    cat_cols: List[str],
    used_cols: List[str],
    epochs: int = 10,
    batch_size: int = 256,
    pac: int = 1,
    seed: int = 0,
    verbose: bool = True,
    discriminator_steps: int = 1,
) -> Tuple[CTGAN, CTGANArtifacts]:
    """
    Fits CTGAN (ctgan package) on a prepared dataframe.
    IMPORTANT: pac is forced to 1 by default to avoid assertion failures with small datasets.
    """
    # pac must be >=1; pac=1 avoids the discriminator batch multiple constraint
    pac = int(max(1, pac))
    batch_size = int(batch_size)

    proc, artifacts = _prep_for_ctgan(train_df, used_cols=used_cols, cat_cols=cat_cols, seed=seed)

    # Reduce batch_size when many cols to avoid OOM / stalls (allow up to requested batch_size, e.g. 500)
    n_cols = proc.shape[1]
    if n_cols > 200:
        bs = min(batch_size, 128)
    elif n_cols > 150:
        bs = min(batch_size, 500)
    else:
        bs = batch_size

    if verbose:
        print(f"[CTGAN] Training rows: {len(proc)} | cols: {n_cols} | pac={pac} | batch_size={bs}")
        print(f"[CTGAN] discrete cols: {len(artifacts.cat_cols)} | continuous cols: {len(artifacts.cont_cols)}")

    model = CTGAN(
        epochs=epochs,
        batch_size=bs,
        pac=pac,
        discriminator_steps=int(max(1, discriminator_steps)),
        log_frequency=True,
        verbose=verbose,
        enable_gpu=False,
    )
    model.fit(proc, discrete_columns=artifacts.cat_cols)
    return model, artifacts


def sample_ctgan(
    model: CTGAN,
    n: int,
    artifacts: CTGANArtifacts,
    chunk: int = 5000,
    verbose: bool = True,
) -> pd.DataFrame:
    remaining = int(n)
    out = []
    i = 0
    while remaining > 0:
        take = min(chunk, remaining)
        i += 1
        if verbose:
            print(f"[CTGAN] Sampling chunk {i}: {take} rows (remaining after: {remaining - take})")
        samp = model.sample(take)
        out.append(samp)
        remaining -= take

    synth = pd.concat(out, axis=0, ignore_index=True)
    # ensure correct column order
    synth = synth[artifacts.used_cols].copy()

    # postprocess numeric columns: coerce to numeric + fill
    for c in artifacts.cont_cols:
        synth[c] = pd.to_numeric(synth[c], errors="coerce").fillna(artifacts.medians.get(c, 0.0))

    # categorical: keep as string
    for c in artifacts.cat_cols:
        synth[c] = synth[c].astype("string").fillna("__MISSING__")

    return synth


def _apply_recency(
    pos_df: pd.DataFrame,
    recency_frac: float,
    time_col: str,
    min_pos: int,
    verbose: bool,
) -> pd.DataFrame:
    """Restrict to last recency_frac of positives by time. Fallback to full if too few."""
    if recency_frac is None or recency_frac >= 1.0:
        return pos_df
    if time_col not in pos_df.columns:
        raise ValueError(f"[recency] time_col={time_col} not in dataframe")
    n_pos = len(pos_df)
    n_recent = max(min_pos, int(np.ceil(recency_frac * n_pos)))
    if n_recent >= n_pos:
        return pos_df
    pos_sorted = pos_df.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    pos_recent = pos_sorted.tail(n_recent)
    if verbose:
        print(f"[recency] n_pos={n_pos} -> n_recent={n_recent} (frac={recency_frac})")
    return pos_recent


def make_synthetic_positives(
    train_df: pd.DataFrame,
    cat_cols: List[str],
    used_cols: List[str],
    target_pos_rate: float,
    max_synth: int = 50000,
    epochs: int = 10,
    batch_size: int = 256,
    pac: int = 1,
    seed: int = 0,
    verbose: bool = True,
    discriminator_steps: int = 1,
    recency_frac: float | None = None,
    time_col: str = "TransactionDT",
    min_pos_for_recency: int = 50,
) -> pd.DataFrame:
    """
    Train CTGAN on REAL POSITIVES ONLY, then sample additional positives to reach target_pos_rate.
    If recency_frac in (0,1), use only last recency_frac of positives (by time_col) for generator.
    Returns a dataframe with TARGET_COL included and =1.
    """
    assert TARGET_COL in train_df.columns, f"Missing {TARGET_COL}"

    df = train_df.copy()
    pos_df = df[df[TARGET_COL] == 1].copy()
    neg_df = df[df[TARGET_COL] == 0].copy()

    n_pos_full = len(pos_df)
    pos_df = _apply_recency(pos_df, recency_frac, time_col, min_pos_for_recency, verbose)
    n_pos = len(pos_df)
    n_neg = len(neg_df)
    if n_pos < 50:
        raise ValueError(f"[CTGAN] Too few positives to train CTGAN reliably: n_pos={n_pos}")

    target = float(target_pos_rate)
    total = n_pos_full + n_neg
    target_pos = int(np.ceil(target * total / max(1e-9, (1.0 - target))))
    synth_add = max(0, target_pos - n_pos_full)
    synth_add = int(min(synth_add, max_synth))

    if verbose:
        print(f"[CTGAN] n_pos={n_pos} (gen), n_pos_full={n_pos_full}, n_neg={n_neg}, target_pos={n_pos_full + synth_add}, synth_add={synth_add}")

    if synth_add == 0:
        return pd.DataFrame(columns=used_cols + [TARGET_COL])

    # warn if cat columns missing
    missing = [c for c in cat_cols if c not in pos_df.columns]
    if missing and verbose:
        print(f"[CTGAN] WARNING: missing columns (ignored): {missing}")

    # Fit on positives only (features only)
    model, artifacts = fit_ctgan(
        train_df=pos_df,
        cat_cols=cat_cols,
        used_cols=used_cols,
        epochs=epochs,
        batch_size=batch_size,
        pac=pac,
        seed=seed,
        verbose=verbose,
        discriminator_steps=discriminator_steps,
    )

    synth_x = sample_ctgan(model, n=synth_add, artifacts=artifacts, verbose=verbose)
    synth_x[TARGET_COL] = 1

    # return ONLY synthetic positives
    return synth_x[used_cols + [TARGET_COL]].copy()
