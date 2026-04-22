#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from ctgan import CTGAN

from fidelity_eval import fidelity_summary
from src.folds import get_temporal_folds
from src.preprocess_synth import preprocess_fold, get_cat_cols_for_synth
from src.synth_tabddpm import make_synthetic_positives_tabddpm


TARGET_COL = "isFraud"
TIME_COL = "TransactionDT"
N_FOLDS = 4
OUT_CSV = "results/protocol/fidelity_results_retrained.csv"


def _repo_root() -> str:
    return os.path.dirname(__file__)


def _load_train_df() -> pd.DataFrame:
    root = _repo_root()
    data_dir = os.path.join(root, "data")
    for name in ("train_merged.parquet", "train_transaction.csv"):
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
    raise FileNotFoundError("No train data in data/ (need train_merged.parquet or train_transaction.csv)")


def _prep_ctgan_input(
    df: pd.DataFrame, used_cols: List[str], cat_cols: List[str]
) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, float]]:
    proc = df[used_cols].copy()
    cat_cols = [c for c in cat_cols if c in proc.columns]
    cont_cols = [c for c in proc.columns if c not in cat_cols]

    # categorical as strings
    for c in cat_cols:
        proc[c] = proc[c].astype("string").fillna("__MISSING__")

    medians: Dict[str, float] = {}
    for c in cont_cols:
        proc[c] = pd.to_numeric(proc[c], errors="coerce")
        med = float(proc[c].median(skipna=True)) if proc[c].notna().any() else 0.0
        medians[c] = med
        proc[c] = proc[c].replace([np.inf, -np.inf], np.nan).fillna(med)

    return proc, cat_cols, cont_cols, medians


def _sample_ctgan_proper(
    train_df: pd.DataFrame,
    used_cols: List[str],
    cat_cols: List[str],
    n_synth: int,
    seed: int = 42,
) -> pd.DataFrame:
    pos_df = train_df[train_df[TARGET_COL] == 1].copy()
    if len(pos_df) < 50 or n_synth <= 0:
        return pd.DataFrame(columns=used_cols + [TARGET_COL])

    x, discrete_cols, cont_cols, medians = _prep_ctgan_input(pos_df, used_cols, cat_cols)
    model = CTGAN(
        epochs=300,
        batch_size=500,
        log_frequency=True,
        discriminator_steps=5,
        verbose=True,
        enable_gpu=False,
    )
    model.fit(x, discrete_columns=discrete_cols)
    synth_x = model.sample(int(n_synth))[used_cols].copy()

    for c in cont_cols:
        synth_x[c] = pd.to_numeric(synth_x[c], errors="coerce").fillna(medians.get(c, 0.0))
    for c in discrete_cols:
        synth_x[c] = synth_x[c].astype("string").fillna("__MISSING__")

    synth_x[TARGET_COL] = 1
    return synth_x[used_cols + [TARGET_COL]]


def _sample_tabddpm_proper(
    train_df: pd.DataFrame,
    used_cols: List[str],
    cat_cols: List[str],
    n_synth: int,
    seed: int = 42,
) -> pd.DataFrame:
    if n_synth <= 0:
        return pd.DataFrame(columns=used_cols + [TARGET_COL])
    # Reuse existing tabddpm pipeline with requested proper hyperparameters.
    return make_synthetic_positives_tabddpm(
        train_df=train_df,
        cat_cols=cat_cols,
        used_cols=used_cols,
        target_pos_rate=0.05,  # same operating point as quick protocol
        max_synth=int(n_synth),
        seed=seed,
        timesteps=1000,
        epochs=100,
        hidden_dims=[512, 512, 512],
        verbose=True,
    )


def main() -> None:
    root = _repo_root()
    out_path = os.path.join(root, OUT_CSV)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = _load_train_df()
    folds = get_temporal_folds(df, n_folds=N_FOLDS, time_col=TIME_COL)
    records: List[Dict[str, float]] = []

    for fold_info in folds:
        fold = int(fold_info["fold"])
        train_df = fold_info["train_df"]
        val_df = fold_info["val_df"]
        train_df, val_df, used_cols = preprocess_fold(train_df, val_df)
        cat_cols = get_cat_cols_for_synth(train_df, used_cols)

        real_fraud = train_df[train_df[TARGET_COL] == 1].copy()
        real_legit = train_df[train_df[TARGET_COL] == 0].copy()
        if len(real_fraud) < 50:
            continue

        # Keep sample size controlled for fidelity diagnostics.
        n_synth = int(min(5000, max(500, len(real_fraud))))

        # CTGAN
        ct_synth = _sample_ctgan_proper(
            train_df=train_df,
            used_cols=used_cols,
            cat_cols=cat_cols,
            n_synth=n_synth,
        )
        if len(ct_synth) > 0:
            ct_sum = fidelity_summary(
                synthetic_fraud=ct_synth,
                real_fraud=real_fraud,
                real_legit=real_legit,
                method="ctgan_retrained",
                fold=fold,
            )
            records.append(
                {
                    "method": "ctgan_retrained",
                    "fold": fold,
                    "mean_dcr": ct_sum.get("dcr_mean"),
                    "median_dcr": ct_sum.get("dcr_median"),
                    "mean_nndr": ct_sum.get("nndr_mean"),
                    "median_nndr": ct_sum.get("nndr_median"),
                    "n_synthetic": int(len(ct_synth)),
                    # No filtering in this retraining script; keep schema parity.
                    "n_after_filter": int(len(ct_synth)),
                }
            )

        # TabDDPM
        td_synth = _sample_tabddpm_proper(
            train_df=train_df,
            used_cols=used_cols,
            cat_cols=cat_cols,
            n_synth=n_synth,
        )
        if len(td_synth) > 0:
            td_sum = fidelity_summary(
                synthetic_fraud=td_synth,
                real_fraud=real_fraud,
                real_legit=real_legit,
                method="tabddpm_retrained",
                fold=fold,
            )
            records.append(
                {
                    "method": "tabddpm_retrained",
                    "fold": fold,
                    "mean_dcr": td_sum.get("dcr_mean"),
                    "median_dcr": td_sum.get("dcr_median"),
                    "mean_nndr": td_sum.get("nndr_mean"),
                    "median_nndr": td_sum.get("nndr_median"),
                    "n_synthetic": int(len(td_synth)),
                    # No filtering in this retraining script; keep schema parity.
                    "n_after_filter": int(len(td_synth)),
                }
            )

    out_df = pd.DataFrame(
        records,
        columns=[
            "method",
            "fold",
            "mean_dcr",
            "median_dcr",
            "mean_nndr",
            "median_nndr",
            "n_synthetic",
            "n_after_filter",
        ],
    )
    out_df.to_csv(out_path, index=False)
    print(f"[DONE] Wrote retrained fidelity results -> {out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
