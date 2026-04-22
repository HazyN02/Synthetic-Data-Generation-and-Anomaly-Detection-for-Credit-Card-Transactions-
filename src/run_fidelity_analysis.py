#!/usr/bin/env python3
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.folds import get_temporal_folds
from src.preprocess_synth import preprocess_fold, get_cat_cols_for_synth
from src.synth_ctgan import make_synthetic_positives
from src.synth_tabddpm import make_synthetic_positives_tabddpm


TARGET_COL = "isFraud"
TIME_COL = "TransactionDT"


@dataclass
class FidelityCfg:
    n_folds: int = 4
    target_pos_rate: float = 0.10
    max_synth: int = 5000
    ctgan_epochs: int = 3
    tabddpm_epochs: int = 2
    tabddpm_timesteps: int = 50
    tabddpm_hidden: Tuple[int, int] = (512, 512)
    seed: int = 42


def _load_train_df() -> pd.DataFrame:
    root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(root, "data")
    for name in ("train_merged.parquet", "train_transaction.csv"):
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
    raise FileNotFoundError("No train data in data/ (need train_merged.parquet or train_transaction.csv)")


def _quantile_l1(real_s: pd.Series, synth_s: pd.Series) -> float:
    qs = np.linspace(0.05, 0.95, 19)
    r = np.quantile(real_s.to_numpy(dtype=float), qs)
    s = np.quantile(synth_s.to_numpy(dtype=float), qs)
    return float(np.mean(np.abs(r - s)))


def _tv_distance(real_s: pd.Series, synth_s: pd.Series) -> float:
    pr = real_s.astype("string").fillna("__MISSING__").value_counts(normalize=True)
    ps = synth_s.astype("string").fillna("__MISSING__").value_counts(normalize=True)
    keys = pr.index.union(ps.index)
    vr = pr.reindex(keys, fill_value=0.0).to_numpy()
    vs = ps.reindex(keys, fill_value=0.0).to_numpy()
    return float(0.5 * np.abs(vr - vs).sum())


def _corr_mad(real_df: pd.DataFrame, synth_df: pd.DataFrame, num_cols: List[str]) -> float:
    if len(num_cols) < 2:
        return float("nan")
    r = real_df[num_cols].corr().fillna(0.0).to_numpy()
    s = synth_df[num_cols].corr().fillna(0.0).to_numpy()
    iu = np.triu_indices_from(r, k=1)
    return float(np.mean(np.abs(r[iu] - s[iu])))


def _detect_auc(real_df: pd.DataFrame, synth_df: pd.DataFrame, cols: List[str], seed: int) -> float:
    both = pd.concat(
        [real_df[cols].assign(_y=0), synth_df[cols].assign(_y=1)],
        axis=0,
        ignore_index=True,
    )
    X = pd.get_dummies(both[cols], dummy_na=True)
    y = both["_y"].to_numpy()
    x_tr, x_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=20,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(x_tr, y_tr)
    p = clf.predict_proba(x_te)[:, 1]
    return float(roc_auc_score(y_te, p))


def _fidelity_metrics(real_pos: pd.DataFrame, synth_pos: pd.DataFrame, used_cols: List[str], seed: int) -> Dict[str, float]:
    real_pos = real_pos[used_cols].copy()
    synth_pos = synth_pos[used_cols].copy()

    num_cols = []
    cat_cols = []
    for c in used_cols:
        if pd.api.types.is_numeric_dtype(real_pos[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    for c in num_cols:
        med = float(pd.to_numeric(real_pos[c], errors="coerce").median())
        real_pos[c] = pd.to_numeric(real_pos[c], errors="coerce").fillna(med)
        synth_pos[c] = pd.to_numeric(synth_pos[c], errors="coerce").fillna(med)

    qd = [_quantile_l1(real_pos[c], synth_pos[c]) for c in num_cols] if num_cols else [np.nan]
    tv = [_tv_distance(real_pos[c], synth_pos[c]) for c in cat_cols] if cat_cols else [np.nan]
    corr = _corr_mad(real_pos, synth_pos, num_cols)
    auc = _detect_auc(real_pos, synth_pos, used_cols, seed=seed)
    return {
        "num_quantile_l1": float(np.nanmean(qd)),
        "cat_tv_distance": float(np.nanmean(tv)),
        "corr_mad": float(corr),
        "real_vs_synth_auc": float(auc),
        "n_real_pos": int(len(real_pos)),
        "n_synth_pos": int(len(synth_pos)),
    }


def main() -> None:
    cfg = FidelityCfg()
    root = os.path.dirname(os.path.dirname(__file__))
    out_dir = os.path.join(root, "paper", "tables")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "synthetic_fidelity.csv")
    out_summary = os.path.join(out_dir, "synthetic_fidelity_summary.csv")

    df = _load_train_df()
    folds_raw = get_temporal_folds(df, n_folds=cfg.n_folds, time_col=TIME_COL)
    rows: List[Dict[str, float]] = []

    for fold_info in folds_raw:
        fold = int(fold_info["fold"])
        train_df = fold_info["train_df"]
        val_df = fold_info["val_df"]
        train_df, val_df, used_cols = preprocess_fold(train_df, val_df)
        cat_cols = get_cat_cols_for_synth(train_df, used_cols)
        real_pos = train_df[train_df[TARGET_COL] == 1].copy()
        if len(real_pos) < 80:
            continue

        print(f"[fidelity] fold={fold} | real_pos={len(real_pos)}")

        ct_synth = make_synthetic_positives(
            train_df=train_df,
            cat_cols=cat_cols,
            used_cols=used_cols,
            target_pos_rate=cfg.target_pos_rate,
            max_synth=cfg.max_synth,
            epochs=cfg.ctgan_epochs,
            batch_size=512,
            pac=1,
            seed=cfg.seed,
            verbose=False,
        )
        if len(ct_synth) > 0:
            m = _fidelity_metrics(real_pos, ct_synth, used_cols, seed=cfg.seed)
            rows.append({"fold": fold, "method": "ctgan", **m})

        td_synth = make_synthetic_positives_tabddpm(
            train_df=train_df,
            cat_cols=cat_cols,
            used_cols=used_cols,
            target_pos_rate=cfg.target_pos_rate,
            max_synth=cfg.max_synth,
            seed=cfg.seed,
            timesteps=cfg.tabddpm_timesteps,
            epochs=cfg.tabddpm_epochs,
            hidden_dims=list(cfg.tabddpm_hidden),
            verbose=False,
        )
        if len(td_synth) > 0:
            m = _fidelity_metrics(real_pos, td_synth, used_cols, seed=cfg.seed)
            rows.append({"fold": fold, "method": "tabddpm", **m})

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise RuntimeError("No fidelity rows produced.")
    out_df.to_csv(out_path, index=False)

    summary = (
        out_df.groupby("method", as_index=False)[
            ["num_quantile_l1", "cat_tv_distance", "corr_mad", "real_vs_synth_auc", "n_real_pos", "n_synth_pos"]
        ]
        .mean()
        .sort_values("method")
    )
    summary.to_csv(out_summary, index=False)

    print(f"[fidelity] wrote {out_path}")
    print(f"[fidelity] wrote {out_summary}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
