from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def split_feature_types(df: pd.DataFrame, cols: List[str]) -> Tuple[List[str], List[str]]:
    num_cols: List[str] = []
    cat_cols: List[str] = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols


def _safe_numeric(s: pd.Series) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce")
    med = float(x.median()) if x.notna().any() else 0.0
    x = x.fillna(med).replace([np.inf, -np.inf], med)
    return x.to_numpy(dtype=float)


def quantile_l1(real_s: pd.Series, synth_s: pd.Series) -> float:
    qs = np.linspace(0.05, 0.95, 19)
    r = np.quantile(_safe_numeric(real_s), qs)
    s = np.quantile(_safe_numeric(synth_s), qs)
    return float(np.mean(np.abs(r - s)))


def normalized_wasserstein(real_s: pd.Series, synth_s: pd.Series) -> float:
    r = _safe_numeric(real_s)
    s = _safe_numeric(synth_s)
    w = float(wasserstein_distance(r, s))
    # Normalize by robust scale to keep columns comparable
    scale = float(np.quantile(r, 0.95) - np.quantile(r, 0.05))
    return float(w / (abs(scale) + 1e-9))


def tv_distance(real_s: pd.Series, synth_s: pd.Series) -> float:
    pr = real_s.astype("string").fillna("__MISSING__").value_counts(normalize=True)
    ps = synth_s.astype("string").fillna("__MISSING__").value_counts(normalize=True)
    keys = pr.index.union(ps.index)
    vr = pr.reindex(keys, fill_value=0.0).to_numpy()
    vs = ps.reindex(keys, fill_value=0.0).to_numpy()
    return float(0.5 * np.abs(vr - vs).sum())


def js_divergence(real_s: pd.Series, synth_s: pd.Series) -> float:
    pr = real_s.astype("string").fillna("__MISSING__").value_counts(normalize=True)
    ps = synth_s.astype("string").fillna("__MISSING__").value_counts(normalize=True)
    keys = pr.index.union(ps.index)
    p = pr.reindex(keys, fill_value=0.0).to_numpy(dtype=float)
    q = ps.reindex(keys, fill_value=0.0).to_numpy(dtype=float)
    m = 0.5 * (p + q)
    eps = 1e-12
    kl_pm = np.sum(p * (np.log(p + eps) - np.log(m + eps)))
    kl_qm = np.sum(q * (np.log(q + eps) - np.log(m + eps)))
    return float(0.5 * (kl_pm + kl_qm))


def corr_mad(real_df: pd.DataFrame, synth_df: pd.DataFrame, num_cols: List[str]) -> float:
    if len(num_cols) < 2:
        return float("nan")
    real_num = real_df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    synth_num = synth_df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    r = real_num.corr(method="spearman").fillna(0.0).to_numpy()
    s = synth_num.corr(method="spearman").fillna(0.0).to_numpy()
    iu = np.triu_indices_from(r, k=1)
    return float(np.mean(np.abs(r[iu] - s[iu])))


def real_vs_synth_auc(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    cols: List[str],
    seed: int = 42,
) -> float:
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
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=10,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(x_tr, y_tr)
    proba = clf.predict_proba(x_te)[:, 1]
    return float(roc_auc_score(y_te, proba))


def rare_category_coverage(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    cat_cols: List[str],
    rare_threshold: float = 0.01,
) -> float:
    if not cat_cols:
        return float("nan")
    coverages: List[float] = []
    for c in cat_cols:
        r = real_df[c].astype("string").fillna("__MISSING__")
        s = synth_df[c].astype("string").fillna("__MISSING__")
        p = r.value_counts(normalize=True)
        rare = set(p[p <= rare_threshold].index)
        if not rare:
            continue
        hit = len(rare.intersection(set(s.unique())))
        coverages.append(hit / max(1, len(rare)))
    return float(np.mean(coverages)) if coverages else float("nan")


def schema_validity_rate(
    synth_df: pd.DataFrame,
    numeric_bounds: Dict[str, Tuple[float, float]],
    allowed_categories: Dict[str, List[str]],
) -> float:
    if len(synth_df) == 0:
        return 0.0
    valid = np.ones(len(synth_df), dtype=bool)
    for c, (lo, hi) in numeric_bounds.items():
        if c not in synth_df.columns:
            continue
        x = pd.to_numeric(synth_df[c], errors="coerce")
        valid &= x.notna().to_numpy()
        valid &= (x >= lo).to_numpy() & (x <= hi).to_numpy()
    for c, allowed in allowed_categories.items():
        if c not in synth_df.columns:
            continue
        x = synth_df[c].astype("string").fillna("__MISSING__")
        valid &= x.isin(set(allowed)).to_numpy()
    return float(np.mean(valid))


def compute_fidelity_metrics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    used_cols: List[str],
    seed: int,
) -> Dict[str, float]:
    real_df = real_df[used_cols].copy()
    synth_df = synth_df[used_cols].copy()
    num_cols, cat_cols = split_feature_types(real_df, used_cols)

    quantiles = [quantile_l1(real_df[c], synth_df[c]) for c in num_cols] if num_cols else [np.nan]
    wass = [normalized_wasserstein(real_df[c], synth_df[c]) for c in num_cols] if num_cols else [np.nan]
    tvs = [tv_distance(real_df[c], synth_df[c]) for c in cat_cols] if cat_cols else [np.nan]
    jss = [js_divergence(real_df[c], synth_df[c]) for c in cat_cols] if cat_cols else [np.nan]

    numeric_bounds = {}
    for c in num_cols:
        x = _safe_numeric(real_df[c])
        numeric_bounds[c] = (float(np.min(x)), float(np.max(x)))
    allowed_categories = {
        c: sorted(real_df[c].astype("string").fillna("__MISSING__").unique().tolist())
        for c in cat_cols
    }

    return {
        "num_quantile_l1": float(np.nanmean(quantiles)),
        "num_wasserstein_norm": float(np.nanmean(wass)),
        "cat_tv_distance": float(np.nanmean(tvs)),
        "cat_js_divergence": float(np.nanmean(jss)),
        "corr_mad": float(corr_mad(real_df, synth_df, num_cols)),
        "real_vs_synth_auc": float(real_vs_synth_auc(real_df, synth_df, used_cols, seed=seed)),
        "rare_cat_coverage": float(rare_category_coverage(real_df, synth_df, cat_cols)),
        "schema_validity_rate": float(schema_validity_rate(synth_df, numeric_bounds, allowed_categories)),
        "n_real": int(len(real_df)),
        "n_synth": int(len(synth_df)),
    }
