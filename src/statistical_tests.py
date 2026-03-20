"""
Statistical tests for protocol results: paired t-test and permutation test.
Used to compare methods across folds (e.g. CTGAN vs baseline, TabDDPM vs SMOTE).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional

METRIC = "pr_auc"


def paired_t_test(
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[float, float]:
    """Paired t-test: H0 = mean(a - b) == 0. Returns (statistic, p_value)."""
    from scipy import stats
    d = a - b
    if len(d) < 2 or np.std(d) == 0:
        return float(np.mean(d)), 1.0
    t, p = stats.ttest_rel(a, b)
    return float(t), float(p)


def permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_perms: int = 10000,
    seed: Optional[int] = 42,
) -> Tuple[float, float]:
    """
    Paired permutation test: H0 = mean(a - b) == 0.
    Returns (mean_delta, p_value).
    """
    rng = np.random.default_rng(seed)
    d = a - b
    obs_mean = np.mean(d)
    if len(d) < 2:
        return float(obs_mean), 1.0

    perm_means = []
    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(d))
        perm_d = d * signs
        perm_means.append(np.mean(perm_d))

    perm_means = np.array(perm_means)
    p = float(np.mean(np.abs(perm_means) >= np.abs(obs_mean)))
    return float(obs_mean), max(p, 1.0 / (n_perms + 1))


def run_comparisons(
    df: pd.DataFrame,
    metric: str = METRIC,
    baseline_method: str = "baseline",
    methods: Optional[list[str]] = None,
    use_permutation: bool = True,
    n_perms: int = 10000,
) -> pd.DataFrame:
    """
    Compare each method vs baseline (and optionally CTGAN vs SMOTE).
    Expects df with columns: fold, method, {metric}.
    Returns table with method_a, method_b, mean_delta, p_value, significant.
    """
    if metric not in df.columns:
        return pd.DataFrame()

    piv = df.pivot_table(index="fold", columns="method", values=metric)
    available = [c for c in piv.columns if c in piv.columns and piv[c].notna().all()]

    if methods is None:
        methods = [m for m in ["baseline", "smote", "ctgan", "tabddpm"] if m in available]

    rows = []
    # vs baseline
    if baseline_method in piv.columns:
        base_vals = piv[baseline_method].values
        for m in methods:
            if m == baseline_method or m not in piv.columns:
                continue
            vals = piv[m].values
            if len(vals) != len(base_vals) or np.any(np.isnan(vals)) or np.any(np.isnan(base_vals)):
                continue
            # Delta = mean(method - baseline), positive => method beats baseline
            if use_permutation:
                delta, p = permutation_test(vals, base_vals, n_perms=n_perms)
            else:
                t, p = paired_t_test(vals, base_vals)
                delta = float(np.mean(vals - base_vals))
            rows.append({
                "method_a": baseline_method,
                "method_b": m,
                "mean_delta": delta,
                "p_value": p,
                "significant_005": p < 0.05,
                "n_folds": len(vals),
            })

    # CTGAN vs SMOTE
    for (ma, mb) in [("ctgan", "smote"), ("tabddpm", "smote"), ("tabddpm", "ctgan")]:
        if ma not in piv.columns or mb not in piv.columns:
            continue
        va, vb = piv[ma].values, piv[mb].values
        if len(va) != len(vb) or np.any(np.isnan(va)) or np.any(np.isnan(vb)):
            continue
        if use_permutation:
            delta, p = permutation_test(vb, va, n_perms=n_perms)  # va - vb
        else:
            t, p = paired_t_test(va, vb)
            delta = np.mean(va - vb)
        rows.append({
            "method_a": ma,
            "method_b": mb,
            "mean_delta": delta,
            "p_value": p,
            "significant_005": p < 0.05,
            "n_folds": len(va),
        })

    return pd.DataFrame(rows)
