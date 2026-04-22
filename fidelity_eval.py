from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors


TARGET_CANDIDATES = {"isFraud", "target", "label", "y"}


def _drop_target_like(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in TARGET_CANDIDATES]
    return df[cols].copy()


def _align_feature_columns(
    left: pd.DataFrame, right: pd.DataFrame, extra: Iterable[pd.DataFrame] | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.DataFrame]]:
    frames = [left.copy(), right.copy()]
    extras = list(extra) if extra is not None else []
    frames.extend([e.copy() for e in extras])
    all_cols = sorted(set().union(*[set(f.columns) for f in frames]))
    aligned = [f.reindex(columns=all_cols) for f in frames]
    return aligned[0], aligned[1], aligned[2:]


def _encode_for_distance(*dfs: pd.DataFrame) -> List[np.ndarray]:
    # Joint one-hot so all matrices share identical encoded feature space.
    tagged = []
    for i, df in enumerate(dfs):
        tmp = df.copy()
        for c in tmp.columns:
            if pd.api.types.is_numeric_dtype(tmp[c]):
                x = pd.to_numeric(tmp[c], errors="coerce")
                med = float(x.median()) if x.notna().any() else 0.0
                tmp[c] = x.replace([np.inf, -np.inf], np.nan).fillna(med)
            else:
                tmp[c] = tmp[c].astype("string").fillna("__MISSING__")
        tmp["__src__"] = i
        tagged.append(tmp)
    joint = pd.concat(tagged, axis=0, ignore_index=True)
    encoded = pd.get_dummies(joint.drop(columns=["__src__"]), dummy_na=True)
    split = []
    start = 0
    for df in dfs:
        n = len(df)
        split.append(encoded.iloc[start : start + n].to_numpy(dtype=float))
        start += n
    return split


def _nearest_distances(query: np.ndarray, ref: np.ndarray, k: int = 1) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(ref)
    dists, _ = nn.kneighbors(query, return_distance=True)
    return dists[:, -1]


def compute_dcr(synthetic_fraud: pd.DataFrame, real_fraud: pd.DataFrame) -> np.ndarray:
    """
    Distance-to-Closest-Real (DCR):
    For each synthetic fraud row, return distance to nearest real fraud row.
    """
    syn = _drop_target_like(synthetic_fraud)
    rf = _drop_target_like(real_fraud)
    syn, rf, _ = _align_feature_columns(syn, rf)
    syn_x, rf_x = _encode_for_distance(syn, rf)
    return _nearest_distances(syn_x, rf_x, k=1)


def compute_nndr(
    synthetic_fraud: pd.DataFrame, real_fraud: pd.DataFrame, real_legit: pd.DataFrame
) -> np.ndarray:
    """
    Nearest-Neighbor Distance Ratio (NNDR):
    (distance to nearest real fraud) / (distance to nearest real legit).
    """
    syn = _drop_target_like(synthetic_fraud)
    rf = _drop_target_like(real_fraud)
    rl = _drop_target_like(real_legit)
    syn, rf, [rl] = _align_feature_columns(syn, rf, extra=[rl])
    syn_x, rf_x, rl_x = _encode_for_distance(syn, rf, rl)

    d_fraud = _nearest_distances(syn_x, rf_x, k=1)
    d_legit = _nearest_distances(syn_x, rl_x, k=1)
    return d_fraud / (d_legit + 1e-12)


def per_column_ks(synthetic_fraud: pd.DataFrame, real_fraud: pd.DataFrame) -> pd.DataFrame:
    """
    KS statistic + p-value per feature column, sorted worst-first by KS.
    """
    syn = _drop_target_like(synthetic_fraud)
    rf = _drop_target_like(real_fraud)
    syn, rf, _ = _align_feature_columns(syn, rf)

    rows = []
    for col in syn.columns:
        s = syn[col]
        r = rf[col]
        # Numeric columns use direct numeric KS.
        if pd.api.types.is_numeric_dtype(s) and pd.api.types.is_numeric_dtype(r):
            s_num = pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy()
            r_num = pd.to_numeric(r, errors="coerce").fillna(0.0).to_numpy()
        else:
            # For categorical/mixed: compare distributions via stable joint coding.
            joint = pd.concat([r.astype("string"), s.astype("string")], axis=0).fillna("__MISSING__")
            codes, _ = pd.factorize(joint, sort=True)
            r_num = codes[: len(r)].astype(float)
            s_num = codes[len(r) :].astype(float)
        ks = ks_2samp(r_num, s_num)
        rows.append(
            {
                "column": col,
                "ks_stat": float(ks.statistic),
                "p_value": float(ks.pvalue),
            }
        )

    out = pd.DataFrame(rows).sort_values(["ks_stat", "p_value"], ascending=[False, True]).reset_index(drop=True)
    return out


def fidelity_summary(
    synthetic_fraud: pd.DataFrame,
    real_fraud: pd.DataFrame,
    real_legit: pd.DataFrame,
    method: str,
    fold: int,
) -> Dict[str, object]:
    """
    Compute DCR/NNDR/KS and print a compact summary.
    """
    dcr = compute_dcr(synthetic_fraud, real_fraud)
    nndr = compute_nndr(synthetic_fraud, real_fraud, real_legit)
    ks_df = per_column_ks(synthetic_fraud, real_fraud)

    summary = {
        "method": method,
        "fold": int(fold),
        "n_synth": int(len(synthetic_fraud)),
        "dcr_mean": float(np.mean(dcr)) if len(dcr) else np.nan,
        "dcr_median": float(np.median(dcr)) if len(dcr) else np.nan,
        "dcr_p95": float(np.quantile(dcr, 0.95)) if len(dcr) else np.nan,
        "nndr_mean": float(np.mean(nndr)) if len(nndr) else np.nan,
        "nndr_median": float(np.median(nndr)) if len(nndr) else np.nan,
        "nndr_p95": float(np.quantile(nndr, 0.95)) if len(nndr) else np.nan,
        "ks_mean": float(ks_df["ks_stat"].mean()) if len(ks_df) else np.nan,
        "ks_max": float(ks_df["ks_stat"].max()) if len(ks_df) else np.nan,
        "worst_ks_columns": ks_df.head(5),
    }

    compact = pd.DataFrame(
        [
            {
                "method": summary["method"],
                "fold": summary["fold"],
                "n_synth": summary["n_synth"],
                "dcr_mean": summary["dcr_mean"],
                "dcr_p95": summary["dcr_p95"],
                "nndr_mean": summary["nndr_mean"],
                "ks_mean": summary["ks_mean"],
                "ks_max": summary["ks_max"],
            }
        ]
    )
    print("\n[Fidelity Summary]")
    print(compact.to_string(index=False))
    print("\n[Worst KS Columns]")
    print(summary["worst_ks_columns"].to_string(index=False))

    return summary


def filter_by_dcr(
    synthetic_fraud: pd.DataFrame, real_fraud: pd.DataFrame, percentile: float = 90
) -> pd.DataFrame:
    """
    Keep synthetic rows whose DCR is <= percentile of nearest-neighbor
    distances among real-fraud rows (plausibility threshold).
    """
    if len(synthetic_fraud) == 0:
        return synthetic_fraud.copy()

    rf = _drop_target_like(real_fraud)
    rf = rf.copy()
    if len(rf) < 2:
        # Not enough real fraud points to estimate a meaningful threshold.
        return synthetic_fraud.copy()

    # Real-to-real nearest-neighbor distances (k=2: first neighbor is itself).
    rf_x = _encode_for_distance(rf)[0]
    real_nn = _nearest_distances(rf_x, rf_x, k=2)
    threshold = float(np.percentile(real_nn, percentile))

    dcr = compute_dcr(synthetic_fraud, real_fraud)
    keep_mask = dcr <= threshold
    return synthetic_fraud.loc[keep_mask].reset_index(drop=True)
