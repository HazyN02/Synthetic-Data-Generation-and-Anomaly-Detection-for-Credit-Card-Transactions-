"""
HDBSCAN sensitivity analysis: vary min_cluster_size in [200, 500, 1000].
Uses identical data loading and preprocessing as run_cluster_analysis_hdbscan.py.
Outputs: results/hdbscan_sensitivity.csv
"""
from __future__ import annotations

import gc
import os
import sys
from typing import List

import numpy as np
import pandas as pd

os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.preprocess_synth import (
    DROP_COLS,
    HASH_COLS,
    MAX_COLS,
    PRIORITY_COLS,
    TARGET_COL,
    TIME_COL,
    preprocess_for_synth,
)


# ---------------------------------------------------------------------------
# Column-projected parquet load (identical to run_cluster_analysis_hdbscan.py)
# ---------------------------------------------------------------------------

def _select_columns_to_load(all_cols: List[str]) -> List[str]:
    drop = set(DROP_COLS)
    present = [c for c in all_cols if c not in drop]

    def ok(c: str) -> bool:
        if c.startswith("id_"):
            return c in set(PRIORITY_COLS)
        if c.startswith("V"):
            try:
                return int(c[1:]) <= 50
            except ValueError:
                return True
        return True

    priority = [c for c in PRIORITY_COLS if c in present]
    rest = [c for c in present if c not in set(priority) and ok(c)]
    selected = priority + rest[: MAX_COLS - len(priority)]
    must_have = {TARGET_COL, TIME_COL, *HASH_COLS}
    for c in must_have:
        if c in all_cols and c not in selected:
            selected.append(c)
    return [c for c in selected if c in all_cols]


def _load_train_data(data_dir: str) -> pd.DataFrame:
    for name in ["train_merged.parquet", "train_transaction.csv"]:
        p = os.path.join(data_dir, name)
        if not os.path.exists(p):
            continue
        print(f"[HDBSCAN_SENS] Loading: {p}", flush=True)
        if p.endswith(".parquet"):
            import pyarrow.parquet as _pq
            schema_cols = _pq.ParquetFile(p).schema_arrow.names
            cols = _select_columns_to_load(schema_cols)
            print(f"[HDBSCAN_SENS] Reading {len(cols)} / {len(schema_cols)} columns", flush=True)
            df = pd.read_parquet(p, columns=cols)
        else:
            df = pd.read_csv(p)
        for c in df.select_dtypes(include=["float64"]).columns:
            df[c] = df[c].astype("float32")
        for c in df.select_dtypes(include=["int64"]).columns:
            try:
                df[c] = pd.to_numeric(df[c], downcast="integer")
            except Exception:
                pass
        gc.collect()
        print(f"[HDBSCAN_SENS] df shape={df.shape}", flush=True)
        return df
    raise FileNotFoundError("No train data found in data/")


# ---------------------------------------------------------------------------
# Feature matrix — identical to fraud_cluster_analysis._float_matrix_aligned
# ---------------------------------------------------------------------------

def _infer_obj_num_cols(df: pd.DataFrame, used_cols: List[str]):
    feature_cols = [c for c in used_cols if c != TARGET_COL]
    obj_cols = [c for c in feature_cols if df[c].dtype == "object" or df[c].dtype.name == "category"]
    num_cols = [c for c in feature_cols if c not in obj_cols]
    return obj_cols, num_cols


def _float_matrix(df: pd.DataFrame, used_cols: List[str], enc: OrdinalEncoder | None,
                  obj_cols: List[str], num_cols: List[str], fit_enc: bool):
    feature_cols = [c for c in used_cols if c != TARGET_COL]
    Xdf = df[feature_cols].copy()
    for c in num_cols:
        Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce").fillna(-1)
    if obj_cols:
        tr_cat = Xdf[obj_cols].astype(str).fillna("__MISSING__")
        if fit_enc:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            Xdf[obj_cols] = enc.fit_transform(tr_cat)
        else:
            assert enc is not None
            Xdf[obj_cols] = enc.transform(tr_cat)
    X = Xdf[feature_cols].fillna(-1).astype(np.float32).values
    return X, enc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    data_dir = os.path.join(_ROOT, "data")
    out_csv = os.path.join(_ROOT, "results", "hdbscan_sensitivity.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Load + sort
    df_raw = _load_train_data(data_dir)
    df_sorted = df_raw.sort_values(TIME_COL).reset_index(drop=True)
    del df_raw
    gc.collect()

    # Preprocess (same as canonical run)
    full_prep, used_cols = preprocess_for_synth(df_sorted)
    del df_sorted
    gc.collect()

    n_total = len(full_prep)
    n_fraud = int((full_prep[TARGET_COL] == 1).sum())
    print(f"[HDBSCAN_SENS] preprocessed: rows={n_total}, cols={len(used_cols)}, fraud={n_fraud}", flush=True)

    # Build scaled fraud feature matrix (identical to fit_clusterer)
    fraud = full_prep[full_prep[TARGET_COL] == 1].copy()
    obj_cols, num_cols = _infer_obj_num_cols(fraud, used_cols)
    X, enc = _float_matrix(fraud, used_cols, None, obj_cols, num_cols, fit_enc=True)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    print(f"[HDBSCAN_SENS] Feature matrix: {Xs.shape}", flush=True)
    del fraud, X
    gc.collect()

    # Degeneracy thresholds (same as fit_clusterer)
    NOISE_FRAC_THRESH = 0.50
    BIGGEST_FRAC_THRESH = 0.95
    MIN_CLUSTERS = 2  # need > 1 real cluster

    rows = []
    for mcs in [200, 500, 1000]:
        print(f"\n[HDBSCAN_SENS] min_cluster_size={mcs} ...", flush=True)
        hdb = HDBSCAN(min_cluster_size=mcs, min_samples=None, metric="euclidean", n_jobs=1)
        labels = hdb.fit_predict(Xs)

        uniq = np.unique(labels)
        real_clusters = uniq[uniq >= 0]
        n_clusters = int(len(real_clusters))
        n_noise = int((labels == -1).sum())
        noise_frac = float(n_noise / max(1, len(labels)))

        if n_clusters >= 1:
            sizes = np.array([int((labels == c).sum()) for c in real_clusters])
            biggest_frac = float(sizes.max() / max(1, sizes.sum()))
        else:
            sizes = np.array([], dtype=int)
            biggest_frac = 1.0

        degenerate = n_clusters <= 1 or noise_frac > NOISE_FRAC_THRESH or biggest_frac > BIGGEST_FRAC_THRESH

        cluster_sizes_str = ",".join(str(s) for s in sorted(sizes, reverse=True)) if len(sizes) > 0 else ""

        print(f"  n_clusters={n_clusters}, n_noise={n_noise} ({noise_frac:.3f}), "
              f"biggest_frac={biggest_frac:.3f}, degenerate={degenerate}", flush=True)
        if len(sizes) > 0:
            print(f"  cluster sizes: {cluster_sizes_str}", flush=True)

        rows.append({
            "min_cluster_size": mcs,
            "n_fraud_total": n_fraud,
            "n_clusters_found": n_clusters,
            "n_noise": n_noise,
            "noise_frac": round(noise_frac, 4),
            "biggest_cluster_frac": round(biggest_frac, 4),
            "cluster_sizes": cluster_sizes_str,
            "degenerate": degenerate,
            "note": "degenerate->KMeans fallback in canonical run" if degenerate else "HDBSCAN used",
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)
    print(f"\n[HDBSCAN_SENS] Results:\n{df_out.to_string(index=False)}", flush=True)
    print(f"\n[HDBSCAN_SENS] Saved to {out_csv}", flush=True)


if __name__ == "__main__":
    main()
