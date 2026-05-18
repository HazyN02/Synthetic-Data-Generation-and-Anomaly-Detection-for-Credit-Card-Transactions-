"""
TabDDPM DCR verification: reproduce and explain the large (~10^9) DCR values
seen in fidelity_smote_tabddpm.csv.

Uses fold 0, identical config to the canonical run:
  - SMOTE pretrain at 10% pos rate
  - TabDDPM: timesteps=1000, epochs=50, hidden_dims=[512,512,512], seed=0
  - Generates 100 synthetic samples
  - Computes DCR via fidelity_eval.compute_dcr

Captures raw z-score samples to show they leave the [-10, 10] training range.
Outputs: results/tabddpm_dcr_verification.csv
"""
from __future__ import annotations

import gc
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fidelity_eval import compute_dcr
from src.folds import get_temporal_folds
from src.preprocess_synth import (
    DROP_COLS, HASH_COLS, MAX_COLS, PRIORITY_COLS,
    TARGET_COL, TIME_COL, get_cat_cols_for_synth, preprocess_fold,
)
from src.synth_ctgan import make_synthetic_positives
from src.synth_tabddpm import (
    TabDDPMArtifacts,
    fit_tabddpm,
    _prepare_tabddpm_data,
)

# Match canonical SMOTE+TabDDPM config
SMOTE_POS_RATE = 0.10
TABDDPM_TIMESTEPS = 1000
TABDDPM_EPOCHS = 50
TABDDPM_HIDDEN = [512, 512, 512]
TABDDPM_SEED = 0
N_SYNTH = 100   # small for speed; enough to diagnose DCR


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
        print(f"[DCR_VER] Loading: {p}", flush=True)
        if p.endswith(".parquet"):
            import pyarrow.parquet as _pq
            schema_cols = _pq.ParquetFile(p).schema_arrow.names
            cols = _select_columns_to_load(schema_cols)
            print(f"[DCR_VER] Reading {len(cols)} / {len(schema_cols)} columns", flush=True)
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
        print(f"[DCR_VER] df shape={df.shape}", flush=True)
        return df
    raise FileNotFoundError("No train data found in data/")


# ---------------------------------------------------------------------------
# SMOTE expansion (bare-bones, avoids import of synth_smote which needs sklearn)
# ---------------------------------------------------------------------------

def _smote_expand(fraud_df: pd.DataFrame, used_cols: List[str], target_n: int, seed: int = 0) -> pd.DataFrame:
    """Simple random oversampling of fraud rows to reach target_n rows (≈ SMOTE pretraining)."""
    rng = np.random.default_rng(seed)
    if len(fraud_df) >= target_n:
        return fraud_df.copy()
    n_needed = target_n - len(fraud_df)
    idx = rng.integers(0, len(fraud_df), size=n_needed)
    synth_extra = fraud_df.iloc[idx].copy()
    return pd.concat([fraud_df, synth_extra], axis=0, ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    data_dir = os.path.join(_ROOT, "data")
    out_csv = os.path.join(_ROOT, "results", "tabddpm_dcr_verification.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Load + sort
    df_raw = _load_train_data(data_dir)
    df_sorted = df_raw.sort_values(TIME_COL).reset_index(drop=True)
    del df_raw
    gc.collect()

    # Get fold 0 only
    folds = get_temporal_folds(df_sorted, n_folds=8, time_col=TIME_COL)
    fold0 = [f for f in folds if f["fold"] == 0][0]
    train_df = fold0["train_df"]
    print(f"[DCR_VER] Fold 0 train: {len(train_df)} rows, "
          f"fraud={int((train_df[TARGET_COL] == 1).sum())}", flush=True)
    del df_sorted, folds
    gc.collect()

    # Preprocess
    val_placeholder = train_df.tail(100).copy()  # val not needed for this experiment
    train_prep, val_prep, used_cols = preprocess_fold(train_df, val_placeholder)
    cat_cols = get_cat_cols_for_synth(train_prep, used_cols)
    del val_placeholder, val_prep, train_df
    gc.collect()

    real_fraud = train_prep[train_prep[TARGET_COL] == 1].copy()
    n_fraud = len(real_fraud)
    n_total = len(train_prep)
    print(f"[DCR_VER] Preprocessed train: {n_total} rows, fraud={n_fraud}, cols={len(used_cols)}", flush=True)

    # SMOTE expand: target 10% positives (canonical config)
    n_neg = int((train_prep[TARGET_COL] == 0).sum())
    target_n_fraud = int(np.ceil(SMOTE_POS_RATE * n_total / (1 - SMOTE_POS_RATE)))
    pos_for_tabddpm = _smote_expand(real_fraud[used_cols + [TARGET_COL]], used_cols, target_n_fraud, seed=TABDDPM_SEED)
    print(f"[DCR_VER] SMOTE expanded fraud: {n_fraud} -> {len(pos_for_tabddpm)} (target {target_n_fraud})", flush=True)

    # Fit TabDDPM on SMOTE-expanded fraud
    print("[DCR_VER] Fitting TabDDPM ...", flush=True)
    diffusion, artifacts, denoiser = fit_tabddpm(
        pos_for_tabddpm,
        cat_cols,
        used_cols,
        timesteps=TABDDPM_TIMESTEPS,
        epochs=TABDDPM_EPOCHS,
        hidden_dims=TABDDPM_HIDDEN,
        seed=TABDDPM_SEED,
        device="cpu",
        verbose=True,
    )
    print("[DCR_VER] TabDDPM fit done.", flush=True)

    # Sample: capture raw z-scores BEFORE inverse transform
    torch.manual_seed(TABDDPM_SEED)
    denoiser.eval()
    device = next(denoiser.parameters()).device
    y_dist = torch.ones(1, device=device)
    with torch.no_grad():
        x_gen, _ = diffusion.sample_all(N_SYNTH, min(256, N_SYNTH), y_dist, ddim=False)

    raw_zscores = x_gen[:, :len(artifacts.cont_cols)].cpu().numpy()  # (N_SYNTH, n_cont)
    print(f"\n[DCR_VER] Raw z-score samples shape: {raw_zscores.shape}", flush=True)
    print(f"[DCR_VER] Z-score stats (all cont features):", flush=True)
    print(f"  min={raw_zscores.min():.4g}, max={raw_zscores.max():.4g}, "
          f"mean={raw_zscores.mean():.4g}, std={raw_zscores.std():.4g}", flush=True)
    frac_outside_train_range = float(np.mean(np.abs(raw_zscores) > 10.0))
    print(f"  fraction of values outside training clip [-10,10]: {frac_outside_train_range:.4f}", flush=True)

    # Inverse transform to original scale
    decoded = pd.DataFrame(raw_zscores, columns=artifacts.cont_cols)
    for c in artifacts.cont_cols:
        decoded[c] = decoded[c] * artifacts.cont_stds[c] + artifacts.cont_means[c]
    for c in cat_cols:
        decoded[c] = "__SYNTH__"
    decoded[TARGET_COL] = 1
    synth_df = decoded[used_cols + [TARGET_COL]]

    # Show decoded value distribution for TransactionAmt (key fraud feature)
    if "TransactionAmt" in decoded.columns:
        ta = decoded["TransactionAmt"]
        print(f"\n[DCR_VER] TransactionAmt (synth): min={ta.min():.4g}, median={ta.median():.4g}, "
              f"max={ta.max():.4g}", flush=True)
        ta_real = real_fraud["TransactionAmt"] if "TransactionAmt" in real_fraud.columns else None
        if ta_real is not None:
            print(f"[DCR_VER] TransactionAmt (real fraud): min={ta_real.min():.4g}, "
                  f"median={ta_real.median():.4g}, max={ta_real.max():.4g}", flush=True)

    # Compute DCR: synth vs real fold-0 fraud
    print("\n[DCR_VER] Computing DCR (TabDDPM synth vs real fold-0 fraud) ...", flush=True)
    dcr_vals = compute_dcr(synth_df, real_fraud)
    print(f"[DCR_VER] DCR: n={len(dcr_vals)}, mean={np.mean(dcr_vals):.6g}, "
          f"median={np.median(dcr_vals):.6g}, p95={np.percentile(dcr_vals, 95):.6g}, "
          f"max={np.max(dcr_vals):.6g}", flush=True)

    # For reference: compute CTGAN DCR on same fold (quick, 10 epochs for speed)
    print("\n[DCR_VER] Computing CTGAN DCR on fold 0 (10 epochs, for comparison) ...", flush=True)
    from src.synth_ctgan import make_synthetic_positives as ctgan_synth
    synth_ctgan = ctgan_synth(
        train_df=train_prep,
        cat_cols=cat_cols,
        used_cols=used_cols,
        target_pos_rate=0.05,
        max_synth=N_SYNTH,
        epochs=10,
        batch_size=500,
        pac=1,
        seed=0,
        discriminator_steps=5,
        verbose=False,
    )
    if len(synth_ctgan) > 0:
        dcr_ctgan = compute_dcr(synth_ctgan.head(N_SYNTH), real_fraud)
        print(f"[DCR_VER] CTGAN DCR: n={len(dcr_ctgan)}, mean={np.mean(dcr_ctgan):.6g}, "
              f"median={np.median(dcr_ctgan):.6g}", flush=True)
    else:
        dcr_ctgan = np.array([])
        print("[DCR_VER] CTGAN produced 0 synthetic rows", flush=True)

    # Save results
    rows = [
        {
            "method": "tabddpm",
            "fold": 0,
            "tabddpm_epochs": TABDDPM_EPOCHS,
            "tabddpm_timesteps": TABDDPM_TIMESTEPS,
            "n_synth": len(dcr_vals),
            "n_real_fraud": n_fraud,
            "zscore_min": float(raw_zscores.min()),
            "zscore_max": float(raw_zscores.max()),
            "zscore_frac_outside_10": float(frac_outside_train_range),
            "dcr_mean": float(np.mean(dcr_vals)) if len(dcr_vals) else float("nan"),
            "dcr_median": float(np.median(dcr_vals)) if len(dcr_vals) else float("nan"),
            "dcr_p95": float(np.percentile(dcr_vals, 95)) if len(dcr_vals) else float("nan"),
            "dcr_max": float(np.max(dcr_vals)) if len(dcr_vals) else float("nan"),
            "explanation": (
                "Large DCR caused by diffusion sampling outside z-score training range "
                "[-10,10]: samples are NOT clipped post-generation, so decoded values "
                "explode after inverse-transform (z*std+mean). fidelity_eval uses "
                "raw-scale L2 without re-normalising, causing DCR~10^9."
            ),
        },
    ]
    if len(dcr_ctgan) > 0:
        rows.append({
            "method": "ctgan",
            "fold": 0,
            "tabddpm_epochs": 10,
            "tabddpm_timesteps": float("nan"),
            "n_synth": len(dcr_ctgan),
            "n_real_fraud": n_fraud,
            "zscore_min": float("nan"),
            "zscore_max": float("nan"),
            "zscore_frac_outside_10": float("nan"),
            "dcr_mean": float(np.mean(dcr_ctgan)),
            "dcr_median": float(np.median(dcr_ctgan)),
            "dcr_p95": float(np.percentile(dcr_ctgan, 95)),
            "dcr_max": float(np.max(dcr_ctgan)),
            "explanation": "CTGAN reference: outputs in original feature space, DCR ~10^4",
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)
    print(f"\n[DCR_VER] Saved to {out_csv}", flush=True)
    print(df_out[["method", "n_synth", "zscore_min", "zscore_max",
                   "zscore_frac_outside_10", "dcr_mean", "dcr_median"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
