"""
CTGAN 5-epoch ablation — for comparison against canonical 150-epoch run.

Only change vs canonical run_protocol.py:
    ctgan_epochs: 150 -> 5

Everything else is IDENTICAL:
    - same 8 folds (get_temporal_folds, n_folds=8)
    - same preprocess_fold / get_cat_cols_for_synth
    - same train_and_eval (LightGBM, n_estimators=300, lr=0.05, etc.)
    - same make_synthetic_positives kwargs (batch_size=500, disc_steps=5, pac=1, seed=0)
    - same target_pos_rate=0.05 (canonical single rate — best from 150-epoch run)
    - same MAX_SYNTH_POS=50000

Outputs:
    results/ctgan_5epoch_ablation.csv   — 8 rows (one per fold, ctgan only)
    results/ctgan_5epoch_ablation_log.txt
"""
from __future__ import annotations

import os
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from datetime import datetime

import gc

import numpy as np
import pandas as pd
import pyarrow.parquet as _pq
from scipy.stats import wilcoxon

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.folds import get_temporal_folds
from src.train import train_and_eval
from src.synth_ctgan import make_synthetic_positives
from src.preprocess_synth import (
    preprocess_fold,
    get_cat_cols_for_synth,
    DROP_COLS,
    HASH_COLS,
    PRIORITY_COLS,
    MAX_COLS,
)

# ── Config (copy-paste from run_protocol.py; only CTGAN_EPOCHS changes) ──────
TARGET_COL              = "isFraud"
TIME_COL                = "TransactionDT"
N_FOLDS                 = 8
TARGET_POS_RATE         = 0.05          # canonical single rate (best from 150-epoch run)
MAX_SYNTH_POS           = 50_000
CTGAN_EPOCHS            = 5             # ← THE ONLY CHANGE
CTGAN_BATCH_SIZE        = 500           # identical to canonical
CTGAN_DISCRIMINATOR_STEPS = 5          # identical to canonical
CTGAN_PAC               = 1             # identical to canonical
CTGAN_SEED              = 0             # identical to canonical

BASELINE_MEAN_PR_AUC    = 0.5786        # from canonical run (ground truth)
CTGAN_150_MEAN_PR_AUC   = 0.5838        # from canonical run (ground truth)

OUT_CSV  = os.path.join(_ROOT, "results", "ctgan_5epoch_ablation.csv")
OUT_LOG  = os.path.join(_ROOT, "results", "ctgan_5epoch_ablation_log.txt")


# ── Column-projected parquet loader (avoids OOM on full 436-col load) ─────────
def _select_cols(all_cols: list[str]) -> list[str]:
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
    rest     = [c for c in present if c not in set(priority) and ok(c)]
    selected = priority + rest[: MAX_COLS - len(priority)]
    must     = {TARGET_COL, TIME_COL, *HASH_COLS}
    for c in must:
        if c in all_cols and c not in selected:
            selected.append(c)
    return [c for c in selected if c in all_cols]


def _load_data(data_dir: str) -> pd.DataFrame:
    for name in ["train_merged.parquet", "train_transaction.csv"]:
        p = os.path.join(data_dir, name)
        if not os.path.exists(p):
            continue
        if p.endswith(".parquet"):
            schema_cols = _pq.ParquetFile(p).schema_arrow.names
            cols = _select_cols(schema_cols)
            _log(f"Reading {len(cols)} / {len(schema_cols)} columns from parquet")
            df = pd.read_parquet(p, columns=cols)
        else:
            df = pd.read_csv(p)
        for c in df.select_dtypes("float64").columns:
            df[c] = df[c].astype("float32")
        for c in df.select_dtypes("int64").columns:
            df[c] = pd.to_numeric(df[c], downcast="integer")
        gc.collect()
        return df
    raise FileNotFoundError("No data in data/ (need train_merged.parquet or train_transaction.csv)")

# ── Logging ───────────────────────────────────────────────────────────────────
_log_lines: list[str] = []

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    _log_lines.append(line)

def _flush_log() -> None:
    with open(OUT_LOG, "w") as f:
        f.write("\n".join(_log_lines) + "\n")

# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.perf_counter()
    _log("=" * 65)
    _log(f"CTGAN {CTGAN_EPOCHS}-epoch ablation  (canonical config, epochs only changed)")
    _log(f"target_pos_rate={TARGET_POS_RATE}  batch_size={CTGAN_BATCH_SIZE}  "
         f"disc_steps={CTGAN_DISCRIMINATOR_STEPS}  pac={CTGAN_PAC}  seed={CTGAN_SEED}")
    _log(f"n_folds={N_FOLDS}  max_synth={MAX_SYNTH_POS}")
    _log("=" * 65)

    # ── Load data (column-projected to avoid OOM on 436-col parquet) ──────────
    data_dir = os.path.join(_ROOT, "data")
    _log(f"Loading data from {data_dir}")
    df = _load_data(data_dir)
    _log(f"Loaded: {df.shape} | fraud={int((df[TARGET_COL]==1).sum()):,}")

    # ── Folds ─────────────────────────────────────────────────────────────────
    folds_raw = get_temporal_folds(df, n_folds=N_FOLDS, time_col=TIME_COL)
    _log(f"Constructed {N_FOLDS} temporal folds ({N_FOLDS+1} equal-size chunks)")

    rows: list[dict] = []
    baseline_pr_aucs: list[float] = []
    ctgan_pr_aucs:    list[float] = []

    for fold_info in folds_raw:
        fold      = fold_info["fold"]
        train_raw = fold_info["train_df"]
        val_raw   = fold_info["val_df"]

        fold_t0 = time.perf_counter()
        _log(f"\n{'='*60}")
        _log(f"FOLD {fold}  (train_raw={len(train_raw):,}, val_raw={len(val_raw):,})")

        # ── Preprocessing (identical to run_protocol.py) ──────────────────────
        train_df, val_df, used_cols = preprocess_fold(train_raw, val_raw)
        cat_cols = get_cat_cols_for_synth(train_df, used_cols)

        n_train       = len(train_df)
        n_fraud_train = int((train_df[TARGET_COL] == 1).sum())
        n_test        = len(val_df)
        n_fraud_test  = int((val_df[TARGET_COL] == 1).sum())

        _log(f"  preprocessed: {len(used_cols)} features | "
             f"train={n_train:,} ({n_fraud_train} fraud) | "
             f"val={n_test:,} ({n_fraud_test} fraud)")

        # ── Baseline ──────────────────────────────────────────────────────────
        base_res   = train_and_eval(train_df, val_df)
        base_prauc = float(base_res["pr_auc"])
        baseline_pr_aucs.append(base_prauc)
        _log(f"  BASELINE  PR-AUC={base_prauc:.6f}")

        # ── CTGAN 5 epochs ────────────────────────────────────────────────────
        synth_pos = make_synthetic_positives(
            train_df             = train_df,
            cat_cols             = cat_cols,
            used_cols            = used_cols,
            target_pos_rate      = TARGET_POS_RATE,
            max_synth            = MAX_SYNTH_POS,
            epochs               = CTGAN_EPOCHS,           # 5
            batch_size           = CTGAN_BATCH_SIZE,        # 500
            pac                  = CTGAN_PAC,               # 1
            seed                 = CTGAN_SEED,              # 0
            discriminator_steps  = CTGAN_DISCRIMINATOR_STEPS,  # 5
            verbose              = True,
        )
        n_synth = len(synth_pos)
        mixed   = pd.concat([train_df, synth_pos], axis=0, ignore_index=True)
        res     = train_and_eval(mixed, val_df)
        prauc   = float(res["pr_auc"])
        ctgan_pr_aucs.append(prauc)

        elapsed_fold = time.perf_counter() - fold_t0
        _log(f"  CTGAN-5   PR-AUC={prauc:.6f}  synth_added={n_synth:,}  fold_time={elapsed_fold:.0f}s")

        row = {
            "fold":          fold,
            "pr_auc":        prauc,
            "method":        "ctgan",
            "epochs":        CTGAN_EPOCHS,
            "n_train":       n_train,
            "n_fraud_train": n_fraud_train,
            "n_test":        n_test,
            "n_fraud_test":  n_fraud_test,
        }
        rows.append(row)
        _flush_log()

    # ── Save CSV ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    _log(f"\nSaved -> {OUT_CSV}")

    # ── Verification ──────────────────────────────────────────────────────────
    check = pd.read_csv(OUT_CSV)
    assert len(check) == N_FOLDS,          f"Expected {N_FOLDS} rows, got {len(check)}"
    assert check["pr_auc"].isna().sum() == 0, "NaN PR-AUC values found"
    assert check["pr_auc"].between(0.3, 0.95).all(), \
        f"Implausible PR-AUC values: {check['pr_auc'].tolist()}"
    _log(f"Verification PASSED: {len(check)} rows, no NaNs, all PR-AUC in plausible range")

    # ── Summary ───────────────────────────────────────────────────────────────
    arr         = np.array(ctgan_pr_aucs)
    base_arr    = np.array(baseline_pr_aucs)
    mean_prauc  = float(arr.mean())
    std_prauc   = float(arr.std(ddof=1))
    delta_vs_baseline     = mean_prauc - BASELINE_MEAN_PR_AUC
    delta_vs_ctgan_150    = mean_prauc - CTGAN_150_MEAN_PR_AUC

    # Wilcoxon signed-rank test (5-epoch CTGAN vs in-fold baseline)
    diffs = arr - base_arr
    try:
        if len(diffs) >= 4 and not np.all(diffs == 0):
            stat, pval = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
        else:
            stat, pval = float("nan"), float("nan")
    except Exception as e:
        stat, pval = float("nan"), float("nan")
        _log(f"  [WARN] Wilcoxon failed: {e}")

    summary_lines = [
        "",
        "=" * 65,
        f"CTGAN {CTGAN_EPOCHS}-EPOCH ABLATION — FINAL RESULTS",
        "=" * 65,
        "",
        "Per-fold PR-AUC (CTGAN 5 epochs):",
    ]
    for i, (b, c) in enumerate(zip(baseline_pr_aucs, ctgan_pr_aucs)):
        summary_lines.append(f"  Fold {i}: baseline={b:.6f}  ctgan5={c:.6f}  Δ={c-b:+.6f}")
    summary_lines += [
        "",
        f"  Mean PR-AUC (CTGAN-5)    : {mean_prauc:.6f}",
        f"  Std  PR-AUC (CTGAN-5)    : {std_prauc:.6f}",
        f"  Δ vs baseline (0.5786)   : {delta_vs_baseline:+.6f}",
        f"  Δ vs CTGAN-150 (0.5838)  : {delta_vs_ctgan_150:+.6f}",
        f"  Wilcoxon p-value (vs base): {pval:.4f}" if not np.isnan(pval) else
        "  Wilcoxon p-value (vs base): not computable",
        "",
        f"  Total wall time: {(time.perf_counter()-t0)/60:.1f} min",
        "=" * 65,
    ]
    for line in summary_lines:
        _log(line)

    _flush_log()
    _log(f"Log -> {OUT_LOG}")


if __name__ == "__main__":
    main()
