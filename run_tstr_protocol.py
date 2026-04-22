#!/usr/bin/env python3
"""
Train-on-Synthetic-Test-on-Real (TSTR): for each fold, train LightGBM on
synthetic fraud + real legitimate only (no real fraud in training), evaluate on real holdout.

CTGAN: 150 epochs, same hyperparameters as run_ctgan_fidelity_gated_protocol.py.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from src.folds import get_temporal_folds
from src.preprocess_synth import get_cat_cols_for_synth, preprocess_fold
from src.synth_ctgan import fit_ctgan, sample_ctgan
from src.train import train_and_eval

TARGET_COL = "isFraud"
TIME_COL = "TransactionDT"
N_FOLDS = 8
TARGET_POS_RATES = [0.05, 0.10, 0.20]
MAX_SYNTH_POS = 50000
CTGAN_EPOCHS = 150
CTGAN_BATCH_SIZE = 500
CTGAN_DISCRIMINATOR_STEPS = 5
CTGAN_PAC = 1
CTGAN_SEED = 0


def _now_str() -> str:
    return datetime.now().strftime("%d-%m-%Y %H:%M")


def _get_metric(res: Dict[str, Any], candidates: List[str]) -> Optional[float]:
    for k in candidates:
        if k in res:
            try:
                return float(res[k])
            except Exception:
                return None
    return None


def _append_row_csv(row: Dict[str, Any], path: str) -> None:
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(path)
    df_row.to_csv(path, mode="a", header=write_header, index=False)


def _tstr_synth_count(n_neg: int, target_pos_rate: float, max_synth: int) -> int:
    """Positives needed so P / (P + n_neg) ~= target_pos_rate (integer rows)."""
    r = float(target_pos_rate)
    if r <= 0.0:
        return 0
    p = int(math.ceil(r * n_neg / max(1e-12, (1.0 - r))))
    return int(min(p, max_synth))


def main() -> None:
    parser = argparse.ArgumentParser(description="TSTR: CTGAN synth fraud + real legit only, test on real val.")
    parser.add_argument("--run-id", type=str, default="20260330_180216")
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--start-fold", type=int, default=0)
    args = parser.parse_args()

    run_id = args.run_id
    n_folds = int(args.n_folds)
    start_fold = int(args.start_fold)

    root = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.join(root, "results", "protocol", f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    results_csv = os.path.join(run_dir, "results_tstr.csv")

    data_dir = os.path.join(root, "data")
    df = None
    for name in ["train_merged.parquet", "train_transaction.csv"]:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            print(f"Loading: {p}")
            df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
            break
    if df is None:
        raise FileNotFoundError("No train data in data/")

    assert TARGET_COL in df.columns
    assert TIME_COL in df.columns

    folds_raw = get_temporal_folds(df, n_folds=n_folds, time_col=TIME_COL)
    print(f"[tstr] run_id={run_id} | folds={n_folds} | rates={TARGET_POS_RATES}")
    print(f"[tstr] results -> {results_csv}")

    t0 = time.perf_counter()

    for fold_info in folds_raw:
        fold = fold_info["fold"]
        if fold < start_fold:
            continue
        train_df = fold_info["train_df"]
        val_df = fold_info["val_df"]
        if len(val_df) == 0:
            continue

        print("\n" + "=" * 60)
        print(f"===== FOLD {fold} (ctgan_tstr) =====")

        train_df, val_df, used_cols = preprocess_fold(train_df, val_df)
        cat_cols = get_cat_cols_for_synth(train_df, used_cols)
        print(f"[INFO] train={len(train_df)}, val={len(val_df)}, features={len(used_cols)}")

        pos_df = train_df[train_df[TARGET_COL] == 1].copy()
        neg_df = train_df[train_df[TARGET_COL] == 0].copy()
        n_neg = len(neg_df)
        train_pos = len(pos_df)

        if train_pos < 50:
            raise ValueError(f"[TSTR/CTGAN] Too few positives to fit CTGAN: n_pos={train_pos}")

        for target_rate in TARGET_POS_RATES:
            print("\n" + "-" * 50)
            print(f"[ctgan_tstr] target_pos_rate={target_rate}")

            synth_add = _tstr_synth_count(n_neg, target_rate, MAX_SYNTH_POS)
            print(
                f"[ctgan_tstr] n_neg={n_neg}, synth_add={synth_add} "
                f"(target train pos rate ~ {synth_add / max(1, n_neg + synth_add):.4f})"
            )

            if synth_add == 0:
                print("[ctgan_tstr] synth_add=0, skipping")
                continue

            model, artifacts = fit_ctgan(
                train_df=pos_df,
                cat_cols=cat_cols,
                used_cols=used_cols,
                epochs=CTGAN_EPOCHS,
                batch_size=CTGAN_BATCH_SIZE,
                pac=CTGAN_PAC,
                seed=CTGAN_SEED,
                verbose=True,
                discriminator_steps=CTGAN_DISCRIMINATOR_STEPS,
            )
            synth_x = sample_ctgan(model, n=synth_add, artifacts=artifacts, verbose=True)
            synth_x[TARGET_COL] = 1
            synth_only = synth_x[used_cols + [TARGET_COL]].copy()

            tstr_train = pd.concat([neg_df, synth_only], axis=0, ignore_index=True)

            res = train_and_eval(tstr_train, val_df)
            pr_auc_m = _get_metric(res, ["pr_auc", "prauc", "prAUC"])
            recall_m = _get_metric(res, ["recall_at_1pct_fpr", "recall@1%fpr", "recall_at_1fpr", "recall_at_1_fpr"])
            print(f"ctgan_tstr PR-AUC: {pr_auc_m:.4f}, Recall@1%FPR: {recall_m:.4f}")

            _append_row_csv(
                {
                    "timestamp": _now_str(),
                    "fold": fold,
                    "delay_days": 0,
                    "run_id": run_id,
                    "method": "ctgan_tstr",
                    "target_pos_rate": float(target_rate),
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "train_pos": int(train_pos),
                    "train_neg": int(n_neg),
                    "synth_rows": int(synth_add),
                    "final_train_rows": int(len(tstr_train)),
                    "final_pos_rate": float((tstr_train[TARGET_COL] == 1).mean()),
                    "pr_auc": pr_auc_m,
                    "recall_at_1pct_fpr": recall_m,
                    "notes": "tstr; train=synth_fraud+real_legit_only; ctgan_epochs=150",
                },
                results_csv,
            )

    elapsed = int(time.perf_counter() - t0)
    print(f"\n[DONE] ctgan_tstr elapsed: {elapsed // 60}m {elapsed % 60}s")


if __name__ == "__main__":
    main()
