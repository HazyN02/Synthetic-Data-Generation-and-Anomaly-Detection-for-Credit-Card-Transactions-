# src/run_smote_tabddpm_protocol.py
"""
Standalone protocol track: SMOTE training set to 10% positives, then TabDDPM fit on
SMOTE-expanded fraud rows, then generate + evaluate (same schema as main protocol).

Writes only:
  results/protocol/run_<run_id>/results_smote_tabddpm.csv
  results/protocol/run_<run_id>/fidelity_smote_tabddpm.csv
Does not modify results.csv, fidelity_results.csv, or other existing artifacts.
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from fidelity_eval import fidelity_summary, filter_by_dcr
from src.folds import get_temporal_folds
from src.preprocess_synth import get_cat_cols_for_synth, preprocess_fold
from src.synth_smote import build_smote_expanded_fraud_df
from src.synth_tabddpm import make_synthetic_positives_tabddpm
from src.train import train_and_eval

TARGET_COL = "isFraud"
TIME_COL = "TransactionDT"
N_FOLDS = 8
TARGET_POS_RATES = [0.05, 0.10, 0.20]
MAX_SYNTH_POS = 50000
SMOTE_PRETRAIN_POS_RATE = 0.10
TABDDPM_SEED = 0
TABDDPM_KWARGS: Dict[str, Any] = {
    "timesteps": 1000,
    "epochs": 50,
    "hidden_dims": [512, 512, 512],
}


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


def main() -> None:
    parser = argparse.ArgumentParser(description="SMOTE(10%) then TabDDPM — standalone results files only.")
    parser.add_argument(
        "--run-id",
        type=str,
        default="20260330_180216",
        help="Run directory id (results/protocol/run_<id>/...)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=N_FOLDS,
        help="Number of temporal folds (default 8)",
    )
    parser.add_argument(
        "--smote-pretrain-rate",
        type=float,
        default=SMOTE_PRETRAIN_POS_RATE,
        help="SMOTE target positive rate on train before TabDDPM fit",
    )
    parser.add_argument(
        "--start-fold",
        type=int,
        default=0,
        help="Start from this fold index (useful after smoke tests)",
    )
    args = parser.parse_args()

    run_id = args.run_id
    n_folds = int(args.n_folds)
    smote_pre = float(args.smote_pretrain_rate)
    start_fold = int(args.start_fold)

    root = os.path.dirname(os.path.dirname(__file__))
    protocol_dir = os.path.join(root, "results", "protocol")
    run_dir = os.path.join(protocol_dir, f"run_{run_id}")
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory required (create or fix run-id): {run_dir}")

    results_csv = os.path.join(run_dir, "results_smote_tabddpm.csv")
    fidelity_csv = os.path.join(run_dir, "fidelity_smote_tabddpm.csv")

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
    print(f"[smote_tabddpm] run_id={run_id} | folds={n_folds} | rates={TARGET_POS_RATES}")
    print(f"[smote_tabddpm] results -> {results_csv}")
    print(f"[smote_tabddpm] fidelity -> {fidelity_csv}")

    fidelity_rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    for fold_info in folds_raw:
        fold = fold_info["fold"]
        if fold < start_fold:
            continue
        train_df = fold_info["train_df"]
        val_df = fold_info["val_df"]
        print("\n" + "=" * 60)
        print(f"===== FOLD {fold} (smote_tabddpm) =====")
        if len(val_df) == 0:
            continue

        train_df, val_df, used_cols = preprocess_fold(train_df, val_df)
        cat_cols = get_cat_cols_for_synth(train_df, used_cols)
        print(f"[INFO] train={len(train_df)}, val={len(val_df)}, features={len(used_cols)}")

        fit_fraud = build_smote_expanded_fraud_df(
            train_df,
            val_df,
            smote_train_pos_rate=smote_pre,
            k_neighbors=5,
            random_state=42,
            max_synth=None,
        )
        print(
            f"[smote_tabddpm] SMOTE pretrain: fraud rows for TabDDPM fit: "
            f"{len(fit_fraud)} (train pos before SMOTE: "
            f"{int((train_df[TARGET_COL] == 1).sum())})"
        )

        for target_rate in TARGET_POS_RATES:
            print("\n" + "-" * 50)
            print(f"[smote_tabddpm] TabDDPM target_pos_rate={target_rate}")

            synth_pos = make_synthetic_positives_tabddpm(
                train_df=train_df,
                cat_cols=cat_cols,
                used_cols=used_cols,
                target_pos_rate=target_rate,
                max_synth=MAX_SYNTH_POS,
                seed=TABDDPM_SEED,
                verbose=True,
                fit_pos_df=fit_fraud,
                recency_frac=None,
                **TABDDPM_KWARGS,
            )

            if len(synth_pos) > 0:
                real_fraud = train_df[train_df[TARGET_COL] == 1]
                real_legit = train_df[train_df[TARGET_COL] == 0]
                fsum = fidelity_summary(
                    synthetic_fraud=synth_pos,
                    real_fraud=real_fraud,
                    real_legit=real_legit,
                    method="smote_tabddpm",
                    fold=fold,
                )
                synth_f = filter_by_dcr(synth_pos, real_fraud, percentile=90)
                n_after = int(len(synth_f))
                n_before = int(len(synth_pos))
                print(f"[FIDELITY][smote_tabddpm][fold={fold}] survived={n_after}, discarded={n_before - n_after}")
                fidelity_rows.append(
                    {
                        "method": "smote_tabddpm",
                        "fold": int(fold),
                        "mean_dcr": fsum.get("dcr_mean"),
                        "median_dcr": fsum.get("dcr_median"),
                        "mean_nndr": fsum.get("nndr_mean"),
                        "median_nndr": fsum.get("nndr_median"),
                        "n_synthetic": n_before,
                        "n_after_filter": n_after,
                    }
                )

            mixed_train = pd.concat([train_df, synth_pos], axis=0, ignore_index=True)
            res = train_and_eval(mixed_train, val_df)
            pr_auc_m = _get_metric(res, ["pr_auc", "prauc", "prAUC"])
            recall_m = _get_metric(res, ["recall_at_1pct_fpr", "recall@1%fpr", "recall_at_1fpr", "recall_at_1_fpr"])
            print(f"smote_tabddpm PR-AUC: {pr_auc_m:.4f}, Recall@1%FPR: {recall_m:.4f}")

            _append_row_csv(
                {
                    "timestamp": _now_str(),
                    "fold": fold,
                    "delay_days": 0,
                    "run_id": run_id,
                    "method": "smote_tabddpm",
                    "target_pos_rate": float(target_rate),
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "train_pos": int((train_df[TARGET_COL] == 1).sum()),
                    "train_neg": int((train_df[TARGET_COL] == 0).sum()),
                    "synth_rows": int(len(synth_pos)),
                    "final_train_rows": int(len(mixed_train)),
                    "final_pos_rate": float((mixed_train[TARGET_COL] == 1).mean()),
                    "pr_auc": pr_auc_m,
                    "recall_at_1pct_fpr": recall_m,
                    "notes": f"smote_pretrain_pos_rate={smote_pre}",
                },
                results_csv,
            )

    if fidelity_rows:
        fidelity_df = pd.DataFrame(fidelity_rows)
        write_header = not os.path.exists(fidelity_csv)
        fidelity_df.to_csv(fidelity_csv, mode="a", header=write_header, index=False)
        print(f"[FIDELITY] appended -> {fidelity_csv} ({len(fidelity_rows)} new rows)")

    elapsed = int(time.perf_counter() - t0)
    print(f"\n[DONE] smote_tabddpm elapsed: {elapsed // 60}m {elapsed % 60}s")


if __name__ == "__main__":
    main()
