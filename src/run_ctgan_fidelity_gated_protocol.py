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
from src.synth_ctgan import make_synthetic_positives
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone CTGAN fidelity-gated run.")
    parser.add_argument("--run-id", type=str, default="20260330_180216")
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--start-fold", type=int, default=0)
    args = parser.parse_args()

    run_id = args.run_id
    n_folds = int(args.n_folds)
    start_fold = int(args.start_fold)

    root = os.path.dirname(os.path.dirname(__file__))
    run_dir = os.path.join(root, "results", "protocol", f"run_{run_id}")
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory required: {run_dir}")

    results_csv = os.path.join(run_dir, "results_ctgan_gated.csv")
    fidelity_csv = os.path.join(run_dir, "fidelity_ctgan_gated.csv")

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
    print(f"[ctgan_fidelity_gated] run_id={run_id} | folds={n_folds} | rates={TARGET_POS_RATES}")
    print(f"[ctgan_fidelity_gated] results -> {results_csv}")
    print(f"[ctgan_fidelity_gated] fidelity -> {fidelity_csv}")

    fidelity_rows: List[Dict[str, Any]] = []
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
        print(f"===== FOLD {fold} (ctgan_fidelity_gated) =====")

        train_df, val_df, used_cols = preprocess_fold(train_df, val_df)
        cat_cols = get_cat_cols_for_synth(train_df, used_cols)
        print(f"[INFO] train={len(train_df)}, val={len(val_df)}, features={len(used_cols)}")

        for target_rate in TARGET_POS_RATES:
            print("\n" + "-" * 50)
            print(f"[ctgan_fidelity_gated] target_pos_rate={target_rate}")

            synth_pos = make_synthetic_positives(
                train_df=train_df,
                cat_cols=cat_cols,
                used_cols=used_cols,
                target_pos_rate=target_rate,
                max_synth=MAX_SYNTH_POS,
                epochs=CTGAN_EPOCHS,
                batch_size=CTGAN_BATCH_SIZE,
                pac=CTGAN_PAC,
                seed=CTGAN_SEED,
                discriminator_steps=CTGAN_DISCRIMINATOR_STEPS,
                verbose=True,
            )

            synth_pos_filtered = synth_pos
            n_before = int(len(synth_pos))
            n_after = int(len(synth_pos))
            if n_before > 0:
                real_fraud = train_df[train_df[TARGET_COL] == 1]
                real_legit = train_df[train_df[TARGET_COL] == 0]
                fsum = fidelity_summary(
                    synthetic_fraud=synth_pos,
                    real_fraud=real_fraud,
                    real_legit=real_legit,
                    method="ctgan_fidelity_gated",
                    fold=fold,
                )
                synth_pos_filtered = filter_by_dcr(
                    synth_pos,
                    real_fraud,
                    percentile=90,
                )
                n_after = int(len(synth_pos_filtered))
                print(
                    f"[FIDELITY][ctgan_fidelity_gated][fold={fold}] "
                    f"survived={n_after}, discarded={n_before - n_after}"
                )
                fidelity_rows.append(
                    {
                        "method": "ctgan_fidelity_gated",
                        "fold": int(fold),
                        "mean_dcr": fsum.get("dcr_mean"),
                        "median_dcr": fsum.get("dcr_median"),
                        "mean_nndr": fsum.get("nndr_mean"),
                        "median_nndr": fsum.get("nndr_median"),
                        "n_synthetic": n_before,
                        "n_after_filter": n_after,
                    }
                )

            mixed_train = pd.concat([train_df, synth_pos_filtered], axis=0, ignore_index=True)
            res = train_and_eval(mixed_train, val_df)
            pr_auc_m = _get_metric(res, ["pr_auc", "prauc", "prAUC"])
            recall_m = _get_metric(res, ["recall_at_1pct_fpr", "recall@1%fpr", "recall_at_1fpr", "recall_at_1_fpr"])
            print(f"ctgan_fidelity_gated PR-AUC: {pr_auc_m:.4f}, Recall@1%FPR: {recall_m:.4f}")

            _append_row_csv(
                {
                    "timestamp": _now_str(),
                    "fold": fold,
                    "delay_days": 0,
                    "run_id": run_id,
                    "method": "ctgan_fidelity_gated",
                    "target_pos_rate": float(target_rate),
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "train_pos": int((train_df[TARGET_COL] == 1).sum()),
                    "train_neg": int((train_df[TARGET_COL] == 0).sum()),
                    "synth_rows": n_after,
                    "final_train_rows": int(len(mixed_train)),
                    "final_pos_rate": float((mixed_train[TARGET_COL] == 1).mean()),
                    "pr_auc": pr_auc_m,
                    "recall_at_1pct_fpr": recall_m,
                    "notes": f"dcr_gated_p90; raw_synth_rows={n_before}",
                },
                results_csv,
            )

    if fidelity_rows:
        fidelity_df = pd.DataFrame(fidelity_rows)
        write_header = not os.path.exists(fidelity_csv)
        fidelity_df.to_csv(fidelity_csv, mode="a", header=write_header, index=False)
        print(f"[FIDELITY] appended -> {fidelity_csv} ({len(fidelity_rows)} new rows)")

    elapsed = int(time.perf_counter() - t0)
    print(f"\n[DONE] ctgan_fidelity_gated elapsed: {elapsed // 60}m {elapsed % 60}s")


if __name__ == "__main__":
    main()
