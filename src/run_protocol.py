# src/run_protocol.py

from __future__ import annotations

# Disable multiprocessing/threading that causes CTGAN crashes (loky, OpenMP)
import os
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from src.folds import get_temporal_folds
from src.train import train_and_eval
from src.synth_ctgan import make_synthetic_positives
from src.synth_tabddpm import make_synthetic_positives_tabddpm
from src.synth_smote import train_and_eval_smote
from src.preprocess_synth import preprocess_fold, get_cat_cols_for_synth



# ----------------------------
# CONFIG
# ----------------------------

TARGET_COL = "isFraud"
TIME_COL = "TransactionDT"

# Explicit categorical columns (do NOT guess later)
CATEGORICAL_COLS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType",
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29",
    "id_34", "id_35", "id_36", "id_37", "id_38",
]

# Default: full run
N_FOLDS = 4
TARGET_POS_RATES = [0.05, 0.10, 0.20]
MAX_SYNTH_POS = 50000
CTGAN_EPOCHS = 10
CTGAN_BATCH_SIZE = 512  # Larger batch = faster (if memory allows)
CTGAN_PAC = 1
CTGAN_SEED = 0
TABDDPM_SEED = 0
# TabDDPM defaults: timesteps=100, epochs=5, hidden=[1024,1024]

# Where to save (organized by model)
RESULTS_DIR = "results"
PROTOCOL_DIR = os.path.join(RESULTS_DIR, "protocol")
RESULTS_CSV = os.path.join(PROTOCOL_DIR, "results.csv")
CONFIG_JSON = os.path.join(PROTOCOL_DIR, "config.json")
RUNTIME_ESTIMATE_TXT = os.path.join(PROTOCOL_DIR, "runtime_estimate.txt")

# ----------------------------
# HELPERS
# ----------------------------

def _ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PROTOCOL_DIR, exist_ok=True)

def _now_str() -> str:
    return datetime.now().strftime("%d-%m-%Y %H:%M")

def _get_metric(res: Dict[str, Any], candidates: List[str]) -> Optional[float]:
    """Robust metric getter to avoid KeyErrors from naming changes."""
    for k in candidates:
        if k in res:
            try:
                return float(res[k])
            except Exception:
                return None
    return None

def _append_row_csv(row: Dict[str, Any], csv_path: str | None = None) -> None:
    path = csv_path or RESULTS_CSV
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(path)
    df_row.to_csv(path, mode="a", header=write_header, index=False)

def _print_fold_header(i: int) -> None:
    print("\n" + "=" * 60)
    print(f"===== FOLD {i} =====")

def _estimate_runtime(n_folds: int, n_rates: int, ctgan_per_run: int, tabddpm_per_run: int) -> str:
    """Rough runtime estimate (minutes)."""
    base_per_fold = 2
    total = n_folds * (base_per_fold + n_rates * (ctgan_per_run + tabddpm_per_run))
    return f"~{total} min ({total // 60}h {total % 60}m)"


# ----------------------------
# MAIN
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="2 folds, 1 rate, fastest (≈1h)")
    parser.add_argument("--medium", action="store_true", help="4 folds, 3 rates, faster training (≈4–5h)")
    parser.add_argument("--full", action="store_true", help="4 folds, 3 rates, full training (≈10h)")
    parser.add_argument(
        "--max-convergence",
        action="store_true",
        help="4 folds, 1 rate (0.10), CTGAN/TabDDPM at 50 epochs for convergence sanity-check (≈6–8h)",
    )
    parser.add_argument("--recency-ablation", action="store_true", help="add ctgan/tabddpm/smote with recency_frac=0.3")
    parser.add_argument(
        "--delay-days",
        type=int,
        default=0,
        help="Label delay gap in days between end of train window and start of validation window (per fold).",
    )
    parser.add_argument(
        "--target-pos-rates",
        type=str,
        default=None,
        help="Override target_pos_rates (comma-separated, e.g. 0.03,0.05,0.10,0.15,0.20) for sensitivity ablation.",
    )
    args = parser.parse_args()

    delay_days = max(0, int(getattr(args, "delay_days", 0)))

    # Mode selection: max-convergence > full > medium > quick
    if args.max_convergence:
        mode = "max"
        n_folds = N_FOLDS
        # Focused convergence check on the main operating point
        target_pos_rates = [0.10]
        max_synth = MAX_SYNTH_POS
        ctgan_epochs = 50
        ctgan_per_run, tabddpm_per_run = 25, 50
        tabddpm_kwargs = {"timesteps": 75, "epochs": 50, "hidden_dims": [768, 768]}
    elif args.full:
        mode = "full"
        n_folds = N_FOLDS
        target_pos_rates = TARGET_POS_RATES
        max_synth = MAX_SYNTH_POS
        ctgan_epochs = CTGAN_EPOCHS
        ctgan_per_run, tabddpm_per_run = 15, 35
        tabddpm_kwargs = {}
    elif args.medium:
        mode = "medium"
        n_folds = N_FOLDS
        # For delay runs, keep protocol lighter and fix to 5% only
        target_pos_rates = [0.05] if delay_days > 0 else TARGET_POS_RATES
        max_synth = MAX_SYNTH_POS
        ctgan_epochs = 7  # balance speed vs quality; full mode uses 10
        ctgan_per_run, tabddpm_per_run = 10, 20
        tabddpm_kwargs = {"timesteps": 75, "epochs": 4, "hidden_dims": [768, 768]}
    elif args.quick:
        mode = "quick"
        n_folds = 2
        target_pos_rates = [0.05]
        max_synth = 5000
        ctgan_epochs = 5
        ctgan_per_run, tabddpm_per_run = 8, 15
        tabddpm_kwargs = {"timesteps": 50, "epochs": 3, "hidden_dims": [512, 512]}
    else:
        mode = "full"
        n_folds = N_FOLDS
        target_pos_rates = TARGET_POS_RATES
        max_synth = MAX_SYNTH_POS
        ctgan_epochs = CTGAN_EPOCHS
        ctgan_per_run, tabddpm_per_run = 15, 35
        tabddpm_kwargs = {}

    if getattr(args, "target_pos_rates", None):
        try:
            target_pos_rates = [float(x.strip()) for x in args.target_pos_rates.split(",") if x.strip()]
        except (ValueError, AttributeError):
            pass

    _ensure_results_dir()

    # Timestamped run dir for full/medium (reproducibility + backup)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(PROTOCOL_DIR, f"run_{run_id}") if mode in ("full", "medium") else None
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        run_results_csv = os.path.join(run_dir, "results.csv")
        run_config_json = os.path.join(run_dir, "config.json")
        run_runtime_txt = os.path.join(run_dir, "runtime.txt")
        # Backup main results before run
        if os.path.exists(RESULTS_CSV):
            backup_path = os.path.join(PROTOCOL_DIR, f"results_backup_{run_id}.csv")
            shutil.copy2(RESULTS_CSV, backup_path)
            print(f"Backed up existing results -> {backup_path}")
    else:
        run_results_csv = RESULTS_CSV
        run_config_json = CONFIG_JSON
        run_runtime_txt = RUNTIME_ESTIMATE_TXT

    est = _estimate_runtime(n_folds, len(target_pos_rates), ctgan_per_run, tabddpm_per_run)
    print(f"\n[RUNTIME ESTIMATE] {est} (mode={mode})\n")
    with open(run_runtime_txt, "w") as f:
        f.write(f"Run ID: {run_id}\nMode: {mode}\nEstimated runtime: {est}\n")
        if run_dir:
            f.write(f"Results: {run_results_csv}\nConfig: {run_config_json}\n")

    # Save config for reproducibility
    recency_ablation = getattr(args, "recency_ablation", False)
    config = {
        "run_id": run_id,
        "mode": mode,
        "recency_ablation": recency_ablation,
        "delay_days": delay_days,
        "command": " ".join(sys.argv),
        "start_time": _now_str(),
        "target_col": TARGET_COL,
        "time_col": TIME_COL,
        "categorical_cols": CATEGORICAL_COLS,
        "n_folds": n_folds,
        "target_pos_rates": target_pos_rates,
        "max_synth_pos": max_synth,
        "runtime_estimate": est,
        "ctgan": {
            "epochs": ctgan_epochs,
            "batch_size": CTGAN_BATCH_SIZE,
            "pac": CTGAN_PAC,
            "seed": CTGAN_SEED,
        },
        "tabddpm": {
            "seed": TABDDPM_SEED,
            **tabddpm_kwargs,
        },
        "results_csv": run_results_csv,
        "results_dir": run_dir or PROTOCOL_DIR,
    }
    with open(run_config_json, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config -> {run_config_json}")

    # Load (same logic as run_smote_baseline for consistency)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    df = None
    for name in ["train_merged.parquet", "train_transaction.csv"]:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            print(f"Loading: {p}")
            df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
            break
    if df is None:
        raise FileNotFoundError("No train data in data/ (need train_merged.parquet or train_transaction.csv)")
    print(f"Total rows: {len(df)}")

    assert TARGET_COL in df.columns, f"Missing target column: {TARGET_COL}"
    assert TIME_COL in df.columns, f"Missing time column: {TIME_COL}"

    # Fold on RAW data; preprocess per fold (fit on train, transform val) to avoid leakage
    folds_raw = get_temporal_folds(df, n_folds=n_folds, time_col=TIME_COL)
    print(f"Running {n_folds} temporal folds (using {n_folds + 1} time chunks)")

    t_start = time.perf_counter()

    for fold_info in folds_raw:
        fold = fold_info["fold"]
        train_df = fold_info["train_df"]
        val_df = fold_info["val_df"]

        _print_fold_header(fold)

        if len(val_df) == 0:
            # Should never happen with K+1 chunks, but keep it safe.
            print(f"[WARN] Fold {fold} has empty val set. Skipping.")
            continue

        # Optional label-delay: drop recent training rows close to validation start
        if delay_days > 0:
            t_val_start = val_df[TIME_COL].min()
            delay_seconds = delay_days * 86400
            cutoff = t_val_start - delay_seconds
            before_rows = len(train_df)
            train_df = train_df[train_df[TIME_COL] <= cutoff].reset_index(drop=True)
            train_pos = int((train_df[TARGET_COL] == 1).sum())
            if len(train_df) == 0 or train_pos < 50:
                print(
                    f"[WARN] Fold {fold}: delay_days={delay_days} leaves too few train samples "
                    f"(rows={len(train_df)}, pos={train_pos}) after cutoff; skipping fold."
                )
                continue
            print(
                f"[INFO] Fold {fold}: applied delay_days={delay_days}, "
                f"train_rows {before_rows} -> {len(train_df)}, train_pos={train_pos}"
            )

        # Per-fold preprocessing (fit on train, transform val) to avoid leakage
        train_df, val_df, used_cols = preprocess_fold(train_df, val_df)
        print(f"[INFO] Fold {fold}: preprocessed -> {len(used_cols)} features, train={len(train_df)}, val={len(val_df)}")
        cat_cols = get_cat_cols_for_synth(train_df, used_cols)

        # ----------------------------
        # BASELINE
        # ----------------------------
        base = train_and_eval(train_df, val_df)

        pr_auc = _get_metric(base, ["pr_auc", "prauc", "prAUC"])
        recall = _get_metric(base, ["recall_at_1pct_fpr", "recall@1%fpr", "recall_at_1fpr", "recall_at_1_fpr"])

        print(f"BASELINE PR-AUC: {pr_auc:.4f}, Recall@1%FPR: {recall:.4f}")

        _append_row_csv({
            "timestamp": _now_str(),
            "fold": fold,
            "delay_days": delay_days,
            "run_id": run_id,
            "method": "baseline",
            "target_pos_rate": "",
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "train_pos": int((train_df[TARGET_COL] == 1).sum()),
            "train_neg": int((train_df[TARGET_COL] == 0).sum()),
            "synth_rows": 0,
            "final_train_rows": len(train_df),
            "final_pos_rate": float((train_df[TARGET_COL] == 1).mean()),
            "pr_auc": pr_auc,
            "recall_at_1pct_fpr": recall,
            "notes": "",
        }, run_results_csv)

        # ----------------------------
        # CTGAN TARGET RATES
        # ----------------------------
        for target_rate in target_pos_rates:
            print("\n" + "-" * 50)
            print(f"[CTGAN] target_pos_rate={target_rate}")

            synth_pos = make_synthetic_positives(
                train_df=train_df,
                cat_cols=cat_cols,
                used_cols=used_cols,
                target_pos_rate=target_rate,
                max_synth=max_synth,
                epochs=ctgan_epochs,
                batch_size=CTGAN_BATCH_SIZE,
                pac=CTGAN_PAC,
                seed=CTGAN_SEED,
                verbose=True,
            )

            mixed_train = pd.concat([train_df, synth_pos], axis=0, ignore_index=True)
            res = train_and_eval(mixed_train, val_df)
            pr_auc_m = _get_metric(res, ["pr_auc", "prauc", "prAUC"])
            recall_m = _get_metric(res, ["recall_at_1pct_fpr", "recall@1%fpr", "recall_at_1fpr", "recall_at_1_fpr"])
            print(f"CTGAN+REAL PR-AUC: {pr_auc_m:.4f}, Recall@1%FPR: {recall_m:.4f}")

            _append_row_csv({
                "timestamp": _now_str(),
                "fold": fold,
                "delay_days": delay_days,
                "run_id": run_id,
                "method": "ctgan",
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
                "notes": "",
            }, run_results_csv)

            if recency_ablation:
                print("\n" + "-" * 50)
                print(f"[CTGAN recency=0.3] target_pos_rate={target_rate}")
                synth_pos = make_synthetic_positives(
                    train_df=train_df,
                    cat_cols=cat_cols,
                    used_cols=used_cols,
                    target_pos_rate=target_rate,
                    max_synth=max_synth,
                    epochs=ctgan_epochs,
                    batch_size=CTGAN_BATCH_SIZE,
                    pac=CTGAN_PAC,
                    seed=CTGAN_SEED,
                    verbose=True,
                    recency_frac=0.3,
                    time_col=TIME_COL,
                )
                mixed_train = pd.concat([train_df, synth_pos], axis=0, ignore_index=True)
                res = train_and_eval(mixed_train, val_df)
                pr_auc_m = _get_metric(res, ["pr_auc", "prauc", "prAUC"])
                recall_m = _get_metric(res, ["recall_at_1pct_fpr", "recall@1%fpr", "recall_at_1fpr", "recall_at_1_fpr"])
                print(f"CTGAN_recency03 PR-AUC: {pr_auc_m:.4f}, Recall@1%FPR: {recall_m:.4f}")
                _append_row_csv({
                    "timestamp": _now_str(),
                    "fold": fold,
                    "delay_days": delay_days,
                    "run_id": run_id,
                    "method": "ctgan_recency03",
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
                    "notes": "recency_frac=0.3",
                }, run_results_csv)

        # ----------------------------
        # TabDDPM TARGET RATES
        # ----------------------------
        for target_rate in target_pos_rates:
            print("\n" + "-" * 50)
            print(f"[TabDDPM] target_pos_rate={target_rate}")

            synth_pos = make_synthetic_positives_tabddpm(
                train_df=train_df,
                cat_cols=cat_cols,
                used_cols=used_cols,
                target_pos_rate=target_rate,
                max_synth=max_synth,
                seed=TABDDPM_SEED,
                verbose=True,
                **tabddpm_kwargs,
            )

            mixed_train = pd.concat([train_df, synth_pos], axis=0, ignore_index=True)

            res = train_and_eval(mixed_train, val_df)
            pr_auc_m = _get_metric(res, ["pr_auc", "prauc", "prAUC"])
            recall_m = _get_metric(res, ["recall_at_1pct_fpr", "recall@1%fpr", "recall_at_1fpr", "recall_at_1_fpr"])

            print(f"TabDDPM+REAL PR-AUC: {pr_auc_m:.4f}, Recall@1%FPR: {recall_m:.4f}")

            _append_row_csv({
                "timestamp": _now_str(),
                "fold": fold,
                "delay_days": delay_days,
                "run_id": run_id,
                "method": "tabddpm",
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
                "notes": "",
            }, run_results_csv)

            if recency_ablation:
                print("\n" + "-" * 50)
                print(f"[TabDDPM recency=0.3] target_pos_rate={target_rate}")
                synth_pos = make_synthetic_positives_tabddpm(
                    train_df=train_df,
                    cat_cols=cat_cols,
                    used_cols=used_cols,
                    target_pos_rate=target_rate,
                    max_synth=max_synth,
                    seed=TABDDPM_SEED,
                    verbose=True,
                    recency_frac=0.3,
                    time_col=TIME_COL,
                    **tabddpm_kwargs,
                )
                mixed_train = pd.concat([train_df, synth_pos], axis=0, ignore_index=True)
                res = train_and_eval(mixed_train, val_df)
                pr_auc_m = _get_metric(res, ["pr_auc", "prauc", "prAUC"])
                recall_m = _get_metric(res, ["recall_at_1pct_fpr", "recall@1%fpr", "recall_at_1fpr", "recall_at_1_fpr"])
                print(f"TabDDPM_recency03 PR-AUC: {pr_auc_m:.4f}, Recall@1%FPR: {recall_m:.4f}")
                _append_row_csv({
                    "timestamp": _now_str(),
                    "fold": fold,
                    "delay_days": delay_days,
                    "run_id": run_id,
                    "method": "tabddpm_recency03",
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
                    "notes": "recency_frac=0.3",
                }, run_results_csv)

        # ----------------------------
        # SMOTE (same feature space as baseline/CTGAN/TabDDPM)
        # ----------------------------
        for target_rate in target_pos_rates:
            print("\n" + "-" * 50)
            print(f"[SMOTE] target_pos_rate={target_rate}")

            res = train_and_eval_smote(
                train_df=train_df,
                val_df=val_df,
                target_pos_rate=target_rate,
                k_neighbors=5,
                max_synth=max_synth,
            )
            pr_auc_m = _get_metric(res, ["pr_auc", "prauc", "prAUC"])
            recall_m = _get_metric(res, ["recall_at_1pct_fpr", "recall@1%fpr", "recall_at_1fpr", "recall_at_1_fpr"])

            print(f"SMOTE PR-AUC: {pr_auc_m:.4f}, Recall@1%FPR: {recall_m:.4f}")

            _append_row_csv({
                "timestamp": _now_str(),
                "fold": fold,
                "delay_days": delay_days,
                "run_id": run_id,
                "method": "smote",
                "target_pos_rate": float(target_rate),
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "train_pos": int((train_df[TARGET_COL] == 1).sum()),
                "train_neg": int((train_df[TARGET_COL] == 0).sum()),
                "synth_rows": 0,  # SMOTE doesn't report count; train_and_eval_smote doesn't return it
                "final_train_rows": len(train_df),
                "final_pos_rate": target_rate,
                "pr_auc": pr_auc_m,
                "recall_at_1pct_fpr": recall_m,
                "notes": "",
            }, run_results_csv)

            if recency_ablation:
                print("\n" + "-" * 50)
                print(f"[SMOTE recency=0.3] target_pos_rate={target_rate}")
                res = train_and_eval_smote(
                    train_df=train_df,
                    val_df=val_df,
                    target_pos_rate=target_rate,
                    k_neighbors=5,
                    max_synth=max_synth,
                    recency_frac=0.3,
                    time_col=TIME_COL,
                )
                pr_auc_m = _get_metric(res, ["pr_auc", "prauc", "prAUC"])
                recall_m = _get_metric(res, ["recall_at_1pct_fpr", "recall@1%fpr", "recall_at_1fpr", "recall_at_1_fpr"])
                print(f"SMOTE_recency03 PR-AUC: {pr_auc_m:.4f}, Recall@1%FPR: {recall_m:.4f}")
                _append_row_csv({
                    "timestamp": _now_str(),
                    "fold": fold,
                    "delay_days": delay_days,
                    "run_id": run_id,
                    "method": "smote_recency03",
                    "target_pos_rate": float(target_rate),
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "train_pos": int((train_df[TARGET_COL] == 1).sum()),
                    "train_neg": int((train_df[TARGET_COL] == 0).sum()),
                    "synth_rows": 0,
                    "final_train_rows": len(train_df),
                    "final_pos_rate": target_rate,
                    "pr_auc": pr_auc_m,
                    "recall_at_1pct_fpr": recall_m,
                    "notes": "recency_frac=0.3",
                }, run_results_csv)

    t_elapsed = time.perf_counter() - t_start
    elapsed_str = f"{int(t_elapsed // 60)}m {int(t_elapsed % 60)}s"
    print(f"\n[DONE] Total elapsed: {elapsed_str}")
    with open(run_runtime_txt, "a") as f:
        f.write(f"Actual elapsed: {elapsed_str}\n")
    if run_dir:
        # Copy run results into main cumulative log
        run_df = pd.read_csv(run_results_csv)
        write_header = not os.path.exists(RESULTS_CSV)
        run_df.to_csv(RESULTS_CSV, mode="a", header=write_header, index=False)
        print(f"Appended run results -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
