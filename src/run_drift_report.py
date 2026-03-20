# src/run_drift_report.py
"""
Drift report: domain AUC, leak hunting, PSI per temporal fold.
Use --protocol-folds to align with run_protocol (same fold boundaries for drift-harm analysis).
"""
import argparse
import os

import pandas as pd

from src.folds import get_temporal_folds
from src.drift import domain_classifier_auc, single_feature_auc_scan, population_stability_index


OUT_PATH = "experiments/results/drift_report.csv"
CONTINUOUS_PSI_COLS = ["TransactionAmt", "TransactionDT", "dist1", "dist2", "D1", "D2", "D3"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol-folds",
        action="store_true",
        help="Use same fold split as run_protocol (n_folds=4) for drift-harm alignment",
    )
    parser.add_argument("--n-folds", type=int, default=4, help="Number of folds (used with --protocol-folds)")
    args = parser.parse_args()

    os.makedirs("experiments/results", exist_ok=True)

    df = pd.read_parquet("data/train_merged.parquet")

    if args.protocol_folds:
        folds = get_temporal_folds(df, n_folds=args.n_folds, time_col="TransactionDT")
        print(f"Using protocol-aligned folds (n_folds={args.n_folds})")
    else:
        # Legacy: split.time_based_cv style (N_SPLITS from config)
        from src.split import time_based_cv
        splits = time_based_cv(df)
        folds = [
            {"fold": i, "train_df": df.iloc[tr_idx].reset_index(drop=True), "val_df": df.iloc[va_idx].reset_index(drop=True)}
            for i, (tr_idx, va_idx) in enumerate(splits)
        ]
        print("Using legacy time_based_cv splits")

    rows = []

    for fold_info in folds:
        fold = fold_info["fold"]
        train_df = fold_info["train_df"]
        val_df = fold_info["val_df"]

        print(f"\n===== DRIFT REPORT: Fold {fold} =====")
        row = {"fold": fold}

        # Leak-hunting: find single columns that separate domains
        top = single_feature_auc_scan(train_df, val_df, top_k=10)
        print("Top single-feature AUC (leak hunting):")
        for c, a in top:
            print(f"  {c}: {a:.4f}")
        row["top_leak_feature"] = top[0][0] if len(top) else ""
        row["top_leak_auc"] = top[0][1] if len(top) else 0.0

        # Proper domain AUC excluding time keys / ids
        auc = domain_classifier_auc(train_df, val_df)
        row["domain_auc_holdout_no_time"] = auc
        print(f"Domain AUC (holdout, no TransactionDT/IDs): {auc:.4f}")

        # PSI
        for col in CONTINUOUS_PSI_COLS:
            if col in train_df.columns:
                psi = population_stability_index(train_df[col], val_df[col])
                row[f"psi_{col}"] = psi
                print(f"PSI({col}): {psi:.4f}")

        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT_PATH, index=False)

    print("\nSaved:", OUT_PATH)


if __name__ == "__main__":
    main()
