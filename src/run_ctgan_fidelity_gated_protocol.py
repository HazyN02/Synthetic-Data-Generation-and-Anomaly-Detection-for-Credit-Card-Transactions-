from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

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


def _run_analysis(run_dir: str) -> None:
    """
    A2 ablation summary: compare ungated vs DCR-p90-gated CTGAN at target_pos_rate=0.05.

    Loads:
      results.csv               — baseline + ungated CTGAN per fold
      results_ctgan_gated.csv   — gated CTGAN per fold
      fidelity_ctgan_gated.csv  — raw vs filtered sample counts per fold

    Outputs:
      Printed per-fold delta table
      results/a2_per_fold_deltas.csv
      Summary line: n_synth, mean delta, Wilcoxon p for both gated and ungated
    """
    canonical_csv = os.path.join(run_dir, "results.csv")
    gated_csv = os.path.join(run_dir, "results_ctgan_gated.csv")
    fidelity_csv = os.path.join(run_dir, "fidelity_ctgan_gated.csv")

    missing = [p for p in [canonical_csv, gated_csv, fidelity_csv] if not os.path.exists(p)]
    if missing:
        print(f"[A2 ANALYSIS] Missing files, skipping analysis: {missing}")
        return

    canon = pd.read_csv(canonical_csv)
    gated_df = pd.read_csv(gated_csv)
    fidelity_df = pd.read_csv(fidelity_csv)

    RATE = 0.05

    # --- Baseline per fold (one row per fold, no target_pos_rate column) ---
    if "method" in canon.columns:
        bl_rows = canon[canon["method"].str.lower().str.contains("baseline", na=False)]
    else:
        bl_rows = canon[canon["target_pos_rate"].isna()]
    bl = bl_rows.drop_duplicates("fold").set_index("fold")["pr_auc"]

    # --- Ungated CTGAN@0.05 ---
    ug_rows = canon[
        canon["method"].str.lower().str.contains("ctgan", na=False) &
        (canon["target_pos_rate"].round(3) == round(RATE, 3))
    ].drop_duplicates("fold").set_index("fold")

    # --- Gated CTGAN@0.05 ---
    g_rows = gated_df[
        (gated_df["target_pos_rate"].round(3) == round(RATE, 3))
    ].drop_duplicates("fold").set_index("fold")

    # --- Raw vs filtered sample counts from fidelity CSV (rate=0.05 rows only) ---
    # fidelity rows are written once per (fold, rate) combination; for rate=0.05
    # we take the first row per fold (rows are ordered by fold then rate).
    fid_05 = fidelity_df.groupby("fold").first()  # first rate written is 0.05
    # cross-check: if n_synthetic in gated notes is available, use that
    if "notes" in gated_df.columns:
        note_05 = gated_df[gated_df["target_pos_rate"].round(3) == round(RATE, 3)].copy()
        note_05["raw_synth"] = (
            note_05["notes"]
            .str.extract(r"raw_synth_rows=(\d+)", expand=False)
            .astype(float)
        )
        raw_by_fold = note_05.drop_duplicates("fold").set_index("fold")["raw_synth"]
    else:
        raw_by_fold = fid_05["n_synthetic"] if "n_synthetic" in fid_05.columns else pd.Series(dtype=float)

    folds = sorted(set(bl.index) & set(ug_rows.index) & set(g_rows.index))

    per_fold_rows = []
    for f in folds:
        bl_val = float(bl.loc[f])
        ug_val = float(ug_rows.loc[f, "pr_auc"])
        g_val = float(g_rows.loc[f, "pr_auc"])
        n_raw = int(raw_by_fold.loc[f]) if f in raw_by_fold.index else int(ug_rows.loc[f, "synth_rows"]) if "synth_rows" in ug_rows.columns else 0
        n_gated = int(g_rows.loc[f, "synth_rows"]) if "synth_rows" in g_rows.columns else 0
        vol_red = 100.0 * (n_raw - n_gated) / max(1, n_raw)
        per_fold_rows.append({
            "fold": f,
            "pr_auc_baseline": round(bl_val, 6),
            "pr_auc_ungated": round(ug_val, 6),
            "pr_auc_gated_p90": round(g_val, 6),
            "delta_ungated": round(ug_val - bl_val, 6),
            "delta_gated_p90": round(g_val - bl_val, 6),
            "n_synth_raw": n_raw,
            "n_synth_gated": n_gated,
            "vol_reduction_pct": round(vol_red, 1),
        })

    df_folds = pd.DataFrame(per_fold_rows)
    deltas_ug = df_folds["delta_ungated"].values
    deltas_g = df_folds["delta_gated_p90"].values

    # Wilcoxon one-sided (greater): H1 = deltas > 0
    try:
        _, p_ug = wilcoxon(deltas_ug, alternative="greater")
    except Exception:
        p_ug = float("nan")
    try:
        _, p_g = wilcoxon(deltas_g, alternative="greater")
    except Exception:
        p_g = float("nan")

    mean_n_raw = int(np.mean(df_folds["n_synth_raw"]))
    mean_n_gated = int(np.mean(df_folds["n_synth_gated"]))
    mean_vol_red = 100.0 * (mean_n_raw - mean_n_gated) / max(1, mean_n_raw)

    # --- Per-fold table ---
    print("\n" + "=" * 75)
    print("A2 ABLATION: DCR p90 Quality Gate — Per-Fold Delta Table")
    print("=" * 75)
    header = (
        f"{'Fold':>4}  {'Baseline':>8}  {'Ungated':>8}  {'Gated-p90':>9}  "
        f"{'dUngated':>9}  {'dGated':>8}  {'N_raw':>7}  {'N_gated':>7}  {'Vol%':>6}"
    )
    print(header)
    print("-" * 75)
    for r in per_fold_rows:
        print(
            f"{r['fold']:>4}  {r['pr_auc_baseline']:>8.4f}  {r['pr_auc_ungated']:>8.4f}  "
            f"{r['pr_auc_gated_p90']:>9.4f}  {r['delta_ungated']:>+9.4f}  "
            f"{r['delta_gated_p90']:>+8.4f}  {r['n_synth_raw']:>7d}  "
            f"{r['n_synth_gated']:>7d}  {r['vol_reduction_pct']:>5.1f}%"
        )
    print("-" * 75)
    mean_ug = float(np.mean(deltas_ug))
    mean_g = float(np.mean(deltas_g))
    print(
        f"{'Mean':>4}  {'':>8}  {'':>8}  {'':>9}  {mean_ug:>+9.4f}  "
        f"{mean_g:>+8.4f}  {mean_n_raw:>7d}  {mean_n_gated:>7d}  {mean_vol_red:>5.1f}%"
    )

    # --- Summary lines (requested format) ---
    print("\n" + "=" * 75)
    print("A2 ABLATION SUMMARY (target_pos_rate=0.05, Wilcoxon one-sided, n=8 folds)")
    print("=" * 75)
    print(f"  A2 ungated:  n_synth = {mean_n_raw}, delta = {mean_ug:+.4f}, p = {p_ug:.6f}")
    print(f"  A2 p90 gate: n_synth = {mean_n_gated}, delta = {mean_g:+.4f}, p = {p_g:.6f}")
    print(f"  Volume reduction: ({mean_n_raw}-{mean_n_gated})/{mean_n_raw} * 100 = {mean_vol_red:.1f}%")

    # Explain direction of effect if gate hurts
    if mean_g < mean_ug:
        print(
            f"\n  NOTE: The DCR p90 gate REDUCES mean delta ({mean_g:+.4f} vs {mean_ug:+.4f} ungated). "
            f"Mechanism: the gate discards {mean_vol_red:.1f}% of synthetic samples "
            f"({mean_n_raw - mean_n_gated} rows/fold on average), shrinking the augmented "
            f"training set. The classifier performance loss from reduced volume outweighs "
            f"any quality benefit from filtering low-fidelity samples at this gate threshold."
        )

    # --- Save per-fold CSV ---
    out_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_csv = os.path.join(out_root, "results", "a2_per_fold_deltas.csv")
    df_folds.to_csv(out_csv, index=False)
    print(f"\n  Per-fold table saved to: {out_csv}")
    print("=" * 75)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone CTGAN fidelity-gated run.")
    parser.add_argument("--run-id", type=str, default="20260330_180216")
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--start-fold", type=int, default=0)
    parser.add_argument(
        "--analyse-only",
        action="store_true",
        help="Skip the protocol run; only compute A2 ablation statistics from existing CSVs.",
    )
    args = parser.parse_args()

    run_id = args.run_id
    root = os.path.dirname(os.path.dirname(__file__))
    run_dir = os.path.join(root, "results", "protocol", f"run_{run_id}")

    if args.analyse_only:
        _run_analysis(run_dir)
        return

    n_folds = int(args.n_folds)
    start_fold = int(args.start_fold)

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

    _run_analysis(run_dir)
