# src/run_xgb_sensitivity.py
"""
XGBoost fold 7 completion + fold sensitivity (n=12, n=15).

Memory-critical fix: save X_tr to disk before CTGAN training so the
full 524K-row feature matrix is not resident during CTGAN's gradient pass.
df_raw is also freed per-step so it is never alive during heavy computation.

Resume-safe: results/revision/checkpoint_xgb_sensitivity.json

Usage:
  python -u -m src.run_xgb_sensitivity [--steps fold7,n12,n15]
"""
from __future__ import annotations
import os, sys, gc, json, time, argparse, tempfile

os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"]     = "1"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.folds import get_temporal_folds
from src.preprocess_synth import (
    TARGET_COL, TIME_COL, PRIORITY_COLS,
    get_cat_cols_for_synth, preprocess_for_synth,
)
from src.synth_ctgan import fit_ctgan, sample_ctgan

PARQUET_PATH = os.path.join(_ROOT, "data", "train_merged.parquet")
OUT_DIR      = os.path.join(_ROOT, "results", "revision")
CHECKPOINT   = os.path.join(OUT_DIR, "checkpoint_xgb_sensitivity.json")
ROB_CSV      = os.path.join(OUT_DIR, "classifier_robustness.csv")

COLS_TO_READ  = 75
CTGAN_EPOCHS  = 150
CTGAN_BATCH   = 250
TARGET_RATE   = 0.05

os.makedirs(OUT_DIR, exist_ok=True)

# ── Checkpoint helpers ────────────────────────────────────────────────────────
def load_ck():
    return json.load(open(CHECKPOINT)) if os.path.exists(CHECKPOINT) else {}

def save_ck(state):
    json.dump(state, open(CHECKPOINT, "w"), indent=2)

# ── Parquet load ──────────────────────────────────────────────────────────────
def load_data():
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(PARQUET_PATH)
    all_cols = pf.schema_arrow.names
    selected = list(dict.fromkeys(
        [TIME_COL, TARGET_COL] + [c for c in PRIORITY_COLS if c in all_cols] + all_cols
    ))[:COLS_TO_READ + 2]
    df = pf.read(columns=selected).to_pandas()
    print(f"[XGB-SENS] Loaded parquet: {df.shape}")
    return df

# ── XGBoost (tree_method='hist' for memory efficiency) ────────────────────────
def run_xgb_pr(X_tr, y_tr, X_te, y_te):
    n_neg = int((y_tr == 0).sum())
    n_pos = max(1, int((y_tr == 1).sum()))
    clf = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        scale_pos_weight=n_neg / n_pos,
        eval_metric="aucpr", random_state=42,
        verbosity=0, n_jobs=1,
        tree_method="hist",
        device="cpu",
    )
    clf.fit(X_tr, y_tr)
    score = float(average_precision_score(y_te, clf.predict_proba(X_te)[:, 1]))
    del clf; gc.collect()
    return score

# ── Wilcoxon summary ──────────────────────────────────────────────────────────
def summarise(rows, label):
    deltas = [r["delta"] for r in rows if r.get("delta") is not None]
    if len(deltas) < 4:
        print(f"  [{label}] only {len(deltas)} deltas -- skip")
        return None, None, None
    try:
        _, p = stats.wilcoxon(deltas, alternative="greater")
    except Exception:
        p = 1.0
    n_pos  = sum(d > 0 for d in deltas)
    mean_d = float(np.mean(deltas))
    print(f"  [{label}] n={len(deltas)}, mean={mean_d:+.4f}, "
          f"p={p:.4f}, {n_pos}/{len(deltas)} pos")
    return round(p, 6), round(mean_d, 6), n_pos

# ── One fold — memory-safe ────────────────────────────────────────────────────
def run_fold(k, train_df, val_df):
    """
    Memory layout during CTGAN:
      - X_tr saved to temp .npy, deleted from RAM
      - df_raw must already be freed by caller before calling this
      - fraud_df (~18K rows) is the only training data in RAM during CTGAN
    """
    n_fraud_val = int((val_df[TARGET_COL] == 1).sum())
    if n_fraud_val < 30:
        print(f"    fold {k}: only {n_fraud_val} fraud in val -- skip")
        return None

    t0 = time.time()
    gc.collect()

    train_proc, used_cols = preprocess_for_synth(train_df)
    del train_df; gc.collect()
    val_proc, _ = preprocess_for_synth(val_df)
    del val_df; gc.collect()

    cat_cols  = get_cat_cols_for_synth(train_proc, used_cols)
    feat_cols = [c for c in used_cols if c != TARGET_COL]
    cat_pres  = [c for c in cat_cols if c in train_proc.columns]

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    # ── Build feature matrices ────────────────────────────────────────────────
    X_tr_df = train_proc[feat_cols].copy()
    if cat_pres:
        X_tr_df[cat_pres] = enc.fit_transform(X_tr_df[cat_pres].astype(str))
    X_tr = X_tr_df.values.astype(float);  del X_tr_df
    y_tr = train_proc[TARGET_COL].values.astype(int)

    X_va_df = val_proc[feat_cols].copy()
    if cat_pres:
        X_va_df[cat_pres] = enc.transform(X_va_df[cat_pres].astype(str))
    X_va = X_va_df.values.astype(float);  del X_va_df
    y_va = val_proc[TARGET_COL].values.astype(int)
    del val_proc; gc.collect()

    # ── Prepare CTGAN inputs BEFORE freeing train_proc ────────────────────────
    n_pos_tr   = int((train_proc[TARGET_COL] == 1).sum())
    n_neg_tr   = int((train_proc[TARGET_COL] == 0).sum())
    n_synth_tgt = min(50000, max(0,
        int(n_neg_tr * TARGET_RATE / (1 - TARGET_RATE)) - n_pos_tr))
    fraud_df = train_proc[train_proc[TARGET_COL] == 1][used_cols].copy()

    # ── Save X_tr to disk so it is NOT in RAM during CTGAN ───────────────────
    # Use project results dir (F: drive, ~38 GB free) — C: has <200 MB free
    tmpfile = os.path.join(OUT_DIR, f"_xtr_fold{k}_tmp.npy")
    np.save(tmpfile, X_tr)
    del X_tr, train_proc; gc.collect()
    print(f"    fold {k}: CTGAN (n_fraud={n_pos_tr}, n_synth={n_synth_tgt})...",
          end=" ", flush=True)

    # ── CTGAN — only fraud_df, X_va, y_tr, y_va in RAM ───────────────────────
    ctgan_model, ctgan_arts = fit_ctgan(
        train_df=fraud_df, cat_cols=cat_cols, used_cols=used_cols,
        epochs=CTGAN_EPOCHS, batch_size=CTGAN_BATCH,
        discriminator_steps=5, pac=1, seed=0, verbose=False,
    )
    del fraud_df; gc.collect()
    synth_df = sample_ctgan(ctgan_model, n=n_synth_tgt, artifacts=ctgan_arts)
    del ctgan_model, ctgan_arts; gc.collect()

    # ── Reload X_tr from disk ─────────────────────────────────────────────────
    X_tr = np.load(tmpfile)
    try:
        os.remove(tmpfile)
    except Exception:
        pass

    # ── Baseline XGBoost ──────────────────────────────────────────────────────
    baseline_pr = run_xgb_pr(X_tr, y_tr, X_va, y_va)

    # ── Augmented XGBoost ─────────────────────────────────────────────────────
    n_synth = len(synth_df)
    if n_synth > 0:
        synth_enc = synth_df[feat_cols].copy()
        if cat_pres:
            synth_enc[cat_pres] = enc.transform(synth_enc[cat_pres].astype(str))
        X_syn  = synth_enc.values.astype(float);  del synth_enc, synth_df
        X_aug  = np.vstack([X_tr, X_syn]);         del X_syn
        y_aug  = np.concatenate([y_tr, np.ones(X_aug.shape[0] - len(X_tr), dtype=int)])
        ctgan_pr = run_xgb_pr(X_aug, y_aug, X_va, y_va)
        del X_aug, y_aug; gc.collect()
    else:
        ctgan_pr = np.nan

    del X_tr, X_va, y_tr, y_va; gc.collect()

    delta   = ctgan_pr - baseline_pr if not np.isnan(ctgan_pr) else np.nan
    elapsed = time.time() - t0
    print(f"base={baseline_pr:.4f} ctgan={ctgan_pr:.4f} "
          f"delta={delta:+.4f} n_synth={n_synth} ({elapsed:.0f}s)")

    return {
        "fold": k,
        "n_fraud_val": n_fraud_val,
        "n_synth": n_synth,
        "baseline_pr": round(float(baseline_pr), 6),
        "ctgan_pr": round(float(ctgan_pr), 6) if not np.isnan(ctgan_pr) else None,
        "delta": round(float(delta), 6) if not np.isnan(delta) else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Complete XGBoost fold 7
# ══════════════════════════════════════════════════════════════════════════════
def run_fold7(ck):
    print("\n=== STEP 1: XGBoost fold 7 (OOM fix) ===")

    xgb_existing = {}
    if os.path.exists(ROB_CSV):
        df_rob = pd.read_csv(ROB_CSV)
        xgb_rows = df_rob[df_rob["classifier"] == "xgboost"].sort_values("fold")
        xgb_existing = {int(r["fold"]): float(r["delta"]) for _, r in xgb_rows.iterrows()}
        print(f"Existing XGBoost folds: {sorted(xgb_existing.keys())}")
        for fk, dv in sorted(xgb_existing.items()):
            print(f"  fold {fk}: {dv:+.4f}")
    else:
        xgb_existing = {0:0.0112, 1:0.0159, 2:0.0117, 3:0.0042, 4:0.0140, 5:0.0144, 6:-0.0000}
        print("WARNING: using hardcoded fallback deltas")

    if "fold7_result" in ck:
        print(f"Fold 7: cached delta={ck['fold7_result'].get('delta')}")
        fold7_result = ck["fold7_result"]
    else:
        # Load data, extract fold 7, FREE df_raw before run_fold
        df_raw = load_data()
        folds  = get_temporal_folds(df_raw, n_folds=8, time_col=TIME_COL)
        fi     = folds[7]
        train_df = fi["train_df"].copy()
        val_df   = fi["val_df"].copy()
        del df_raw, folds; gc.collect()   # FREE before heavy compute
        print(f"Fold 7: train={len(train_df)}, val={len(val_df)}")

        fold7_result = run_fold(7, train_df, val_df)
        ck["fold7_result"] = fold7_result
        save_ck(ck)

    # Complete 8-fold summary
    all_8 = [xgb_existing.get(k) for k in range(7)]
    if fold7_result and fold7_result.get("delta") is not None:
        all_8.append(fold7_result["delta"])

    rows_8 = [{"fold": k, "delta": d, "n_folds": 8}
               for k, d in enumerate(all_8) if d is not None]
    print("\n8-fold XGBoost (complete):")
    summarise([{"delta": d} for d in all_8 if d is not None], "XGBoost 8-fold")

    pd.DataFrame(rows_8).to_csv(os.path.join(OUT_DIR, "xgb_fold7.csv"), index=False)
    print("Saved: results/revision/xgb_fold7.csv")
    return all_8


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: XGBoost fold sensitivity n=12 and n=15
# ══════════════════════════════════════════════════════════════════════════════
def run_sensitivity(ck, n_folds_list=(12, 15)):
    print("\n=== STEP 2: XGBoost fold sensitivity ===")
    all_rows = []

    for n_folds in n_folds_list:
        key  = f"xgb_sens_{n_folds}"
        done = {int(k2) for k2 in ck.get(key, {}).keys()}
        fold_rows = [ck[key][str(k2)] for k2 in sorted(done)] if key in ck else []
        print(f"\n  n_folds={n_folds} | done: {sorted(done)}")

        # Process one fold at a time — load parquet, extract fold, free parquet
        # Pre-copying all folds at once causes OOM (expanding windows sum to 7.5x data)
        for k in range(n_folds):
            if k in done:
                print(f"    fold {k}: cached")
                continue

            df_raw = load_data()
            folds  = get_temporal_folds(df_raw, n_folds=n_folds, time_col=TIME_COL)
            train_df = folds[k]["train_df"].copy()
            val_df   = folds[k]["val_df"].copy()
            del df_raw, folds; gc.collect()   # FREE before heavy compute

            result = run_fold(k, train_df, val_df)
            if result is None:
                continue

            result["n_folds"] = n_folds
            fold_rows.append(result)
            if key not in ck:
                ck[key] = {}
            ck[key][str(k)] = result
            save_ck(ck)
            gc.collect()

        valid = [r for r in fold_rows if r.get("delta") is not None]
        summarise(valid, f"XGBoost n={n_folds}")
        all_rows.extend(fold_rows)

    if all_rows:
        pd.DataFrame(all_rows).to_csv(
            os.path.join(OUT_DIR, "xgb_fold_sensitivity.csv"), index=False
        )
        print("Saved: results/revision/xgb_fold_sensitivity.csv")
    return all_rows


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(all_8, sens_rows):
    print("\n=== PAPER-READY NUMBERS ===")
    valid_8 = [d for d in all_8 if d is not None]
    if valid_8:
        try:
            _, p8 = stats.wilcoxon(valid_8, alternative="greater")
        except Exception:
            p8 = 1.0
        n_pos8 = sum(d > 0 for d in valid_8)
        print(f"XGBoost 8-fold: n={len(valid_8)}, mean={np.mean(valid_8):+.4f}, "
              f"p={p8:.4f}, {n_pos8}/{len(valid_8)} pos")

    for n_folds in [12, 15]:
        rows = [r for r in sens_rows if r.get("n_folds") == n_folds]
        deltas = [r["delta"] for r in rows if r.get("delta") is not None]
        if deltas:
            try:
                _, p = stats.wilcoxon(deltas, alternative="greater")
            except Exception:
                p = 1.0
            n_pos = sum(d > 0 for d in deltas)
            print(f"XGBoost n={n_folds}:  n={len(deltas)}, mean={np.mean(deltas):+.4f}, "
                  f"p={p:.4f}, {n_pos}/{len(deltas)} pos")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default="fold7,n12,n15",
                        help="Comma-separated: fold7, n12, n15")
    args   = parser.parse_args()
    steps  = set(args.steps.lower().split(","))

    ck = load_ck()

    all_8     = []
    sens_rows = []

    if "fold7" in steps:
        all_8 = run_fold7(ck)

    n_folds_to_run = []
    if "n12" in steps:
        n_folds_to_run.append(12)
    if "n15" in steps:
        n_folds_to_run.append(15)
    if n_folds_to_run:
        sens_rows = run_sensitivity(ck, n_folds_list=n_folds_to_run)

    print_summary(all_8, sens_rows)
    print("\n=== DONE ===")
