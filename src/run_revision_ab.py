# src/run_revision_ab.py
"""
Revision Experiments A and B — overnight run.

A: Fold sensitivity at n=12 and n=15 (baseline + CTGAN per fold)
B: Multi-classifier robustness (XGBoost, LogReg) on canonical 8 folds

Resume-safe via results/revision/checkpoint_ab.json.
Expected runtime: ~10-14 hours (CTGAN ~20 min/fold).

Usage:
  python -m src.run_revision_ab [--exp A] [--exp B] [--exp AB]
"""
from __future__ import annotations
import os, sys, gc, json, time, argparse
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"]     = "1"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.folds import get_temporal_folds
from src.preprocess_synth import (
    TARGET_COL, TIME_COL, PRIORITY_COLS,
    get_cat_cols_for_synth, preprocess_for_synth,
)
from src.synth_ctgan import make_synthetic_positives

PARQUET_PATH  = os.path.join(_ROOT, "data", "train_merged.parquet")
OUT_DIR       = os.path.join(_ROOT, "results", "revision")
CHECKPOINT    = os.path.join(OUT_DIR, "checkpoint_ab.json")
COLS_TO_READ  = 75   # reduced to avoid OOM on large fold training sets
CTGAN_EPOCHS  = 150
CTGAN_BATCH   = 250   # reduced to ease memory pressure on large folds
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
    print(f"[AB] Loaded parquet: {df.shape}")
    return df

# ── Feature matrix builder (handles LightGBM native cats + ordinal for others)
def build_Xy(proc_df: pd.DataFrame, cat_cols: list, used_cols: list,
             encoder=None, scaler=None, fit=True):
    """Return (X, y, encoder, scaler). encoder/scaler fitted if fit=True."""
    feat_cols = [c for c in used_cols if c != TARGET_COL]
    y = proc_df[TARGET_COL].values.astype(int)
    X_df = proc_df[feat_cols].copy()

    # Ordinal-encode cats (needed for XGBoost / LogReg)
    if encoder is None:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value",
                                 unknown_value=-1)
    cat_present = [c for c in cat_cols if c in X_df.columns]
    if cat_present:
        if fit:
            X_df[cat_present] = encoder.fit_transform(X_df[cat_present].astype(str))
        else:
            X_df[cat_present] = encoder.transform(X_df[cat_present].astype(str))

    X = X_df.values.astype(float)
    if scaler is None:
        scaler = StandardScaler()
    if fit:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    return X, y, encoder, scaler

# ── LightGBM config (identical to canonical run) ─────────────────────────────
LGBM_PARAMS = dict(
    objective="binary", n_estimators=300, learning_rate=0.05,
    num_leaves=64, min_child_samples=200, subsample=0.8,
    colsample_bytree=0.8, n_jobs=1, random_state=42, verbosity=-1,
)

def fit_lgbm(X_tr, y_tr, cat_cols_idx=None):
    clf = lgb.LGBMClassifier(**LGBM_PARAMS)
    clf.fit(X_tr, y_tr)
    return clf

def run_lgbm_pr(X_tr, y_tr, X_te, y_te):
    clf = fit_lgbm(X_tr, y_tr)
    return float(average_precision_score(y_te, clf.predict_proba(X_te)[:, 1]))

# ── XGBoost ───────────────────────────────────────────────────────────────────
def run_xgb_pr(X_tr, y_tr, X_te, y_te):
    from xgboost import XGBClassifier
    n_neg = int((y_tr == 0).sum())
    n_pos = max(1, int((y_tr == 1).sum()))
    clf = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        scale_pos_weight=n_neg / n_pos,
        eval_metric="aucpr", random_state=42,
        verbosity=0, use_label_encoder=False, n_jobs=1,
    )
    clf.fit(X_tr, y_tr)
    return float(average_precision_score(y_te, clf.predict_proba(X_te)[:, 1]))

# ── LogReg ────────────────────────────────────────────────────────────────────
def run_logreg_pr(X_tr, y_tr, X_te, y_te):
    # Replace NaN/inf with 0 before LogReg (can't handle missing)
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)
    clf = LogisticRegression(
        class_weight="balanced", max_iter=1000,
        C=0.1, random_state=42, n_jobs=1,
    )
    clf.fit(X_tr, y_tr)
    return float(average_precision_score(y_te, clf.predict_proba(X_te)[:, 1]))

# ── Wilcoxon summary ──────────────────────────────────────────────────────────
def summarise(rows, label):
    deltas = [r["delta"] for r in rows if r.get("delta") is not None]
    if len(deltas) >= 4:
        try:
            _, p = stats.wilcoxon(deltas, alternative="greater")
        except Exception:
            p = 1.0
        n_pos = sum(d > 0 for d in deltas)
        print(f"  [{label}] n={len(deltas)}, mean={np.mean(deltas):+.4f}, "
              f"p={p:.4f}, {n_pos}/{len(deltas)} pos")
        return round(float(p), 6), round(float(np.mean(deltas)), 6), n_pos
    return None, None, None

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A: Fold sensitivity (n=12 and n=15)
# ══════════════════════════════════════════════════════════════════════════════
def run_experiment_a(df_raw, ck):
    print("\n=== EXPERIMENT A: Fold sensitivity ===")
    all_rows = []

    for n_folds in [12, 15]:
        key = f"A_n{n_folds}"
        done_folds = set(ck.get(key, {}).keys())
        fold_rows  = list(ck.get(key, {}).values()) if key in ck else []
        fold_rows  = [r for sublist in fold_rows for r in (sublist if isinstance(sublist, list) else [sublist])]

        print(f"\n  n_folds={n_folds} | already done folds: {sorted(done_folds)}")
        folds = get_temporal_folds(df_raw, n_folds=n_folds, time_col=TIME_COL)

        for fold_info in folds:
            k = fold_info["fold"]
            if str(k) in done_folds:
                print(f"    fold {k}: skipping (done)")
                continue

            train_df = fold_info["train_df"]
            val_df   = fold_info["val_df"]
            n_fraud_test = int((val_df[TARGET_COL] == 1).sum())
            if n_fraud_test < 30:
                print(f"    fold {k}: only {n_fraud_test} fraud in val — skip")
                continue

            t0 = time.time()
            gc.collect()
            # Preprocess — work on copies then free originals
            train_proc, used_cols = preprocess_for_synth(train_df)
            del train_df; gc.collect()
            val_proc,   _         = preprocess_for_synth(val_df)
            del val_df; gc.collect()
            cat_cols = get_cat_cols_for_synth(train_proc, used_cols)
            feat_cols = [c for c in used_cols if c != TARGET_COL]

            # Feature matrices (ordinal-encoded for compatibility)
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            cat_pres = [c for c in cat_cols if c in train_proc.columns]
            X_tr = train_proc[feat_cols].copy()
            X_va = val_proc[feat_cols].copy()
            if cat_pres:
                X_tr[cat_pres] = enc.fit_transform(X_tr[cat_pres].astype(str))
                X_va[cat_pres] = enc.transform(X_va[cat_pres].astype(str))
            X_tr = X_tr.values.astype(float)
            X_va = X_va.values.astype(float)
            y_tr = train_proc[TARGET_COL].values.astype(int)
            y_va = val_proc[TARGET_COL].values.astype(int)

            # Baseline
            baseline_pr = run_lgbm_pr(X_tr, y_tr, X_va, y_va)

            # CTGAN augmented
            synth_df = make_synthetic_positives(
                train_df=train_proc, cat_cols=cat_cols, used_cols=used_cols,
                target_pos_rate=TARGET_RATE, epochs=CTGAN_EPOCHS,
                batch_size=CTGAN_BATCH, discriminator_steps=5, pac=1, seed=0, verbose=False,
            )
            if len(synth_df) > 0:
                synth_proc = synth_df[feat_cols].copy()
                del synth_df; gc.collect()
                if cat_pres:
                    synth_proc[cat_pres] = enc.transform(synth_proc[cat_pres].astype(str))
                X_aug = np.vstack([X_tr, synth_proc.values.astype(float)])
                del synth_proc; gc.collect()
                y_aug = np.concatenate([y_tr, np.ones(X_aug.shape[0] - len(X_tr), dtype=int)])
                n_synth = X_aug.shape[0] - len(X_tr)
                ctgan_pr = run_lgbm_pr(X_aug, y_aug, X_va, y_va)
                del X_aug, y_aug; gc.collect()
                delta = ctgan_pr - baseline_pr
            else:
                ctgan_pr = np.nan
                delta = np.nan
                n_synth = 0

            row = {
                "n_folds": n_folds, "fold": k,
                "n_fraud_train": int((y_tr == 1).sum()),
                "n_fraud_val": n_fraud_test,
                "n_synth": n_synth,
                "baseline_pr": round(baseline_pr, 6),
                "ctgan_pr": round(ctgan_pr, 6) if not np.isnan(ctgan_pr) else None,
                "delta": round(delta, 6) if not np.isnan(delta) else None,
            }
            fold_rows.append(row)
            print(f"    fold {k:>2}: base={baseline_pr:.4f} ctgan={ctgan_pr:.4f} "
                  f"delta={delta:+.4f} n_synth={n_synth} ({time.time()-t0:.0f}s)")

            # Checkpoint after each fold
            if key not in ck:
                ck[key] = {}
            ck[key][str(k)] = row
            save_ck(ck)
            gc.collect()

        valid = [r for r in fold_rows if r.get("delta") is not None]
        p, mean_d, n_pos = summarise(valid, f"A n={n_folds}")
        for r in valid:
            r["wilcoxon_p"] = p
        all_rows.extend(fold_rows)

    pd.DataFrame(all_rows).to_csv(
        os.path.join(OUT_DIR, "fold_sensitivity.csv"), index=False
    )
    print("  Saved: results/revision/fold_sensitivity.csv")
    return all_rows


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B: Multi-classifier robustness
# ══════════════════════════════════════════════════════════════════════════════
def run_experiment_b(df_raw, ck):
    print("\n=== EXPERIMENT B: Multi-classifier robustness ===")
    folds = get_temporal_folds(df_raw, n_folds=8, time_col=TIME_COL)
    # Extract fold data before freeing df_raw to save memory on large folds
    fold_data = [(fi["fold"], fi["train_df"].copy(), fi["val_df"].copy()) for fi in folds]
    del df_raw, folds; gc.collect()

    all_rows = []
    done_key = "B_done_folds"
    done_folds = set(ck.get(done_key, []))

    for k, train_df, val_df in fold_data:
        if k in done_folds:
            print(f"  fold {k}: skipping (done)")
            saved = ck.get(f"B_fold_{k}", [])
            all_rows.extend(saved)
            del train_df, val_df; gc.collect()
            continue

        t0 = time.time()
        gc.collect()

        train_proc, used_cols = preprocess_for_synth(train_df)
        del train_df; gc.collect()
        val_proc,   _         = preprocess_for_synth(val_df)
        del val_df; gc.collect()
        cat_cols  = get_cat_cols_for_synth(train_proc, used_cols)
        feat_cols = [c for c in used_cols if c != TARGET_COL]

        # Ordinal encode for XGBoost/LogReg
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        sca = StandardScaler()
        cat_pres = [c for c in cat_cols if c in train_proc.columns]

        X_tr_raw = train_proc[feat_cols].copy()
        X_va_raw = val_proc[feat_cols].copy()
        if cat_pres:
            X_tr_raw[cat_pres] = enc.fit_transform(X_tr_raw[cat_pres].astype(str))
            X_va_raw[cat_pres] = enc.transform(X_va_raw[cat_pres].astype(str))
        X_tr = X_tr_raw.values.astype(float)
        X_va = X_va_raw.values.astype(float)
        y_tr = train_proc[TARGET_COL].values.astype(int)
        y_va = val_proc[TARGET_COL].values.astype(int)

        X_tr_sc = sca.fit_transform(X_tr)
        X_va_sc = sca.transform(X_va)

        # Generate CTGAN synthetics ONCE for this fold
        print(f"  fold {k}: generating CTGAN synthetics...")
        synth_df = make_synthetic_positives(
            train_df=train_proc, cat_cols=cat_cols, used_cols=used_cols,
            target_pos_rate=TARGET_RATE, epochs=CTGAN_EPOCHS,
            batch_size=CTGAN_BATCH, discriminator_steps=5, pac=1, seed=0, verbose=False,
        )

        fold_rows = []
        for clf_name, run_fn, use_scaled in [
            ("xgboost", run_xgb_pr, False),
            ("logreg",  run_logreg_pr, True),
        ]:
            X_use_tr = X_tr_sc if use_scaled else X_tr
            X_use_va = X_va_sc if use_scaled else X_va

            base_pr = run_fn(X_use_tr, y_tr, X_use_va, y_va)

            if len(synth_df) > 0:
                synth_enc = synth_df[feat_cols].copy()
                if cat_pres:
                    synth_enc[cat_pres] = enc.transform(synth_enc[cat_pres].astype(str))
                X_syn = synth_enc.values.astype(float)
                if use_scaled:
                    X_syn = sca.transform(X_syn)
                X_aug = np.vstack([X_use_tr, X_syn])
                y_aug = np.concatenate([y_tr, np.ones(len(synth_df), dtype=int)])
                ctgan_pr = run_fn(X_aug, y_aug, X_use_va, y_va)
            else:
                ctgan_pr = np.nan

            delta = ctgan_pr - base_pr if not np.isnan(ctgan_pr) else np.nan
            row = {
                "classifier": clf_name, "fold": k,
                "n_synth": len(synth_df),
                "baseline_pr": round(base_pr, 6),
                "ctgan_pr": round(ctgan_pr, 6) if not np.isnan(ctgan_pr) else None,
                "delta": round(delta, 6) if not np.isnan(delta) else None,
            }
            fold_rows.append(row)
            print(f"    fold {k} [{clf_name}]: base={base_pr:.4f} "
                  f"ctgan={ctgan_pr:.4f} delta={delta:+.4f}")

        all_rows.extend(fold_rows)
        ck[f"B_fold_{k}"] = fold_rows
        ck[done_key] = list(done_folds | {k})
        done_folds.add(k)
        save_ck(ck)
        print(f"  fold {k} done ({time.time()-t0:.0f}s)")
        gc.collect()

    # Wilcoxon per classifier
    df_b = pd.DataFrame(all_rows)
    for clf_name in ["xgboost", "logreg"]:
        rows_c = df_b[df_b["classifier"] == clf_name].to_dict("records")
        p, mean_d, n_pos = summarise(rows_c, f"B {clf_name}")

    df_b.to_csv(os.path.join(OUT_DIR, "classifier_robustness.csv"), index=False)
    print("  Saved: results/revision/classifier_robustness.csv")
    return all_rows


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="AB",
                        help="Which experiments to run: A, B, or AB (default)")
    args = parser.parse_args()

    ck = load_ck()
    df_raw = load_data()

    if "A" in args.exp.upper():
        run_experiment_a(df_raw, ck)
    if "B" in args.exp.upper():
        run_experiment_b(df_raw, ck)

    print("\n=== DONE ===")
    print("Paste results/revision/fold_sensitivity.csv and "
          "results/revision/classifier_robustness.csv to Claude.")
