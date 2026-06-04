# src/run_ulb_utility.py
"""
ULB utility experiment: CTGAN, SMOTE, TVAE, LightGBM baseline
under a single chronological split (last 20% as test).
"""
import os, sys, csv, time
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"]     = "1"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"

import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

CSV_PATH = os.path.join(_ROOT, 'data', 'ulb', 'creditcard.csv')
OUT_DIR  = os.path.join(_ROOT, 'results', 'ulb_utility')
OUT_CSV  = os.path.join(OUT_DIR, 'ulb_utility_results.csv')
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load and split ─────────────────────────────────────────────────────────
print("Loading ULB dataset...")
df = pd.read_csv(CSV_PATH)
print(f"Total: {len(df):,} transactions, {int(df['Class'].sum())} fraud "
      f"({df['Class'].mean()*100:.3f}%)")

split_idx = int(len(df) * 0.80)
train = df.iloc[:split_idx].copy()
test  = df.iloc[split_idx:].copy()
print(f"Train: {len(train):,} rows, {int(train['Class'].sum())} fraud")
print(f"Test:  {len(test):,} rows,  {int(test['Class'].sum())} fraud")

feature_cols = [c for c in df.columns if c not in ['Class', 'Time']]
X_train = train[feature_cols].values.astype(float)
y_train = train['Class'].values.astype(int)
X_test  = test[feature_cols].values.astype(float)
y_test  = test['Class'].values.astype(int)

n_fraud_train = int(y_train.sum())
n_neg_train   = int((y_train == 0).sum())
# Target r=0.05: n_synth s.t. (n_fraud + n_synth) / (n_total + n_synth) = 0.05
n_target_fraud = int(n_neg_train * 0.05 / 0.95)
n_synth_target = max(0, n_target_fraud - n_fraud_train)
print(f"\nAugmentation target: +{n_synth_target} synthetic fraud (r=0.05)")

# ── 2. LightGBM config ───────────────────────────────────────────────────────
LGBM_PARAMS = dict(
    objective='binary', n_estimators=300, learning_rate=0.05,
    num_leaves=64, min_child_samples=20,   # relaxed for small ULB dataset
    subsample=0.8, colsample_bytree=0.8,
    n_jobs=1, random_state=42, verbosity=-1,
)

# ── 3. Baseline ──────────────────────────────────────────────────────────────
print("\n=== BASELINE ===")
clf = lgb.LGBMClassifier(**LGBM_PARAMS)
clf.fit(X_train, y_train)
baseline_pr = float(average_precision_score(y_test, clf.predict_proba(X_test)[:, 1]))
print(f"Baseline PR-AUC: {baseline_pr:.4f}")

results = [{'method': 'baseline', 'pr_auc': round(baseline_pr, 6),
            'delta': 0.0, 'n_synth': 0}]

def evaluate(X_aug, y_aug, label, n_synth):
    clf2 = lgb.LGBMClassifier(**LGBM_PARAMS)
    clf2.fit(X_aug, y_aug)
    pr    = float(average_precision_score(y_test, clf2.predict_proba(X_test)[:, 1]))
    delta = pr - baseline_pr
    print(f"{label}: PR-AUC={pr:.4f}  delta={delta:+.4f}  n_synth={n_synth:,}")
    results.append({'method': label, 'pr_auc': round(pr, 6),
                    'delta': round(delta, 6), 'n_synth': n_synth})

# ── 4. SMOTE ─────────────────────────────────────────────────────────────────
print("\n=== SMOTE ===")
k = min(5, n_fraud_train - 1)
smote = SMOTE(
    sampling_strategy={1: n_fraud_train + n_synth_target},
    k_neighbors=k, random_state=42
)
X_sm, y_sm = smote.fit_resample(X_train, y_train)
evaluate(X_sm, y_sm, 'smote', n_synth_target)

# ── 5. CTGAN (150 epochs) — use project's synth_ctgan.py (pac=1, handles small n)
print("\n=== CTGAN (150 epochs) ===")
sys.path.insert(0, _ROOT)
from src.synth_ctgan import fit_ctgan, sample_ctgan

fraud_train_df = train[train['Class'] == 1][feature_cols].copy()
fraud_train_df['isFraud'] = 1  # synth_ctgan expects TARGET_COL='isFraud'

# ULB has no categorical cols — all V1-V28 + Amount are continuous
cat_cols_ulb = []
used_cols_ulb = feature_cols + ['isFraud']

t0 = time.time()
ctgan_model, ctgan_artifacts = fit_ctgan(
    train_df=fraud_train_df,
    cat_cols=cat_cols_ulb,
    used_cols=used_cols_ulb,
    epochs=150, batch_size=500, pac=1,
    discriminator_steps=5, seed=0, verbose=True,
)
print(f"CTGAN training: {(time.time()-t0)/60:.1f} min")

synth_c = sample_ctgan(ctgan_model, n=n_synth_target, artifacts=ctgan_artifacts)
X_aug_c = np.vstack([X_train, synth_c[feature_cols].values.astype(float)])
y_aug_c = np.concatenate([y_train, np.ones(n_synth_target, dtype=int)])
evaluate(X_aug_c, y_aug_c, 'ctgan_150ep', n_synth_target)

# ── 6. TVAE (300 epochs) — use project's synth_tvae.py ───────────────────────
print("\n=== TVAE (300 epochs) ===")
from src.synth_tvae import fit_tvae, sample_tvae

t0 = time.time()
tvae_model, tvae_artifacts = fit_tvae(
    fraud_train_df,
    cat_cols=cat_cols_ulb,
    used_cols=used_cols_ulb,
    epochs=300, batch_size=500, seed=0, verbose=True,
)
print(f"TVAE training: {(time.time()-t0)/60:.1f} min")

synth_t = sample_tvae(tvae_model, n=n_synth_target, artifacts=tvae_artifacts)
X_aug_t = np.vstack([X_train, synth_t[feature_cols].values.astype(float)])
y_aug_t = np.concatenate([y_train, np.ones(n_synth_target, dtype=int)])
evaluate(X_aug_t, y_aug_t, 'tvae_300ep', n_synth_target)

# ── 7. Summary ───────────────────────────────────────────────────────────────
print("\n=== SUMMARY ===")
print(f"{'Method':<20} {'PR-AUC':>8} {'Delta':>8} {'N_synth':>8}")
print("-" * 50)
for r in results:
    print(f"{r['method']:<20} {r['pr_auc']:>8.4f} "
          f"{float(r['delta']):>+8.4f} {r['n_synth']:>8,}")

# ── 8. Save ──────────────────────────────────────────────────────────────────
with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['method', 'pr_auc', 'delta', 'n_synth'])
    w.writeheader()
    w.writerows(results)
print(f"\nSaved: {OUT_CSV}")
