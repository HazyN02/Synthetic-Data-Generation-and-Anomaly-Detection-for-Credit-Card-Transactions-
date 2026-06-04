# src/run_revision_cd.py
"""
Revision Experiments C and D — run from existing data, no retraining.

C: Statistical hardening (bootstrap CI, permutation p, MDE, Cohen's d)
D: Per-fold per-cluster Spearman rank correlation (DCR vs delta)

Both complete in <30 seconds from existing CSV files.
"""
import os, sys, json
import numpy as np
import pandas as pd
import scipy.stats as stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

RD         = os.path.join(_ROOT, 'results', 'protocol', 'run_20260330_180216')
OUT_DIR    = os.path.join(_ROOT, 'results', 'revision')
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Statistical hardening
# ═══════════════════════════════════════════════════════════════════════════════
print("--- EXPERIMENT C: Statistical hardening ---")

# Canonical 8-fold CTGAN deltas (from significance_tests_8fold_verified.csv)
ctgan_deltas = np.array([0.0105, 0.0080, 0.0096, 0.0066,
                         -0.0033, 0.0034, -0.0062, 0.0006])
n = len(ctgan_deltas)
observed_mean = float(np.mean(ctgan_deltas))
delta_std     = float(np.std(ctgan_deltas, ddof=1))

# Bootstrap 95% CI on mean delta
N_BOOT = 50000
boot_means = np.array([
    np.mean(np.random.choice(ctgan_deltas, size=n, replace=True))
    for _ in range(N_BOOT)
])
ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
print(f"  Bootstrap 95% CI on mean delta: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

# Permutation test (sign-flip, one-sided greater)
N_PERM = 50000
perm_means = np.array([
    np.mean(np.abs(ctgan_deltas) * np.random.choice([-1, 1], size=n))
    for _ in range(N_PERM)
])
perm_p = float(np.mean(perm_means >= observed_mean))
perm_p = max(perm_p, 1.0 / (N_PERM + 1))  # floor
print(f"  Permutation p (one-sided, 50k perms): {perm_p:.4f}")

# Wilcoxon (canonical)
_, wilcoxon_p = stats.wilcoxon(ctgan_deltas, alternative='greater')
print(f"  Wilcoxon p (one-sided): {wilcoxon_p:.6f}")

# MDE at n=8, alpha=0.05, power=0.80 (normal approx)
z_alpha = stats.norm.ppf(0.95)
z_beta  = stats.norm.ppf(0.80)
mde_80  = (z_alpha + z_beta) * delta_std / np.sqrt(n)
mde_90  = (z_alpha + stats.norm.ppf(0.90)) * delta_std / np.sqrt(n)
print(f"  MDE at n=8, 80% power: {mde_80:.4f}  (90% power: {mde_90:.4f})")
print(f"  Observed mean {observed_mean:+.4f} vs MDE {mde_80:.4f} "
      f"-> {'ABOVE' if observed_mean >= mde_80 else 'BELOW'}")

# Cohen's d (paired)
cohen_d = observed_mean / delta_std
print(f"  Cohen's d: {cohen_d:.3f}")

# Power at observed effect size (retrospective)
from scipy.stats import norm
power_obs = float(1 - norm.cdf(z_alpha - cohen_d * np.sqrt(n)))
print(f"  Retrospective power at observed d={cohen_d:.3f}, n=8: {power_obs:.3f}")

# Folds needed for 80% power at observed effect size
n_needed_80 = int(np.ceil(((z_alpha + z_beta) / cohen_d) ** 2))
n_needed_95 = int(np.ceil(((z_alpha + stats.norm.ppf(0.95)) / cohen_d) ** 2))
print(f"  Folds needed for 80% power: {n_needed_80}, 95% power: {n_needed_95}")

hardening = {
    'n_folds': n,
    'observed_mean_delta': round(observed_mean, 6),
    'delta_std': round(delta_std, 6),
    'bootstrap_ci_lower': round(float(ci_lo), 6),
    'bootstrap_ci_upper': round(float(ci_hi), 6),
    'wilcoxon_p_onesided': round(float(wilcoxon_p), 6),
    'permutation_p_onesided': round(perm_p, 6),
    'mde_80pct_power': round(mde_80, 6),
    'mde_90pct_power': round(mde_90, 6),
    'cohens_d': round(cohen_d, 4),
    'retrospective_power': round(power_obs, 4),
    'n_folds_needed_80pct_power': n_needed_80,
    'n_folds_needed_95pct_power': n_needed_95,
}
pd.DataFrame([hardening]).to_csv(os.path.join(OUT_DIR, 'bootstrap_ci.csv'), index=False)
print(f"  Saved: results/revision/bootstrap_ci.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# D: Per-fold per-cluster Spearman rank correlation (DCR vs delta)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- EXPERIMENT D: Per-fold cluster rank correlation ---")

cdf = pd.read_csv(os.path.join(RD, 'cluster_per_fold_hdbscan.csv'))

# Per-cluster DCR (mean over folds, clusters 1-3 only — those with allocation)
dcr_rows   = cdf[cdf['record_type'] == 'dcr_ctgan_synthetic'].copy()
dcr_rows['dcr_mean']     = pd.to_numeric(dcr_rows['dcr_mean'], errors='coerce')
dcr_rows['n_synthetic']  = pd.to_numeric(dcr_rows['n_synthetic'], errors='coerce')
dcr_rows['fold']         = pd.to_numeric(dcr_rows['fold'], errors='coerce')

# Per-cluster delta from pr_auc_per_cluster rows
pr_rows = cdf[cdf['record_type'] == 'pr_auc_per_cluster'].copy()
pr_rows['pr_auc'] = pd.to_numeric(pr_rows['pr_auc'], errors='coerce')
pr_rows['fold']   = pd.to_numeric(pr_rows['fold'], errors='coerce')

rank_results = []
allocated_clusters = [1, 2, 3]   # clusters that receive CTGAN synthetics

print(f"  {'Fold':>5}  {'Rho':>6}  {'p':>7}  DCRs                 Deltas")
print("  " + "-" * 65)

for fold in sorted(dcr_rows['fold'].dropna().unique()):
    fold = int(fold)
    dcr_fold = dcr_rows[(dcr_rows['fold'] == fold) &
                        (dcr_rows['cluster_id'].isin(allocated_clusters))]
    pr_base  = pr_rows[(pr_rows['fold'] == fold) &
                       (pr_rows['method'] == 'baseline') &
                       (pr_rows['cluster_id'].isin(allocated_clusters))]
    pr_ctgan = pr_rows[(pr_rows['fold'] == fold) &
                       (pr_rows['method'] == 'ctgan') &
                       (pr_rows['cluster_id'].isin(allocated_clusters))]

    # Align on cluster_id
    clusters_with_data = (
        set(dcr_fold['cluster_id'].dropna().astype(int)) &
        set(pr_base['cluster_id'].dropna().astype(int)) &
        set(pr_ctgan['cluster_id'].dropna().astype(int))
    )
    if len(clusters_with_data) < 3:
        print(f"  {fold:>5}  insufficient data ({len(clusters_with_data)} clusters)")
        continue

    dcr_vals, delta_vals = [], []
    for c in sorted(clusters_with_data):
        dcr_val = dcr_fold[dcr_fold['cluster_id'] == c]['dcr_mean'].values
        b_pr    = pr_base[pr_base['cluster_id'] == c]['pr_auc'].values
        c_pr    = pr_ctgan[pr_ctgan['cluster_id'] == c]['pr_auc'].values
        if len(dcr_val) and len(b_pr) and len(c_pr):
            dcr_vals.append(float(dcr_val[0]))
            delta_vals.append(float(c_pr[0]) - float(b_pr[0]))

    if len(dcr_vals) >= 3:
        rho, p_val = stats.spearmanr(dcr_vals, delta_vals)
        dcr_str   = ' '.join(f'{d:>7,.0f}' for d in dcr_vals)
        delta_str = ' '.join(f'{d:>+.4f}' for d in delta_vals)
        print(f"  {fold:>5}  {rho:>+.3f}  {p_val:>7.3f}  [{dcr_str}]  [{delta_str}]")
        rank_results.append({
            'fold': fold, 'spearman_rho': round(rho, 4),
            'p_value': round(p_val, 4),
            'n_clusters': len(dcr_vals),
            'dcr_vals': str([round(d, 0) for d in dcr_vals]),
            'delta_vals': str([round(d, 4) for d in delta_vals]),
        })

if rank_results:
    df_rank = pd.DataFrame(rank_results)
    mean_rho  = df_rank['spearman_rho'].mean()
    n_neg_rho = int((df_rank['spearman_rho'] < 0).sum())
    print(f"\n  Mean Spearman rho: {mean_rho:+.3f}")
    print(f"  Folds with negative rho: {n_neg_rho}/{len(rank_results)}")
    print(f"  Interpretation: {'higher DCR -> worse delta (expected sign)' if mean_rho < 0 else 'higher DCR -> better delta (unexpected)'}")

    # Meta-analysis: Fisher Z transform
    z_vals = np.arctanh(np.clip(df_rank['spearman_rho'].values, -0.999, 0.999))
    z_mean = z_vals.mean()
    z_se   = 1.0 / np.sqrt(len(z_vals) - 3) if len(z_vals) > 3 else np.nan
    meta_rho = float(np.tanh(z_mean))
    meta_p   = float(2 * (1 - stats.norm.cdf(abs(z_mean / z_se)))) if not np.isnan(z_se) else np.nan
    print(f"  Meta-analytic rho (Fisher Z): {meta_rho:+.3f}, p={meta_p:.3f}")

    df_rank['meta_rho'] = round(meta_rho, 4)
    df_rank.to_csv(os.path.join(OUT_DIR, 'rank_correlation.csv'), index=False)
    print(f"  Saved: results/revision/rank_correlation.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- PAPER-READY OUTPUT ---")
print(f"C — Bootstrap 95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
print(f"C — Permutation p: {perm_p:.4f}  |  Wilcoxon p: {wilcoxon_p:.4f}")
print(f"C — MDE (80% power, n=8): {mde_80:.4f}  |  Observed: {observed_mean:+.4f} ({'above' if observed_mean >= mde_80 else 'BELOW'} MDE)")
print(f"C — Cohen's d: {cohen_d:.3f}  |  Retrospective power: {power_obs:.1%}")
print(f"C — Folds for 80% power: {n_needed_80}")
if rank_results:
    print(f"D — Mean Spearman rho (DCR vs delta, 3-cluster, per fold): {mean_rho:+.3f}")
    print(f"D — Meta-analytic rho: {meta_rho:+.3f}, p={meta_p:.3f}")
    print(f"D — {n_neg_rho}/{len(rank_results)} folds show negative rho (higher DCR -> lower delta)")
