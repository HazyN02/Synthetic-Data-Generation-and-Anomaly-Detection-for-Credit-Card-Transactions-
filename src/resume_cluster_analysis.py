"""
Resume the HDBSCAN fraud-cluster analysis from a prior partial run.

Does three things:
  1. Verifies the on-disk checkpoint (resume_state.json + live CSV).
  2. Reports which folds are done and which remain.
  3. Re-launches src.run_cluster_analysis_hdbscan with the same args.
     The runner itself has fold-level resume via _completed_folds() — folds
     already present in cluster_per_fold_hdbscan.csv as pr_auc_per_cluster
     rows are skipped. So this script is a safe idempotent re-invocation.

After the 8 folds finish, also produces the final per-cluster summary table:
    results/protocol/run_<RUN_ID>/cluster_summary_hdbscan.csv
with columns:
    cluster_id, cluster_size, fraud_density_pct, mean_baseline_pr_auc,
    mean_ctgan_pr_auc, mean_delta_pr_auc, std_delta_pr_auc, mean_dcr_synth

Usage:
    python -u src/resume_cluster_analysis.py
    python -u src/resume_cluster_analysis.py --dry-run    # verify only, no relaunch
    python -u src/resume_cluster_analysis.py --summary-only  # recompute summary from whatever is already on disk
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

RUN_DIR = os.path.join('results', 'protocol', 'run_20260330_180216')
LIVE_CSV = os.path.join(RUN_DIR, 'cluster_per_fold_hdbscan.csv')
STATE_JSON = os.path.join(RUN_DIR, 'resume_state.json')
PARTITION_JSON = os.path.join(RUN_DIR, 'hdbscan_partition.json')
SUMMARY_CSV = os.path.join(RUN_DIR, 'cluster_summary_hdbscan.csv')


def verify_checkpoint() -> dict:
    if not os.path.exists(STATE_JSON):
        print(f"[ERROR] missing {STATE_JSON}", file=sys.stderr)
        sys.exit(2)
    with open(STATE_JSON) as f:
        state = json.load(f)
    ckpt = state['checkpoint_csv']
    if not os.path.exists(ckpt):
        print(f"[ERROR] checkpoint missing: {ckpt}", file=sys.stderr)
        sys.exit(2)
    h = hashlib.sha256()
    with open(ckpt, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    if h.hexdigest() != state['checkpoint_csv_sha256']:
        print(f"[ERROR] checkpoint sha256 mismatch on {ckpt}", file=sys.stderr)
        sys.exit(2)
    print(f"[OK] checkpoint verified: {ckpt}")
    print(f"[OK] sha256 = {state['checkpoint_csv_sha256']}")
    return state


def report_progress(state: dict) -> tuple[list[int], list[int]]:
    if not os.path.exists(LIVE_CSV):
        print(f"[WARN] live csv {LIVE_CSV} not found; falling back to checkpoint.")
        src = state['checkpoint_csv']
    else:
        src = LIVE_CSV
    df = pd.read_csv(src)
    pr = df[df['record_type'] == 'pr_auc_per_cluster']
    done = sorted(int(x) for x in pr['fold'].dropna().unique())
    remaining = [f for f in range(state.get('total_folds', 8)) if f not in done]
    print(f"[INFO] csv           : {src}")
    print(f"[INFO] completed folds : {done}")
    print(f"[INFO] remaining folds : {remaining}")
    return done, remaining


def relaunch():
    cmd = [
        sys.executable, '-u', '-m', 'src.run_cluster_analysis_hdbscan',
        '--n-folds', '8',
        '--ctgan-epochs', '150',
        '--hdbscan-min-cluster-size', '500',
    ]
    print('[RUN]', ' '.join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[ERROR] runner exited {rc}", file=sys.stderr)
        sys.exit(rc)


def build_summary():
    if not os.path.exists(LIVE_CSV):
        print(f"[ERROR] {LIVE_CSV} not found", file=sys.stderr)
        sys.exit(2)
    df = pd.read_csv(LIVE_CSV)
    fa = df[df['record_type'] == 'fraud_cluster_assignment']
    pr = df[df['record_type'] == 'pr_auc_per_cluster']
    dcr = df[df['record_type'] == 'dcr_ctgan_synthetic']

    cluster_sizes = fa['cluster_id'].value_counts().sort_index()
    total_fraud = int(cluster_sizes.sum())

    # wide-ify pr_auc: one row per (fold, cluster, method)
    pr_wide = (
        pr.pivot_table(index=['fold', 'cluster_id'], columns='method', values='pr_auc', aggfunc='first')
          .reset_index()
    )
    pr_wide['delta'] = pr_wide['ctgan'] - pr_wide['baseline']

    agg = pr_wide.groupby('cluster_id').agg(
        mean_baseline_pr_auc=('baseline', 'mean'),
        mean_ctgan_pr_auc=('ctgan', 'mean'),
        mean_delta_pr_auc=('delta', 'mean'),
        std_delta_pr_auc=('delta', 'std'),
        n_folds=('fold', 'nunique'),
    ).reset_index()

    dcr_agg = dcr.groupby('cluster_id').agg(mean_dcr_synth=('dcr_mean', 'mean')).reset_index()

    out = agg.merge(dcr_agg, on='cluster_id', how='left')
    out['cluster_size'] = out['cluster_id'].map(cluster_sizes).astype(int)
    out['fraud_density_pct'] = 100.0 * out['cluster_size'] / total_fraud
    out = out[['cluster_id', 'cluster_size', 'fraud_density_pct',
               'mean_baseline_pr_auc', 'mean_ctgan_pr_auc',
               'mean_delta_pr_auc', 'std_delta_pr_auc',
               'mean_dcr_synth', 'n_folds']].sort_values('cluster_id')
    out.to_csv(SUMMARY_CSV, index=False)
    print(f"[OK] wrote {SUMMARY_CSV}")
    print(out.to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true', help='verify + report only; no relaunch')
    p.add_argument('--summary-only', action='store_true', help='skip relaunch; just build summary from whatever is on disk')
    args = p.parse_args()

    state = verify_checkpoint()
    done, remaining = report_progress(state)

    if args.summary_only:
        build_summary()
        return
    if args.dry_run:
        return
    if not remaining:
        print('[INFO] all folds already done; building summary.')
        build_summary()
        return

    relaunch()
    build_summary()


if __name__ == '__main__':
    main()
