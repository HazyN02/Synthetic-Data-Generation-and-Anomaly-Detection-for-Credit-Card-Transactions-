"""One-shot: snapshot the live HDBSCAN run's on-disk state. Does NOT touch PID 3708."""
import pandas as pd, json, shutil, os, hashlib, datetime

SRC = r'results/protocol/run_20260330_180216/cluster_per_fold_hdbscan.csv'
CKPT_DIR = r'results/protocol/run_20260330_180216/checkpoints'
STAMP = '20260418_0214'

os.makedirs(CKPT_DIR, exist_ok=True)
dst = os.path.join(CKPT_DIR, f'cluster_per_fold_hdbscan.{STAMP}.csv')
shutil.copy2(SRC, dst)

h = hashlib.sha256()
with open(dst, 'rb') as f:
    for blk in iter(lambda: f.read(1 << 20), b''):
        h.update(blk)
digest = h.hexdigest()

df = pd.read_csv(dst)
fa = df[df['record_type'] == 'fraud_cluster_assignment']
pr = df[df['record_type'] == 'pr_auc_per_cluster']
dcr = df[df['record_type'] == 'dcr_ctgan_synthetic']

partition = {
    'algo': 'hdbscan',
    'hyperparams': {'min_cluster_size': 500, 'min_samples': None, 'seed': 42},
    'n_fraud_rows': int(len(fa)),
    'cluster_sizes': {str(k): int(v) for k, v in fa['cluster_id'].value_counts().sort_index().items()},
    'cluster_ids_including_noise': sorted(fa['cluster_id'].unique().tolist()),
    'notes': 'Full per-row cluster assignments are preserved in the checkpoint CSV as record_type=fraud_cluster_assignment (cols: global_row_idx, cluster_id).',
}
with open(r'results/protocol/run_20260330_180216/hdbscan_partition.json', 'w') as f:
    json.dump(partition, f, indent=2)

completed_folds = sorted(int(x) for x in pr['fold'].dropna().unique())
resume_state = {
    'saved_at': datetime.datetime.now().isoformat(timespec='seconds'),
    'live_process': {
        'pid': 3708,
        'started': '2026-04-17T19:28:11',
        'status_when_saved': 'ALIVE - running fold 4 (do not kill)',
        'cpu_seconds_at_snapshot': 7629,
        'rss_bytes_at_snapshot': 3690319872,
    },
    'checkpoint_csv': dst.replace(os.sep, '/'),
    'checkpoint_csv_sha256': digest,
    'live_csv_being_appended': SRC.replace(os.sep, '/'),
    'completed_folds': completed_folds,
    'total_folds': 8,
    'remaining_folds': [f for f in range(8) if f not in completed_folds],
    'partition_file': 'results/protocol/run_20260330_180216/hdbscan_partition.json',
    'cluster_sizes': partition['cluster_sizes'],
    'command_originally_launched': 'python -u -m src.run_cluster_analysis_hdbscan --n-folds 8 --ctgan-epochs 150 --hdbscan-min-cluster-size 500',
    'resume_command': 'python -u -m src.run_cluster_analysis_hdbscan --n-folds 8 --ctgan-epochs 150 --hdbscan-min-cluster-size 500',
    'per_fold_timings_observed_sec': {'0': 456.4, '1': 996.3, '2': 8340.1, '3': 12000.7},
    'notes': [
        'The live runner has fold-level resume built in via _completed_folds() which reads pr_auc_per_cluster rows.',
        'Completed folds 0-3 will be skipped if the resume command is re-run.',
        'HDBSCAN is deterministic given seed+data+hyperparams; the global partition will be rebuilt identically.',
        'Folds 2 and 3 were slow due to a zombie KMeans process (already killed); remaining folds should run at fold-0/1 pace (~10-20 min each).',
        'If PID 3708 finishes naturally overnight, remaining_folds will shrink. Always re-run _save_checkpoint.py before resuming to get a fresh snapshot.',
    ],
}
with open(r'results/protocol/run_20260330_180216/resume_state.json', 'w') as f:
    json.dump(resume_state, f, indent=2)

print('CHECKPOINT SAVED')
print('  dst      =', dst)
print('  sha256   =', digest)
print('  rows     =', len(df))
print('  fraud_assignments =', len(fa))
print('  pr_auc_rows       =', len(pr))
print('  dcr_rows          =', len(dcr))
print('  completed_folds   =', completed_folds)
print('  partition         =', partition['cluster_sizes'])
