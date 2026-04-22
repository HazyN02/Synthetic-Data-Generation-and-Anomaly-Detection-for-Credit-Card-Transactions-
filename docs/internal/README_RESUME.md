# Resume: HDBSCAN fraud-cluster analysis

Snapshot taken **2026-04-18 02:14** while PID 3708 was still running fold 4 of 8.
Do not kill PID 3708 if it's still alive when you come back — it's appending
to the live CSV and will finish the run on its own. Everything below assumes
you're resuming **after** 3708 has either finished or died.

## State on disk

| File | Purpose |
|---|---|
| `results/protocol/run_20260330_180216/cluster_per_fold_hdbscan.csv` | Live CSV being appended by PID 3708 |
| `results/protocol/run_20260330_180216/checkpoints/cluster_per_fold_hdbscan.20260418_0214.csv` | Frozen snapshot (folds 0–3 done) |
| `results/protocol/run_20260330_180216/resume_state.json` | All resume metadata (pid, sha256, completed_folds, command) |
| `results/protocol/run_20260330_180216/hdbscan_partition.json` | HDBSCAN global partition (cluster sizes + hyperparams) |

Completed at snapshot time: folds **0, 1, 2, 3**.  Remaining: **4, 5, 6, 7**.

Global partition (HDBSCAN, `min_cluster_size=500`, seed=42, 4 clusters + noise):

| Cluster | Fraud rows | Share |
|---|---:|---:|
| -1 (noise, "low-density fraud") | 8,231 | 39.8% |
| 0 | 3,091 | 15.0% |
| 1 | 2,497 | 12.1% |
| 2 | 1,657 |  8.0% |
| 3 | 5,187 | 25.1% |

## Step 1 — Is PID 3708 still alive?

```powershell
Get-Process -Id 3708 -ErrorAction SilentlyContinue
```

- **Alive** → let it finish. Check progress:
  ```powershell
  Get-Content -Tail 20 cluster_hdbscan.log
  ```
- **Dead / missing** → proceed to Step 2.

## Step 2 — Verify checkpoint integrity

```
python -u src/resume_cluster_analysis.py --dry-run
```

Expected output:
```
[OK] checkpoint verified: results/protocol/run_20260330_180216/checkpoints/cluster_per_fold_hdbscan.20260418_0214.csv
[OK] sha256 = efa6993fae62809c67549680bf3600ab0343990b924180d6e7c319fb58f9cb63
[INFO] completed folds : [0, 1, 2, 3]   (or more, if 3708 made progress)
[INFO] remaining folds : [4, 5, 6, 7]
```

## Step 3 — Resume

```
python -u src/resume_cluster_analysis.py
```

This re-invokes the existing runner with the original args:

```
python -u -m src.run_cluster_analysis_hdbscan --n-folds 8 --ctgan-epochs 150 --hdbscan-min-cluster-size 500
```

The runner has built-in fold-level resume (`_completed_folds()` reads
`pr_auc_per_cluster` rows). Completed folds are skipped. HDBSCAN is
deterministic given seed+data+hyperparams so the global partition is
rebuilt identically.

Expected per-fold wall time **with the zombie KMeans gone**: 10–20 min.
Estimated remaining: ~1–1.5 h for folds 4–7.

If you'd rather not re-invoke the full runner (e.g. PID 3708 already
finished all 8 folds naturally), just build the summary:

```
python -u src/resume_cluster_analysis.py --summary-only
```

## Step 4 — Final output table

After resume completes, `cluster_summary_hdbscan.csv` will land at:

```
results/protocol/run_20260330_180216/cluster_summary_hdbscan.csv
```

Columns (exactly what was requested):

| Column | Meaning |
|---|---|
| `cluster_id` | HDBSCAN label; `-1` = noise / low-density fraud |
| `cluster_size` | # fraud rows in the cluster (global) |
| `fraud_density_pct` | cluster_size / 20,663 × 100 |
| `mean_baseline_pr_auc` | mean over 8 folds of LightGBM PR-AUC on cluster slice (baseline) |
| `mean_ctgan_pr_auc` | mean over 8 folds of LightGBM PR-AUC on cluster slice (CTGAN-augmented) |
| `mean_delta_pr_auc` | **headline: CTGAN − baseline** |
| `std_delta_pr_auc` | fold-level std |
| `mean_dcr_synth` | mean DCR of CTGAN synth routed to this cluster |
| `n_folds` | should be 8 after resume finishes |

Mechanistic claim to support in the paper: the noise bucket (-1, 40% of
fraud, low-density by construction) should show the most negative
`mean_delta_pr_auc` and the highest `mean_dcr_synth`.

## Re-taking the snapshot

If you want a fresher snapshot (e.g. 3708 finished more folds):

```
python -u src/_save_checkpoint.py
```

This overwrites `resume_state.json` + `hdbscan_partition.json` and drops
a new CSV copy under `checkpoints/`. Timestamp is hard-coded in the script
— edit the `STAMP` constant if you want a new filename instead of an
overwrite.
