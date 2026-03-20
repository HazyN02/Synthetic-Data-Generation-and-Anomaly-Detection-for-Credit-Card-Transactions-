# Frozen results (paper evidence)

**Frozen:** 2026-02-24

This directory is the **canonical snapshot** of protocol results for the paper. Do not edit `frozen_results.csv` after freezing; any new runs (e.g. label delay) should be added as separate files or a new freeze.

## Contents

| File | Description |
|------|-------------|
| `frozen_results.csv` | Main results: 4 temporal folds, baseline + SMOTE + CTGAN + TabDDPM + recency ablations (_recency03). No label delay. |
| `frozen_config.json` | Protocol config for this run (run_id, mode, n_folds, target_pos_rates, delay_days=0). |
| `README.md` | This file. |

## Source run

- **Run ID:** `20260224_101041`
- **Command:** `python3 -m src.run_protocol --medium --recency-ablation`
- **Original path:** `results/protocol/run_20260224_101041/results.csv`
- **delay_days:** 0 (no label delay)

## Methods in frozen_results.csv

- **baseline** — LightGBM on real data only
- **smote** — SMOTE to target_pos_rate 0.05, 0.1, 0.2
- **ctgan** / **tabddpm** — generative oversampling to same target rates
- **\*_recency03** — recency ablation (train oversampler on latest 30% of positives only)

Metrics: `pr_auc`, `recall_at_1pct_fpr`. Columns: `fold`, `method`, `target_pos_rate`, `train_rows`, `val_rows`, `train_pos`, `synth_rows`, `final_train_rows`, `final_pos_rate`, `notes`.

## Drift report (domain AUC)

To merge domain AUC for drift-harm analysis, run:

```bash
.venv/bin/python -m src.run_drift_report --protocol-folds --n-folds 4
```

Output: `experiments/results/drift_report.csv`. Then run `run_unified_analysis` to merge.

## Label-delay runs completed

**7-day (medium, 4 folds) and 14-day (quick, 2 folds) label-delay runs completed successfully.**

Results are in `results/protocol/results.csv` with `delay_days` and `run_id` columns. Use `load_protocol_results(delay_days=7)` or `delay_days=14` in `run_unified_analysis` for delay-specific analysis.
