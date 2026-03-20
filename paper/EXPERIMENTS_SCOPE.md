# Experiments Scope: Main vs Supplementary

This document clarifies which experiments belong in the **main text** vs **appendix/supplementary**.

## Main Text

| Experiment | Table/Figure | Description |
|------------|--------------|-------------|
| Main protocol (baseline vs CTGAN vs TabDDPM vs SMOTE) | `unified_comparison.csv`, `method_comparison_pr_auc.pdf` | Core comparison with temporal folds. |
| Method summary (mean ± std) | `method_summary.csv` | Aggregate PR-AUC and Recall@1%FPR. |
| Statistical comparisons | `statistical_comparisons.csv` | Paired permutation tests (CTGAN vs baseline, TabDDPM vs baseline, etc.). |
| Drift–harm analysis | `drift_harm_analysis.csv`, `drift_vs_harm.pdf` | Domain AUC vs synthetic oversampling harm. |
| When it helps/hurts | `when_it_helps_hurts.csv` | Verdict per method (helps / neutral / hurts). |

## Appendix / Supplementary

| Experiment | Table/Figure | Description |
|------------|--------------|-------------|
| Label-delay ablation (0, 3, 7, 14 days) | `canonical_by_delay.csv`, `label_delay_ablation.pdf` | PR-AUC vs label delay. |
| Recency ablation (recency_frac=0.3) | Included in protocol runs | ctgan_recency03, tabddpm_recency03, smote_recency03. |
| Sliding window (static vs recent) | `sliding_window_comparison.pdf`, `sliding_window_results.csv` | Does retraining on recent data only change conclusions? |
| Sensitivity (extra target_pos_rate / recency_frac) | From `run_sensitivity_ablation` | Robustness to hyperparameters. |
| Max-convergence ablation | From protocol `--max-convergence` | CTGAN/TabDDPM at 50 epochs. |

## Run Mapping

Canonical tables are produced from:

- **Main results**: `results/protocol/results.csv` (no delay, no recency) or `results/protocol/FROZEN/frozen_results.csv`.
- **Run mapping**: `paper/tables/canonical_run_mapping.csv` documents which `run_id`, `delay_days`, `recency_ablation` correspond to each table.

## Commands

```bash
# Full pipeline
python -m src.run_canonical_analysis   # Consolidate all runs, run mapping
python -m src.run_unified_analysis    # Unified tables, drift-harm, stat tests
python -m src.paper_figures           # Figures

# Ablations (run separately)
python -m src.run_protocol --medium --delay-days 7   # Label delay
python -m src.run_protocol --medium --recency-ablation
python -m src.run_sliding_window
python -m src.run_sensitivity_ablation
python -m src.run_label_delay_ablation
```
