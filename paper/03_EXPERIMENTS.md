# 3. Experiments

See `EXPERIMENTS_SCOPE.md` for main vs supplementary experiment placement.

## 3.1 Main Results (Table)

See `paper/tables/unified_comparison.csv` and `paper/tables/method_summary.csv`.

## 3.2 Statistical Comparisons

See `paper/tables/statistical_comparisons.csv` for paired permutation tests (e.g., CTGAN vs baseline, TabDDPM vs SMOTE). Report p-values and note multiple comparisons.

## 3.3 Drift-Harm Analysis

See `paper/tables/drift_harm_analysis.csv` and `paper/figures/drift_vs_harm.pdf`.

## 3.4 When It Helps / Hurts

See `paper/tables/when_it_helps_hurts.csv`.

## 3.5 Sliding Window (Appendix)

Static vs sliding retraining: does training on recent data only change conclusions?  
See `paper/figures/sliding_window_comparison.pdf` and `results/sliding_window_results.csv`.

## 3.6 Label-Delay Ablation (Appendix)

PR-AUC vs label delay (0, 3, 7, 14 days). See `paper/tables/canonical_by_delay.csv` and `paper/figures/label_delay_ablation.pdf`.
