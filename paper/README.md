# Paper: When Synthetic Oversampling Helps or Hurts Rare-Event Fraud Detection

## Structure

- `00_ABSTRACT.md` - Abstract and title
- `01_INTRODUCTION.md` - Introduction
- `02_METHOD.md` - Method
- `03_EXPERIMENTS.md` - Experiments (tables/figures referenced)
- `04_ANALYSIS.md` - Analysis
- `05_DISCUSSION.md` - Discussion

## Tables (generated)

- `tables/unified_comparison.csv` - All methods, all folds
- `tables/drift_harm_analysis.csv` - Drift metrics + harm per method
- `tables/when_it_helps_hurts.csv` - Summary verdict per method
- `tables/method_summary.csv` - Mean ± std by method

## Figures (generated)

- `figures/method_comparison_pr_auc.pdf` - Bar chart: PR-AUC by method and fold
- `figures/drift_vs_harm.pdf` - Scatter: domain AUC vs harm
- `figures/sliding_window_comparison.pdf` - Static vs sliding

## How to regenerate

```bash
# 1. Run unified analysis (merges results, drift-harm, when-it-helps)
python -m src.run_unified_analysis

# 2. Generate figures
python -m src.paper_figures
```

Requires: full protocol results in `results/results.csv`, SMOTE in `results/smote_baseline_results.csv`, drift in `experiments/results/drift_report.csv`.
