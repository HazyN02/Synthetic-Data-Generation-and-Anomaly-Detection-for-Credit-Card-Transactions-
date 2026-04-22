# Paper: Fidelity or Illusion: Why Synthetic Oversampling Benchmarks for Fraud Detection Cannot Be Trusted

## Structure (canonical order)

- `00_ABSTRACT.md` — Abstract and title  
- `01_INTRODUCTION.md` — Introduction  
- `02_RELATED_WORK.md` — Related work  
- `03_METHOD.md` — Method (**locked** canonical §3; dataset, folds, oversampling, drift, metrics)  
- `04_EXPERIMENTS.md` — Experiments (**full** tables, figures, statistics; §4)  
- `05_ANALYSIS.md` — Analysis (interpretation, mechanisms; §5)  
- `06_DISCUSSION.md` — Discussion and limitations  
- `07_CONCLUSION.md` — Conclusion  

**Full single-file draft:** `FULL_PAPER_DRAFT.md` (sync section order with the files above when preparing camera-ready).

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

## Submission LaTeX

- `main.tex` is generated from `FULL_PAPER_DRAFT.md` and includes Table 1–4 and Figures from `paper/figures/` for a clean ACM-style LaTeX workflow.
