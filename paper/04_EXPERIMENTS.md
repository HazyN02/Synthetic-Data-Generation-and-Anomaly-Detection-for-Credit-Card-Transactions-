# 4. Experiments

> **Scope.** Main text focuses on **no label delay** (\(\delta=0\)) unless noted. **Label-delay** and **sliding-window** analyses follow the same pipeline as §3; placement details are in `EXPERIMENTS_SCOPE.md` if you need appendix vs main text.

All numbers below come from **`paper/tables/`** (CSV) and match **`python -m src.run_unified_analysis`** on the saved protocol runs.

## 4.1 Setup (summary)

**Protocol** is **§3** (folds, preprocessing, oversampling, metrics). **Main** tables pick the **best** **target fraud rate** in \(\{5\%,10\%,20\%\}\) **per fold and method** by validation **PR‑AUC**; **recency** uses \(\rho=0.3\). **Tests:** **10,000**‑permutation **paired** comparisons on fold‑level **PR‑AUC** (`src/statistical_tests.py`). **Selection‑on‑validation** caveats: **§5.6**.

## 4.2 Main results: mean PR‑AUC and Recall@1% FPR

**Table 1. Mean PR‑AUC and Recall@1% FPR across four temporal folds (no label delay).**  
Table reports **mean ± std** over folds. **PR‑AUC** is the **primary** metric; **Recall@1% FPR** is the **fixed**‑FPR operating point from §3.5.

| Method | PR‑AUC (mean ± std) | Recall@1% FPR (mean ± std) |
|--------|---------------------|----------------------------|
| Baseline | 0.544 ± 0.025 | 0.451 ± 0.023 |
| CTGAN | 0.570 ± 0.027 | 0.478 ± 0.022 |
| TabDDPM | 0.565 ± 0.029 | 0.472 ± 0.026 |
| SMOTE | 0.567 ± 0.024 | 0.476 ± 0.017 |
| CTGAN (recency 0.3) | 0.569 ± 0.025 | 0.478 ± 0.018 |
| TabDDPM (recency 0.3) | 0.564 ± 0.026 | 0.471 ± 0.023 |
| SMOTE (recency 0.3) | 0.552 ± 0.031 | 0.470 ± 0.026 |

**Figure.** `figures/method_comparison_pr_auc.pdf` (and `.png`) bar‑charts **PR‑AUC** by **method** and **fold** for quick visual comparison.

*Source: `paper/tables/method_summary.csv`, `unified_comparison.csv`.*

## 4.3 Fold-by-fold PR‑AUC and domain shift

**Table 2. Fold-by-fold PR‑AUC and domain shift.**  
Table lists **PR‑AUC** per **fold** for **core** methods (best **target rate** per cell). **Domain AUC** (train vs. validation, **no** raw time in features) quantifies **shift** for that fold.

| Fold | Domain AUC | Baseline | CTGAN | TabDDPM | SMOTE |
|------|------------|----------|-------|---------|-------|
| 0 | 0.824 | 0.547 | 0.560 | 0.554 | 0.553 |
| 1 | 0.885 | 0.578 | 0.585 | 0.562 | 0.576 |
| 2 | 0.919 | 0.601 | 0.600 | 0.605 | 0.596 |
| 3 | 0.885 | 0.534 | 0.542 | 0.537 | 0.541 |

*Interpretation of fold‑level variation:* **§5.2**.

*Source: `unified_comparison.csv`, `experiments/results/drift_report.csv` (domain AUC column `domain_auc_holdout_no_time`).*

## 4.4 Statistical comparisons

**Table 3. Paired permutation tests on fold-level PR‑AUC differences.**  
Table summarizes **paired** tests on **fold‑level** mean **PR‑AUC** differences. **Mean Δ** is **method A − method B** (convention from `statistical_comparisons.csv`).

| Comparison | Mean Δ (PR‑AUC) | *p* | Significant at α = 0.05? |
|------------|-----------------|-----|---------------------------|
| SMOTE − baseline | +0.0229 | 0.12 | No |
| CTGAN − baseline | +0.0267 | 0.12 | No |
| TabDDPM − baseline | +0.0210 | 0.12 | No |
| CTGAN − SMOTE | +0.0038 | 0.12 | No |
| TabDDPM − SMOTE | −0.0019 | 0.75 | No |
| TabDDPM − CTGAN | −0.0057 | 0.62 | No |

*Source: `paper/tables/statistical_comparisons.csv`.*

**Power note.** These tests are performed on fold-level differences with **n = 4** points; permutation-test p-values therefore have limited power for small effects. We interpret the results as “no strong evidence of improvement,” and primarily rely on effect-size magnitude and fold variability rather than p-values.

## 4.5 Drift and oversampling effect relative to the baseline

**Pearson** **r** (**domain AUC** vs. method **PR‑AUC − baseline PR‑AUC**, **n = 4** fold-level points): **CTGAN** **-0.89**, **TabDDPM** **-0.80**, **SMOTE** **-0.86** (`drift_correlations.csv`; **exploratory**). We treat this as **consistent with** drift-dependent behavior rather than evidence of a robust correlation.

**Figure.** `figures/drift_vs_harm.pdf` (y-axis is **baseline − method**, so **negative** values mean oversampling **helps**).

*Source: `paper/tables/drift_harm_analysis.csv`.*

## 4.6 When oversampling helps or hurts (verdict)

Aggregating **mean** **(baseline PR‑AUC − method PR‑AUC)** across folds (see `when_it_helps_hurts.csv`): **CTGAN**, **TabDDPM**, **SMOTE**, and their **recency** variants are labeled **helps** under our **±0.01** band; **SMOTE (recency 0.3)** is labeled **neutral**.

## 4.7 Label-delay ablation

**δ ∈ {0, 7, 14}** days (§3.1). **Baseline** **mean** **PR‑AUC** (aggregated rows): **~0.547 / ~0.534 / ~0.488** at **0 / 7 / 14** (`canonical_by_delay.csv`; **14‑day** **sparse**). **Interpretation:** **§5.5**.

**Figure.** `figures/label_delay_ablation.pdf`.

*Source: `paper/tables/canonical_by_delay.csv`.*

## 4.8 Sliding vs. static training (supplementary)

**Five** splits; **mean** **PR‑AUC** **~0.583** (**static**) vs **~0.570** (**sliding**) (`results/sliding_window/results.csv`).

**Figure.** `figures/sliding_window_comparison.pdf`.

## 4.9 Synthetic-sample fidelity diagnostics (supplementary)

To address sample-quality concerns for **CTGAN** and **TabDDPM**, we run a lightweight fidelity audit on synthetic positives (`src/run_fidelity_analysis.py`) across the same **4 temporal folds**. For tractability, this uses **target rate 10%**, **max 5000 synthetic positives/fold**, and reduced generator budgets (**CTGAN epochs=3**, **TabDDPM epochs=2, timesteps=50**). We report:

- **Numeric quantile L1** (lower is better),
- **Categorical TV distance** (lower is better),
- **Correlation MAD** on numeric features (lower is better),
- **Real-vs-synthetic AUC** from a discriminator (closer to 0.5 is better).

| Method | Numeric quantile L1 ↓ | Categorical TV ↓ | Correlation MAD ↓ | Real-vs-synth AUC (ideal 0.5) |
|--------|------------------------|------------------|-------------------|-------------------------------|
| CTGAN | 9,829.68 | 0.145 | 0.112 | 1.000 |
| TabDDPM | 18,916,262.15 | 1.000 | 0.118 | 1.000 |

These diagnostics indicate **low distribution fidelity** in this lightweight setting, especially for TabDDPM on mixed-type features. This helps explain why downstream gains remain inconsistent and why we treat generative results as protocol-dependent rather than universally reliable.

*Source: `paper/tables/synthetic_fidelity.csv`, `paper/tables/synthetic_fidelity_summary.csv`.*

---

**Reproducibility.** Regenerate tables: `python -m src.run_unified_analysis`. Regenerate figures: `python -m src.paper_figures`.
