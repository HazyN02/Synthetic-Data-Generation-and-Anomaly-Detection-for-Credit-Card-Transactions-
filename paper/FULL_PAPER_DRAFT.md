# When Synthetic Oversampling Helps or Hurts Rare-Event Fraud Detection Under Temporal Shift

---

## Abstract

Extreme class imbalance (~3.5% fraud; ~1:27 fraud-to-legitimate) lets a majority-only predictor reach ~96.5% accuracy while failing on fraud, so PR-AUC drives evaluation. SMOTE and deep tabular generators (CTGAN, TabDDPM) are widely used; whether generators justify their complexity over SMOTE under temporal shift remains an open empirical question, because most prior work uses random splits. **Research question:** when does oversampling improve fraud detection under leakage-safe temporal evaluation? On IEEE-CIS (~590k Vesta transactions), we use four temporal folds, one shared LightGBM and preprocessing pipeline, comparing baseline, SMOTE, CTGAN, and TabDDPM, with label-delay and recency ablations. Relative to a class-weighted baseline, CTGAN and TabDDPM show moderate mean PR-AUC improvements (~+2–3 points), but no baseline comparison is statistically significant with n=4 folds (p≈0.12). SMOTE remains competitive; recency synthesis is often neutral overall (with SMOTE-recency hurting in some folds). A domain-drift signal is **consistent with** drift-dependent effects (CTGAN Pearson r≈-0.89 on n=4 fold-level points; exploratory). **Takeaway:** synthetic oversampling can help under temporal evaluation, but results remain fold- and drift-dependent; the contribution is a controlled temporal protocol and candid, low-power evidence.

**Keywords:** Fraud detection, synthetic oversampling, temporal evaluation, CTGAN, TabDDPM, SMOTE, class imbalance, distribution shift

---

## 1. Introduction

Credit card fraud detection is defined by **severe class imbalance** and **temporal non-stationarity**. Fraud is the **minority** class: roughly **3.5%** of transactions are fraudulent—about **one fraud for every ~27 legitimate** transactions (**~96.5% legitimate, ~3.5% fraud**). The obvious useless baseline is to **always predict “legitimate” (non-fraud)**—i.e. never flag fraud. That rule is **correct on ~96–97% of rows** because most transactions are legitimate, so **accuracy looks high**, but it **detects none of the fraud** (zero recall on the minority class). That is why **raw accuracy is a poor headline metric** here, and why evaluations emphasize **PR-AUC**, **recall at a fixed false-positive rate**, and **cost-sensitive** choices instead.

A standard response to imbalance is **synthetic oversampling**: augment the minority class with **interpolated** examples (**SMOTE**) or **learned** generators (**CTGAN**, **TabDDPM**). Deep models can in principle represent **multi-modal** or **non-linear** structure in the minority class better than local linear interpolation. Under **i.i.d.** validation, that promise has motivated a large literature on tabular GANs and diffusion. **Whether those gains survive realistic financial settings—where transactions are time-ordered, distributions drift, and labels arrive late—is largely unsettled**, because many studies still evaluate with **random train/test splits** that **leak the future** into training and **overstate** both accuracy and the benefit of augmentation. **We treat comparison of SMOTE versus deep generators under temporal evaluation as an open empirical question**, not as a given that “more sophisticated” synthesis wins.

**Research question.** *When does synthetic oversampling genuinely improve fraud detection under realistic, time-respecting evaluation?*

We address this on the **IEEE-CIS** benchmark (≈590 k e‑commerce transactions from Vesta, ≈3.5% fraud) using **leakage‑safe temporal cross‑validation** over four folds: in each fold, models train on an earlier time block and are scored only on a strictly later block. Every method uses the **same fixed preprocessing**, the **same LightGBM classifier**, and the same search over oversampling strengths, so differences are not confounded by ad‑hoc feature engineering. We compare a **real‑data baseline** (no oversampling; same LightGBM as all methods) to **SMOTE**, **CTGAN**, and **TabDDPM**, and we add two robustness checks aligned with the Fraud Detection Handbook: a **label‑delay protocol** that withholds the most recent training days before validation, and a **recency‑aware synthesis ablation** that trains generators on only the most recent fraction of positive examples.

**Findings (detail).** All **tables**, **tests**, and **figures** are in **§4**; **interpretation** of patterns is in **§5**. The **abstract** states the **headline** outcomes.

**Contributions.** (1) **Empirical:** controlled comparison of baseline, SMOTE, CTGAN, and TabDDPM under **temporal** evaluation, with **recency** and **label‑delay** checks (**§4**). (2) **Diagnostic:** **domain AUC** vs. oversampling impact (**§4.5**, **§5.3**). (3) **Methodological:** a **leakage‑safe** protocol and **frank** reporting of **small** and **null** effects.
---

## 2. Related Work

**Fraud detection, benchmarks, and time.** The IEEE-CIS Fraud Detection dataset [4] is a widely used public benchmark of e‑commerce transactions (Vesta, Kaggle); a large fraction of published work still reports scores from **random** train/test splits. Calibrated and sampling‑aware learning under imbalance has been studied for credit risk and fraud [5]; more recent work stresses that **evaluation must respect time**—otherwise models can exploit **future information** and **overstate** performance [6]. Our work sits in that line: we treat **temporal leakage** as a first‑class problem.

**Synthetic oversampling.** SMOTE [1] interpolates between minority examples in feature space and remains a standard baseline for imbalanced classification [8]. For tabular data, **CTGAN** and **TVAE** [2] learn conditional generators; **TabDDPM** [3] uses diffusion. These methods are often benchmarked on **synthetic tabular tasks** or **i.i.d. splits**; **head‑to‑head comparison under time‑ordered fraud data** is still relatively rare, which motivates our **empirical** question.

**Why compare deep generators to SMOTE** (beyond “more complex is better”). Interpolation assumes local linear structure in feature space; **generators** can in principle **hallucinate** diverse minority modes, which may help when **training and test distributions are similar**. Under **temporal shift**, that advantage is **not guaranteed**—generators may **fit past fraud** and **amplify mismatch**—so **SMOTE versus CTGAN/TabDDPM** is an **open** question we **do not** assume resolved.

**Label delay and drift.** The Fraud Detection Handbook [6] recommends **time‑respecting** splits and explicit **label latency**; we follow that spirit with a **label‑delay** variant. **Domain classifier** scores are a common **proxy for covariate shift** [7]; we use **domain AUC** between train and validation blocks to ask **when** augmentation **helps or hurts**, in line with shift‑aware ML [7].

**Positioning.** We do **not** propose a new oversampling algorithm; we **empirically** compare **existing** methods under **temporal** evaluation (**§3–4**).
---

## 3. Method

> **Locked (canonical).** This section is the source of truth for §3 in the camera-ready paper.

### 3.1 Dataset and temporal evaluation

**Data.** We use the **IEEE-CIS Fraud Detection** training split (Kaggle): real e‑commerce transactions from Vesta, **≈590 k** rows, **≈3.5%** fraudulent (`isFraud = 1`)—the same **minority‑class** regime as in §1. We do **not** use the Kaggle **test** file (labels withheld); all scores are on **held‑out blocks** carved from the training data. The time column **`TransactionDT`** is a **monotonically increasing** offset in **seconds** (competition‑relative, not a calendar clock); it **orders** rows. The target is **binary fraud**.

**Temporal folds (expanding window).** We sort by `TransactionDT` and divide the timeline into **\(K{+}1\)** **equal‑sized** row blocks (\(K=4\) folds). **Fold** \(i\) **trains** on the **concatenation** of blocks **0 through \(i\)** and **validates** on block **\(i{+}1\)**. So the training set **grows** with later folds; validation is always the **next** unseen segment—**strictly in the future** of that fold’s training data. There is **no random shuffling**. **Every** method shares the **same** fold definitions.

**Preprocessing and leakage.** Feature construction uses one **shared** pipeline (§3.2 below). The pipeline is **fit only on the training rows of each fold** and **applied** to validation—**never** fit on validation or on the full table before splitting. That blocks **leakage** from **global** statistics that would **peek** at future transactions.

**Where synthetic data appears.** SMOTE and generative models run **only** on **training** rows **after** preprocessing. **Synthetic fraud rows are never added to validation**; reported metrics are always **real** transactions at **true** labels.

**Label delay.** To mimic **label latency**, we remove training rows that are **too recent** relative to validation. Let \(t_{\mathrm{val\_start}} = \min(\texttt{TransactionDT})\) on the validation block. For delay \(\delta\) **days**, we keep only training rows with \(\texttt{TransactionDT} \le t_{\mathrm{val\_start}} - \delta \times 86400\) **seconds**. (Equivalently: a **gap** of \(\delta\) days between the **end** of usable training time and the **start** of validation.) We report \(\delta \in \{0, 7, 14\}\). If too few rows or **fewer than 50** training frauds remain, we **skip** that fold.

### 3.2 Preprocessing and downstream classifier

**Features.** We **drop** high‑cardinality / sparse columns, **hash or group** selected categoricals (e.g. email‑domain fields), and keep **≈100** numerical/categorical columns suitable for **LightGBM** and the **generators**. Categorical handling is aligned between **tree** training and **synthetic** pipelines where required.

**Classifier (fixed across methods).** All oversampling conditions use the **same** **LightGBM** hyperparameters so that differences reflect **data augmentation**, not a **different learner**. Our implementation uses **`n_estimators=300`**, **`learning_rate=0.05`**, **`num_leaves=64`**, **`min_child_samples=200`**, **`subsample=0.8`**, **`colsample_bytree=0.8`**, **`objective=binary`**, **`random_state=42`** (see appendix for a one‑line table). The **baseline** is also **class‑weighted** via **`scale_pos_weight = n_neg / n_pos`** while keeping **real** labels only and **no** synthetic oversampling.

### 3.3 Oversampling protocols

**Target fraud rates.** For SMOTE, CTGAN, and TabDDPM we vary the **post‑augmentation** minority proportion via a **target positive rate** \(\in \{5\%, 10\%, 20\%\}\). **Main** tables in this paper **aggregate** results by taking, **per fold and method**, the **best** validation **PR‑AUC** across that grid (same validation fold used for scoring). **5%** is the **canonical** setting we emphasize for **label‑delay** runs and for **interpretability**; sensitivity to the rate is part of the **experimental** design.

**SMOTE [1].** **Synthetic Minority Over‑sampling** runs in the **preprocessed** feature space. We use **`k_neighbors=5`** (capped by minority count − 1), **`random_state=42`**, and optionally cap synthetic count (**`max_synth`**) for speed. SMOTE **only** touches **training** rows.

**CTGAN [2].** We fit the **ctgan** implementation on **fraud (positive) training** rows only in the **generator** feature space, then **sample** synthetic fraud until the **target** positive rate is reached, and **concatenate** with **real** training data before fitting LightGBM. In our **canonical** (medium) protocol runs used for the main tables, CTGAN uses **`epochs=7`**, **`batch_size=512`**, **`pac=1`**, **`seed=0`**. **GPU** is disabled for reproducibility across environments.

**TabDDPM [3].** We train a **Gaussian** tabular diffusion model on **positive** training rows, **sample** synthetic positives to the **same** target rates, then train LightGBM. In our **canonical** (medium) protocol runs used for the main tables, TabDDPM uses **`timesteps=75`**, **`epochs=4`**, **`hidden_dims=[768,768]`**, **Adam** **lr** \(10^{-4}\) and **batch_size=1024** (gradient clip as in the implementation).

To address concerns about undertraining, we ran a **max-convergence sanity check** with much larger budgets (**CTGAN epochs=50**, **TabDDPM epochs=50**). Mean PR-AUC shifts remained small and inconsistent, suggesting that the small effect sizes in the main protocol are not driven solely by insufficient generator training time.

**Recency ablation (\(\rho=0.3\)).** For each generator and for SMOTE, we restrict **positive** training examples to the **latest 30%** by `TransactionDT` (with a **minimum** count fallback to **all** positives). **Negatives** stay **unchanged** for SMOTE; generators see only **recent** fraud for training.

### 3.4 Drift quantification

We train a **domain classifier** (**LightGBM**) to separate **training** (domain 0) from **validation** (domain 1), **excluding** pure time/ID features. We report **ROC‑AUC** (**domain AUC**): **~0.5** suggests **little** shift; **higher** values indicate **stronger** covariate shift. We relate this to **per‑fold** changes in **PR‑AUC** vs. the baseline (**exploratory**, **four** folds).

### 3.5 Evaluation metrics and statistics

**Primary:** **PR‑AUC** (Average Precision)—**threshold‑free** and standard under **extreme** imbalance.

**Operating point:** **Recall at 1% FPR**—fraction of frauds caught when the **false‑positive rate** on transactions is **1%**.

We report **mean ± std** **across** folds. **Method** comparisons use **paired permutation tests** over folds (**§4 Experiments**).
---

## 4. Experiments

> **Scope.** Main text focuses on **no label delay** (\(\delta=0\)) unless noted. **Label-delay** and **sliding-window** analyses follow the same pipeline as §3; placement details are in `EXPERIMENTS_SCOPE.md` if you need appendix vs main text.

All numbers below come from **`paper/tables/`** (CSV) and match **`python -m src.run_unified_analysis`** on the saved protocol runs.

### 4.1 Setup (summary)

**Protocol** is **§3** (folds, preprocessing, oversampling, metrics). **Main** tables pick the **best** **target fraud rate** in \(\{5\%,10\%,20\%\}\) **per fold and method** by validation **PR‑AUC**; **recency** uses \(\rho=0.3\). **Tests:** **10,000**‑permutation **paired** comparisons on fold‑level **PR‑AUC** (`src/statistical_tests.py`). **Selection‑on‑validation** caveats: **§5.6**.

### 4.2 Main results: mean PR‑AUC and Recall@1% FPR

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

### 4.3 Fold-by-fold PR‑AUC and domain shift

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

### 4.4 Statistical comparisons

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

### 4.5 Drift and oversampling effect relative to the baseline

**Pearson** **r** (**domain AUC** vs. method **PR‑AUC − baseline PR‑AUC**, **n = 4** fold-level points): **CTGAN** **-0.89**, **TabDDPM** **-0.80**, **SMOTE** **-0.86** (`drift_correlations.csv`; **exploratory**). We treat this as **consistent with** drift-dependent behavior rather than evidence of a robust correlation.

**Figure.** `figures/drift_vs_harm.pdf` (y-axis is **baseline − method**, so **negative** values mean oversampling **helps**).

*Source: `paper/tables/drift_harm_analysis.csv`.*

### 4.6 When oversampling helps or hurts (verdict)

Aggregating **mean** **(baseline PR‑AUC − method PR‑AUC)** across folds (see `when_it_helps_hurts.csv`): **CTGAN**, **TabDDPM**, **SMOTE**, and their **recency** variants are labeled **helps** under our **±0.01** band; **SMOTE (recency 0.3)** is labeled **neutral**.

### 4.7 Label-delay ablation

**δ ∈ {0, 7, 14}** days (§3.1). **Baseline** **mean** **PR‑AUC** (aggregated rows): **~0.547 / ~0.534 / ~0.488** at **0 / 7 / 14** (`canonical_by_delay.csv`; **14‑day** **sparse**). **Interpretation:** **§5.5**.

**Figure.** `figures/label_delay_ablation.pdf`.

*Source: `paper/tables/canonical_by_delay.csv`.*

### 4.8 Sliding vs. static training (supplementary)

**Five** splits; **mean** **PR‑AUC** **~0.583** (**static**) vs **~0.570** (**sliding**) (`results/sliding_window/results.csv`).

**Figure.** `figures/sliding_window_comparison.pdf`.

### 4.9 Synthetic-sample fidelity diagnostics (supplementary)

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
---

## 5. Analysis

> **Role.** §4 reports **numbers**; here we interpret **patterns** without repeating tables. Broader implications and limitations are in §6.

### 5.1 Why low-power results matter

The paired tests in §4.4 do not reach α = 0.05, despite moderate mean PR-AUC deltas versus the baseline. Under temporal evaluation, that is still informative: it characterizes what generic SMOTE, CTGAN, and TabDDPM pipelines achieve on a standard benchmark when leakage is controlled. Importantly, fold-level **n = 4** gives these tests limited power, so we treat p-values as weak evidence and instead emphasize effect sizes, fold variability, and protocol realism.

### 5.2 Fold heterogeneity

Rankings move by fold (§4.3) because validation regimes differ and training grows in the expanding window. Oversampling perturbs training only; it cannot stabilize rankings across shifts. Random train/test splits suppress this variance—one reason offline leaderboards can mislead.

### 5.3 Generators, drift, and drift-dependent effects

Domain AUC spans roughly 0.82–0.92 (§4.3). Because our drift metric is correlated with fold difficulty, the correlation sign depends on how “effect” is defined; in our tables we report deltas relative to the baseline, so negative r values (n=4) are exploratory and best read as: **higher drift is associated with larger absolute augmentation effects** (often benefits in this run), not as a stable causal relationship. A plausible hypothesis remains that generators fit the training fraud distribution; under covariate shift, synthetic points can either improve coverage or amplify mismatch, depending on how fraud modes evolve. SMOTE interpolates between real points; GANs and diffusion sample new points—different failure modes.

### 5.4 Recency

Recency restricts positives to the latest **30%**. Under temporal evaluation, this creates an asymmetry: **CTGAN-recency** and **TabDDPM-recency** remain close to their non-recency variants (mean PR‑AUC ≈ 0.569 and 0.564), whereas **SMOTE-recency** drops noticeably (mean PR‑AUC ≈ 0.552). In §4.6, SMOTE-recency is therefore **neutral overall** (though it hurts in some folds). A plausible reading is that thinning positives reduces minority coverage for interpolation-based SMOTE more severely than it damages the generators’ learned mapping.

### 5.5 Label delay

Delay removes labeled training rows before validation (§3.1). Oversampling cannot recover features from transactions that were never in training; the parallel decline across methods in §4.7 matches that story.

### 5.6 Selection on validation

Choosing the best target fraud rate per fold and method on validation PR-AUC (§3.3, §4.1) can be optimistic relative to a fixed rate chosen a priori. Mitigations: an inner temporal split for rate selection, or reporting a fixed 5% configuration—see §3.3 for the honest statement.

### 5.7 Why fidelity checks matter

The supplementary fidelity audit (§4.9) gives a useful mechanistic signal: synthetic samples are easy to distinguish from real frauds (real-vs-synth AUC \(\approx 1.0\) for both CTGAN and TabDDPM in that run), and TabDDPM shows especially weak categorical fidelity there. This does not invalidate the downstream benchmark results, but it clarifies why augmentation effects can be unstable: utility can improve in some folds even when synthetic realism is imperfect, and weak fidelity can cap generalization under shift.
---

## 6. Discussion

### 6.1 Synthesis

Sections 4–5 give results and analysis; we do not repeat them here. In short: on IEEE-CIS under our protocol, generative oversampling shows moderate average gains over the baseline, but these gains are not statistically significant with four folds (n=4); recency and label-delay behavior follow §5; drift–performance links are hypothesis-generating. Generalizing beyond this benchmark or to production would require new studies.

### 6.2 Practical implications for practitioners

- **Generators are optional, not default.** CTGAN and TabDDPM require substantially more training iterations and operational effort than SMOTE; in our study they show **moderate mean PR‑AUC gains** over baseline but **no** statistically significant baseline improvements with n=4 folds. The extra overhead is hard to justify unless it works in *your* temporally split validation (see §4).
- **Start with simpler tools (and a good protocol)**: A leakage‑safe temporal validation scheme and a well‑tuned tree model on real data already go a long way. If more recall is needed, SMOTE (or a carefully regularized variant) on a well‑engineered feature space is a sensible first step before deploying heavy generative models.
- **Monitor drift explicitly**: Even though our drift analysis is exploratory, the pattern reinforces a basic operational lesson: when train–test drift is high, augmentation effects are more variable and can flip sign. Monitoring a domain‑classifier AUC or related shift metrics should be standard practice for deciding when to trust augmentation and when to retrain or simplify.
- **Treat synthetic oversampling as a surgical tool, not a default**: Given the modest and inconsistent gains we observe, oversampling—especially via deep generators—should be deployed for clearly identified pain points (e.g., specific rare fraud typologies), not as a blanket solution to imbalance. In regulated financial settings, synthetic-data pipelines can also raise privacy and compliance considerations (e.g., GDPR-style data minimization), so prefer simpler methods unless deep generation offers clear, temporally robust benefits.

### 6.3 Ethical considerations (for AIES-style review)

Because synthetic data can be a privacy-adjacent artifact and can change what downstream models learn from rare events, ethical implications deserve explicit attention even in a performance-focused evaluation. We do not study demographic fairness directly, nor do we quantify how synthetic fraud impacts bias in protected attributes; however, practitioners should treat synthetic augmentation as a potential risk factor for amplifying dataset-specific correlations that correlate with sensitive groups. In addition, fraud detection systems operate under asymmetric costs: false negatives typically have immediate financial and customer impact, while false positives create operational burden and customer friction. Our results suggest that oversampling—especially deep generation—should be deployed only when it yields **clear** and **temporally robust** gains under leakage-safe protocols, since uncontrolled augmentation could otherwise increase harm.

### 6.4 Limitations

- **Single primary dataset**: Our main conclusions are drawn from a single, albeit important, public benchmark (IEEE‑CIS). While we briefly examine other datasets in a separate analysis, we deliberately avoid over‑generalizing beyond IEEE‑CIS and emphasize that results on other fraud portfolios may differ.
- **Tabular, transaction‑level view only**: We work with the anonymized tabular representation provided by Vesta, without access to raw sequences, card‑holder histories beyond the engineered features, or graph structure. Some failure modes of synthetic oversampling—such as violating long‑range dependencies across accounts or merchants—are therefore only partially observable.
- **Limited folds:** We use **four** temporal folds; this bounds statistical power and makes paired p-values less informative for small effects. Drift correlations are exploratory.
- **Lightweight fidelity diagnostics:** We added supplementary distribution-fidelity checks for synthetic positives, but they are coarse and use reduced generator budgets for tractability; stronger, fully tuned fidelity evaluation remains future work.
- **Restricted label‑delay and recency settings**: We implement 7‑day and 14‑day delays and a single recency fraction (30%) for tractability. In production, label delays and temporal decay patterns may be much more complex. It remains possible that more finely tuned delay windows or recency strategies would yield larger benefits, though at the cost of additional complexity and overfitting risk.
- **No explicit adversary modeling or privacy analysis**: We do not model adaptive fraudsters reacting to deployed models, nor do we quantify privacy properties of synthetic samples. These are important for deployment, but orthogonal to our primary goal of understanding when oversampling helps or hurts under temporal shift.

### 6.5 Directions for future work

- **Multi‑dataset temporal evaluation**: Extending this protocol to other real, time‑stamped fraud datasets (e.g., those in the Fraud Dataset Benchmark) would test whether our observations about SMOTE, CTGAN, and TabDDPM hold beyond IEEE‑CIS or are dataset‑specific.
- **Richer, structure‑aware generators**: Applying sequence‑ or graph‑based generative models that operate on card‑ or merchant‑level histories might better preserve the structures that matter for fraud, and could be compared head‑to‑head with tabular generators under the same temporal protocol.
- **More realistic label‑delay regimes**: Combining longer and heterogeneous delay windows with rolling model updates, as in the Fraud Detection Handbook, would bring evaluation even closer to production and could reveal regimes where synthetic oversampling is more or less valuable.
- **Targeted augmentation for rare typologies**: Rather than globally oversampling all frauds, future work could focus on augmenting specific, business‑critical fraud segments (e.g., new merchant categories, cross‑border transactions) and measuring impact at that granularity.
---

## 7. Conclusion

We study when synthetic oversampling improves or degrades rare-event fraud detection under **leakage-safe temporal** evaluation on the **IEEE-CIS** benchmark. Across four temporal folds, deep generative oversampling (**CTGAN**, **TabDDPM**) yields **moderate average gains** over the baseline but does **not** show statistically significant improvements under our paired tests (n=4). **SMOTE** remains a **competitive** and simple alternative, and **recency-aware** synthesis shows **limited or mixed** benefit: CTGAN/TabDDPM recency helps within our protocol, while **SMOTE-recency** is neutral overall but harms in some folds.

To explain these patterns, we relate oversampling impact to **train–validation drift** using a **domain-classifier** signal: generator performance is more consistent with drift-dependent effects (exploratory with four folds). Finally, under a **label-delay** protocol, oversampling does not recover the performance lost when recent labeled training data are withheld.

Overall, our results suggest that practitioners should treat synthetic oversampling—especially deep tabular generators—as an **evidence-driven option** rather than a default remedy for imbalance. We recommend that fraud-detection studies report performance under **time-respecting protocols** and label-delay regimes before claiming benefits from sophisticated synthetic-data generators.
---

## References

[1] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer. SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16:321–357, 2002.

[2] L. Xu, M. Skoularidou, A. Cuesta-Infante, and K. Veeramachaneni. Modeling Tabular Data using Conditional GAN. In *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*. (Introduces CTGAN and TVAE, a VAE adapted for tabular data.)

[3] A. Kotelnikov, D. Baranchuk, I. Rubachev, and A. Babenko. TabDDPM: Modelling Tabular Data with Diffusion Models. In *Proceedings of the 40th International Conference on Machine Learning (ICML 2023)*, Vol. 202, pp. 17564–17579.

[4] IEEE-CIS Fraud Detection. Kaggle competition dataset, 2019. https://www.kaggle.com/c/ieee-fraud-detection

[5] A. Dal Pozzolo, O. Caelen, R. A. Johnson, and G. Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In *2015 IEEE Symposium Series on Computational Intelligence (SSCI)*. IEEE, 2015.

[6] A. Dal Pozzolo, Y.-A. Le Borgne, O. Caelen, Y. Kessaci, F. Oblé, and G. Bontempi. *Reproducible Machine Learning for Credit Card Fraud Detection - Practical Handbook*. GitHub, 2020. https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook

[7] S. Rabanser, S. Günnemann, and Z. C. Lipton. Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift. In *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*.

[8] H. He and E. A. Garcia. Learning from Imbalanced Data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9):1263–1284, 2009.

---

*Word count (approx.): ~1,800 (excluding references). Expand or trim as needed for target page limit.*
