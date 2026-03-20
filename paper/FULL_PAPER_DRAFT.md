# When Synthetic Oversampling Helps or Hurts Rare-Event Fraud Detection Under Temporal Shift

---

## Abstract

Credit card fraud detection faces extreme class imbalance (~3.5% fraud) and non-stationary data: fraud patterns evolve over time, so models trained on the past degrade on future transactions. A common mitigation is synthetic oversampling (e.g., SMOTE, CTGAN, TabDDPM) to rebalance the minority class, but most evaluations ignore temporal structure and label delay. We ask: *when does synthetic oversampling genuinely improve fraud detection under realistic, time-respecting evaluation?*

We study this question on the IEEE-CIS fraud dataset (~590k real e-commerce transactions from Vesta) using leakage-safe temporal cross-validation: in each fold, models train on earlier transactions and are evaluated strictly on later ones. All methods share a single, fixed feature pipeline and a single LightGBM classifier to avoid confounding from feature engineering. We compare (i) a baseline trained only on real data, (ii) SMOTE oversampling, and (iii) deep generative oversampling via CTGAN and TabDDPM. We also implement a label-delay protocol and a recency-aware synthesis ablation (training generators only on the most recent 30% of positives).

Across four temporal folds, deep generative oversampling yields at most modest average gains over the baseline (CTGAN: +0.65 PR-AUC points; TabDDPM: +0.08; SMOTE: +0.27), and none of these differences are statistically significant (p ≥ 0.12). SMOTE is consistently competitive despite its simplicity. Recency-aware synthesis does not systematically help and often hurts; SMOTE restricted to recent positives clearly degrades performance. A domain-classifier "drift AUC" correlates with when CTGAN helps (r ≈ 0.91): in folds with lower train–test shift, CTGAN can slightly improve PR-AUC, whereas under stronger shift it often degrades performance. With only four folds, this correlation is exploratory, not definitive.

We conclude that, on IEEE-CIS under realistic temporal and label-delay protocols, simple interpolation-based oversampling (SMOTE) is a strong baseline and deep tabular generators are far from a silver bullet. The main contribution is methodological: a carefully controlled, leakage-safe evaluation of oversampling methods under temporal shift, together with honest reporting of null and nuanced results.

**Keywords:** Fraud detection, synthetic oversampling, temporal evaluation, CTGAN, TabDDPM, SMOTE, class imbalance, distribution shift.

---

## 1. Introduction

Credit card fraud detection is a critical problem in financial systems, characterized by extreme class imbalance (typically 1–5% fraudulent transactions) and non-stationary data distributions. Fraudsters adapt their tactics over time, so models trained on historical data often degrade when applied to future transactions. A common approach to address imbalance is **synthetic oversampling**: generating additional minority-class examples to rebalance the training set. Methods range from simple interpolation (SMOTE [1]) to deep generative models (CTGAN [2], TabDDPM [3]) that synthesize realistic-looking fraud examples.

Despite the growing use of generative oversampling in tabular settings, most evaluations use random train/test splits and ignore temporal structure. In production, models are trained on past data and evaluated on future data; validation sets must lie strictly in the future of training to avoid leakage. Moreover, real-world label latency—the delay between a transaction and its fraud label—means that the most recent training data may be unavailable. Evaluations that ignore these constraints can overstate the benefits of synthetic oversampling.

**Research Question.** We ask: *When does synthetic oversampling genuinely improve fraud detection under realistic, time-respecting evaluation?*

**Contributions.** We make three contributions:

1. **Empirical characterization:** We compare baseline (real data only), SMOTE, CTGAN, and TabDDPM on the IEEE-CIS fraud dataset under leakage-safe temporal cross-validation. We find that deep generative oversampling yields at most modest average gains (0–1 PR-AUC points) over the baseline, while SMOTE is consistently competitive. None of the method comparisons reach statistical significance across four folds.

2. **Drift-harm relationship:** We quantify distribution shift via a domain-classifier AUC and show that higher shift correlates with synthetic oversampling degrading performance. For CTGAN, this correlation is r ≈ 0.91 (exploratory, n=4 folds). This provides a practical signal: when train–test drift is high, augmentation learned from the past may be misleading.

3. **Label-delay and recency ablations:** We implement a label-delay protocol (0, 7, 14 days) and a recency-aware synthesis ablation (training generators only on the most recent 30% of positives). Recency-aware synthesis does not systematically help and often hurts; SMOTE restricted to recent positives is clearly detrimental. Label delay degrades all methods proportionally; synthetic oversampling does not rescue performance under delay.

We argue that future fraud-detection work should report performance under temporal protocols before claiming gains from sophisticated synthetic data generators.

---

## 2. Related Work

**Fraud Detection and the IEEE-CIS Benchmark.** The IEEE-CIS Fraud Detection dataset [4] is a widely used benchmark of real e-commerce transactions from Vesta, with ~590k training samples and ~3.5% fraud rate. Numerous studies have applied gradient boosting, neural networks, and ensemble methods to this dataset. A recurring theme is the importance of temporal evaluation: random splits can leak future information and overstate performance [5].

**Synthetic Oversampling.** SMOTE [1] interpolates between minority examples in feature space and remains a standard baseline for imbalanced classification [8]. Deep generative approaches for tabular data include CTGAN and TVAE [2] (conditional GAN and tabular VAE from Xu et al.) and TabDDPM [3] (diffusion-based generation). These are often evaluated on synthetic data benchmarks or with random splits; few studies systematically compare them under temporal evaluation for fraud.

**Temporal Evaluation and Label Delay.** The Fraud Detection Handbook [6] recommends time-respecting splits and explicit modeling of label latency. We adopt a similar protocol: temporal folds where validation lies strictly in the future of training, and an optional label-delay variant that drops the most recent days from training to simulate label latency.

**Distribution Shift and Domain Adaptation.** Domain classifier AUC is a common proxy for covariate shift [7]. We use it to quantify train–validation shift and examine whether it predicts when synthetic oversampling helps or hurts.

---

## 3. Method

### 3.1 Dataset and Temporal Evaluation

**Dataset.** We use the IEEE-CIS Fraud Detection dataset (Kaggle), comprising ~590k transactions with ~3.5% labeled as fraudulent (`isFraud = 1`). The time variable `TransactionDT` is a monotonically increasing delta in seconds that orders transactions.

**Temporal Folds.** We follow the Fraud Detection Handbook's recommendation to respect time in both training and validation. Preprocessing is fit per fold on the training block only and then applied to the validation block, avoiding leakage from global preprocessing. We construct four temporal folds: in fold i, we train on an initial time prefix and validate on the immediately following block. There is no random shuffling—validation always lies strictly in the future of training. All methods use the same fold boundaries.

**Label-Delay Protocol.** For a given delay δ (in days), we identify the start time of the validation block and remove from training any transaction with `TransactionDT` within δ days of that start. This mimics real-world label latency. We run 0-, 7-, and 14-day delays and report when folds become too sparse.

### 3.2 Models and Oversampling Protocols

All classifiers are **LightGBM** models with the same hyperparameters. Before modeling, we apply a single shared preprocessing pipeline: drop high-cardinality columns, apply hashing/grouping to categoricals, and limit the feature set to ~100 mixed numerical/categorical columns. The pipeline is fit only on training data in each fold and applied to validation data.

| Method | Description |
|--------|-------------|
| **Baseline** | LightGBM on real data only (class weights). |
| **SMOTE** | Interpolation-based oversampling to target fraud rates 5%, 10%, 20%. |
| **CTGAN** | Conditional Tabular GAN trained on positives only; synthetic positives added to reach same target rates. |
| **TabDDPM** | Tabular diffusion model trained on positives only; used analogously to CTGAN. |

**Recency-Aware Ablation.** We test whether training generators only on the most recent 30% of positives (by `TransactionDT`) helps. For SMOTE, we restrict to recent positives before interpolation. We use ρ = 0.3 with a safeguard that falls back to all positives if too few fraud cases remain.

### 3.3 Drift Quantification

We quantify distribution shift via a **domain classifier AUC**: for each fold, we label training as domain 0 and validation as domain 1, train a LightGBM to distinguish them, and report ROC-AUC. A value near 0.5 indicates little shift; higher values indicate stronger covariate shift. Given only four folds, correlations with oversampling impact are exploratory.

### 3.4 Evaluation Metrics

- **PR-AUC (Average Precision):** Summarizes the full precision–recall curve; more informative than ROC-AUC under strong imbalance.
- **Recall@1% FPR:** Fraction of frauds caught when constraining false positive rate to 1%; approximates production operating regions.

---

## 4. Experiments

### 4.1 Setup

- **Dataset:** IEEE-CIS Fraud Detection, ~590k transactions, ~3.5% fraud.
- **Folds:** 4 temporal folds; preprocessing fit per fold on train only.
- **Classifier:** LightGBM (shared hyperparameters across methods).
- **Oversampling target rates:** 5%, 10%, 20% (best per method per fold reported in main tables).

### 4.2 Main Results

Table 1 reports mean PR-AUC and Recall@1% FPR across four temporal folds. Baseline achieves 0.564 ± 0.031 PR-AUC and 0.472 ± 0.022 Recall@1% FPR. CTGAN yields 0.570 ± 0.027 PR-AUC; TabDDPM 0.565 ± 0.029; SMOTE 0.567 ± 0.024. All oversampling methods are within ~1 PR-AUC point of each other and the baseline. Recency-aware variants (ctgan_recency03, tabddpm_recency03) perform similarly to their non-recency counterparts; smote_recency03 is clearly worse (0.552 ± 0.031).

**Table 1. Mean PR-AUC and Recall@1% FPR across four temporal folds (no label delay).**

| Method | PR-AUC (mean ± std) | Recall@1% FPR (mean ± std) |
|--------|---------------------|----------------------------|
| Baseline | 0.564 ± 0.031 | 0.472 ± 0.022 |
| CTGAN | 0.570 ± 0.027 | 0.478 ± 0.023 |
| TabDDPM | 0.565 ± 0.029 | 0.472 ± 0.021 |
| SMOTE | 0.567 ± 0.024 | 0.476 ± 0.019 |
| CTGAN (recency 0.3) | 0.569 ± 0.025 | 0.478 ± 0.023 |
| TabDDPM (recency 0.3) | 0.564 ± 0.026 | 0.471 ± 0.022 |
| SMOTE (recency 0.3) | 0.552 ± 0.031 | 0.470 ± 0.025 |

### 4.3 Fold-by-Fold Results

Table 2 shows PR-AUC per fold. No method consistently dominates. In Fold 0 (domain AUC 0.82), all oversampling methods slightly beat baseline. In Fold 1 (domain AUC 0.89), CTGAN leads; TabDDPM underperforms. In Fold 2 (domain AUC 0.92), TabDDPM is best. In Fold 3 (domain AUC 0.88), CTGAN and SMOTE match; TabDDPM is slightly below. SMOTE restricted to recency (smote_recency03) is the worst in Folds 0 and 1.

**Table 2. PR-AUC by fold and method (best target rate per method).**

| Fold | Baseline | CTGAN | TabDDPM | SMOTE | SMOTE (rec) |
|------|----------|-------|---------|-------|-------------|
| 0 | 0.543 | 0.555 | 0.554 | 0.553 | 0.518 |
| 1 | 0.578 | 0.585 | 0.562 | 0.576 | 0.552 |
| 2 | 0.601 | 0.600 | 0.605 | 0.596 | 0.593 |
| 3 | 0.534 | 0.542 | 0.537 | 0.541 | 0.543 |

### 4.4 Statistical Comparisons

We run paired permutation tests (10,000 permutations) across folds for each method pair. Table 3 reports mean delta (method − baseline) and p-values. None of the comparisons are significant at α = 0.05. CTGAN shows the largest positive delta vs baseline (+0.0065 PR-AUC, p = 0.245). SMOTE vs baseline: +0.0027, p = 0.50. TabDDPM vs baseline: +0.0008, p = 1.0.

**Table 3. Paired permutation tests: mean PR-AUC delta (method − baseline) and p-values.**

| Comparison | Mean Δ | p-value | Significant? |
|------------|--------|---------|--------------|
| Baseline vs SMOTE | +0.0027 | 0.50 | No |
| Baseline vs CTGAN | +0.0065 | 0.245 | No |
| Baseline vs TabDDPM | +0.0008 | 1.0 | No |
| CTGAN vs SMOTE | −0.0038 | 0.12 | No |
| TabDDPM vs SMOTE | +0.0019 | 0.75 | No |
| TabDDPM vs CTGAN | +0.0057 | 0.62 | No |

### 4.5 Drift-Harm Analysis

We compute the correlation between domain AUC and PR-AUC delta (baseline − method) per fold. Higher domain AUC indicates stronger train–validation shift. For CTGAN, correlation r ≈ 0.91: when shift is high, CTGAN tends to hurt more. For TabDDPM, r ≈ 0.37; for SMOTE, r ≈ 0.80. With only four folds, these are exploratory; the pattern suggests that synthetic oversampling from the training distribution degrades when the validation distribution has shifted.

### 4.6 When It Helps or Hurts

We classify each method as "helps" (mean delta < −0.01), "neutral", or "hurts" (mean delta > 0.01). CTGAN, TabDDPM, and SMOTE are all neutral. SMOTE (recency 0.3) is the only method that clearly hurts (mean delta +0.012, verdict: hurts).

### 4.7 Label-Delay Ablation

Table 4 reports mean PR-AUC by label delay (0, 7, 14 days). All methods decline as delay increases. At 0 days, baseline 0.555; at 7 days, 0.542; at 14 days, 0.518. Synthetic oversampling does not rescue performance under delay; methods track the baseline. SMOTE (recency) remains worst at all delays.

**Table 4. Mean PR-AUC by label delay.**

| Delay | Baseline | CTGAN | TabDDPM | SMOTE | SMOTE (rec) |
|-------|----------|-------|---------|-------|-------------|
| 0 days | 0.555 | 0.559 | 0.556 | 0.559 | 0.545 |
| 7 days | 0.542 | 0.544 | 0.539 | 0.543 | 0.516 |
| 14 days | 0.518 | 0.520 | 0.515 | 0.522 | 0.488 |

### 4.8 Sliding vs Static Retraining

We compare static training (all past data) vs sliding-window training (fixed recent window) across five folds. Static: 0.582 ± 0.024 PR-AUC; Sliding: 0.570 ± 0.023. Static slightly outperforms; sliding does not clearly help under our fold structure.

---

## 5. Discussion

**Deep generative oversampling is not a game changer.** On IEEE-CIS under time-respecting validation, CTGAN and TabDDPM deliver at most modest average gains over a strong real-data baseline. SMOTE is consistently competitive. This contrasts with benchmarks that use random splits, where generative methods can appear more beneficial.

**Recency-aware synthesis mostly fails.** Training CTGAN/TabDDPM or SMOTE only on the most recent 30% of positives does not systematically help and often hurts. For SMOTE, restricting to recent positives is clearly detrimental. Naïve recency filtering can remove valuable minority modes.

**Drift-harm is visible but exploratory.** Domain-classifier AUC correlates with when synthetic oversampling degrades performance, especially for CTGAN. With four folds, this is hypothesis-generating; monitoring drift in production may help decide when to trust augmentation.

**Practical implications.** Practitioners should not assume deep generators will fix imbalance. A leakage-safe temporal protocol and a well-tuned tree model (with class weights or SMOTE) are sensible first steps. Synthetic oversampling—especially via deep generators—should be deployed for clearly identified pain points, not as a default.

---

## 6. Limitations

- **Single dataset:** Conclusions are drawn from IEEE-CIS; results on other fraud portfolios may differ.
- **Limited folds:** Four temporal folds constrain statistical power; correlations are exploratory.
- **Restricted ablations:** We use 7- and 14-day label delays and a single recency fraction (0.3).
- **No adversary or privacy analysis:** We do not model adaptive fraudsters or quantify synthetic sample privacy.

---

## 7. Conclusion

We evaluated synthetic oversampling (SMOTE, CTGAN, TabDDPM) for fraud detection under leakage-safe temporal cross-validation on the IEEE-CIS dataset. Deep generative oversampling yields at most modest gains; SMOTE is a strong baseline. Recency-aware synthesis does not help and often hurts. Domain-classifier drift correlates with when CTGAN degrades performance. We recommend that future fraud-detection work report performance under temporal protocols before claiming gains from synthetic data generators.

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
