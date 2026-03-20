# Fraud Detection Results — Summary

This document summarizes all model experiments for the fraud-synth project.

## 1. Main Protocol (Baseline vs CTGAN vs TabDDPM)

Oversampling methods: real data + synthetic fraud from generative models.

| Method | Target Pos Rate | PR-AUC (mean ± std) | Recall@1%FPR (mean ± std) |
|--------|-----------------|----------------------|----------------------------|
| baseline | — | 0.5517 ± 0.0247 | 0.4583 ± 0.0225 |
| ctgan | 5% | 0.5492 ± 0.0261 | 0.4568 ± 0.0266 |
| ctgan | 10% | 0.5631 ± 0.0254 | 0.4699 ± 0.0232 |
| ctgan | 20% | 0.5601 ± 0.0273 | 0.4654 ± 0.0253 |
| ctgan_recency03 | 5% | 0.5539 ± 0.0270 | 0.4628 ± 0.0257 |
| ctgan_recency03 | 10% | 0.5651 ± 0.0241 | 0.4743 ± 0.0166 |
| ctgan_recency03 | 20% | 0.5643 ± 0.0275 | 0.4710 ± 0.0221 |
| smote | 5% | 0.5498 ± 0.0253 | 0.4596 ± 0.0226 |
| smote | 10% | 0.5640 ± 0.0239 | 0.4739 ± 0.0191 |
| smote | 20% | 0.5605 ± 0.0254 | 0.4707 ± 0.0182 |
| smote_recency03 | 5% | 0.5336 ± 0.0353 | 0.4520 ± 0.0316 |
| smote_recency03 | 10% | 0.5454 ± 0.0286 | 0.4646 ± 0.0234 |
| smote_recency03 | 20% | 0.5448 ± 0.0222 | 0.4658 ± 0.0206 |
| tabddpm | 5% | 0.5452 ± 0.0280 | 0.4510 ± 0.0299 |
| tabddpm | 10% | 0.5590 ± 0.0241 | 0.4665 ± 0.0205 |
| tabddpm | 20% | 0.5600 ± 0.0216 | 0.4694 ± 0.0183 |
| tabddpm_recency03 | 5% | 0.5501 ± 0.0282 | 0.4572 ± 0.0280 |
| tabddpm_recency03 | 10% | 0.5581 ± 0.0229 | 0.4649 ± 0.0229 |
| tabddpm_recency03 | 20% | 0.5589 ± 0.0228 | 0.4667 ± 0.0211 |

## 2. SMOTE Baseline

Non-generative oversampling: SMOTE (k-NN interpolation of minority class).

| Method | Target Pos Rate | PR-AUC (mean ± std) | Recall@1%FPR (mean ± std) |
|--------|-----------------|----------------------|----------------------------|
| baseline | — | 0.5866 ± 0.0247 | 0.4939 ± 0.0249 |
| smote | 5% | 0.5867 ± 0.0314 | 0.4890 ± 0.0277 |
| smote | 10% | 0.5831 ± 0.0290 | 0.4860 ± 0.0324 |
| smote | 20% | 0.5829 ± 0.0307 | 0.4908 ± 0.0286 |

## 3. Sliding Window vs Static Retraining

- **Static**: train on all past data for each fold.
- **Sliding**: train on a fixed recent window.

| Strategy | PR-AUC (mean ± std) | Recall@1%FPR (mean ± std) |
|----------|----------------------|----------------------------|
| sliding | 0.5704 ± 0.0227 | 0.4856 ± 0.0228 |
| static | 0.5818 ± 0.0238 | 0.4906 ± 0.0122 |

## How These Results Serve the Project

| Component | Purpose |
|-----------|---------|
| **Baseline** | Honest baseline (real data only) — reviewers expect this. |
| **SMOTE** | Standard non-generative oversampling baseline — must beat this. |
| **CTGAN** | GAN-based synthetic fraud — compares deep generative vs interpolation. |
| **TabDDPM** | Diffusion-based synthetic fraud — state-of-the-art tabular generation. |
| **Sliding vs Static** | Temporal robustness: does retraining on recent data help? |

**Metrics** (fraud detection standard):
- **PR-AUC**: Precision-Recall AUC (handles imbalance well)
- **Recall@1%FPR**: Recall when false positive rate = 1%

**Project goal**: Establish that generative oversampling (CTGAN, TabDDPM) improves fraud detection vs baseline and SMOTE, under temporal evaluation.
