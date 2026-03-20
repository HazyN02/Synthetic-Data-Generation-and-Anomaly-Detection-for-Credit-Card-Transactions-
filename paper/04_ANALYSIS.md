# 4. Analysis

## 4.1 Synthetic Oversampling: SMOTE Helps; CTGAN/TabDDPM Marginal

*[Fill with protocol results after re-run with unified SMOTE]*

- **SMOTE**: Consistently improves over baseline (mean PR-AUC +2–3%). Use 5% or 10% target rate.
- **CTGAN**: Marginal benefit (mean +0.7%); gains shrink with higher drift. Use 5% target rate when drift is low.
- **TabDDPM**: Minimal benefit (≈0% mean). No clear pattern across folds.

**Reason:** Generative models (CTGAN, TabDDPM) train on the *past* distribution; under temporal shift, synthetic fraud does not match future fraud. SMOTE interpolates in feature space, which generalizes better across moderate shift.

## 4.2 Drift Predicts CTGAN Harm

- **Correlation:** domain AUC vs (baseline − CTGAN) PR-AUC: r ≈ 0.99 (n=4 folds). Higher drift → CTGAN helps less or hurts.
- **Interpretation:** When domain classifier can easily distinguish train from validation, synthetic oversampling from the training distribution is likely to degrade performance.
- **Caveat:** n=4 is small; correlation is exploratory. Report p-value and std.

## 4.3 Target Rate Ablation

- **5% target rate:** Least harmful for CTGAN/TabDDPM; sometimes helps.
- **10–20%:** More often degrade performance.
- **Table:** method × target_rate × fold; mean delta vs baseline.

## 4.4 Sliding vs Static (Optional)

- If sliding aligned with protocol folds: compare on high-drift vs low-drift folds.
- De-emphasize if fold structure differs; keep as supplementary.
