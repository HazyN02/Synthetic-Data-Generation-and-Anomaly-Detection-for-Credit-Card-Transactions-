# 5. Discussion

## 5.1 What We Learn About Synthetic Oversampling on IEEE‑CIS

- **Deep generative oversampling is not a game changer**: On IEEE‑CIS under time‑respecting validation, CTGAN and TabDDPM deliver at most modest average gains over a strong real‑data LightGBM baseline—typically on the order of 0–1 PR‑AUC points, sometimes positive, sometimes slightly negative. This is far from the dramatic improvements that might be inferred from synthetic‑data benchmarks without temporal structure.
- **SMOTE is a surprisingly strong baseline**: Despite its simplicity, SMOTE in the shared 100‑feature space is consistently competitive with CTGAN/TabDDPM. In several folds it matches or slightly exceeds the generative methods, especially when the domain‑shift (train vs. validation) is large.
- **Recency‑aware synthesis mostly fails in this setting**: Training CTGAN/TabDDPM only on the most recent 30% of positives does not systematically help and often harms. For SMOTE, restricting to recent positives before interpolation is clearly detrimental. This suggests that, at least on IEEE‑CIS, naïve recency filtering is not a free win and can remove valuable minority modes.
- **Drift‑harm is visible but noisy**: Domain‑classifier AUC provides a crude but useful proxy for covariate shift. Folds with lower domain AUC sometimes see small CTGAN gains, while folds with higher shift more often see harm. With only four folds, though, any numerical correlation should be treated as **exploratory evidence**, not a robust law.

## 5.2 Practical Implications for Practitioners

- **Do not assume deep generators will fix imbalance**: On a realistic, temporally evaluated fraud benchmark, CTGAN/TabDDPM offer only small and fragile gains over a strong baseline. They add modeling and serving complexity (training stability, hyperparameters, sampling) that is hard to justify unless they deliver clear and robust benefits in your specific environment.
- **Start with simpler tools (and a good protocol)**: A leakage‑safe temporal validation scheme and a well‑tuned tree model with class weights already go a long way. If more recall is needed, SMOTE (or a carefully regularized variant) on a well‑engineered feature space is a sensible first step before deploying heavy generative models.
- **Monitor drift explicitly**: Even though our drift‑harm analysis is exploratory, the pattern reinforces a basic operational lesson: when train–test drift is high, augmentation learned from the past can be misleading. Monitoring a domain‑classifier AUC or related shift metrics should be standard practice for deciding when to trust augmentation and when to retrain or simplify.
- **Treat synthetic oversampling as a surgical tool, not a default**: Given the modest and inconsistent gains we observe, oversampling—especially via deep generators—should be deployed for clearly identified pain points (e.g., specific rare fraud typologies), not as a blanket solution to imbalance.

## 5.3 Limitations

- **Single primary dataset**: Our main conclusions are drawn from a single, albeit important, public benchmark (IEEE‑CIS). While we briefly examine other datasets in a separate analysis, we deliberately avoid over‑generalizing beyond IEEE‑CIS and emphasize that results on other fraud portfolios may differ.
- **Tabular, transaction‑level view only**: We work with the anonymized tabular representation provided by Vesta, without access to raw sequences, card‑holder histories beyond the engineered features, or graph structure. Some failure modes of synthetic oversampling—such as violating long‑range dependencies across accounts or merchants—are therefore only partially observable.
- **Limited number of temporal folds**: With four folds and a fixed set of hyperparameters, we can detect clear qualitative patterns but cannot provide tight uncertainty bounds on small effect sizes. All empirical correlations (e.g., between domain AUC and CTGAN benefit) should be interpreted as hypotheses that require confirmation on additional datasets or more folds.
- **Restricted label‑delay and recency settings**: We implement 7‑day and 14‑day delays and a single recency fraction (30%) for tractability. In production, label delays and temporal decay patterns may be much more complex. It remains possible that more finely tuned delay windows or recency strategies would yield larger benefits, though at the cost of additional complexity and overfitting risk.
- **No explicit adversary modeling or privacy analysis**: We do not attempt to model adaptive fraudsters reacting to deployed models, nor do we quantify privacy properties of synthetic samples. Both are crucial for deploying synthetic‑data pipelines in the real world but orthogonal to our main goal of understanding when oversampling helps or hurts under temporal shift.

## 5.4 Directions for Future Work

- **Multi‑dataset temporal evaluation**: Extending this protocol to other real, time‑stamped fraud datasets (e.g., those in the Fraud Dataset Benchmark) would test whether our observations about SMOTE, CTGAN, and TabDDPM hold beyond IEEE‑CIS or are dataset‑specific.
- **Richer, structure‑aware generators**: Applying sequence‑ or graph‑based generative models that operate on card‑ or merchant‑level histories might better preserve the structures that matter for fraud, and could be compared head‑to‑head with tabular generators under the same temporal protocol.
- **More realistic label‑delay regimes**: Combining longer and heterogeneous delay windows with rolling model updates, as in the Fraud Detection Handbook, would bring evaluation even closer to production and could reveal regimes where synthetic oversampling is more or less valuable.
- **Targeted augmentation for rare typologies**: Rather than globally oversampling all frauds, future work could focus on augmenting specific, business‑critical fraud segments (e.g., new merchant categories, cross‑border transactions) and measuring impact at that granularity.

