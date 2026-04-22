# 6. Discussion

## 6.1 Synthesis

Sections 4–5 give results and analysis; we do not repeat them here. In short: on IEEE-CIS under our protocol, generative oversampling shows moderate average gains over the baseline, but these gains are not statistically significant with four folds (n=4); recency and label-delay behavior follow §5; drift–performance links are hypothesis-generating. Generalizing beyond this benchmark or to production would require new studies.

## 6.2 Practical implications for practitioners

- **Generators are optional, not default.** CTGAN and TabDDPM require substantially more training iterations and operational effort than SMOTE; in our study they show **moderate mean PR‑AUC gains** over baseline but **no** statistically significant baseline improvements with n=4 folds. The extra overhead is hard to justify unless it works in *your* temporally split validation (see §4).
- **Start with simpler tools (and a good protocol)**: A leakage‑safe temporal validation scheme and a well‑tuned tree model on real data already go a long way. If more recall is needed, SMOTE (or a carefully regularized variant) on a well‑engineered feature space is a sensible first step before deploying heavy generative models.
- **Monitor drift explicitly**: Even though our drift analysis is exploratory, the pattern reinforces a basic operational lesson: when train–test drift is high, augmentation effects are more variable and can flip sign. Monitoring a domain‑classifier AUC or related shift metrics should be standard practice for deciding when to trust augmentation and when to retrain or simplify.
- **Treat synthetic oversampling as a surgical tool, not a default**: Given the modest and inconsistent gains we observe, oversampling—especially via deep generators—should be deployed for clearly identified pain points (e.g., specific rare fraud typologies), not as a blanket solution to imbalance. In regulated financial settings, synthetic-data pipelines can also raise privacy and compliance considerations (e.g., GDPR-style data minimization), so prefer simpler methods unless deep generation offers clear, temporally robust benefits.

## 6.3 Ethical considerations (for AIES-style review)

Because synthetic data can be a privacy-adjacent artifact and can change what downstream models learn from rare events, ethical implications deserve explicit attention even in a performance-focused evaluation. We do not study demographic fairness directly, nor do we quantify how synthetic fraud impacts bias in protected attributes; however, practitioners should treat synthetic augmentation as a potential risk factor for amplifying dataset-specific correlations that correlate with sensitive groups. In addition, fraud detection systems operate under asymmetric costs: false negatives typically have immediate financial and customer impact, while false positives create operational burden and customer friction. Our results suggest that oversampling—especially deep generation—should be deployed only when it yields **clear** and **temporally robust** gains under leakage-safe protocols, since uncontrolled augmentation could otherwise increase harm.

## 6.4 Limitations

- **Single primary dataset**: Our main conclusions are drawn from a single, albeit important, public benchmark (IEEE‑CIS). While we briefly examine other datasets in a separate analysis, we deliberately avoid over‑generalizing beyond IEEE‑CIS and emphasize that results on other fraud portfolios may differ.
- **Tabular, transaction‑level view only**: We work with the anonymized tabular representation provided by Vesta, without access to raw sequences, card‑holder histories beyond the engineered features, or graph structure. Some failure modes of synthetic oversampling—such as violating long‑range dependencies across accounts or merchants—are therefore only partially observable.
- **Limited folds:** We use **four** temporal folds; this bounds statistical power and makes paired p-values less informative for small effects. Drift correlations are exploratory.
- **Lightweight fidelity diagnostics:** We added supplementary distribution-fidelity checks for synthetic positives, but they are coarse and use reduced generator budgets for tractability; stronger, fully tuned fidelity evaluation remains future work.
- **Restricted label‑delay and recency settings**: We implement 7‑day and 14‑day delays and a single recency fraction (30%) for tractability. In production, label delays and temporal decay patterns may be much more complex. It remains possible that more finely tuned delay windows or recency strategies would yield larger benefits, though at the cost of additional complexity and overfitting risk.
- **No explicit adversary modeling or privacy analysis**: We do not model adaptive fraudsters reacting to deployed models, nor do we quantify privacy properties of synthetic samples. These are important for deployment, but orthogonal to our primary goal of understanding when oversampling helps or hurts under temporal shift.

## 6.5 Directions for future work

- **Multi‑dataset temporal evaluation**: Extending this protocol to other real, time‑stamped fraud datasets (e.g., those in the Fraud Dataset Benchmark) would test whether our observations about SMOTE, CTGAN, and TabDDPM hold beyond IEEE‑CIS or are dataset‑specific.
- **Richer, structure‑aware generators**: Applying sequence‑ or graph‑based generative models that operate on card‑ or merchant‑level histories might better preserve the structures that matter for fraud, and could be compared head‑to‑head with tabular generators under the same temporal protocol.
- **More realistic label‑delay regimes**: Combining longer and heterogeneous delay windows with rolling model updates, as in the Fraud Detection Handbook, would bring evaluation even closer to production and could reveal regimes where synthetic oversampling is more or less valuable.
- **Targeted augmentation for rare typologies**: Rather than globally oversampling all frauds, future work could focus on augmenting specific, business‑critical fraud segments (e.g., new merchant categories, cross‑border transactions) and measuring impact at that granularity.

