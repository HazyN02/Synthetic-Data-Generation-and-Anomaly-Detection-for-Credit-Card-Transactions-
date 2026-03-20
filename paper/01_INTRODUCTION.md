# 1. Introduction

[To be filled with full protocol results]

- Credit card fraud: rare events, evolving attacker behavior, temporal non-stationarity
- Common approach: synthetic oversampling (CTGAN, SMOTE) to balance classes
- Gap: evaluation often uses random splits; real deployment involves temporal shift
- Research question: When does synthetic oversampling help or hurt under realistic temporal evaluation?
- Contributions:
  1. Empirical characterization: CTGAN/TabDDPM often hurt, SMOTE neutral; harm correlates with drift
  2. Drift-harm relationship: domain AUC predicts synthetic oversampling degradation
  3. Sliding-window retraining: can outperform static when validating on far future
