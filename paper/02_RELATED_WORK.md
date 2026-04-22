# 2. Related Work

**Fraud detection, benchmarks, and time.** The IEEE-CIS Fraud Detection dataset [4] is a widely used public benchmark of e‑commerce transactions (Vesta, Kaggle); a large fraction of published work still reports scores from **random** train/test splits. Calibrated and sampling‑aware learning under imbalance has been studied for credit risk and fraud [5]; more recent work stresses that **evaluation must respect time**—otherwise models can exploit **future information** and **overstate** performance [6]. Our work sits in that line: we treat **temporal leakage** as a first‑class problem.

**Synthetic oversampling.** SMOTE [1] interpolates between minority examples in feature space and remains a standard baseline for imbalanced classification [8]. For tabular data, **CTGAN** and **TVAE** [2] learn conditional generators; **TabDDPM** [3] uses diffusion. These methods are often benchmarked on **synthetic tabular tasks** or **i.i.d. splits**; **head‑to‑head comparison under time‑ordered fraud data** is still relatively rare, which motivates our **empirical** question.

**Why compare deep generators to SMOTE** (beyond “more complex is better”). Interpolation assumes local linear structure in feature space; **generators** can in principle **hallucinate** diverse minority modes, which may help when **training and test distributions are similar**. Under **temporal shift**, that advantage is **not guaranteed**—generators may **fit past fraud** and **amplify mismatch**—so **SMOTE versus CTGAN/TabDDPM** is an **open** question we **do not** assume resolved.

**Label delay and drift.** The Fraud Detection Handbook [6] recommends **time‑respecting** splits and explicit **label latency**; we follow that spirit with a **label‑delay** variant. **Domain classifier** scores are a common **proxy for covariate shift** [7]; we use **domain AUC** between train and validation blocks to ask **when** augmentation **helps or hurts**, in line with shift‑aware ML [7].

**Positioning.** We do **not** propose a new oversampling algorithm; we **empirically** compare **existing** methods under **temporal** evaluation (**§3–4**).
