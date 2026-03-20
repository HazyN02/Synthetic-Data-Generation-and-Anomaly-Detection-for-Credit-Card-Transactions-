# Abstract

**Title (draft):** When Synthetic Oversampling Helps or Hurts Rare-Event Fraud Detection Under Temporal Shift

**Abstract:**

Credit card fraud detection faces **extreme class imbalance** (≈3.5% fraud) and **non‑stationary data**: fraud patterns evolve over time, so models trained on the past degrade on future transactions. A common mitigation is **synthetic oversampling** (e.g. SMOTE, CTGAN, TabDDPM) to rebalance the minority class, but most evaluations ignore temporal structure and label delay. We ask: **when does synthetic oversampling genuinely improve fraud detection under realistic, time‑respecting evaluation?**

We study this question on the IEEE‑CIS fraud dataset (≈590k real e‑commerce transactions from Vesta) using **leakage‑safe temporal cross‑validation**: in each fold, models train on earlier transactions and are evaluated strictly on later ones. All methods share a **single, fixed feature pipeline** and a **single LightGBM classifier** to avoid confounding from feature engineering. We compare (i) a baseline trained only on real data, (ii) **SMOTE** oversampling, and (iii) **deep generative oversampling** via CTGAN and TabDDPM that synthesize additional fraud examples. We also introduce two robustness checks inspired by the Fraud Detection Handbook: a **label‑delay protocol** (dropping the most recent days from training) and a **recency‑aware synthesis ablation** that trains generators only on the most recent positives instead of the full history.

Across four temporal folds, **deep generative oversampling yields at most modest average gains** over the real‑data baseline (on the order of 0–1 PR‑AUC points), while **SMOTE is consistently competitive despite its simplicity**. Recency‑aware synthesis with CTGAN/TabDDPM does **not** systematically help and often hurts, and SMOTE restricted to recent positives is clearly detrimental. A domain‑classifier “drift AUC” partially explains when CTGAN helps: in folds with lower train–test shift, CTGAN can slightly improve PR‑AUC, whereas under stronger shift it often degrades performance; however, this correlation is based on only four folds and should be viewed as **exploratory**, not definitive.

We conclude that, on IEEE‑CIS under realistic temporal and label‑delay protocols, **simple interpolation‑based oversampling (SMOTE) is a strong baseline and deep tabular generators are far from a silver bullet**. The main contribution of this work is **methodological**: a carefully controlled, leakage‑safe evaluation of oversampling methods under temporal shift, together with negative results for recency‑aware synthesis. We argue that future fraud‑detection work should report performance under such temporal protocols before claiming gains from sophisticated synthetic data generators.

