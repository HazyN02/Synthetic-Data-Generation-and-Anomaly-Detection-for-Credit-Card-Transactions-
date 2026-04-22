# Abstract

**Title:** Fidelity or Illusion: Why Synthetic Oversampling Benchmarks for Fraud Detection Cannot Be Trusted

**Authors:** Nigamanth Rajagopalan, Yasasvi Kaipa  
**Affiliation:** University of Florida, Department of Computer Science and Informatics  
**Target venue:** IEEE Transactions on Information Forensics and Security (T-IFS)

---

**Abstract:**

Synthetic oversampling is routinely benchmarked on imbalanced fraud datasets using random train/test splits and aggregate fidelity metrics — evaluation choices that mask the gap between synthetic data quality and downstream utility. We expose this gap empirically on the IEEE-CIS benchmark (~590k transactions, 3.5% fraud) using a leakage-safe 8-fold expanding-window temporal protocol with LightGBM and PR-AUC as the primary metric. We evaluate SMOTE, CTGAN (150 epochs), TVAE (300 epochs), and TabDDPM (50 epochs) against a class-weighted real-data baseline. CTGAN achieves the only statistically significant improvement (Δ PR-AUC = +0.0046, p = 0.045), while SMOTE, TVAE, and TabDDPM do not improve significantly over baseline. Critically, all deep generators exhibit high Distance to Closest Record (DCR) scores — TabDDPM's DCR is on the order of 10^18 — indicating that aggregate fidelity statistics do not capture genuine fraud-space coverage. An HDBSCAN cluster analysis of the fraud feature space reveals the mechanistic cause: 54.8% of fraud transactions (the noise bucket and C0) receive zero synthetic allocation across all 8 folds, and the cluster with the highest DCR (C2, DCR = 17,892) shows the worst CTGAN PR-AUC degradation (Δ = −0.0052). Our contribution is a measurement framework — temporal folds, per-class DCR/NNDR diagnostics, and cluster-level routing analysis — that exposes when and why synthetic oversampling benchmarks for fraud detection cannot be trusted.
