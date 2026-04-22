# Publication Plan: Reviewer-Proof, Sophisticated, Brutally Honest

**Target:** ICML / NeurIPS / ICLR workshop or main track (fraud / tabular / temporal ML)  
**Last updated:** 2026-02-20

---

## Executive Summary

This document provides a **brutal, no-appeasement** assessment of the current manuscript, grades it against related work, and outlines concrete steps to make it publication-ready. Every decision is justified. We assume a skeptical reviewer who will reject weak claims, inconsistent methodology, or overstated contributions.

---

## 1. Brutal Self-Assessment: Where We Stand

### 1.1 Claim–Evidence Mismatch (CRITICAL)

| Paper Claim | Current Evidence | Verdict |
|-------------|------------------|---------|
| "CTGAN and TabDDPM *often degrade* performance" | CTGAN mean PR-AUC 0.5679 vs baseline 0.5639 (+0.7%); TabDDPM 0.5642 (≈0). Both *slightly improve or match*. | **FALSE.** Overstated. Say "marginal gains" or "minimal benefit." |
| "SMOTE is *neutral or slightly worse* than baseline" | SMOTE mean 0.5797 vs baseline 0.5639 (+2.8%); best across folds. | **FALSE.** SMOTE clearly helps. Abstract contradicts data. |
| "Drift predicts harm from generative methods" | CTGAN: r=0.994 (domain AUC vs pr_auc_delta). TabDDPM: r=0.30. | **PARTIAL.** Strong for CTGAN; weak for TabDDPM. |
| "Sliding can match/beat static on far-future" | Sliding vs static: different fold structure; not yet aligned. | **UNVERIFIED.** |

### 1.2 Methodological Fatal Flaw: Feature-Space Mismatch

**Problem:** The "unified" comparison mixes two different feature spaces.

| Source | Features | Preprocessing |
|--------|----------|---------------|
| Protocol (baseline, CTGAN, TabDDPM) | ~100 | `preprocess_for_synth`: drop high-card, hash domains, limit cols |
| SMOTE experiment | 430+ | Raw → `prepare_features_aligned` (all cols) |

**Consequence:** Protocol baseline (0.54 fold 0) is artificially lower because it uses 100 features. SMOTE (0.56 fold 0) uses 430 features. Comparing them is **invalid**. A reviewer will reject: "You're comparing models trained on different input spaces."

**Fix (non-negotiable):**  
Run SMOTE **inside the protocol** on the same `preprocess_for_synth` output. All methods must share the same feature pipeline.

### 1.3 Statistical Power

- **4 folds only.** No confidence intervals, no significance tests.
- Correlation r=0.99 with n=4 is **extremely unstable**—one fold flip could collapse it.
- **Must add:** std over folds, bootstrap or permutation CIs where feasible, explicit "n=4, interpret with caution."

### 1.4 Single-Dataset Limitation

- IEEE-CIS only. No external validation (e.g., Kaggle Credit Fraud, PAYSIM, FDB).
- **Grade impact:** Limits generalizability claims. Must state clearly and discuss.

---

## 2. Comparison to Prior Work: Honest Grading

### 2.1 Fraud Detection Handbook (fraud-detection-handbook.github.io)

- **Their standard:** Hold-out validation with **delay period** between train and test (label delay); prequential validation for CIs; Card Precision@100 as key metric.
- **Our gap:** We use contiguous temporal blocks (no explicit delay). We use PR-AUC and Recall@1%FPR. We do **not** report Card Precision@100.
- **Action:** Either (a) add a delay block and Card Precision@100, or (b) justify our choices in Method ("We use contiguous blocks for simplicity; delay period could be added for production settings").

### 2.2 CTCN (Zhao & Guan, PeerJ CS 2023)

- **Their setup:** CTGAN + NCL (Neighborhood Cleaning Rule) + TCN classifier. Three public datasets; Recall, F1, AUC-ROC.
- **Their validation:** Likely random or simple temporal split (paper unclear).
- **Our advantage:** We use **leakage-safe temporal CV** and **domain AUC for drift**. We show CTGAN *fails* under realistic temporal evaluation—directly contradicts their positive narrative.
- **Positioning:** "CTCN reports gains from CTGAN oversampling; we show these gains vanish or reverse under strict temporal evaluation with distribution shift."

### 2.3 T-SMOTE (Temporal SMOTE, IJCAI 2022)

- **Their contribution:** SMOTE variant that respects temporal order; outperforms SMOTE on time-series.
- **Our gap:** We use standard SMOTE. T-SMOTE is a natural extension we do not test.
- **Action:** Cite as future work; optionally run T-SMOTE if implementation available.

### 2.4 Fraud Dataset Benchmark (Grover et al., arXiv 2022)

- **Their contribution:** FDB with standardized splits, multiple fraud tasks.
- **Our gap:** Single dataset (IEEE-CIS). No FDB experiments.
- **Action:** Acknowledge; consider adding one FDB dataset in appendix for robustness (if time permits).

### 2.5 TabDDPM / CTGAN Original Papers

- **TabDDPM:** Benchmarked on non-fraud tabular data with random splits.
- **CTGAN:** Same. No temporal evaluation.
- **Our contribution:** First systematic comparison of TabDDPM and CTGAN for fraud under **temporal shift**. Strong negative result.

---

## 3. Reframed Narrative (Matches the Data)

### 3.1 Correct Abstract (Draft)

> **Fidelity or Illusion: Why Synthetic Oversampling Benchmarks for Fraud Detection Cannot Be Trusted**
>
> Credit card fraud detection faces extreme class imbalance and non-stationary data. A common mitigation is synthetic oversampling (SMOTE, CTGAN, TabDDPM). We ask: *when does synthetic oversampling improve fraud detection under realistic temporal evaluation?*
>
> We evaluate baseline, SMOTE, CTGAN, and TabDDPM on the IEEE-CIS fraud dataset with leakage-safe temporal cross-validation. All methods share a common feature pipeline. We measure distribution shift via domain classifier AUC. Our main findings: (1) **SMOTE consistently improves** over baseline (mean PR-AUC +2.8%); (2) CTGAN and TabDDPM provide **marginal or no benefit** (mean +0.7% and ≈0%); (3) the benefit from CTGAN **negatively correlates with distribution shift**—higher domain AUC (more drift) predicts less gain or harm from CTGAN (r≈0.99, n=4 folds); (4) TabDDPM shows no clear drift–harm relationship.
>
> We conclude that under temporal shift, **simple interpolation (SMOTE) outperforms deep generative oversampling**. The drift–harm correlation for CTGAN suggests augmentation outcome as a **diagnostic for shift severity**: when domain AUC is high, prefer SMOTE or no augmentation over CTGAN.

### 3.2 Core Contributions (Honest)

1. **Empirical characterization:** SMOTE > baseline; CTGAN/TabDDPM ≈ baseline under temporal CV.
2. **Drift–harm correlation:** For CTGAN, higher shift → less benefit (strong correlation).
3. **Practical guidance:** Use SMOTE, not CTGAN/TabDDPM, for fraud oversampling under shift; use domain AUC as a cheap diagnostic.
4. **Methodological standard:** Shared temporal folds, shared features, leakage-safe evaluation.

---

## 4. Methodological Fixes (Non-Negotiable)

### 4.1 Unified Feature Pipeline

**Current:** Protocol uses `preprocess_for_synth` (100 cols); SMOTE uses raw (430+ cols).

**Fix:** Add SMOTE to `run_protocol.py`. After `preprocess_for_synth`, for each fold:
- Baseline: `train_and_eval(train_df, val_df)` (unchanged)
- SMOTE: `train_and_eval_smote(train_df, val_df, target_pos_rate=r)` for r in {0.05, 0.10, 0.20}
- CTGAN, TabDDPM: (unchanged)

**Reason:** `train_and_eval` and `train_and_eval_smote` both use `prepare_features_aligned`, which accepts any DataFrame. The preprocessed `train_df`/`val_df` have ~100 cols; both will work.

**Code change:** In `run_protocol.py`, add SMOTE loop similar to CTGAN, calling `train_and_eval_smote` from `synth_smote`.

### 4.2 Result Path Alignment

- `run_unified_analysis` loads SMOTE from `results/smote_baseline_results.csv`.
- `run_smote_baseline` writes to `results/smote/results.csv`.
- **Fix:** Update `load_smote_results()` to check `results/smote/results.csv` first, then legacy path.
- **Better:** Once SMOTE is in protocol, SMOTE results come from protocol; no separate SMOTE load.

### 4.3 Statistical Reporting

- Report **mean ± std** over folds for each method.
- For drift–harm: report **Pearson r, p-value** (with caveat: n=4, low power).
- Add sentence: "With only 4 temporal folds, correlation estimates are unstable; results should be interpreted as exploratory."

### 4.4 Delay Period (Optional but Strengthening)

- Fraud Detection Handbook: train | delay | test.
- **Reason:** Labels arrive with delay; test should not include "recent" unlabeled.
- **Action:** Add optional `--delay-frac` to hold out last fraction of train as unlabeled gap. Improves alignment with best practice.

---

## 5. Ablations and Additions

### 5.1 Target Rate Ablation (Keep)

- Table: method × target_rate × fold; mean delta.
- **Takeaway:** "5% target rate is least harmful for CTGAN/TabDDPM; 10–20% more often degrade."
- One paragraph + one table.

### 5.2 Recency Ablation (Optional)

- Train generators only on last 30% of training period (by time).
- **Hypothesis:** Synthetic data from recent past matches validation better.
- **If it helps:** Strong addition. **If not:** Report as negative result; supports "synthetic oversampling is unreliable."

### 5.3 Sliding vs Static (Align or De-emphasize)

- Sliding uses different fold logic. Either:
  - **A:** Refactor sliding to use `get_temporal_folds`; compare sliding vs static on same folds.
  - **B:** De-emphasize in main narrative; keep as supplementary.
- Recommendation: **B** unless sliding shows a strong effect—reduces scope and reviewer questions.

### 5.4 Augmentation as Evaluation (Skip for Main)

- "Use synthetic fraud as OOD test set" is a different research question.
- Mention as future work only.

---

## 6. Paper Structure (Revised)

1. **Abstract** — Use Section 3.1 draft. No false claims.
2. **Introduction** — Contributions: (1) empirical characterization, (2) drift–harm for CTGAN, (3) practical guidance, (4) unified temporal evaluation protocol.
3. **Related Work** — CTCN, T-SMOTE, Fraud Handbook, TabDDPM, CTGAN, FDB. Position: we are the first to systematically evaluate generative oversampling under temporal shift.
4. **Method** — Dataset; temporal folds (canonical `get_temporal_folds`); shared preprocessing; drift quantification; metrics (PR-AUC, Recall@1%FPR).
5. **Experiments** — Protocol setup; all methods on same features; 4 folds, 3 target rates.
6. **Analysis** — (6.1) SMOTE > baseline, CTGAN/TabDDPM ≈ baseline; (6.2) drift–harm for CTGAN; (6.3) target rate ablation; (6.4) limitations (n=4, single dataset).
7. **Discussion** — When to augment; limitations; future work (T-SMOTE, FDB, delay period).

---

## 7. Action Checklist (Prioritized)

### Must Do (Blockers)

- [ ] **Add SMOTE to run_protocol** — Same preprocessed data as CTGAN/TabDDPM.
- [ ] **Re-run protocol** — Full run with SMOTE included.
- [ ] **Update abstract and intro** — Match claims to data (SMOTE helps; CTGAN/TabDDPM marginal).
- [ ] **Update run_unified_analysis** — Load SMOTE from `results/smote/results.csv` if still needed before protocol includes SMOTE.
- [ ] **Statistical reporting** — Mean ± std; correlation + p-value; "n=4" caveat.

### Should Do (Strengthens)

- [ ] **Target rate ablation table** — Method × rate × fold.
- [ ] **Recency ablation** — Fold 3, recency_frac=0.3 for CTGAN.
- [ ] **Fill 05_ANALYSIS.md** — Real numbers from protocol.
- [ ] **Generate figures** — `python -m src.paper_figures`.

### Nice to Have

- [ ] **Delay period** — Optional `--delay-frac`.
- [ ] **Card Precision@100** — Add metric; compare to handbook.
- [ ] **Second dataset** — One FDB or Kaggle Credit Fraud run in appendix.

---

## 8. Reviewer Anticipation: Likely Attacks and Responses

| Attack | Response |
|--------|----------|
| "n=4 folds is too few" | Acknowledge. Report CIs or std; state "exploratory; recommend more folds in production." |
| "Single dataset" | State limitation; cite FDB as future work. |
| "Why no T-SMOTE?" | Cite as future work; standard SMOTE is the baseline we compare against. |
| "CTGAN helps in some folds" | Correct. We say "marginal benefit," not "always hurts." Drift–harm explains when it fails. |
| "SMOTE uses different features" | **Fixed** by adding SMOTE to protocol. Must implement before submission. |
| "Correlation n=4 is meaningless" | Report p-value; add caveat. The direction (higher drift → less CTGAN benefit) is interpretable even with low n. |

---

## 9. Final Grade vs Related Work

| Criterion | CTCN | T-SMOTE | Fraud Handbook | **Ours** |
|-----------|------|---------|----------------|----------|
| Temporal evaluation | Weak/unclear | Yes | Yes (delay) | Yes (contiguous) |
| Leakage safety | Unclear | Yes | Yes | Yes |
| Multiple methods | CTGAN only | T-SMOTE vs SMOTE | N/A | Baseline, SMOTE, CTGAN, TabDDPM |
| Drift quantification | No | No | No | **Yes (domain AUC)** |
| Negative result | No | No | N/A | **Yes** |
| Actionable guidance | No | Limited | Yes | **Yes** |

**Verdict:** Our angle—**negative result + drift diagnostic**—is novel and practitioner-relevant. Fix the feature mismatch, align claims with data, and the paper is viable for a strong venue.
