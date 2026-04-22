# KDD-MLF 2026 — Compliance Report (Strict Criteria)

This report maps **every item** in the provided acceptance criteria to the current manuscript materials (`paper/00_ABSTRACT.md`, `01_INTRODUCTION.md`, `02_RELATED_WORK.md`, `03_METHOD.md`, `04_EXPERIMENTS.md`, `05_ANALYSIS.md`, `06_DISCUSSION.md`, `FULL_PAPER_DRAFT.md`, tables, code).  

**Legend:** ✅ Satisfied · ⚠️ Partial / needs tightening · ❌ Gap (action required before submission)

---

## I. Formatting & Submission Compliance

| Criterion | Status | Evidence / notes |
|-----------|--------|------------------|
| **≤ 8 content pages** (ACM two-column, figures/tables included; refs/appendix excluded) | ⚠️ | `FULL_PAPER_DRAFT.md` is Markdown prose (~1,800 words noted at end)—**not** the ACM template. **Action:** Compile the **final** paper in the official LaTeX template and verify page count on the **PDF** (only the PDF counts). |
| **Double-blind** (no names, affiliations, funding IDs, self-identifying citations) | ⚠️ | Not yet audited for camera-ready. **Action:** Strip author blocks; cite own work in third person if any; search PDF for institution, email, GitHub, “our previous work” as first person. |
| **Single PDF** (embedded fonts, no broken refs, no placeholders) | ❌ | Pre-submission artifact. **Action:** `pdflatex` clean build; fix overfull boxes; resolve all `\ref{}`. |
| **No concurrent submission** | ✅ | Process / author responsibility—not a document check. |

---

## II. Problem Framing & Motivation

| Criterion | Status | Evidence / notes |
|-----------|--------|------------------|
| **Imbalance severity explicit** (exact ratio in introduction) | ⚠️ | Abstract/intro state **≈3.5% fraud** (~590k transactions). **Strengthen:** give **explicit ratio** (e.g. ~**1 : 27** fraud:legit, or **96.5%** majority class) so reviewers see it in one glance. |
| **Why naive classifiers fail** (e.g. trivial majority accuracy) | ❌ | Criteria ask to **quantify** (e.g. always predicting “legit” → ~96.5% accuracy but useless PR-AUC). **Action:** Add **one sentence + optional number** in Introduction (or §3.1). |
| **Motivate generative methods beyond SMOTE** (open empirical question) | ⚠️ | Related work mentions CTGAN/TabDDPM; intro could be clearer that **generators are worth testing** because they **may** capture multi-modal minority structure **under i.i.d. assumptions**, but **whether that transfers under temporal shift** is **unknown**—your framing. **Action:** Add a short paragraph: *why study deep generators*, not only SMOTE. |
| **Research gap** (financial ML: temporal non-stationarity, deployment, etc.) | ✅ | Temporal evaluation, label delay, drift—aligned with `01_INTRODUCTION.md`, `03_METHOD.md`, `06_DISCUSSION.md`. |
| **Crisp, falsifiable research question / hypothesis** | ✅ | “When does synthetic oversampling genuinely improve fraud detection under realistic, time-respecting evaluation?” (`00_ABSTRACT.md`, `FULL_PAPER_DRAFT.md`). |

---

## III. Dataset & Experimental Design

| Criterion | Status | Evidence / notes |
|-----------|--------|------------------|
| **Temporal train/test split mandatory** | ✅ | Four folds; train prefix, validation strictly later (`TransactionDT` order); no shuffle (`03_METHOD.md`, `folds.py`). |
| **Exact date/transaction boundaries** | ⚠️ | IEEE-CIS uses **`TransactionDT`** (seconds delta), not calendar dates. **Action:** In Method/Experiments, report **exact fold boundaries** as **TransactionDT ranges** (min/max per train/val per fold) **or** row-index / quantile splits—whatever the code uses—so reviewers see **no leakage**. |
| **Oversampling only on training**; synth never in val/test | ✅ | Generators fit on train positives; SMOTE on train; val is held-out (`03_METHOD.md`, `run_protocol.py`). **Action:** Keep **one explicit sentence** in the camera-ready PDF: “Synthetic samples are **never** added to validation or test.” |
| **No leakage from future in features** | ✅ | Preprocessing **fit per fold on train only** (`03_METHOD.md`). |
| **Baselines** (no oversampling + classifier; SMOTE; each generative method) | ✅ | Baseline (class weights), SMOTE, CTGAN, TabDDPM. Optional strengtheners (undersampling, cost-sensitive beyond weights) are **not required** if baseline + SMOTE + gens are complete—criteria say additional baselines **strengthen** but minimum is met. |
| **Dataset description** (name, source, time span, N, fraud count, rate, preprocessing) | ⚠️ | Name, source, ~N, ~3.5% fraud, preprocessing summarized. **Action:** Add **exact fraud count** (or formula: 0.035 × 590k) and clarify **time span**: relative horizon via `TransactionDT` if calendar span unavailable. |

---

## IV. Evaluation Metrics

| Criterion | Status | Evidence / notes |
|-----------|--------|------------------|
| **PR-AUC primary** | ✅ | Headline metric throughout tables and text. |
| **ROC-AUC secondary** | ⚠️ | Domain classifier uses ROC-AUC for **drift** (appropriate). **Optional:** State that **model** ranking uses PR-AUC, not ROC-AUC for fraud. |
| **Precision and Recall at chosen operating threshold** | ⚠️ | **Recall@1% FPR** reported (`method_summary`, `eval.py`). **Precision at the same threshold** is **not** in current tables. **Action:** Compute and report **Precision@1% FPR** (or precision at the score threshold that yields 1% FPR) **alongside** recall—criteria are explicit here. |
| **F1 or F-β** | ❌ | Not reported. **Action:** Add **F1 at 1% FPR threshold** (or justify **β** for cost asymmetry) **or** state clearly why PR-AUC + Recall@FPR suffice and F1 is redundant—**risky**; safer to add **one row** in the main table. |
| **Statistical significance** (CI or std; no single-run claims) | ✅ | **Mean ± std across folds**; **paired permutation tests** (`statistical_comparisons.csv`, `statistical_tests.py`). Criteria mention McNemar/bootstrap—**permutation across folds** is defensible for **fold-level** paired comparisons; **optional:** add **bootstrap CI** for mean PR-AUC if space. |
| **No cherry-picked thresholds** | ✅ | PR-AUC is threshold-free; fixed FPR regime (1%) stated. |

---

## V. Methodology Description

| Criterion | Status | Evidence / notes |
|-----------|--------|------------------|
| **Hyperparameters fully reported** (SMOTE k, CTGAN epochs, TabDDPM steps/size, LightGBM) | ⚠️ | Code has defaults (e.g. SMOTE `k_neighbors=5`, CTGAN/TabDDPM epochs/timesteps in `run_protocol.py`, `synth_tabddpm.py`). **Action:** **Table or appendix**: LightGBM params, SMOTE k, target rates, **canonical** CTGAN/TabDDPM settings used for **main results** (match the run that produced `paper/tables/*.csv`). |
| **Generator training** (where, on what subset, HP selection temporal?) | ⚠️ | Text describes train-only positives; **HP selection** (grid over target rates) should be stated as **per-fold on training** / **no peeking at test**—**Action:** one explicit sentence. |
| **Fidelity of synthetic data** | ⚠️ | Domain classifier is **drift**, not synthetic realism. **Action:** Add **brief** evidence: e.g. **mean/std of 2–3 key numeric features** (real vs synthetic) **or** one sentence on lack of obvious collapse—criteria ask for **something**. |
| **Downstream classifier fixed** | ✅ | Same LightGBM across methods. |
| **Justify each generative model** (≥1 sentence each) | ⚠️ | CTGAN and TabDDPM cited; **Action:** one line each on **why** (GAN for explicit minority generation; diffusion for multi-modal tabular)—already partly in Related Work; ensure it appears in **Method** or **Related Work**. |

---

## VI. Results & Analysis

| Criterion | Status | Evidence / notes |
|-----------|--------|------------------|
| **Analyze when methods help vs. hurt** (not just tables) | ✅ | Drift–harm, recency, label delay (`05_ANALYSIS` / `FULL_PAPER_DRAFT` §4.5–4.7, `06_DISCUSSION.md`). |
| **Frame SMOTE vs. generative under temporal shift as a finding** | ✅ | Discussion: competitive SMOTE, modest generative gains (`06_DISCUSSION.md`). |
| **Mechanism** (shift, mode collapse, overfitting to train distribution) | ⚠️ | Drift discussed; **mode collapse** / **generator overfitting** could be **one paragraph** of interpretation—not mandatory verbatim but **strengthens** VI. |
| **Ablation or diagnostic** | ✅ | Label-delay, recency, domain AUC, sliding vs static. |
| **Negative results as contributions** | ✅ | Abstract + discussion emphasize null/small effects and protocol value. |
| **Scoped conclusions** (not “generative never works everywhere”) | ✅ | IEEE-CIS + temporal protocol scope (`limitations`). |

---

## VII. Related Work

| Criterion | Status | Evidence / notes |
|-----------|--------|------------------|
| **Canonical fraud / temporal** (Dal Pozzolo et al., etc.) | ⚠️ | [5] Dal Pozzolo 2015; [6] Handbook 2020. **Action:** Add **Bahnsen et al. cost-sensitive learning** (criteria name-check) if appropriate—**verify** paper title/year and **read** before citing. |
| **CTGAN, TabDDPM + tabular financial** | ⚠️ | Original papers cited [2][3]; **tabular finance-specific** oversampling citations **missing**. |
| **2–3 recent (2022–2025) oversampling / imbalanced financial ML** | ❌ | Current refs are older (except TabDDPM 2023). **Action:** Add **2–3** peer-reviewed **2022–2025** papers on oversampling or fraud under **shift/imbalance**; **read** each before citing. |
| **Do not cite unread papers** | ✅ | Author responsibility—only add sources you have read. |

---

## VIII. KDD-MLF Relevance & Framing

| Criterion | Status | Evidence / notes |
|-----------|--------|------------------|
| **Practical implications** | ✅ | `06_DISCUSSION.md` §6.2 (practitioners: SMOTE first, drift monitoring, etc.). **Action:** Ensure a **short standalone** “Practical implications” block in the **8-page** PDF (can be §5 subsection). |
| **Regulatory / compliance** (e.g. GDPR, synthetic data) | ⚠️ | Limitations mention privacy not modeled. **Action:** **One sentence** on regulatory context of training on real transactions vs synthetic augmentation—criteria ask for **credibility**, not a legal analysis. |
| **Deployment / compute** | ⚠️ | Discussion mentions complexity briefly. **Action:** **One sentence:** training cost of CTGAN/TabDDPM vs SMOTE vs baseline, aligned with **“not worth it”** finding. |

---

## IX. Writing Quality

| Criterion | Status | Evidence / notes |
|-----------|--------|------------------|
| **Abstract ~150 words** with problem, method, dataset, key result, takeaway | ❌ | `00_ABSTRACT.md` is **longer** than 150 words (~230–250). **Action:** **Trim** to **≤150–200** (workshop may allow slightly more—**confirm CFP**); keep all five elements. |
| **Every intro claim substantiated in body** | ⚠️ | After adding trivial-accuracy, precision@FPR, F1, fold boundaries, cross-check each intro sentence. |
| **Avoid vague “significantly better” without numbers** | ✅ | Numbers and p-values in intro/draft. |
| **Non-author proofread** | ✅ | Process—not tracked here. |

---

## X. Pre-Submission Checklist

| Item | Status | Notes |
|------|--------|--------|
| Clean LaTeX → PDF; refs resolve | ❌ | Do when LaTeX exists. |
| Anonymization search (institution, email, GitHub) | ❌ | Pre-upload. |
| AI detector + rewrite flagged passages | ⚠️ | Author workflow. |
| Consistent citation style; live URLs | ⚠️ | Handbook GitHub URL in [6]—verify. |
| Read aloud | ✅ | Author workflow. |

---

## Priority action list (to satisfy the strict list)

1. **Abstract:** Trim toward **~150 words** (confirm official word limit in CFP).  
2. **Introduction:** Add **majority-class baseline** accuracy (or trivial classifier) + **explicit imbalance ratio**; strengthen **why generative methods** deserve an empirical test.  
3. **Method / Experiments:** **TransactionDT (or index) boundaries** per fold; **explicit** “synthetic only in train.”  
4. **Metrics:** **Precision@1% FPR** (matching recall) + **F1** (or F-β with justification).  
5. **Hyperparameters:** **Appendix table** (SMOTE k, CTGAN/TabDDPM/LightGBM settings for reported runs).  
6. **Synthetic fidelity:** **Short** statistical check or honest limitation.  
7. **Related work:** **Bahnsen** (if appropriate) + **2–3 papers from 2022–2025** on oversampling / fraud / imbalance (read first).  
8. **KDD-MLF framing:** **One sentence** each on **regulation** and **compute vs benefit**.  
9. **Formatting:** **ACM two-column PDF**, **≤8 pages** content, **double-blind** pass.  
10. **Statistics:** Keep permutation + std; optional bootstrap CIs as supplement.

---

## Summary

| Section | Approx. satisfaction |
|---------|----------------------|
| I Formatting | ⚠️ PDF + blind + page limit not yet proven |
| II Motivation | ⚠️ Add trivial baseline + ratio + generative motivation |
| III Design | ⚠️ Fold boundaries + exact counts |
| IV Metrics | ❌ Precision + F1/Fβ |
| V Method | ⚠️ Full HP table + fidelity blurb |
| VI Results | ✅ Strong (add mechanism wording optional) |
| VII Related work | ❌ Recent citations + Bahnsen |
| VIII MLF | ⚠️ Regulation + compute (short) |
| IX Writing | ⚠️ Abstract length |
| X Checklist | ❌ Pre-submission tasks |

**Bottom line:** The **scientific core** (temporal protocol, PR-AUC-first, paired tests, drift/recency/delay diagnostics, practitioner discussion) **aligns well** with KDD-MLF. The **strict list** still requires **document and metric gaps** (precision/F1, fold boundaries, abstract length, related-work recency, appendix hyperparameters, LaTeX/page/blind compliance). Addressing the **Priority action list** row-by-row will bring the submission in line with the criteria you provided.
