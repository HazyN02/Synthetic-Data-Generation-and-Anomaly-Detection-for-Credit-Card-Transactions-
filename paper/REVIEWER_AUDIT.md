# Reviewer-Style Audit: Rigor, Contribution, and Next Steps

*Acting as a skeptical reviewer to ensure every claim is supported and every proposed addition earns its place.*

---

## 1. CRITICAL: Methodological Gaps

### 1.1 Fold Alignment (Must Fix)

**Issue:** Drift report and protocol use *different fold definitions*.

| Source | Fold logic | Train/Val for "fold 0" |
|--------|------------|------------------------|
| `run_drift_report` | `split.time_based_cv` (N_SPLITS=5) | train=chunk0, val=chunk1 |
| `run_protocol` | K+1 chunks (n_folds=4) | train=chunks0-1, val=chunk2 |

**Consequence:** Drift-harm correlation (domain AUC vs augmentation delta) is **invalid** if drift fold i ≠ protocol fold i. You would be correlating shift in [chunk0 vs chunk1] with performance change in [train on chunks0-1, val on chunk2].

**Fix:** Use the *same* temporal fold logic everywhere. Options:
- **(A)** Add `run_drift_report` mode that uses protocol's `_split_temporal_folds` and pass n_folds=4.
- **(B)** Change protocol to use `time_based_cv` and match drift.
- **(C)** Create a shared `src/folds.py` with one canonical `get_temporal_folds(df, n_folds)` used by protocol, drift, sliding, unified_analysis.

**Action:** Implement (C). Refactor `run_protocol`, `run_drift_report`, `run_sliding_window` to call shared fold logic. Re-run drift report with protocol folds. Re-run unified_analysis.

---

### 1.2 Result Path Inconsistency

**Issue:** `run_unified_analysis` loads `results/results.csv` but protocol writes to `results/protocol/results.csv`. `run_aggregate_results` correctly checks `results/protocol/results.csv` first.

**Fix:** Update `run_unified_analysis.load_protocol_results()` to check `results/protocol/results.csv` (and optionally `results/protocol/run_*/results.csv` for latest run) before legacy `results/results.csv`.

---

## 2. Claim-to-Evidence Mapping

For each paper claim, we need direct evidence. Current status:

| Claim | Evidence needed | Status |
|-------|-----------------|--------|
| CTGAN/TabDDPM often hurt | Per-fold delta = baseline − method; mean > 0 | **Partial** – we have protocol results; aggregate says baseline wins on mean |
| SMOTE neutral or slightly worse | SMOTE vs baseline per fold | **Missing** – SMOTE not run on same folds as protocol (different experiments) |
| Harm correlates with drift | Spearman/Pearson(domain_auc, pr_auc_delta) | **Invalid until 1.1 fixed** – folds don't match |
| Sliding can match/beat static on far future | Sliding vs static per fold; compare on high-drift folds | **Partial** – sliding results exist but fold structure may differ; need to align |

**Action:** Fix 1.1, then re-run drift. Run SMOTE on *protocol* folds (same train/val split) for fair comparison. Re-run sliding on protocol folds if needed.

---

## 3. Proposed Additions: Do They Earn Their Place?

### 3.1 Synthetic Data as Stress Test

**Idea:** Reframe augmentation as a diagnostic—if augmentation hurts, it indicates high shift.

**Contribution check:**
- **Pro:** Turns negative result into actionable insight. Low cost (narrative + one plot).
- **Con:** Requires drift-harm correlation to hold. If correlation is weak or non-existent after fold fix, the "stress test" interpretation collapses.

**Verdict:** **Conditional.** Only add after fold alignment and validated drift-harm correlation. If correlation holds: strong addition. If not: drop or soften to "augmentation outcome varies with fold; we hypothesize shift plays a role."

**Action:** Fix folds → compute drift per protocol fold → plot domain_auc vs (baseline − best_augmented) per fold. Report correlation and p-value. If |r| > 0.5 and p < 0.1, adopt "stress test" framing.

---

### 3.2 Recency Bias / Recency-Aware Synthesis

**Idea:** Train generators only on *recent* positives (e.g., last 30% of training period) to better match validation period.

**Contribution check:**
- **Pro:** Tests a concrete mitigation—shift-aligned synthesis.
- **Con:** Adds complexity. Need to show it helps in high-shift folds (e.g., Fold 3). If it doesn't help, it's a negative result that still supports "synthetic oversampling is unreliable."

**Verdict:** **Worth one ablation.** Run recency-only synthesis on Fold 3 (highest shift in our results). If it beats both baseline and full-history augmentation → strong. If it doesn't → report as negative and discuss.

**Action:** Add `recency_frac` to `make_synthetic_positives` (filter pos_df to last frac of rows by time). Run protocol with recency_frac=0.3 for Fold 3 only (or add a `--recency-ablation` mode). Compare to baseline and full-history CTGAN/TabDDPM.

---

### 3.3 Target Rate Ablation (5% vs 10% vs 20%)

**Idea:** Show that 5% is safer; 10%/20% often hurt.

**Contribution check:**
- **Pro:** We have the data. Clear pattern in our results.
- **Con:** Might be overkill if the main message is "augmentation often hurts." Reviewer could ask: "So the takeaway is use 5% if you must? That's a small footnote."

**Verdict:** **Keep as ablation, one paragraph + one table.** Don't overstate. "Among target rates, 5% was least harmful; 10% and 20% more often degraded performance (Table X)."

---

### 3.4 Augmentation as Evaluation (Synthetic OOD Test Set)

**Idea:** Use synthetic fraud not for training but for evaluation—"Can the detector catch synthetic fraud?" as a robustness check.

**Contribution check:**
- **Pro:** Novel use of synthetic data; ties to OOD robustness.
- **Con:** Different research question. Our paper is about *oversampling for training*. "Augmentation as eval" shifts to "synthetic data for robustness testing." Risk of scope creep. Also: what does "catching synthetic fraud" mean? Detector trained on real data; we evaluate on synthetic. If detector fails on synthetic, is that bad? Synthetic might be distributionally different—so failure could mean "synthetic is OOD" not "detector is bad." Interpretation is murky.

**Verdict:** **Skip for main paper.** It dilutes the core message. Optionally mention as future work: "Using synthetic fraud as an OOD test set to stress-test detector robustness is an interesting direction."

---

### 3.5 When-to-Augment Decision Rule

**Idea:** If domain AUC > threshold, skip augmentation; else try 5%.

**Contribution check:**
- **Pro:** Actionable. Practitioners want a rule.
- **Con:** Threshold is arbitrary without validation. We'd need a separate validation procedure (e.g., tune threshold on held-out folds) to avoid overfitting. With 4 folds, that's fragile.

**Verdict:** **Soften to guidance, not a fixed rule.** "When domain AUC is high (>0.85 in our setup), augmentation more often hurt; when lower, gains were possible. We recommend practitioners run a small pilot: baseline vs augmented on their most recent holdout; if augmented underperforms, prioritize retraining over augmentation."

---

## 4. What Must Exist for the Paper to Be Rigorous

### Must-have (before submission)

1. **Unified fold logic** – One canonical fold split used by protocol, drift, sliding, SMOTE.
2. **Drift per protocol fold** – domain_auc for each of our 4 folds.
3. **Drift-harm analysis** – Scatter: domain_auc vs augmentation_delta. Correlation + p-value. If significant, adopt stress-test framing.
4. **SMOTE on protocol folds** – Same train/val as protocol. Otherwise SMOTE comparison is unfair.
5. **Sliding on protocol folds** – Ensure sliding uses same folds. Compare sliding vs static on high-drift vs low-drift folds.

### Should-have (strengthens paper)

6. **Target rate ablation table** – One table: method × rate × fold, mean delta. One-sentence takeaway.
7. **Recency ablation** – One experiment: Fold 3, recency-only synthesis vs full. One paragraph.

### Nice-to-have (don't let block submission)

8. **Augmentation as eval** – Future work only.
9. **Calibration analysis** – If a reviewer asks, we could add. Not central.

---

## 5. Suggested Paper Structure (Revised)

1. **Abstract** – Stress test framing (conditional on correlation). "We show synthetic oversampling often hurts under temporal shift; the degree of harm correlates with distribution shift. We propose using augmentation outcome as a cheap diagnostic for shift severity."

2. **Intro** – Same. Contributions: (1) empirical characterization, (2) drift-harm correlation, (3) stress-test interpretation, (4) practical guidance.

3. **Method** – Add: "All experiments use a shared temporal fold split (Section X) to ensure fair comparison and valid drift-harm analysis."

4. **Experiments** – Protocol results (done). Drift report (re-run with protocol folds). SMOTE (re-run on protocol folds). Sliding (align folds). Unified table.

5. **Analysis** – 
   - 4.1 Augmentation often hurts (table).
   - 4.2 Drift predicts harm (scatter, correlation). Stress-test interpretation.
   - 4.3 Target rate ablation (one table).
   - 4.4 Recency ablation (if run).
   - 4.5 Sliding vs static by drift level.

6. **Discussion** – When to augment (guidance, not rule). Limitations. Future work (augmentation as eval).

---

## 6. Action Checklist

- [x] Create `src/folds.py` with canonical `get_temporal_folds(df, n_folds, time_col)`.
- [x] Refactor `run_protocol`, `run_drift_report` to use it.
- [x] Update `run_unified_analysis` to load from `results/protocol/results.csv`.
- [x] Re-run `run_drift_report --protocol-folds --n-folds 4`.
- [x] Run `run_unified_analysis` → verify drift-harm correlation.
- [ ] Align `run_sliding_window` with protocol folds (optional; sliding uses different logic).
- [ ] Add recency ablation (Fold 3, recency_frac=0.3) — optional.
- [ ] Update paper sections with final results and figures.

---

## 7. Post-Fix Results (Protocol-Aligned Drift)

**Drift-harm correlation (domain AUC vs pr_auc_delta):**
- **CTGAN: r = 0.994** — Very strong. Higher drift → more harm from CTGAN. **Supports stress-test framing.**
- **TabDDPM: r = 0.30** — Weak. TabDDPM harm less predictable by drift.
- **SMOTE: r = -0.36** — Slight negative. SMOTE helps more when drift is higher (or noise).

**Takeaway:** CTGAN drift-harm correlation is strong enough to adopt the "synthetic data as stress test" framing for *CTGAN*. For TabDDPM, be more cautious—say "harm varies by fold; CTGAN harm correlates with drift."

**Domain AUC by fold (protocol-aligned):**
- Fold 0: 0.824
- Fold 1: 0.885
- Fold 2: 0.919 (highest)
- Fold 3: 0.885
