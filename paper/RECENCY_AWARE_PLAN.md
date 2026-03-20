# Recency-Aware Synthesis: Implementation Plan (Reviewer-Proof)

**Version:** 1.0  
**Last updated:** 2026-02-20

---

## 1. Executive Summary

**Idea:** Restrict the positive-class samples used to train CTGAN/TabDDPM to only the **most recent fraction** (e.g., 30%) of the training period. The generator learns P(X|Y=1) from *recent* fraud only, not from all historical fraud.  
**Hypothesis:** Recent fraud distribution is closer to the validation period than full-history fraud; recency-aware synthesis should help more on high-drift folds.

**Novelty claim:** *First work to restrict generative oversampling (CTGAN, TabDDPM) to recent positive samples for temporal alignment with the validation period in tabular fraud detection.*

---

## 2. Literature Check: Is This Novel?

### 2.1 What Exists (No Hallucination)

| Work | What They Do | How We Differ |
|------|--------------|---------------|
| **THG-OAFN** (Wei & Lee, PLOS One 2025) | Graph-based fraud detection. GraphSMOTE with k-hop constraints. Temporal via GRU in classifier. Sliding-window graph construction. | **Different paradigm:** Graphs, not tabular. Temporal is in *model architecture* (GRU), not in *subset selection for generator training*. They do not restrict oversampling input to recent positives. |
| **T-SMOTE** (IJCAI 2022) | SMOTE variant for time series. Temporal weighting in interpolation; boundary-aware sampling. | **Different mechanism:** Modifies *how* SMOTE interpolates (temporal structure). Uses all minority samples. We restrict *which* samples the generator sees. |
| **VFC-SMOTE** (Evolving streams) | Streaming SMOTE for concept drift. | Streaming/online setting; we use batch temporal folds. |
| **LD_CSMOTE** | Low-density contrastive SMOTE for streaming imbalance + drift. | Streaming; no tabular CTGAN/TabDDPM. |
| **CTCN** (PeerJ 2023) | CTGAN + NCL + TCN. Uses all training positives for CTGAN. | No recency restriction; no temporal alignment of generator input. |
| **CTGAN / TabDDPM originals** | Train on full dataset. No temporal considerations. | No recency; random or static splits. |

### 2.2 Gap

**No prior work restricts the positive samples fed to a GAN/diffusion generator to a recent temporal subset for distribution alignment with the test period.** THG-OAFN uses sliding windows for graph construction but does not apply this idea to tabular CTGAN/TabDDPM. T-SMOTE changes interpolation, not input subset. We are the first to propose **recency-based subset selection for generator training** in tabular synthetic oversampling.

### 2.3 Reviewer Anticipation

| Attack | Response |
|--------|----------|
| "T-SMOTE already does temporal SMOTE" | T-SMOTE modifies SMOTE's interpolation; we restrict which positives the *generator* trains on. Different mechanism, complementary. |
| "THG-OAFN has temporal oversampling" | THG-OAFN is graph-based (GraphSMOTE) with GRU temporal modeling. We are tabular; we subset generator input by time. Different setting. |
| "Why not just use sliding-window training?" | Sliding window changes classifier training. We change *generator* training. Both can be combined; we focus on the generator. |

---

## 3. Method Definition (Precise, Unambiguous)

### 3.1 Notation

- Train set: \( D_{\text{train}} = \{(x_i, y_i, t_i)\} \) with \( t_i = \) `TransactionDT`, sorted by \( t \).
- Positives: \( P = \{i : y_i = 1\} \). \( P \) is ordered by \( t_i \).
- **Recency fraction** \( \phi \in (0, 1] \): use the last \( \phi \) fraction of \( P \) (by time).
- \( P_{\text{recent}} = \) last \( \lceil \phi |P| \rceil \) rows of \( P \).

### 3.2 Algorithm

```
RECENCY-AWARE SYNTHESIS (CTGAN / TabDDPM)
Input: train_df (with TIME_COL, TARGET_COL), recency_frac φ, min_pos_samples M
1. pos_df = train_df[train_df[TARGET_COL] == 1].copy()
2. pos_df = pos_df.sort_values(TIME_COL).reset_index(drop=True)
3. n_pos = len(pos_df)
4. n_recent = max(M, ceil(φ * n_pos))
5. If n_recent > n_pos: n_recent = n_pos
6. pos_recent = pos_df.tail(n_recent)
7. Train generator (CTGAN/TabDDPM) on pos_recent only
8. Sample synth_add synthetic positives
9. Combine: real train (full) + synthetic
10. Train LightGBM on combined set; evaluate on val
```

**Critical:** The *classifier* is always trained on **real train (full) + synthetic**. We do **not** restrict classifier training to recent data. Only the *generator* sees recent positives.

### 3.3 Design Choices and Rationale

| Choice | Rationale |
|--------|-----------|
| **Restrict positives only** | Negatives are abundant; restricting them loses signal. Positives are scarce and their distribution shifts (fraud evolution). |
| **Classifier sees full train + synthetic** | Standard: augmented set = real + synthetic. If we restricted classifier to recent only, that would be sliding-window training, a different ablation. |
| **Sort by time, take tail** | Simple, interpretable. "Recent" = closest to validation period. |
| **min_pos_samples** | CTGAN/TabDDPM need enough positives (e.g. ≥50). If recency subset is too small, fall back to full positives. |
| **Recency frac φ** | Ablation: try 0.2, 0.3, 0.5. 0.3 is default (motivated by "last third of train"). |

---

## 4. Implementation Specification

### 4.1 Interface Change

Add optional parameter to `make_synthetic_positives` and `make_synthetic_positives_tabddpm`:

```python
recency_frac: float | None = None,  # None = use all positives (baseline)
time_col: str = "TransactionDT",
min_pos_for_recency: int = 50,
```

**Behavior:**
- If `recency_frac is None`: use all positives (current behavior).
- If `recency_frac in (0, 1]`:
  1. Sort positives by `time_col`
  2. Take last `max(min_pos_for_recency, ceil(recency_frac * n_pos))` rows
  3. If resulting subset has `>= min_pos_for_recency` rows: use it for generator
  4. Else: fall back to full positives, log warning

### 4.2 Protocol Integration

Add `--recency-ablation` mode to `run_protocol.py`:

- When `--recency-ablation`:
  - Run CTGAN and TabDDPM with `recency_frac=0.3` in addition to full-history (recency_frac=None).
  - Method names: `ctgan`, `ctgan_recency03`, `tabddpm`, `tabddpm_recency03`.
  - Optionally: run on high-drift folds only (e.g. Fold 2, 3) to reduce compute and focus the test.

- When not `--recency-ablation`: unchanged (recency_frac=None for all).

### 4.3 Ablation Grid

| Variant | recency_frac | Description |
|---------|--------------|-------------|
| ctgan | None | Full-history (current) |
| ctgan_recency03 | 0.3 | Last 30% of positives |
| ctgan_recency05 | 0.5 | Last 50% (optional) |
| tabddpm | None | Full-history |
| tabddpm_recency03 | 0.3 | Last 30% |

**SMOTE recency:** For fairness, add `smote_recency03`: restrict SMOTE to recent positives only. Same logic—`pos_df` = last 30% of positives, then SMOTE on that subset. This tests whether recency helps interpolation too, or only generative models.

---

## 5. Evaluation Plan

### 5.1 Primary Comparison

For each fold, compare:
- Baseline (no augmentation)
- CTGAN full-history (recency=None)
- CTGAN recency 0.3
- TabDDPM full-history
- TabDDPM recency 0.3
- SMOTE full-history
- SMOTE recency 0.3 (optional)

**Primary question:** Does recency-aware CTGAN/TabDDPM beat full-history on high-drift folds (e.g. Fold 2, 3)?

### 5.2 Stratified Analysis

- **By drift level:** Correlate `(recency_delta - full_delta)` with domain AUC.  
  Hypothesis: recency helps more when domain AUC is high (more shift).
- **By fold:** Report per-fold. With n=4, avoid overclaiming; report std.

### 5.3 Success Criteria (Honest)

| Outcome | Interpretation |
|---------|----------------|
| Recency beats full-history on high-drift folds | **Novel contribution validated.** Recency alignment helps generative oversampling. |
| Recency ≈ full-history | Inconclusive; report as neutral. |
| Recency worse than full-history | **Still publishable:** "Recency-based subset selection did not help; supports that synthetic oversampling is fragile under shift." Negative result. |

---

## 6. Edge Cases and Failure Modes

### 6.1 Too Few Recent Positives

- **Rule:** If `ceil(recency_frac * n_pos) < min_pos_for_recency`, fall back to full positives.
- **Log:** `[RECENCY] n_recent={n} < min={M}, using full positives`
- **Reason:** CTGAN/TabDDPM need sufficient samples; otherwise generator underfits or crashes.

### 6.2 Time Column Missing

- **Rule:** If `TIME_COL` not in `train_df`, raise clear error when `recency_frac` is set.
- **Preprocessing:** Protocol already has `TransactionDT`; ensure it is preserved in fold splits.

### 6.3 Ties in Time

- **Rule:** Stable sort by (time_col, index). Deterministic for reproducibility.

### 6.4 Same Distribution in Recent vs Full

- If train period is short, recent ≈ full. Recency has no effect.  
- **Mitigation:** Report `n_recent` vs `n_pos` per fold. If nearly equal, state in analysis.

---

## 7. Statistical Rigor

- **n=4 folds:** Report mean ± std. Do not overstate significance.
- **Correlation:** If we compute (domain_AUC, recency_benefit), report r and p; add "exploratory, n=4."
- **Pre-registration-style:** State hypotheses before seeing results: "Recency helps more on high-drift folds."

---

## 8. Paper Narrative Integration

### 8.1 Method Section

> We propose **recency-aware synthesis**: training the generator (CTGAN or TabDDPM) only on the most recent fraction φ of positive samples, ordered by transaction time. This aligns the learned fraud distribution with the validation period, which is temporally adjacent to recent training data. The classifier is trained on the full training set augmented with synthetic samples. We ablate φ ∈ {0.3, 0.5} and compare against full-history (φ=1) and SMOTE with the same recency restriction.

### 8.2 Related Work

> T-SMOTE modifies SMOTE to respect temporal structure in interpolation [cite]. THG-OAFN uses temporal modeling in a graph-based fraud detector [cite]. Unlike these, we restrict the *input* to the generative model to recent positives—a simple, interpretable strategy for temporal alignment in tabular oversampling.

### 8.3 Limitations

- Single dataset (IEEE-CIS).
- Recency fraction φ is a hyperparameter; no theoretical optimal.
- Assumes monotonic recency benefit (closer in time = more similar); may not hold if fraud is cyclical.

---

## 9. Implementation Checklist

- [ ] Add `recency_frac`, `time_col`, `min_pos_for_recency` to `make_synthetic_positives` (synth_ctgan.py)
- [ ] Add same params to `make_synthetic_positives_tabddpm` (synth_tabddpm.py)
- [ ] Add `recency_frac` to `train_and_eval_smote` for SMOTE recency ablation (synth_smote.py)
- [ ] Add `--recency-ablation` to run_protocol.py; run ctgan/tabddpm with recency_frac=0.3
- [ ] Extend run_unified_analysis to handle ctgan_recency03, tabddpm_recency03
- [ ] Document in config.json: `recency_frac`, `min_pos_for_recency`
- [ ] Add unit test: recency_frac=1.0 should match full-history
- [ ] Run protocol with --recency-ablation on at least 2 high-drift folds

---

## 10. Summary

| Aspect | Decision |
|--------|----------|
| **Novelty** | First recency-based subset selection for generator training in tabular fraud oversampling. |
| **Mechanism** | Restrict positives to last φ fraction by time; train generator on subset only. |
| **Default φ** | 0.3; ablate 0.5. |
| **Fallback** | If n_recent < 50, use full positives. |
| **SMOTE** | Include smote_recency for fair comparison. |
| **Success** | Recency beats full on high-drift folds → validated. Else → report as negative result. |
