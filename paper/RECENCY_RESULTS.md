# Recency-Aware Synthesis: Results (2 folds, 1 rate, n=2)

**Run:** `python3 -m src.run_protocol --quick --recency-ablation`  
**Folds:** 0, 1 | **Target rate:** 5%

---

## Raw PR-AUC by Fold

| Fold | Baseline | CTGAN | CTGAN_recency03 | TabDDPM | TabDDPM_recency03 | SMOTE | SMOTE_recency03 |
|------|----------|-------|-----------------|---------|-------------------|-------|-----------------|
| 0 | 0.5369 | 0.5392 | 0.5343 | 0.5323 | 0.5303 | 0.5407 | 0.5066 |
| 1 | 0.5407 | 0.5425 | **0.5515** | 0.5283 | **0.5333** | 0.5408 | 0.5324 |

---

## Verdict (Brutal)

### CTGAN recency
- Fold 0: **Worse** (0.5343 vs 0.5392 full; −0.49%)
- Fold 1: **Better** (0.5515 vs 0.5425 full; +0.90%)
- **Mixed.** Recency helps in one fold, hurts in the other. No clear win.

### TabDDPM recency
- Fold 0: **Worse** (0.5303 vs 0.5323 full; −0.20%)
- Fold 1: **Better** (0.5333 vs 0.5283 full; +0.50%)
- **Mixed.** Same pattern as CTGAN.

### SMOTE recency
- Fold 0: **Much worse** (0.5066 vs 0.5407 full; −3.41%)
- Fold 1: **Worse** (0.5324 vs 0.5408 full; −0.84%)
- **Consistently hurts.** Restricting SMOTE to 30% of positives removes too many real samples; interpolation degrades.

---

## Interpretation

1. **Recency does not systematically help.** For CTGAN/TabDDPM, gains in Fold 1 are offset by losses in Fold 0.
2. **SMOTE + recency is a bad idea.** Using only 30% of positives weakens SMOTE’s ability to interpolate.
3. **Fold-dependent effect.** Fold 1 may have stronger drift; recency could matter more there. With n=2 folds this is speculative.
4. **Caveat:** Quick run (2 folds, 1 rate). Need 4 folds and multiple rates before drawing strong conclusions.

---

## Paper narrative (honest)

> Recency-aware synthesis (generator trained on last 30% of positives) showed **no consistent benefit**. In one fold, CTGAN and TabDDPM improved; in another, they worsened. SMOTE with recency **consistently degraded** performance. We conclude that temporal subsetting of generator input is not a reliable fix for synthetic oversampling under shift; the loss of diversity from discarding older positives can outweigh gains from recency alignment.
