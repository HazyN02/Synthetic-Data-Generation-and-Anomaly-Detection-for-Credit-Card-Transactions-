# Full Evaluation: Results and Interpretation

## 1. Evaluation Setup

**Command:** `python3 -m src.run_protocol --medium --recency-ablation`

| Parameter | Value | Reason |
|-----------|-------|--------|
| Folds | 4 | Temporal chunks; train on past, validate on next |
| Target rates | 5%, 10%, 20% | Ablation over oversampling intensity |
| Methods | baseline, CTGAN, TabDDPM, SMOTE + recency variants | Full + recency_frac=0.3 |
| Est. runtime | ~6h | CTGAN/TabDDPM dominate; recency doubles generator runs |

**Fold structure:** Data sorted by `TransactionDT`. Fold i: train on chunks 0..i, validate on chunk i+1. Earlier folds = less train data, shorter temporal gap to validation. Later folds = more train data, longer gap (more drift).

---

## 2. Partial Results (Run Timed Out)

The full run was started but did not complete within the session. We have:
- **Fold 0:** Complete (all methods × all rates)
- **Fold 1:** Partial (CTGAN, TabDDPM; SMOTE missing)

### 2.1 PR-AUC by Fold (Best Rate per Method)

| Fold | Baseline | CTGAN | CTGAN_recency03 | TabDDPM | TabDDPM_recency03 | SMOTE | SMOTE_recency03 |
|------|----------|-------|-----------------|---------|-------------------|-------|-----------------|
| 0 | 0.5435 | 0.5604 | 0.5531 | 0.5524 | **0.5571** | 0.5531 | 0.5179 |
| 1 | 0.5778 | 0.5849 | 0.5816 | 0.5618 | 0.5641 | — | — |

### 2.2 Recency vs Full (Δ = recency − full)

| Method | Fold 0 | Fold 1 | Mean Δ |
|--------|--------|--------|--------|
| CTGAN | −0.73% | −0.33% | **−0.53%** |
| TabDDPM | +0.47% | +0.23% | **+0.35%** |
| SMOTE | −3.53% | — | −3.53% |

---

## 3. Interpretation

### 3.1 CTGAN Recency
- **Full-history CTGAN** slightly outperforms recency in both folds.
- Hypothesis: CTGAN benefits from more diverse positives; restricting to 30% loses useful modes.
- **Verdict:** Recency does not help CTGAN here.

### 3.2 TabDDPM Recency
- **Recency wins** in both folds (+0.47%, +0.23%).
- TabDDPM may be more sensitive to distribution mismatch; recent positives better match the validation period.
- **Verdict:** Recency helps TabDDPM. Tentative; needs Folds 2–3 for confirmation.

### 3.3 SMOTE Recency
- **Large drop** (−3.5% in Fold 0). Using only 30% of positives removes too many real samples; SMOTE’s interpolation degrades.
- **Verdict:** Recency hurts SMOTE. Do not use.

### 3.4 Best Method Overall
- Baseline: 0.5606
- CTGAN (full): 0.5726 (best in partial run)
- TabDDPM recency: 0.5606 (matches baseline)
- SMOTE (full): 0.5531
- SMOTE recency: 0.5179 (worst)

---

## 4. Completing the Full Run

To finish the evaluation:

```bash
cd /Users/yasasvikaipa/Downloads/fraud-synth-icml
nohup python3 -m src.run_protocol --medium --recency-ablation > results/protocol/run_recency_medium_log.txt 2>&1 &
```

Monitor with:
```bash
tail -f results/protocol/run_recency_medium_log.txt
wc -l results/protocol/run_*/results.csv  # grows as run progresses
```

When done, run unified analysis:
```bash
# Copy run results to main (or point run_unified_analysis at run dir)
cp results/protocol/run_20260222_202529/results.csv results/protocol/results_recency_medium.csv
python3 -m src.run_unified_analysis  # if it loads from run dir
```

---

## 5. Summary Table (Paper-Ready, Partial Data)

| Method | Mean PR-AUC | vs Baseline | Recency Effect |
|--------|-------------|-------------|----------------|
| Baseline | 0.5606 | — | — |
| CTGAN | 0.5726 | +1.2% | Recency **hurts** (−0.5%) |
| CTGAN_recency03 | 0.5673 | +0.7% | — |
| TabDDPM | 0.5571 | −0.4% | Recency **helps** (+0.4%) |
| TabDDPM_recency03 | 0.5606 | 0% | — |
| SMOTE | 0.5531 | −0.8% | Recency **hurts** (−3.5%) |
| SMOTE_recency03 | 0.5179 | −4.3% | — |

**Caveat:** n=2 folds (0, 1); Fold 1 SMOTE missing. Interpret with caution.
