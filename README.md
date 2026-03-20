# When Synthetic Oversampling Helps or Hurts Rare-Event Fraud Detection Under Temporal Shift

Empirical evaluation of SMOTE, CTGAN, and TabDDPM for fraud detection under leakage-safe temporal cross-validation on the IEEE-CIS dataset.

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/fraud-synth-icml.git
cd fraud-synth-icml
pip install -r requirements.txt

# 2. Add data (see data/README.md)
# Download IEEE-CIS from Kaggle; place train_transaction.csv, train_identity.csv in data/

# 3. Run full analysis pipeline
bash scripts/run_full_analysis.sh
```

## Project Structure

```
├── src/                    # Core code
│   ├── run_protocol.py     # Main experiment (baseline, CTGAN, TabDDPM, SMOTE)
│   ├── run_unified_analysis.py
│   ├── run_canonical_analysis.py
│   ├── run_sliding_window.py
│   ├── run_label_delay_ablation.py
│   └── ...
├── paper/                  # Paper draft and figures
│   ├── FULL_PAPER_DRAFT.md
│   ├── tables/             # Generated result tables
│   └── figures/            # Generated figures
├── results/                # Experiment outputs
├── data/                   # IEEE-CIS dataset (add manually)
├── scripts/                # Analysis scripts
└── requirements.txt
```

## Key Commands

| Command | Description |
|---------|-------------|
| `python -m src.run_protocol --medium` | Run main protocol (4 folds, baseline + CTGAN + TabDDPM + SMOTE) |
| `python -m src.run_protocol --quick` | Quick run (2 folds, 1 rate) |
| `python -m src.run_unified_analysis` | Merge results, drift-harm, statistical tests |
| `python -m src.paper_figures` | Generate paper figures |
| `bash scripts/run_full_analysis.sh` | Full pipeline (normalize, analyze, figures) |

## Methods

- **Baseline**: LightGBM on real data only
- **SMOTE**: Interpolation-based oversampling
- **CTGAN**: Conditional Tabular GAN (synthetic fraud)
- **TabDDPM**: Tabular diffusion model (synthetic fraud)

Evaluation uses temporal folds (train on past, validate on future), label-delay protocol, and domain-classifier drift quantification.

## Results Summary

On IEEE-CIS under temporal evaluation:
- CTGAN, TabDDPM, and SMOTE yield at most modest gains over baseline (~0–1 PR-AUC points)
- None of the comparisons are statistically significant (p ≥ 0.12)
- SMOTE is a strong baseline; recency-aware oversampling often hurts
- Higher drift correlates with synthetic oversampling degrading performance

## Citation

If you use this code, please cite:

```
When Synthetic Oversampling Helps or Hurts Rare-Event Fraud Detection Under Temporal Shift.
[Authors]. [Venue], [Year].
```

## License

MIT (or your preferred license)
