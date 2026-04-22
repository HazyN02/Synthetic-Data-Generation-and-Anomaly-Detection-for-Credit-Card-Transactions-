# Fidelity or Illusion: Why Synthetic Oversampling Benchmarks for Fraud Detection Cannot Be Trusted

**Nigamanth Rajagopalan · Yasasvi Kaipa**  
University of Florida, Department of Computer Science and Informatics

*Target venue: IEEE Transactions on Information Forensics and Security (T-IFS)*

---

## Abstract

Synthetic oversampling is routinely benchmarked on imbalanced fraud datasets using random train/test splits and aggregate fidelity metrics — evaluation choices that mask the gap between synthetic data quality and downstream utility. We expose this gap empirically on the IEEE-CIS benchmark (~590k transactions, 3.5% fraud) using a leakage-safe 8-fold expanding-window temporal protocol and LightGBM as a fixed downstream classifier, with PR-AUC as the primary metric. We evaluate SMOTE, CTGAN (150 epochs), TVAE (300 epochs), and TabDDPM (50 epochs) against a class-weighted real-data baseline. CTGAN achieves the only statistically significant improvement (Δ PR-AUC = +0.0046, p = 0.045), while SMOTE, TVAE, and TabDDPM do not improve significantly over baseline. Critically, all deep generators exhibit high Distance to Closest Record (DCR) scores — TabDDPM's DCR is on the order of 10^18 — indicating that "statistically similar" synthesis and genuine fraud-space coverage are not the same thing. An HDBSCAN cluster analysis of the fraud feature space reveals the mechanistic cause: 54.8% of fraud transactions (the noise bucket and C0) receive zero synthetic allocation across all 8 folds, and the cluster with the highest DCR (C2, DCR = 17,892) shows the worst CTGAN PR-AUC degradation (Δ = −0.0052). Our contribution is not a new generator; it is a measurement framework — temporal folds, per-class fidelity diagnostics, and cluster-level routing analysis — that exposes when and why synthetic oversampling benchmarks cannot be trusted.

---

## Key Results

| Method | Architecture | Epochs | Mean PR-AUC | Δ vs Baseline | p-value | Significant | Mean DCR |
|--------|-------------|--------|-------------|---------------|---------|-------------|----------|
| Baseline | Real only | — | 0.5786 | — | — | — | — |
| CTGAN | GAN | 150 | 0.5838 | +0.0046 | 0.045 | ✅ Yes | Moderate |
| SMOTE | Interpolation | — | 0.5816 | +0.0030 | 0.41 | ❌ No | N/A |
| TVAE | VAE | 300 | 0.5797 | +0.0012 | 0.55 | ❌ No | ~2,570 |
| TabDDPM | Diffusion | 50 | 0.5776 | −0.0010 | 0.46 | ❌ No | ~10^18 |
| CTGAN p90 | GAN (gated) | 150 | 0.5824 | +0.0038 | — | ❌ No | — |

*Evaluation: 8-fold expanding-window temporal cross-validation on IEEE-CIS. Significance: two-sided sign test.*

---

## HDBSCAN Cluster Analysis

HDBSCAN partitions 20,663 fraud transactions into 4 dense clusters plus a 39.8% noise bucket of structurally isolated fraud. CTGAN routes zero synthetic samples to the noise bucket and C0 — **54.8% of fraud receives no synthetic allocation**.

| Cluster | Size | Share | Synth/fold | Mean DCR | Baseline PR-AUC | CTGAN Δ |
|---------|------|-------|-----------|---------|-----------------|---------|
| −1 Noise | 8,231 | 39.8% | 0 | — | 0.391 | −0.0027 |
| C0 Dense | 3,091 | 15.0% | 0 | — | 0.464 | −0.0038 |
| C1 Dense | 2,497 | 12.1% | 4,926 | 11,791 | 0.034 | +0.0017 |
| C2 Dense | 1,657 | 8.0% | 2,701 | 17,892 | 0.060 | −0.0052 |
| C3 Dense | 5,187 | 25.1% | 8,613 | 12,012 | 0.651 | ≈ 0.000 |

**Mechanistic finding:** CTGAN concentrates synthesis in the three densest clusters but still produces high-DCR samples there. Cluster C2, with the highest DCR (17,892), shows the worst downstream degradation. Low-density and structurally isolated fraud is simply not learned by CTGAN at all.

---

## Repository Structure

```
fraud-synth-icml/
├── src/
│   ├── run_protocol.py                    # Main 8-fold protocol (baseline + CTGAN + SMOTE)
│   ├── run_tvae_protocol.py               # TVAE 8-fold runner
│   ├── run_smote_tabddpm_protocol.py      # SMOTE + TabDDPM 8-fold runner
│   ├── run_ctgan_fidelity_gated_protocol.py  # CTGAN with DCR gating
│   ├── run_cluster_analysis_hdbscan.py    # HDBSCAN fraud-cluster analysis
│   ├── run_fidelity_analysis.py           # DCR / NNDR fidelity metrics
│   ├── resume_cluster_analysis.py         # Checkpoint-aware resume for cluster runs
│   ├── synth_ctgan.py / synth_tvae.py / synth_smote.py / synth_tabddpm.py
│   ├── train.py                           # LightGBM training + evaluation
│   ├── fidelity/                          # DCR, NNDR metric implementations
│   └── ...
├── fraud_cluster_analysis.py              # Core clustering helpers (imported by runners)
├── paper/
│   ├── main_doubleblind.tex               # Double-blind IEEE submission
│   ├── main_named.tex                     # Named version
│   ├── FULL_PAPER_DRAFT.md
│   ├── tables/                            # Generated result tables
│   │   └── fidelity/                      # Per-method fidelity tables
│   └── figures/                           # PR-AUC plots, drift analysis
├── results/
│   └── protocol/
│       ├── run_20260330_180216/           # CANONICAL 8-fold run
│       │   ├── results.csv                # Per-fold per-method PR-AUC
│       │   ├── results_tvae.csv
│       │   ├── results_smote_tabddpm.csv
│       │   ├── cluster_summary_hdbscan.csv   # Final cluster table
│       │   ├── cluster_per_fold_hdbscan.csv  # Per-fold cluster results
│       │   └── fidelity_*.csv             # Per-method DCR/NNDR
│       ├── significance_tests_8fold.csv   # Statistical test results
│       ├── literature_audit.csv           # 12-paper audit table
│       └── FROZEN/                        # Frozen canonical snapshot
├── configs/fidelity/                      # CTGAN/TabDDPM fidelity grid configs
├── data/                                  # IEEE-CIS data (not tracked — see below)
│   └── README.md                          # Kaggle download instructions
├── scripts/
│   └── run_full_analysis.sh
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/HazyN02/Synthetic-Data-Generation-and-Anomaly-Detection-for-Credit-Card-Transactions-.git
cd Synthetic-Data-Generation-and-Anomaly-Detection-for-Credit-Card-Transactions-
pip install -r requirements.txt

# 2. Download IEEE-CIS data from Kaggle
#    https://www.kaggle.com/c/ieee-fraud-detection
#    Place train_transaction.csv + train_identity.csv in data/

# 3. Merge transaction + identity tables
python -m src.run_merge

# 4. Run main 8-fold protocol
python -u -m src.run_protocol --run-id main_8fold --n-folds 8
```

---

## Key Commands

| Command | Description |
|---------|-------------|
| `python -u -m src.run_protocol --n-folds 8` | Main protocol: baseline + CTGAN (8 expanding-window folds) |
| `python -u -m src.run_tvae_protocol --n-folds 8` | TVAE 8-fold run |
| `python -u -m src.run_smote_tabddpm_protocol --n-folds 8` | SMOTE + TabDDPM 8-fold run |
| `python -u -m src.run_ctgan_fidelity_gated_protocol` | CTGAN with DCR-based fidelity gating |
| `python -u -m src.run_cluster_analysis_hdbscan --n-folds 8 --ctgan-epochs 150 --hdbscan-min-cluster-size 500` | HDBSCAN cluster analysis |
| `python -u -m src.run_fidelity_analysis` | Compute DCR / NNDR for all methods |
| `python -u src/resume_cluster_analysis.py --dry-run` | Verify cluster checkpoint |
| `python -u src/resume_cluster_analysis.py --summary-only` | Rebuild cluster summary from completed folds |
| `python -m src.paper_figures` | Regenerate paper figures |

---

## Methodology

### Temporal Evaluation Protocol
Training and validation always respect time order. 8 expanding-window folds: each fold trains on all data up to a cutpoint and validates on the immediately following block. No future information leaks into training. This is in contrast to the random 60/20/20 splits used by most prior tabular synthesis work (see `results/protocol/literature_audit.csv`).

### Fidelity Metrics: DCR and NNDR
- **DCR (Distance to Closest Record):** For each synthetic sample, the L2 distance to the nearest real training sample. High DCR = synthetic data is far from real fraud = poor coverage.
- **NNDR (Nearest-Neighbour Distance Ratio):** Ratio of distance to closest real sample vs. closest other synthetic sample. Values close to 1 indicate synthetic samples cluster among themselves rather than covering real fraud space.
- **Per-class fidelity** is computed on the fraud class only — not the full table. Prior work (CTGAN, TabDDPM, CTAB-GAN) reports aggregate column statistics; per-class DCR/NNDR is the critical missing diagnostic.

### HDBSCAN Fraud Clustering
HDBSCAN (min_cluster_size=500, seed=42) is fit on the fraud-only feature space (standardised + ordinal-encoded). Degeneracy check: falls back to KMeans(k=5) if n_clusters ≤ 1, noise fraction > 0.5, or largest cluster > 95%. Synthetic samples are routed to clusters via nearest-centroid assignment for per-cluster PR-AUC and DCR measurement.

---

## Reproducing Results

All canonical results are in `results/protocol/run_20260330_180216/`. Steps to reproduce from scratch:

```bash
# Step 1: Baseline + CTGAN (main protocol)
python -u -m src.run_protocol --n-folds 8 --ctgan-epochs 150

# Step 2: TVAE
python -u -m src.run_tvae_protocol --n-folds 8 --tvae-epochs 300

# Step 3: SMOTE + TabDDPM
python -u -m src.run_smote_tabddpm_protocol --n-folds 8 --tabddpm-steps 50

# Step 4: CTGAN with fidelity gating (DCR p90 threshold)
python -u -m src.run_ctgan_fidelity_gated_protocol --n-folds 8

# Step 5: Fidelity analysis (DCR / NNDR per method)
python -u -m src.run_fidelity_analysis

# Step 6: HDBSCAN cluster analysis
python -u -m src.run_cluster_analysis_hdbscan --n-folds 8 --ctgan-epochs 150 --hdbscan-min-cluster-size 500

# Step 7: Significance tests (reads from results CSVs; no retraining)
python -c "from src.statistical_tests import run_tests; run_tests()"

# Step 8: Paper figures
python -m src.paper_figures
```

Expected runtime per fold: ~10–50 min depending on method and hardware (tested on Windows, Python 3.11, 32 GB RAM). HDBSCAN full run: ~8–10 h for 8 folds.

---

## Citation

```bibtex
@article{rajagopalan2026fidelity,
  title     = {Fidelity or Illusion: Why Synthetic Oversampling Benchmarks for
               Fraud Detection Cannot Be Trusted},
  author    = {Rajagopalan, Nigamanth and Kaipa, Yasasvi},
  journal   = {IEEE Transactions on Information Forensics and Security},
  year      = {2026},
  note      = {Under review},
  institution = {University of Florida, Department of Computer Science and Informatics}
}
```

---

## License

MIT
