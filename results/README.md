# Results Directory

## Layout

| Path | Contents |
|------|----------|
| `protocol/` | Baseline, CTGAN, TabDDPM (main protocol) |
| `smote/` | Baseline vs SMOTE oversampling |
| `sliding_window/` | Static vs sliding retraining |
| `SUMMARY.md` | **Human-readable summary** — start here |
| `aggregate_summary.csv` | All experiments in one CSV |

## Quick Commands

```bash
# Run full protocol (baseline + CTGAN + TabDDPM)
python3 -m src.run_protocol

# Run quick protocol (~50 min: 2 folds, 1 rate, faster models)
python3 -m src.run_protocol --quick

# Run SMOTE baseline
python3 -m src.run_smote_baseline

# Run sliding window comparison
python3 -m src.run_sliding_window

# Regenerate SUMMARY.md from existing results
python3 -m src.run_aggregate_results
```
