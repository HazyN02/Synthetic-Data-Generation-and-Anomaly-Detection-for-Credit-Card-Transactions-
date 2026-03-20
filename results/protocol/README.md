# Protocol Results

## Run modes (time estimates)

| Mode | Command | Folds | Target rates | Est. runtime |
|------|---------|-------|--------------|--------------|
| `--quick` | `python -m src.run_protocol --quick` | 2 | [0.05] | ~50 min |
| `--medium` | `python -m src.run_protocol --medium` | 4 | [0.05, 0.10, 0.20] | ~4–6 h |
| `--full` | `python -m src.run_protocol --full` | 4 | [0.05, 0.10, 0.20] | ~10 h |

## Reproducibility

For `--medium` and `--full`, each run creates a timestamped directory:

```
results/protocol/run_YYYYMMDD_HHMMSS/
├── config.json     # Full config (seeds, epochs, command)
├── results.csv     # This run's results only
└── runtime.txt     # Estimated + actual elapsed time
```

Before each run, `results/protocol/results.csv` is backed up to  
`results_backup_YYYYMMDD_HHMMSS.csv`.

To reproduce a run, use the same command and data. All random seeds and hyperparameters are in `config.json`.
