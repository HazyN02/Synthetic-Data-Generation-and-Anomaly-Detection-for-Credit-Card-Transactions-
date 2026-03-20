# Data Directory

Place the **IEEE-CIS Fraud Detection** dataset here.

## How to obtain

1. Go to [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
2. Download the competition data (requires Kaggle account)
3. Place these files in this directory:
   - `train_transaction.csv`
   - `train_identity.csv`
   - `test_transaction.csv` (optional, for full pipeline)

## Preprocessing

To create `train_merged.parquet` (merged transaction + identity):

```bash
python -m src.run_merge
```

Or use the merge logic in your preprocessing pipeline. The protocol uses `train_merged.parquet` or falls back to `train_transaction.csv`.
