# Second Dataset: Extending the Protocol to Other Fraud Datasets

To strengthen claims, run the protocol on a second public fraud dataset with timestamps.

## Recommended Sources

### Fraud Dataset Benchmark (FDB)

Amazon Science’s [Fraud Dataset Benchmark](https://github.com/amazon-science/fraud-dataset-benchmark) aggregates multiple public fraud datasets:

- **Credit Card Fraud Detection** – 227K train, 57K test, 28 features  
- **Fraud ecommerce** – 121K train, 30K test, 6 features  
- **IEEE-CIS Fraud Detection** – already used (561K train, 29K test)

Use FDB’s Python loaders for consistent train/test splits and evaluation.

### Other Public Datasets

- **Kaggle Credit Card Fraud** – [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) – 284K rows, 31 features; add a synthetic timestamp if needed.
- **Synthetic Financial Datasets** – e.g. [Synthetic Financial Datasets for Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1) – includes `step` (time) and `type`.

## Requirements

The protocol expects:

1. **Time column** – for temporal folds (e.g. `TransactionDT`, `step`, `timestamp`)
2. **Binary target** – fraud (1) vs non-fraud (0)
3. **Feature matrix** – numerical and categorical columns compatible with LightGBM

## Integration Steps

1. **Data loader** – Implement a loader in `src/data.py` that returns a DataFrame with:
   - `time_col` (int or datetime)
   - `target_col` (0/1)
   - Feature columns aligned with IEEE-CIS schema where possible

2. **Config** – Add a dataset switch in `src/config.py` or `src/run_protocol.py`:
   ```python
   DATASET = os.environ.get("FRAUD_DATASET", "ieee_cis")
   ```

3. **Folds** – Reuse `src/folds.py` (temporal split); ensure `time_col` matches.

4. **Run** – `python -m src.run_protocol --medium` with `FRAUD_DATASET=credit_card` (or similar).

## Partial Results

Even partial results (e.g. 2 folds, 1 target_pos_rate) on a second dataset help validate that conclusions generalize beyond IEEE-CIS.
