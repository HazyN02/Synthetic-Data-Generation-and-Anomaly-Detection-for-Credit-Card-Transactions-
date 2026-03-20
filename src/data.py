import pandas as pd


def load_raw_data(data_dir="data"):
    train_trans = pd.read_csv(f"{data_dir}/train_transaction.csv")
    train_ident = pd.read_csv(f"{data_dir}/train_identity.csv")

    test_trans = pd.read_csv(f"{data_dir}/test_transaction.csv")
    test_ident = pd.read_csv(f"{data_dir}/test_identity.csv")

    return train_trans, train_ident, test_trans, test_ident


def merge_data(trans, ident):
    df = trans.merge(ident, on="TransactionID", how="left")

    id_cols = [c for c in df.columns if c.startswith("id_")]
    df["has_identity"] = (~df[id_cols].isna().all(axis=1)).astype(int)

    df["nan_count_row"] = df.isna().sum(axis=1)

    return df


def save_parquet(df, path):
    df.to_parquet(path, index=False)
