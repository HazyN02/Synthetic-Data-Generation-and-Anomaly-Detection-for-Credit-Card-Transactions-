from src.data import load_raw_data, merge_data, save_parquet


def main():
    print("Starting merge...")

    train_trans, train_ident, test_trans, test_ident = load_raw_data()

    print("Raw train transaction shape:", train_trans.shape)
    print("Raw train identity shape:", train_ident.shape)
    print("Raw test transaction shape:", test_trans.shape)
    print("Raw test identity shape:", test_ident.shape)

    train_df = merge_data(train_trans, train_ident)
    test_df = merge_data(test_trans, test_ident)

    print("Merged train shape:", train_df.shape)
    print("Merged test shape:", test_df.shape)

    save_parquet(train_df, "data/train_merged.parquet")
    save_parquet(test_df, "data/test_merged.parquet")

    print("Parquet files saved successfully.")
    print("Merge complete.")


if __name__ == "__main__":
    main()
