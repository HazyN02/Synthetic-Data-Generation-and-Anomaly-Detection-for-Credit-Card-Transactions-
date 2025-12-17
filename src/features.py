def add_time_features(df):
    # mutate in place (no df.copy())
    df["hour"] = (df["TransactionDT"] // 3600) % 24
    df["day"] = df["TransactionDT"] // 86400
    df["weekday"] = df["day"] % 7
    return df


def add_frequency_encoding(df, cols):
    # mutate in place (no df.copy())
    for col in cols:
        freq = df[col].value_counts()
        df[f"{col}_freq"] = df[col].map(freq).fillna(0).astype("int32")
    return df
