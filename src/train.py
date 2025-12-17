import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score

from src.config import TARGET_COL
from src.split import time_based_cv
from src.features import add_time_features, add_frequency_encoding


def prepare_data(df):
    y = df[TARGET_COL].values

    # IMPORTANT: avoid copying the whole merged df
    X = df.drop(columns=[TARGET_COL, "TransactionID"]).copy(deep=False)

    # Feature engineering (in-place)
    add_time_features(X)

    freq_cols = ["card1", "card2", "addr1", "P_emaildomain"]
    freq_cols = [c for c in freq_cols if c in X.columns]
    add_frequency_encoding(X, freq_cols)

    # Encode categoricals (must handle NaN for LabelEncoder)
    cat_cols = X.select_dtypes(include="object").columns
    for col in cat_cols:
        X[col] = X[col].fillna("NA")
        X[col] = LabelEncoder().fit_transform(X[col]).astype("int32")

    # DO NOT: X = X.fillna(-1)  (this is what crashed you)
    # Leave NaNs for LightGBM (it handles them)

    # Downcast numeric dtypes to reduce memory
    for col in X.columns:
        dt = X[col].dtype
        if dt == "float64":
            X[col] = X[col].astype("float32")
        elif dt == "int64":
            X[col] = X[col].astype("int32")

    return X, y


def train_baseline():
    df = pd.read_parquet("data/train_merged.parquet")
    splits = time_based_cv(df)

    pr_aucs = []
    recalls_1 = []

    for i, (tr, va) in enumerate(splits):
        print(f"\n===== Fold {i} =====")

        train_df = df.iloc[tr]
        val_df = df.iloc[va]

        X_tr, y_tr = prepare_data(train_df)
        X_va, y_va = prepare_data(val_df)

        pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            random_state=42,
            n_jobs=-1,
            force_col_wise=True
        )

        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_va)[:, 1]

        pr = average_precision_score(y_va, preds)
        pr_aucs.append(pr)

        # Recall @ 1% FPR: threshold based on negatives
        thresh = np.quantile(preds[y_va == 0], 0.99)
        recall_1 = (preds[y_va == 1] > thresh).mean()
        recalls_1.append(recall_1)

        print(f"PR-AUC: {pr:.4f}")
        print(f"Recall@1%FPR: {recall_1:.4f}")

    print("\n===== FINAL RESULTS =====")
    print(f"PR-AUC mean ± std: {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f}")
    print(f"Recall@1%FPR mean ± std: {np.mean(recalls_1):.4f} ± {np.std(recalls_1):.4f}")


if __name__ == "__main__":
    train_baseline()

