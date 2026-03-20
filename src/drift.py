# src/drift.py

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


DEFAULT_EXCLUDE = {
    "isFraud",
    "TransactionDT",   # trivially separates time windows
    "TransactionID",   # if present
    "__domain__",
}


def _encode_objects_to_codes(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    obj_cols = X.select_dtypes(include="object").columns
    for c in obj_cols:
        X[c] = X[c].astype("category").cat.codes
    return X


def domain_classifier_auc(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    exclude_cols=None,
    max_rows_per_domain: int = 70_000,   # matches your logs
    test_size: float = 0.3,
    random_state: int = 42,
):
    """
    Leak-free domain classifier AUC with:
    - subsampling per domain
    - holdout evaluation
    - explicit exclusion of time/ID columns
    """

    if exclude_cols is None:
        exclude_cols = set(DEFAULT_EXCLUDE)
    else:
        exclude_cols = set(exclude_cols) | set(DEFAULT_EXCLUDE)

    tr = train_df.copy()
    va = val_df.copy()
    tr["__domain__"] = 0
    va["__domain__"] = 1

    if len(tr) > max_rows_per_domain:
        tr = tr.sample(n=max_rows_per_domain, random_state=random_state)
    if len(va) > max_rows_per_domain:
        va = va.sample(n=max_rows_per_domain, random_state=random_state)

    df = pd.concat([tr, va], axis=0, ignore_index=True)

    y = df["__domain__"].astype(int).values
    X = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors="ignore")

    # Encode categoricals + float32 for memory
    X = _encode_objects_to_codes(X).fillna(-1)
    X = X.astype(np.float32, copy=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    model = lgb.LGBMClassifier(
        n_estimators=400,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        force_col_wise=True,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    preds = model.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, preds))


def single_feature_auc_scan(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    max_rows_per_domain: int = 70_000,
    random_state: int = 42,
    top_k: int = 15,
):
    """
    Finds columns that trivially separate time windows.
    Returns top_k columns by single-feature AUC.
    """

    tr = train_df.copy()
    va = val_df.copy()
    tr["__domain__"] = 0
    va["__domain__"] = 1

    if len(tr) > max_rows_per_domain:
        tr = tr.sample(n=max_rows_per_domain, random_state=random_state)
    if len(va) > max_rows_per_domain:
        va = va.sample(n=max_rows_per_domain, random_state=random_state)

    df = pd.concat([tr, va], axis=0, ignore_index=True)
    y = df["__domain__"].astype(int).values

    results = []

    for col in df.columns:
        if col in DEFAULT_EXCLUDE:
            continue

        x = df[col]

        # Encode object columns
        if x.dtype == "object":
            x = x.astype("category").cat.codes

        # Fill NaN
        x = pd.Series(x).fillna(-1).astype(np.float32).values

        # If constant, skip
        if np.all(x == x[0]):
            continue

        # AUC on same samples (single feature only; OK for leak hunting)
        try:
            auc = roc_auc_score(y, x)
            auc = max(auc, 1.0 - auc)  # symmetric
            results.append((col, float(auc)))
        except Exception:
            pass

    results.sort(key=lambda t: t[1], reverse=True)
    return results[:top_k]


def population_stability_index(train: pd.Series, test: pd.Series, bins: int = 10, eps: float = 1e-6):
    """
    PSI for continuous variables only.
    Defensive and stable.
    """
    try:
        train = train.dropna().values
        test = test.dropna().values

        if len(train) == 0 or len(test) == 0:
            return 0.0
        if np.all(train == train[0]) or np.all(test == test[0]):
            return 0.0

        cuts = np.percentile(train, np.linspace(0, 100, bins + 1))
        cuts = np.unique(cuts)
        if len(cuts) <= 2:
            return 0.0

        train_bins = pd.cut(train, cuts, include_lowest=True)
        test_bins = pd.cut(test, cuts, include_lowest=True)

        train_counts = train_bins.value_counts()
        test_counts = test_bins.value_counts()

        train_dist = train_counts / train_counts.sum()
        test_dist = test_counts / test_counts.sum()

        all_bins = train_dist.index.union(test_dist.index)
        train_dist = train_dist.reindex(all_bins, fill_value=eps)
        test_dist = test_dist.reindex(all_bins, fill_value=eps)

        psi = np.sum((train_dist - test_dist) * np.log(train_dist / test_dist))
        if np.isnan(psi) or np.isinf(psi):
            return 0.0
        return float(psi)
    except Exception:
        return 0.0

