"""Stage 6 — Build a leakage-safe pipeline and train the chosen model.

Handles classification, regression, clustering, time-series forecasting
(lag features + ordered split) and text classification (TF-IDF).
"""
from __future__ import annotations
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from .. import config
from .loader import column_types
from .model_select import build_estimator, NO_K_CLUSTERERS
from . import rigor


def build_preprocessor(df: pd.DataFrame, feature_cols: list[str]) -> ColumnTransformer:
    """Impute + scale numeric columns; impute + one-hot encode categoricals."""
    types = column_types(df[feature_cols])
    numeric = types["numeric"]
    categorical = types["categorical"] + types["datetime"]
    numeric_tf = Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())])
    # dense output so every estimator works (GaussianNB / MLP need dense input)
    categorical_tf = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                               ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                        max_categories=20, sparse_output=False))])
    return ColumnTransformer([("num", numeric_tf, numeric),
                              ("cat", categorical_tf, categorical)], remainder="drop")


def _apply_balance(estimator, balanced: bool):
    """Enable class_weight='balanced' on estimators that support it."""
    if balanced and "class_weight" in estimator.get_params():
        estimator.set_params(class_weight="balanced")
    return estimator


def make_timeseries_features(df: pd.DataFrame, target: str, time_col: str,
                             n_lags: int = 3) -> tuple[pd.DataFrame, list[str]]:
    """Sort by time and build lag + rolling-mean features for forecasting."""
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col, target]).sort_values(time_col).reset_index(drop=True)
    feat_cols = [c for c in column_types(d)["numeric"] if c != target]  # exogenous numerics
    # calendar features so the model can learn weekly / monthly seasonality
    d["cal_dayofweek"] = d[time_col].dt.dayofweek
    d["cal_day"] = d[time_col].dt.day
    d["cal_month"] = d[time_col].dt.month
    feat_cols += ["cal_dayofweek", "cal_day", "cal_month"]
    for k in range(1, n_lags + 1):
        d[f"{target}_lag{k}"] = d[target].shift(k)
        feat_cols.append(f"{target}_lag{k}")
    d[f"{target}_roll{n_lags}"] = d[target].shift(1).rolling(n_lags).mean()
    feat_cols.append(f"{target}_roll{n_lags}")
    d = d.dropna(subset=feat_cols).reset_index(drop=True)
    return d, feat_cols


def timeseries_fit(d: pd.DataFrame, feat_cols: list[str], target: str,
                   estimator, split: int):
    """Fit on the *differenced* target (y - lag1) for stationarity so any
    regressor — including trees, which can't extrapolate a trend — works, then
    reconstruct level predictions. Returns (fitted_pipe, y_true, y_pred)."""
    import numpy as np
    lag1 = d[f"{target}_lag1"]
    y = d[target]
    delta = y - lag1
    X_tr = d[feat_cols].iloc[:split]
    pipe = Pipeline([("pre", build_preprocessor(d, feat_cols)), ("model", estimator)])
    pipe.fit(X_tr, delta.iloc[:split])
    pred_delta = pipe.predict(d[feat_cols].iloc[split:])
    y_pred = lag1.iloc[split:].to_numpy() + pred_delta
    y_true = y.iloc[split:].to_numpy()
    return pipe, y_true, y_pred


def balance_preview(df: pd.DataFrame, target: str, strategy: str) -> dict:
    """Show the effect of the imbalance strategy without training a model.

    Splits exactly as training does, then reports the class counts of the training
    split before and after resampling, plus the untouched test split. This is the
    proof that balancing happened on train only and the test set keeps its real
    ratio. Returns {strategy, before, after, test, changed}.
    """
    def counts(s):
        vc = pd.Series(list(s)).value_counts()
        return {str(k): int(v) for k, v in vc.items()}

    d = df.dropna(subset=[target]).reset_index(drop=True)
    feature_cols = [c for c in d.columns if c != target]
    X, y = d[feature_cols], d[target]
    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=stratify)

    before, test = counts(y_train), counts(y_test)
    after, used = dict(before), strategy
    try:
        if strategy == "smote" and rigor.smote_available():
            from imblearn.over_sampling import SMOTE
            Xt = build_preprocessor(d, feature_cols).fit_transform(X_train)
            minority = int(pd.Series(list(y_train)).value_counts().min())
            k = max(1, min(5, minority - 1))
            _, y_res = SMOTE(random_state=config.RANDOM_STATE, k_neighbors=k).fit_resample(Xt, y_train)
            after, used = counts(y_res), "smote"
        elif strategy == "oversample" or (strategy == "smote" and not rigor.smote_available()):
            _, y_res = rigor.oversample(X_train, y_train)
            after, used = counts(y_res), "oversample"
        # none / class_weight leave the rows unchanged (after == before)
    except Exception:
        _, y_res = rigor.oversample(X_train, y_train)
        after, used = counts(y_res), "oversample"

    return {"strategy": used, "before": before, "after": after, "test": test,
            "changed": after != before}


def _fit_supervised(df, feature_cols, X_train, y_train, est, task_type, strategy):
    """Fit a classification/regression pipeline, applying the resampling strategy
    on the training split only. Returns (pipeline, strategy_actually_used)."""
    pre = build_preprocessor(df, feature_cols)
    if task_type == "classification" and strategy == "smote":
        if rigor.smote_available():
            try:
                from imblearn.pipeline import Pipeline as ImbPipeline
                from imblearn.over_sampling import SMOTE
                minority = int(pd.Series(list(y_train)).value_counts().min())
                k = max(1, min(5, minority - 1))
                pipe = ImbPipeline([("pre", pre),
                                    ("smote", SMOTE(random_state=config.RANDOM_STATE, k_neighbors=k)),
                                    ("model", est)])
                pipe.fit(X_train, y_train)
                return pipe, "smote"
            except Exception:
                pass
        strategy = "oversample"  # imblearn missing or SMOTE failed
    if task_type == "classification" and strategy == "oversample":
        X_res, y_res = rigor.oversample(X_train, y_train)
        pipe = Pipeline([("pre", pre), ("model", est)])
        pipe.fit(X_res, y_res)
        return pipe, "oversample"
    # none, class_weight (already set on est), or regression
    pipe = Pipeline([("pre", pre), ("model", est)])
    pipe.fit(X_train, y_train)
    return pipe, strategy


def train_model(df: pd.DataFrame, task_type: str, model_name: str,
                target: str | None = None, time_col: str | None = None,
                text_col: str | None = None, balanced: bool = False,
                n_clusters: int = 3, params: dict | None = None,
                estimator=None, balance: str | None = None) -> dict:
    """Train and return everything the evaluation stage needs.

    `params` optionally overrides the model's hyperparameters (used after tuning).
    `estimator` optionally supplies a fully-built estimator (used for ensembles),
    bypassing the registry lookup, balancing, and params.
    `balance` picks the imbalance strategy for classification: none | class_weight |
    oversample | smote. Left None, it falls back to class_weight when `balanced`.
    """
    def _apply_params(est):
        if params:
            est.set_params(**params)
        return est

    if task_type == "clustering":
        feature_cols = [c for c in df.columns if c != target]
        pre = build_preprocessor(df, feature_cols)
        X = pre.fit_transform(df[feature_cols])
        est = build_estimator("clustering", model_name,
                              **({} if model_name in NO_K_CLUSTERERS else {"k": n_clusters}))
        labels = est.fit_predict(X)
        return {"task_type": task_type, "model_name": model_name, "estimator": est,
                "preprocessor": pre, "X": X, "labels": labels, "feature_cols": feature_cols}

    if task_type == "text":
        df = df.dropna(subset=[target]).reset_index(drop=True)
        X, y = df[[text_col]].astype(str), df[target]
        strat = y if y.nunique() > 1 else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=strat)
        pre = ColumnTransformer([("tfidf", TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), stop_words="english"), text_col)])
        est = estimator or _apply_params(_apply_balance(build_estimator("classification", model_name), balanced))
        pipe = Pipeline([("pre", pre), ("model", est)])
        pipe.fit(X_tr, y_tr)
        return {"task_type": "text", "model_name": model_name, "pipeline": pipe,
                "X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te,
                "feature_cols": [text_col], "target": target}

    if task_type == "timeseries":
        d, feat_cols = make_timeseries_features(df, target, time_col)
        split = int(len(d) * (1 - config.TEST_SIZE))
        est = estimator or _apply_params(build_estimator("timeseries", model_name))
        pipe, y_true, y_pred = timeseries_fit(d, feat_cols, target, est, split)
        return {"task_type": "timeseries", "model_name": model_name, "pipeline": pipe,
                "X_train": d[feat_cols].iloc[:split], "X_test": d[feat_cols].iloc[split:],
                "y_test": y_true, "y_pred": y_pred,
                "time_index": pd.to_datetime(d[time_col]).iloc[split:].astype(str).tolist(),
                "feature_cols": feat_cols, "target": target}

    # classification / regression
    df = df.dropna(subset=[target]).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c != target]
    X, y = df[feature_cols], df[target]
    stratify = y if task_type == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=stratify)
    strategy = "none"
    if estimator is not None:
        est = estimator
        pipe = Pipeline([("pre", build_preprocessor(df, feature_cols)), ("model", est)])
        pipe.fit(X_train, y_train)
    else:
        est = build_estimator(task_type, model_name)
        if task_type == "classification":
            strategy = rigor.resolve_balance(balanced, balance)
            if strategy == "class_weight":
                est = _apply_balance(est, True)
        est = _apply_params(est)
        pipe, strategy = _fit_supervised(df, feature_cols, X_train, y_train,
                                         est, task_type, strategy)
    return {"task_type": task_type, "model_name": model_name, "pipeline": pipe,
            "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
            "feature_cols": feature_cols, "target": target, "balance": strategy}
