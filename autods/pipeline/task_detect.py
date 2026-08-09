"""Stage 2 — Detect the ML task type, target column, and special columns.

Supported task types: classification, regression, clustering, timeseries, text.
"""
from __future__ import annotations
import pandas as pd
from .. import config
from .loader import column_types

_TIME_NAME_HINTS = {"date", "time", "timestamp", "datetime", "month", "day",
                    "year", "week", "ds", "period"}


def suggest_target(df: pd.DataFrame) -> str | None:
    """Heuristic: the last column is the most common target convention."""
    if df.shape[1] < 2:
        return None
    return df.columns[-1]


def detect_time_col(df: pd.DataFrame) -> str | None:
    """Find a datetime-like column: real datetime dtype, a name hint, or an
    object column that mostly parses as dates."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    for col in df.columns:
        # only parse string/object columns — never coerce integers (e.g.
        # "tenure_years") into epoch dates just because the name hints at time
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        if col.lower() in _TIME_NAME_HINTS or any(h in col.lower() for h in _TIME_NAME_HINTS):
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.8:
                    return col
            except Exception:
                continue
    return None


def detect_text_col(df: pd.DataFrame, exclude: str | None = None) -> str | None:
    """Find a free-text column: object dtype whose values are long and mostly
    unique (as opposed to a low-cardinality category)."""
    best, best_len = None, 0
    for col in df.select_dtypes(include="object").columns:
        if col == exclude:
            continue
        s = df[col].dropna().astype(str)
        if s.empty:
            continue
        avg_len = s.str.len().mean()
        avg_words = s.str.split().str.len().mean()
        # free text = long, multi-word values with reasonable variety
        if avg_len >= 20 and avg_words >= 3 and s.nunique() > 15 and avg_len > best_len:
            best, best_len = col, avg_len
    return best


def detect_task(df: pd.DataFrame, target: str | None = None) -> dict:
    """Return a rich task descriptor.

    Keys: task_type, target, reason, candidates, time_col, text_col, imbalanced.
    """
    if target and target not in df.columns:
        raise ValueError(f"Target '{target}' not in dataset")
    if target is None:
        target = suggest_target(df)

    time_col = detect_time_col(df)
    text_col = detect_text_col(df, exclude=target)

    base = {"target": target, "time_col": time_col, "text_col": text_col,
            "imbalanced": False}

    if target is None:
        return {**base, "task_type": "clustering",
                "reason": "No target column available — defaulting to unsupervised clustering.",
                "candidates": ["clustering"]}

    s = df[target]
    n_unique = s.nunique(dropna=True)
    numeric_target = pd.api.types.is_numeric_dtype(s)

    if numeric_target and n_unique > config.CLASSIFICATION_MAX_UNIQUE:
        if time_col:
            return {**base, "task_type": "timeseries",
                    "reason": (f"Numeric target '{target}' with a time column "
                               f"'{time_col}' → treating as time-series forecasting."),
                    "candidates": ["timeseries", "regression"]}
        return {**base, "task_type": "regression",
                "reason": (f"Target '{target}' is numeric with {n_unique} distinct "
                           f"values → regression."),
                "candidates": ["regression", "clustering"]}

    # classification-like target
    vc = s.value_counts(normalize=True)
    imbalanced = bool(len(vc) and vc.iloc[0] > 0.7)
    if text_col:
        return {**base, "task_type": "text", "imbalanced": imbalanced,
                "reason": (f"Target '{target}' is categorical and column '{text_col}' "
                           f"holds free text → text classification (TF-IDF)."),
                "candidates": ["text", "classification"]}
    return {**base, "task_type": "classification", "imbalanced": imbalanced,
            "reason": (f"Target '{target}' has {n_unique} distinct values → classification"
                       + (" (imbalanced)." if imbalanced else ".")),
            "candidates": ["classification", "clustering"]}
