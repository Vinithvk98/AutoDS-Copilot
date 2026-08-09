"""Stage 4 — Preprocessing: analyse nulls, recommend fixes, apply approved plan.

The heavy lifting (imputation, encoding, scaling) is done inside an sklearn
Pipeline at train time to avoid data leakage. This module produces a
human-readable, editable *plan* that the user approves before training.
"""
from __future__ import annotations
import pandas as pd
from .loader import column_types
from .. import config


def recommend_plan(df: pd.DataFrame, target: str | None = None, task: dict | None = None) -> dict:
    """Return a per-column recommendation dict the user can review/override."""
    types = column_types(df)
    plan = {"drop_columns": [], "impute": {}, "notes": []}

    # fold in the data-quality guardrails: constant, ID-like, duplicate, and
    # leaking columns are pre-suggested for dropping (the user can uncheck them)
    if task is not None:
        from . import quality
        for col, reason in quality.drop_suggestions(df, task):
            if col not in plan["drop_columns"]:
                plan["drop_columns"].append(col)
                plan["notes"].append(f"{reason} → recommend dropping.")

    n = len(df)
    for col in df.columns:
        if col == target:
            continue
        miss_pct = 100 * df[col].isna().mean()
        # Very sparse columns -> suggest dropping
        if miss_pct > 60:
            plan["drop_columns"].append(col)
            plan["notes"].append(f"'{col}' is {miss_pct:.0f}% missing → recommend dropping.")
            continue
        if miss_pct > 0:
            if col in types["numeric"]:
                strategy = "median"
            else:
                strategy = "most_frequent"
            plan["impute"][col] = strategy
            plan["notes"].append(
                f"'{col}' is {miss_pct:.1f}% missing → impute with {strategy}.")

    # High-cardinality categoricals are flagged (kept, but noted)
    for col in types["categorical"]:
        if col == target:
            continue
        if df[col].nunique() > config.MAX_CATEGORICAL_CARDINALITY:
            plan["notes"].append(
                f"'{col}' has high cardinality ({df[col].nunique()} values) — "
                "consider grouping rare levels or dropping.")

    if df.duplicated().sum():
        plan["notes"].append(f"{int(df.duplicated().sum())} duplicate rows detected — "
                             "they will be removed.")
        plan["drop_duplicates"] = True
    else:
        plan["drop_duplicates"] = False

    if not plan["notes"]:
        plan["notes"].append("Data looks clean — no missing values or duplicates found.")
    return plan


def apply_plan(df: pd.DataFrame, plan: dict) -> pd.DataFrame:
    """Apply the (approved) plan's row-level operations. Column imputation for
    modelling is handled in the training Pipeline; here we only drop columns /
    duplicates so downstream EDA reflects user choices."""
    out = df.copy()
    if plan.get("drop_duplicates"):
        out = out.drop_duplicates().reset_index(drop=True)
    drop = [c for c in plan.get("drop_columns", []) if c in out.columns]
    if drop:
        out = out.drop(columns=drop)
    return out
