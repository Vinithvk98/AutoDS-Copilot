"""Data quality guardrails.

Runs before modelling and flags the silent mistakes that quietly ruin a model:
target leakage, constant columns, ID-like columns, duplicate columns, very high
cardinality, and heavy outliers. Findings are shown to the user, and the risky
structural ones (leakage, constant, ID, duplicate) are suggested for dropping.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import config


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def run_checks(df: pd.DataFrame, task: dict) -> list[dict]:
    """Return a list of findings: {level, title, detail, column, suggest_drop}.
    level is one of 'risk', 'warn', 'info'."""
    findings: list[dict] = []
    n = len(df)
    if n == 0:
        return findings
    target = task.get("target")
    ttype = task.get("task_type")
    features = [c for c in df.columns if c != target]

    # --- structural checks per column ---
    seen_signatures: dict[tuple, str] = {}
    for c in features:
        s = df[c]
        nunique = int(s.nunique(dropna=True))

        if nunique <= 1:
            findings.append({"level": "warn", "column": c, "suggest_drop": True,
                             "title": f"'{c}' is constant",
                             "detail": "It holds a single value, so it carries no "
                                       "information and only adds noise."})
            continue

        if nunique == n and not _is_numeric(s):
            findings.append({"level": "warn", "column": c, "suggest_drop": True,
                             "title": f"'{c}' looks like an identifier",
                             "detail": "Every row has a different value, which is "
                                       "usually an ID and lets a model memorise rows "
                                       "instead of learning."})
            continue

        # duplicate columns (identical values)
        try:
            sig = tuple(pd.util.hash_pandas_object(s, index=False).values.tolist())
            if sig in seen_signatures:
                findings.append({"level": "warn", "column": c, "suggest_drop": True,
                                 "title": f"'{c}' duplicates '{seen_signatures[sig]}'",
                                 "detail": "The two columns are identical, so one is "
                                           "redundant."})
                continue
            seen_signatures[sig] = c
        except Exception:
            pass

        # high cardinality category
        if not _is_numeric(s) and nunique > config.MAX_CATEGORICAL_CARDINALITY:
            findings.append({"level": "info", "column": c, "suggest_drop": False,
                             "title": f"'{c}' has many categories",
                             "detail": f"{nunique} distinct values. Consider grouping "
                                       "rare ones, or it may be free text."})

    # --- target leakage ---
    if target and target in df.columns and ttype in ("classification", "regression"):
        tser = df[target]
        for c in features:
            s = df[c]
            if _is_numeric(s) and _is_numeric(tser) and s.nunique() > 2:
                try:
                    r = float(pd.Series(s).corr(tser))
                except Exception:
                    r = 0.0
                if abs(r) >= 0.98:
                    findings.append({"level": "risk", "column": c, "suggest_drop": True,
                                     "title": f"'{c}' may leak the target",
                                     "detail": f"It moves almost perfectly with "
                                               f"{target} (correlation {r:.2f}). If it "
                                               "would not be known before the outcome, "
                                               "it should be dropped."})
            elif not _is_numeric(s) and 1 < s.nunique() <= 50 and s.nunique() < n:
                # every category maps to a single target value = perfect predictor
                try:
                    pure = df.groupby(c, observed=True)[target].nunique(dropna=True)
                    if len(pure) > 1 and (pure <= 1).all():
                        findings.append({"level": "risk", "column": c, "suggest_drop": True,
                                         "title": f"'{c}' may leak the target",
                                         "detail": "Each of its values maps to a single "
                                                   f"{target}, which is too good to be "
                                                   "true and usually means leakage."})
                except Exception:
                    pass

    # --- outliers (report the worst offender only, informational) ---
    worst = None
    for c in [f for f in features if _is_numeric(df[f]) and df[f].nunique() > 5]:
        col = df[c].dropna()
        if col.empty:
            continue
        q1, q3 = col.quantile(0.25), col.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        frac = float(((col < q1 - 1.5 * iqr) | (col > q3 + 1.5 * iqr)).mean())
        if frac > 0.05 and (worst is None or frac > worst[1]):
            worst = (c, frac)
    if worst:
        findings.append({"level": "info", "column": worst[0], "suggest_drop": False,
                         "title": f"'{worst[0]}' has notable outliers",
                         "detail": f"About {round(worst[1] * 100)} percent of its "
                                   "values sit far from the rest. Worth a look before "
                                   "trusting averages."})

    # rank risks first
    order = {"risk": 0, "warn": 1, "info": 2}
    findings.sort(key=lambda f: order.get(f["level"], 3))
    return findings


def drop_suggestions(df: pd.DataFrame, task: dict) -> list[tuple[str, str]]:
    """Columns worth dropping, from the checks, as (column, reason)."""
    out = []
    for f in run_checks(df, task):
        if f.get("suggest_drop"):
            out.append((f["column"], f["title"]))
    return out
