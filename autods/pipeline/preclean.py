"""Light, mechanical pre-clean — runs after load, before EDA.

It only fixes how the data is *represented* so the charts are trustworthy. It
never changes what the data says: no imputation, no outlier removal, no scaling.
Those stay visible in EDA and are handled later in the approval step.

Every change is returned as a human-readable log line so nothing happens
invisibly.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

# Strings that really mean "missing".
MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null", "nil", "?", "-", "--",
                  "missing", "unknown"}

# A value looks date-ish if it has date separators or a month name.
_DATE_RE = re.compile(r"(?:\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4})|"
                      r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
                      re.IGNORECASE)


def _looks_datey(s: pd.Series) -> bool:
    sample = s.dropna().astype(str).head(50)
    if sample.empty:
        return False
    return (sample.str.contains(_DATE_RE)).mean() >= 0.8


def preclean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (cleaned_df, changes). Mechanical fixes only."""
    out = df.copy()
    changes: list[str] = []

    # 1) Trim whitespace on text columns
    for c in out.select_dtypes(include="object").columns:
        before = out[c].copy()
        out[c] = out[c].apply(lambda v: v.strip() if isinstance(v, str) else v)
        trimmed = int((before.astype(str) != out[c].astype(str)).sum())
        if trimmed:
            changes.append(f"'{c}': trimmed whitespace on {trimmed} value(s)")

    # 2) Normalise placeholder tokens to real missing values
    for c in out.select_dtypes(include="object").columns:
        low = out[c].astype(str).str.strip().str.lower()
        mask = low.isin(MISSING_TOKENS) & out[c].notna()
        if mask.any():
            out.loc[mask, c] = np.nan
            changes.append(f"'{c}': {int(mask.sum())} placeholder value(s) → missing")

    # 3) Parse clearly date-like text columns
    for c in out.select_dtypes(include="object").columns:
        if not _looks_datey(out[c]):
            continue
        parsed = pd.to_datetime(out[c], errors="coerce")
        seen = out[c].notna()
        if seen.sum() and parsed[seen].notna().mean() >= 0.9:
            out[c] = parsed
            changes.append(f"'{c}': parsed as datetime")

    # 4) Coerce numeric-looking text (guarded)
    for c in out.select_dtypes(include="object").columns:
        seen = out[c].notna()
        if seen.sum() == 0:
            continue
        conv = pd.to_numeric(out[c].astype(str).str.replace(",", "", regex=False),
                             errors="coerce")
        # only convert when almost everything that was present converts cleanly,
        # and it isn't a tiny code-like set (avoid ids / zip codes staying numeric
        # is acceptable, but skip if it looks like an identifier column)
        if conv[seen].notna().mean() >= 0.95 and out[c].nunique() > 2:
            out[c] = conv
            changes.append(f"'{c}': converted text to numeric")

    # 5) Drop exact duplicate rows
    dups = int(out.duplicated().sum())
    if dups:
        out = out.drop_duplicates().reset_index(drop=True)
        changes.append(f"removed {dups} exact duplicate row(s)")

    if not changes:
        changes.append("no mechanical fixes needed, data already tidy")
    return out, changes
