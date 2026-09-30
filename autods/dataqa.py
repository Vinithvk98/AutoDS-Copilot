"""Safe, vetted analysis of the loaded dataset for the conversational copilot.

The copilot answers questions about the *actual data* by routing a question to
one of these whitelisted operations, never by executing model-written code. Each
tool validates its arguments against the real columns, runs a single pandas
operation, and returns a compact, JSON-safe result carrying the real numbers.
Anything invalid returns an ``error`` the copilot can relay honestly.
"""
from __future__ import annotations
import pandas as pd

# Aggregations a question is allowed to request.
_AGGS = {"mean", "median", "sum", "count", "min", "max", "std"}
_MAX_GROUPS = 25   # cap rows returned from a grouping so answers stay compact
_MAX_LEVELS = 25   # cap distinct values reported for a category


def _num(v):
    """Coerce a pandas/numpy scalar to a JSON-safe rounded native number."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, 4)


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def schema(df: pd.DataFrame) -> dict:
    """A compact description of the dataset for the router to reason over."""
    cols = []
    for c in df.columns:
        s = df[c]
        numeric = pd.api.types.is_numeric_dtype(s)
        info = {"name": c, "type": "numeric" if numeric else "categorical"}
        if numeric:
            info["min"], info["max"] = _num(s.min()), _num(s.max())
        else:
            vals = [str(x) for x in s.dropna().unique()[:8]]
            info["examples"] = vals
        cols.append(info)
    return {"n_rows": int(len(df)), "columns": cols}


# ---------------------------------------------------------------------------
#  Tools — each returns {"summary": str, "data": ...} or {"error": str}
# ---------------------------------------------------------------------------
def overview(df: pd.DataFrame) -> dict:
    num = _numeric_cols(df)
    return {"summary": f"{len(df)} rows and {len(df.columns)} columns.",
            "data": {"rows": int(len(df)), "columns": list(df.columns),
                     "numeric": num,
                     "categorical": [c for c in df.columns if c not in num]}}


def column_summary(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        return {"error": f"There is no column named '{column}'."}
    s = df[column]
    miss = int(s.isna().sum())
    if pd.api.types.is_numeric_dtype(s):
        d = {"count": int(s.count()), "missing": miss, "mean": _num(s.mean()),
             "median": _num(s.median()), "std": _num(s.std()),
             "min": _num(s.min()), "max": _num(s.max())}
        return {"summary": f"'{column}' is numeric with mean {d['mean']} and "
                           f"range {d['min']} to {d['max']}.", "data": d}
    vc = s.value_counts().head(_MAX_LEVELS)
    data = {str(k): int(v) for k, v in vc.items()}
    return {"summary": f"'{column}' is categorical with {s.nunique()} distinct values.",
            "data": {"missing": miss, "top_values": data}}


def value_counts(df: pd.DataFrame, column: str, normalize: bool = False) -> dict:
    if column not in df.columns:
        return {"error": f"There is no column named '{column}'."}
    vc = df[column].value_counts(normalize=normalize).head(_MAX_LEVELS)
    data = {str(k): (_num(v) if normalize else int(v)) for k, v in vc.items()}
    unit = "share" if normalize else "count"
    return {"summary": f"Distribution of '{column}' by {unit}.", "data": data}


def group_stat(df: pd.DataFrame, by: str, column: str, agg: str = "mean") -> dict:
    if by not in df.columns:
        return {"error": f"There is no column named '{by}'."}
    if column not in df.columns:
        return {"error": f"There is no column named '{column}'."}
    if agg not in _AGGS:
        return {"error": f"Unsupported aggregation '{agg}'."}
    if agg != "count" and not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"'{column}' is not numeric, so '{agg}' does not apply. "
                         "Try count, or a numeric column."}
    if df[by].nunique() > _MAX_GROUPS:
        return {"error": f"'{by}' has too many distinct values to group by."}
    g = df.groupby(by)[column].agg(agg).sort_values(ascending=False)
    data = {str(k): _num(v) for k, v in g.items()}
    return {"summary": f"{agg} of '{column}' by '{by}'.", "data": data}


def rate(df: pd.DataFrame, column: str, equals=None) -> dict:
    """Proportion of rows where ``column`` equals a value, or the mean of a 0/1
    column. Handy for churn rate and similar."""
    if column not in df.columns:
        return {"error": f"There is no column named '{column}'."}
    s = df[column]
    if equals is None:
        if not pd.api.types.is_numeric_dtype(s):
            return {"error": f"Give a value to measure the rate of in '{column}'."}
        r = _num(s.mean())
        return {"summary": f"Mean of '{column}' (its positive rate if it is 0/1).",
                "data": {"rate": r}}
    # match tolerant of type (compare as string)
    mask = s.astype(str) == str(equals)
    r = _num(mask.mean())
    return {"summary": f"Share of rows where '{column}' equals {equals}.",
            "data": {"rate": r, "matches": int(mask.sum()), "of": int(len(s))}}


def correlation(df: pd.DataFrame, column_a: str, column_b: str) -> dict:
    for c in (column_a, column_b):
        if c not in df.columns:
            return {"error": f"There is no column named '{c}'."}
        if not pd.api.types.is_numeric_dtype(df[c]):
            return {"error": f"'{c}' is not numeric, so correlation does not apply."}
    r = _num(df[column_a].corr(df[column_b]))
    return {"summary": f"Pearson correlation between '{column_a}' and '{column_b}'.",
            "data": {"correlation": r}}


def top_correlations(df: pd.DataFrame, target: str, k: int = 8) -> dict:
    if target not in df.columns:
        return {"error": f"There is no column named '{target}'."}
    if not pd.api.types.is_numeric_dtype(df[target]):
        return {"error": f"'{target}' is not numeric, so correlations do not apply."}
    num = [c for c in _numeric_cols(df) if c != target]
    corr = {c: _num(df[c].corr(df[target])) for c in num}
    corr = {c: v for c, v in corr.items() if v is not None}
    ordered = dict(sorted(corr.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k])
    return {"summary": f"Features most correlated with '{target}'.", "data": ordered}


# ---------------------------------------------------------------------------
#  Dispatcher — the only entry point the router uses
# ---------------------------------------------------------------------------
TOOLS = {
    "overview": overview,
    "column_summary": column_summary,
    "value_counts": value_counts,
    "group_stat": group_stat,
    "rate": rate,
    "correlation": correlation,
    "top_correlations": top_correlations,
}

# What each tool accepts, for the router prompt and light validation.
TOOL_SPECS = {
    "overview": {},
    "column_summary": {"column": "str"},
    "value_counts": {"column": "str", "normalize": "bool (optional)"},
    "group_stat": {"by": "str", "column": "str", "agg": "one of mean median sum count min max std"},
    "rate": {"column": "str", "equals": "value (optional)"},
    "correlation": {"column_a": "str", "column_b": "str"},
    "top_correlations": {"target": "str"},
}


def run_tool(df: pd.DataFrame, tool: str, args: dict | None) -> dict:
    """Validate and run one whitelisted tool. Never raises."""
    fn = TOOLS.get(tool)
    if fn is None:
        return {"error": f"Unknown tool '{tool}'."}
    args = {k: v for k, v in (args or {}).items() if k in TOOL_SPECS.get(tool, {})}
    try:
        return fn(df, **args)
    except TypeError as e:
        return {"error": f"Bad arguments for '{tool}': {e}"}
    except Exception as e:
        return {"error": f"Could not run '{tool}': {e}"}
