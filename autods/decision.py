"""Decision Studio analytics engine.

The second mode of AutoDS. Given a business dataset it builds an overview a
non-technical person can read and present: KPI tiles, a trend, category
breakdowns, key findings, and a grounded recommendation with an estimated impact.

Everything here is pure and grounded. Every number is computed from the data, no
model required. The LLM, when present, only rewrites the narrative, it never
invents a figure. See docs/DECISION_STUDIO.md for the full plan.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

_ID_HINT = ("id", "uuid", "guid", "code", "key", "index")
_RATE_HINT = ("rate", "pct", "percent", "ratio", "share", "score", "churn", "prob")
_MAX_DIM = 20          # a categorical with more distinct values is not a dimension
_MAX_BREAKDOWN = 12    # cap groups shown in a breakdown


def _round(x, n=4):
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return None


def _is_id(name, s) -> bool:
    low = name.lower()
    if any(h in low for h in _ID_HINT):
        return True
    # all-unique integer or text columns look like identifiers; continuous floats
    # (a real measure) can also be all-unique, so they are never flagged here
    if len(s) > 10 and s.nunique(dropna=True) == len(s):
        return s.dtype == object or pd.api.types.is_integer_dtype(s)
    return False


def _is_rate(name, s) -> bool:
    low = name.lower()
    if any(h in low for h in _RATE_HINT):
        return True
    vals = pd.to_numeric(s, errors="coerce").dropna()
    return bool(len(vals)) and vals.min() >= 0 and vals.max() <= 1


def _detect_time(df: pd.DataFrame):
    """Return (column_name, parsed_series) for the best date-like column, or None."""
    best = None
    for c in df.columns:
        s = df[c]
        low = c.lower()
        hint = any(h in low for h in ("date", "time", "year", "month", "day", "period", "week", "quarter"))
        if not hint and not (s.dtype == object or str(s.dtype).startswith("datetime")):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s, errors="coerce")
        ok = parsed.notna().sum()
        if ok >= max(5, 0.6 * len(s)) and parsed.dt.normalize().nunique() > 1:
            if best is None or ok > best[2]:
                best = (c, parsed, ok)
    return (best[0], best[1]) if best else None


def roles(df: pd.DataFrame) -> dict:
    """Split columns into time, measures (numeric outcomes) and dimensions (groups)."""
    time = _detect_time(df)
    time_col = time[0] if time else None
    measures, dims = [], []
    for c in df.columns:
        if c == time_col:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not _is_id(c, s):
            measures.append(c)
        elif not pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) <= _MAX_DIM:
            dims.append(c)
    return {"time": time_col, "time_series": time[1] if time else None,
            "measures": measures, "dimensions": dims}


def _agg_kind(name, s) -> str:
    return "mean" if _is_rate(name, s) else "sum"


def _headline(df, measure) -> dict:
    s = pd.to_numeric(df[measure], errors="coerce")
    kind = _agg_kind(measure, df[measure])
    if kind == "mean":
        return {"label": measure, "value": _round(s.mean() * 100, 1), "unit": "percent", "agg": "mean"}
    return {"label": measure, "value": _round(s.sum(), 2), "unit": "", "agg": "sum"}


def _period_key(parsed: pd.Series):
    """Choose a readable period grain: year if multi-year, else month."""
    if parsed.dt.year.nunique() > 1:
        return parsed.dt.year.astype(str), "year"
    return parsed.dt.to_period("M").astype(str), "month"


def build_overview(df: pd.DataFrame) -> dict:
    """Return the full Decision Studio overview for a dataset."""
    r = roles(df)
    measures, dims, time_col = r["measures"], r["dimensions"], r["time"]
    out = {"n_rows": int(len(df)), "n_cols": int(df.shape[1]),
           "measures": measures, "dimensions": dims, "time": time_col,
           "kpis": [], "trend": None, "breakdowns": [], "findings": [],
           "recommendation": None}
    if not measures:
        out["findings"].append("No numeric measures were found, so there is nothing to total or trend.")
        return out

    primary = next((m for m in measures if _is_rate(m, df[m])), measures[0])

    # ---- KPIs, with change versus the previous period when there is time -----
    periods = None
    if time_col is not None:
        keys, grain = _period_key(r["time_series"])
        out["period_grain"] = grain
        periods = sorted(keys.dropna().unique().tolist())
    for m in measures[:4]:
        k = _headline(df, m)
        if periods and len(periods) >= 2:
            s = pd.to_numeric(df[m], errors="coerce")
            by = s.groupby(keys)
            agg = (by.mean() if k["agg"] == "mean" else by.sum())
            cur, prev = agg.get(periods[-1]), agg.get(periods[-2])
            if cur is not None and prev not in (None, 0):
                k["change_pct"] = _round((cur - prev) / abs(prev) * 100, 1)
                k["direction"] = "up" if cur >= prev else "down"
                k["period_label"] = periods[-1]
        out["kpis"].append(k)

    # ---- trend of the primary measure over time -----------------------------
    if periods and len(periods) >= 2:
        s = pd.to_numeric(df[primary], errors="coerce")
        kind = _agg_kind(primary, df[primary])
        agg = (s.groupby(keys).mean() if kind == "mean" else s.groupby(keys).sum())
        agg = agg.reindex(periods)
        mult = 100 if kind == "mean" else 1
        out["trend"] = {"measure": primary, "unit": "percent" if kind == "mean" else "",
                        "labels": periods, "values": [_round(v * mult, 2) for v in agg.values]}

    # ---- breakdowns of the primary measure by each dimension ----------------
    kind = _agg_kind(primary, df[primary])
    mult = 100 if kind == "mean" else 1
    scored = []
    for d in dims:
        if df[d].nunique() > _MAX_BREAKDOWN:
            continue
        g = pd.to_numeric(df[primary], errors="coerce").groupby(df[d])
        agg = (g.mean() if kind == "mean" else g.sum()).sort_values(ascending=False)
        data = {str(k): _round(v * mult, 2) for k, v in agg.items()}
        if len(data) >= 2:
            vals = list(data.values())
            scored.append({"dimension": d, "measure": primary,
                           "unit": "percent" if kind == "mean" else "",
                           "data": data, "spread": _round(max(vals) - min(vals), 2)})
    scored.sort(key=lambda b: (b["spread"] or 0), reverse=True)
    out["breakdowns"] = scored[:3]

    out["findings"] = _findings(df, out, primary, kind)
    out["recommendation"] = _recommend(df, out, primary, kind)
    return out


def _findings(df, out, primary, kind) -> list[str]:
    f = []
    f.append(f"The dataset has {out['n_rows']} rows across {out['n_cols']} columns, "
             f"with {len(out['measures'])} measures and {len(out['dimensions'])} ways to group them.")
    # trend direction
    t = out.get("trend")
    if t and len(t["values"]) >= 2 and t["values"][0] is not None and t["values"][-1] is not None:
        first, last = t["values"][0], t["values"][-1]
        word = "risen" if last > first else "fallen" if last < first else "held flat"
        f.append(f"{primary} has {word} from {first} to {last} between {t['labels'][0]} and {t['labels'][-1]}.")
    # top breakdown
    if out["breakdowns"]:
        b = out["breakdowns"][0]
        hi = max(b["data"], key=b["data"].get)
        lo = min(b["data"], key=b["data"].get)
        f.append(f"{primary} varies most by {b['dimension']}, from {b['data'][lo]} for {lo} "
                 f"to {b['data'][hi]} for {hi}.")
    return f


def _recommend(df, out, primary, kind) -> dict | None:
    """The biggest lever: the group farthest from the average on the primary
    measure, with a rough, clearly labeled estimate of closing that gap."""
    if not out["breakdowns"]:
        return None
    b = out["breakdowns"][0]
    dim = b["dimension"]
    mult = 100 if kind == "mean" else 1
    s = pd.to_numeric(df[primary], errors="coerce")
    overall = (s.mean() if kind == "mean" else s.sum() / max(df[dim].nunique(), 1)) * mult
    # the group with the widest gap from the average
    gaps = {g: abs(v - overall) for g, v in b["data"].items()}
    worst = max(gaps, key=gaps.get)
    v = b["data"][worst]
    n_worst = int((df[dim].astype(str) == worst).sum())
    if kind == "mean":
        # pp change to the overall rate if this group moved to the average
        effect = _round((overall - v) * (n_worst / max(len(df), 1)), 2)
        est = (f"about {abs(effect)} percentage points on the overall {primary}")
        math = (f"gap {_round(overall - v, 2)} points times the group's share "
                f"{n_worst} of {len(df)} rows")
    else:
        effect = _round((overall - v) * 1.0, 2)
        est = f"about {abs(effect)} in total {primary}"
        math = f"gap {effect} between {worst} and the per group average {_round(overall, 2)}"
    move = "lift" if v < overall else "rein in"
    return {
        "headline": f"Focus on {worst} within {dim}",
        "detail": (f"{worst} sits at {v} on {primary}, the widest gap from the average {_round(overall, 2)}. "
                   f"It covers {n_worst} of {len(df)} rows, so it is where a change moves the overall number most."),
        "move": f"{move.capitalize()} {primary} for {worst} in {dim}.",
        "estimate": f"Estimated effect, {est}.",
        "math": f"Estimate only, {math}.",
    }
