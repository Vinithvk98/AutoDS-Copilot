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
# measures where a lower number is the better outcome
_LOWER_BETTER_HINT = ("churn", "cost", "expense", "defect", "complaint", "return", "refund",
                      "delay", "late", "error", "fail", "cancel", "loss", "attrition", "default",
                      "fraud", "wait", "downtime", "incident", "risk")
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


def lower_is_better(name) -> bool:
    low = name.lower()
    return any(h in low for h in _LOWER_BETTER_HINT)


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


def primary_measure(df: pd.DataFrame, measures=None):
    """The headline measure, the first rate like column, else the first measure."""
    measures = roles(df)["measures"] if measures is None else measures
    return next((m for m in measures if _is_rate(m, df[m])), measures[0] if measures else None)


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

    primary = primary_measure(df, measures)
    out["primary"] = primary
    out["overall"] = _headline(df, primary)

    # ---- KPIs, with change versus the previous period when there is time -----
    periods = None
    if time_col is not None:
        keys, grain = _period_key(r["time_series"])
        out["period_grain"] = grain
        periods = sorted(keys.dropna().unique().tolist())
    for m in ([primary] + [x for x in measures if x != primary])[:4]:
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
            counts = {str(k): int(v) for k, v in g.count().items()}
            scored.append({"dimension": d, "measure": primary,
                           "unit": "percent" if kind == "mean" else "",
                           "agg": kind, "data": data, "counts": counts,
                           "spread": _round(max(vals) - min(vals), 2)})
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
    # the group that lags the average most in the good direction. For a measure
    # where lower is better (churn, cost, delays) that is the highest group.
    down = lower_is_better(primary)
    lag = {g: (v - overall) if down else (overall - v) for g, v in b["data"].items()}
    worst = max(lag, key=lag.get)
    v = b["data"][worst]
    eff = estimate_effect(df, primary, dim, worst, overall)
    n_worst, n_all = eff["group_rows"], eff["total_rows"]
    effect = _round(eff["change"], 2)
    if kind == "mean":
        # pp change to the overall rate if this group moved to the average
        est = (f"about {abs(effect)} percentage points on the overall {primary}")
        math = (f"gap {_round(abs(overall - v), 2)} points times the group's share "
                f"{n_worst} of {n_all} rows")
    else:
        est = f"about {abs(effect)} in total {primary}"
        math = f"gap {abs(effect)} between {worst} and the per group average {_round(overall, 2)}"
    move = "bring down" if v > overall else "lift"
    return {
        "headline": f"Focus on {worst} within {dim}",
        "detail": (f"{worst} sits at {v} on {primary} against an average of {_round(overall, 2)}, "
                   f"the widest gap on the wrong side. It covers {n_worst} of {n_all} rows, so closing that gap moves the "
                   f"overall number the most."),
        "move": f"{move.capitalize()} {primary} for {worst} in {dim}, toward the average.",
        "estimate": f"Estimated effect, {est}.",
        "math": f"Estimate only, {math}.",
        "dimension": dim, "group": worst, "lower_is_better": down,
    }


# ---------------------------------------------------------------------------
# Phase 2, what if and goal seeking
#
# Both rest on one exact identity. The overall mean of a measure is the row
# weighted average of its group means, and the overall total is the sum of its
# group totals. So moving one group's value moves the overall number by the
# change times that group's weight (its share of rows for a mean, one for a
# total). This is arithmetic on the data, not a causal claim, and every result
# is labeled as an estimate.
# ---------------------------------------------------------------------------

def _group_table(df, measure, dimension):
    """Per group mean, total, and row count of a measure, keyed by group text."""
    if measure not in df.columns:
        raise ValueError(f"{measure} is not a column in this data")
    if dimension not in df.columns:
        raise ValueError(f"{dimension} is not a column in this data")
    s = pd.to_numeric(df[measure], errors="coerce")
    keep = s.notna() & df[dimension].notna()
    g = s[keep].groupby(df.loc[keep, dimension].astype(str))
    return pd.DataFrame({"mean": g.mean(), "total": g.sum(), "rows": g.count()})


def _overall(df, measure, kind, mult):
    s = pd.to_numeric(df[measure], errors="coerce")
    return float(s.mean() * mult) if kind == "mean" else float(s.sum()), int(s.notna().sum())


def _unit_word(kind):
    return " points" if kind == "mean" else ""


def _num(v):
    v = _round(v, 2)
    return str(int(v)) if v is not None and v == int(v) else str(v)


def _fmt(v, kind):
    return f"{_num(v)} percent" if kind == "mean" else _num(v)


def estimate_effects(df, measure, dimension, changes: dict) -> dict:
    """What if several groups moved at once. `changes` maps a group to its new
    value, in the same units the breakdown shows (percent for a rate, the group
    total for a summed measure). Groups do not overlap, so effects add up."""
    kind = _agg_kind(measure, df[measure])
    mult = 100 if kind == "mean" else 1
    tab = _group_table(df, measure, dimension)
    current, n_all = _overall(df, measure, kind, mult)
    vals = pd.to_numeric(df[measure], errors="coerce")
    bounded = kind == "mean" and vals.min() >= 0 and vals.max() <= 1   # a true 0 to 1 rate
    moved, delta = [], 0.0
    for group, new_value in changes.items():
        group = str(group)
        if group not in tab.index:
            raise ValueError(f"{group} is not a group of {dimension}")
        new_value = float(new_value)
        if kind == "mean":
            if bounded:
                new_value = min(max(new_value, 0.0), 100.0)
            was = float(tab.at[group, "mean"]) * mult
            weight = tab.at[group, "rows"] / max(n_all, 1)
        else:
            was = float(tab.at[group, "total"])
            weight = 1.0
        part = (new_value - was) * weight
        delta += part
        moved.append({"group": group, "from": _round(was, 2), "to": _round(new_value, 2),
                      "rows": int(tab.at[group, "rows"]), "effect": _round(part, 4)})
    new = current + delta
    unit = "percent" if kind == "mean" else ""
    if not moved or abs(delta) < 1e-12:
        text = f"No change. The overall {measure} stays at {_fmt(current, kind)}."
    else:
        word = "rise" if delta > 0 else "fall"
        text = (f"The overall {measure} would {word} from {_fmt(current, kind)} to {_fmt(new, kind)}, "
                f"a change of {_round(abs(delta), 2)}{_unit_word(kind)}.")
    if kind == "mean":
        math = ", plus ".join(f"{m['group']} moves {_round(m['to'] - m['from'], 2)} points times its share "
                         f"{m['rows']} of {n_all} rows" for m in moved)
    else:
        math = ", plus ".join(f"{m['group']} total moves by {_round(m['to'] - m['from'], 2)}" for m in moved)
    return {"measure": measure, "dimension": dimension, "unit": unit, "agg": kind,
            "current": _round(current, 4), "new": _round(new, 4), "change": _round(delta, 4),
            "total_rows": n_all, "groups": moved, "text": text,
            "math": f"Estimate only, {math}." if moved else "Estimate only, nothing was moved.",
            "note": "This is arithmetic on the current data, not a measured cause and effect."}


def estimate_effect(df, measure, dimension, group, new_value) -> dict:
    """What if one group's value on `measure` moved to `new_value`. Returns the
    estimated overall number before and after, with the arithmetic behind it."""
    out = estimate_effects(df, measure, dimension, {group: new_value})
    g = out["groups"][0]
    out.update({"group": g["group"], "group_from": g["from"], "group_to": g["to"],
                "group_rows": g["rows"]})
    return out


def _dims_for(df, measure):
    return [d for d in roles(df)["dimensions"]
            if 2 <= df[d].nunique(dropna=True) <= _MAX_BREAKDOWN]


def _seek_in(tab, kind, mult, n_all, needed):
    """Smallest move within one dimension. Groups behind the best performer are
    brought toward it, biggest lever first, until the gap is closed. The best
    group's per row level is the ceiling, since it is already achieved in the data.
    Returns (changes, achieved)."""
    up = needed > 0
    per_row = tab["mean"] * mult                         # level per row, display units
    bench = per_row.max() if up else per_row.min()
    weight = tab["rows"] / max(n_all, 1) if kind == "mean" else tab["rows"].astype(float)
    room = (bench - per_row) * weight                    # most each group can add
    room = room[room.abs() > 1e-12].sort_values(ascending=not up)
    changes, achieved = [], 0.0
    for g, r in room.items():
        if abs(achieved) >= abs(needed) - 1e-12:
            break
        take = r if abs(achieved + r) <= abs(needed) else needed - achieved
        new_level = per_row[g] + take / weight[g]
        if kind == "mean":
            frm, to = per_row[g], new_level
        else:
            frm, to = tab.at[g, "total"], new_level * tab.at[g, "rows"]
        changes.append({"group": g, "from": _round(frm, 2), "to": _round(to, 2),
                        "rows": int(tab.at[g, "rows"]), "effect": _round(take, 4),
                        "benchmark": _round(bench if kind == "mean" else bench * tab.at[g, "rows"], 2)})
        achieved += take
    return changes, achieved


def goal_seek(df, measure, target) -> dict:
    """What would it take for the overall `measure` to reach `target` (percent for
    a rate, the total for a summed measure). Looks across the driver dimensions and
    returns the smallest feasible move, the one that changes the fewest groups and
    then the fewest points, where no group goes past the best level any group in
    that dimension already reaches."""
    measure = measure or primary_measure(df)
    if measure is None:
        raise ValueError("There is no numeric measure to set a goal on")
    kind = _agg_kind(measure, df[measure])
    mult = 100 if kind == "mean" else 1
    current, n_all = _overall(df, measure, kind, mult)
    target = float(target)
    needed = target - current
    unit = "percent" if kind == "mean" else ""
    base = {"measure": measure, "unit": unit, "agg": kind, "current": _round(current, 4),
            "target": _round(target, 4), "needed": _round(needed, 4),
            "note": "This is arithmetic on the current data, not a measured cause and effect."}
    if abs(needed) < 5e-4:   # within display rounding
        return {**base, "feasible": True, "dimension": None, "changes": [], "result": _round(current, 4),
                "text": f"The overall {measure} is already at {_fmt(current, kind)}, so nothing needs to change.",
                "math": "Estimate only, the target equals the current value.", "options": []}

    options = []
    for d in _dims_for(df, measure):
        tab = _group_table(df, measure, d)
        if len(tab) < 2:
            continue
        changes, achieved = _seek_in(tab, kind, mult, n_all, needed)
        moved = sum(abs(c["to"] - c["from"]) for c in changes)
        options.append({"dimension": d, "changes": changes, "achieved": achieved,
                        "feasible": abs(achieved - needed) < 1e-6, "n_groups": len(changes),
                        "moved": moved})
    if not options:
        return {**base, "feasible": False, "dimension": None, "changes": [], "result": _round(current, 4),
                "text": f"There is no way to group {measure} here, so there is no lever to pull.",
                "math": "Estimate only, no dimension with two to twelve groups was found.", "options": []}

    feas = [o for o in options if o["feasible"]]
    if feas:
        best = min(feas, key=lambda o: (o["n_groups"], o["moved"]))
    else:
        best = max(options, key=lambda o: abs(o["achieved"]))
    result = current + best["achieved"]
    word = "raise" if needed > 0 else "lower"
    steps = ", and ".join(f"move {c['group']} from {_fmt(c['from'], kind)} to {_fmt(c['to'], kind)}"
                          for c in best["changes"])
    if best["feasible"]:
        text = (f"To {word} the overall {measure} from {_fmt(current, kind)} to {_fmt(target, kind)}, "
                f"the smallest move is within {best['dimension']}. {steps[0].upper() + steps[1:]}.")
    else:
        text = (f"Reaching {_fmt(target, kind)} is beyond what the data supports. Even bringing every "
                f"{best['dimension']} group to the best level already seen gets the overall {measure} "
                f"to about {_fmt(result, kind)}.")
    if kind == "mean":
        math = ", plus ".join(f"{c['group']} moves {_round(c['to'] - c['from'], 2)} points times its share "
                              f"{c['rows']} of {n_all} rows" for c in best["changes"])
    else:
        math = ", plus ".join(f"{c['group']} total moves by {_round(c['to'] - c['from'], 2)}"
                              for c in best["changes"])
    return {**base, "feasible": best["feasible"], "dimension": best["dimension"],
            "changes": best["changes"], "result": _round(result, 4), "text": text,
            "math": f"Estimate only, {math}.",
            "options": [{"dimension": o["dimension"], "feasible": o["feasible"],
                         "groups": o["n_groups"], "reaches": _round(current + o["achieved"], 2)}
                        for o in options]}
