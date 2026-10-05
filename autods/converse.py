"""The conversational data copilot.

Turns a natural-language question into a real answer about the *loaded dataset*.
The model never runs code. It routes the question to whitelisted tools in
:mod:`autods.dataqa`, we execute them safely, and the computed numbers ground the
spoken answer. Four behaviours:

  single   one tool answers the question (churn rate by plan)
  multi    a why question chains several analyses to find the drivers
  clarify  an unclear question asks back with concrete options
  suggest  starter questions built from the dataset so users know what to ask
"""
from __future__ import annotations
import json
import re

import pandas as pd

from . import llm, dataqa

_ROUTER_SYSTEM = (
    "You decide how to answer a question about a tabular dataset. Choose ONE "
    "analysis tool when the question is about the dataset's actual values, or "
    "answer none for general or conceptual questions. Reply with ONLY a single "
    "JSON object and nothing else."
)

_WHY = re.compile(r"\b(why|driver|drivers?|drive[sn]?|reason|reasons?|explain|cause[sd]?|"
                  r"factor|factors?|influenc\w*|lead\w* to|associated with)\b", re.I)
_DATA_WORDS = re.compile(r"\b(rate|average|mean|median|sum|count|distribution|breakdown|"
                         r"by|per|top|highest|lowest|correlat\w*|how many|most|group)\b", re.I)


# ---------------------------------------------------------------------------
#  small schema helpers
# ---------------------------------------------------------------------------
def _low_card_cats(df, target=None, maxc: int = 12) -> list[str]:
    return [c for c in df.columns if c != target
            and not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= maxc]


def _numeric_target(df, target) -> bool:
    return bool(target) and target in df.columns and pd.api.types.is_numeric_dtype(df[target])


# ---------------------------------------------------------------------------
#  suggested starter questions (deterministic, no model needed)
# ---------------------------------------------------------------------------
def suggest_questions(df, task, k: int = 5) -> list[str]:
    if df is None:
        return []
    target = (task or {}).get("target")
    ttype = (task or {}).get("task_type")
    cats = _low_card_cats(df, target)
    qs: list[str] = []
    if target and ttype in ("classification", "text"):
        for c in cats[:2]:
            qs.append(f"what is the {target} rate by {c}")
        qs.append(f"what drives {target}")
        qs.append(f"which features correlate most with {target}")
    elif target and ttype in ("regression", "timeseries"):
        for c in cats[:2]:
            qs.append(f"what is the average {target} by {c}")
        qs.append(f"what drives {target}")
        qs.append(f"which features correlate most with {target}")
    else:
        qs.append("give me an overview of the data")
        for c in cats[:3]:
            qs.append(f"show the distribution of {c}")
    # de-duplicate, keep order
    seen, out = set(), []
    for q in qs:
        if q not in seen:
            seen.add(q); out.append(q)
    return out[:k]


# ---------------------------------------------------------------------------
#  routing to a single tool
# ---------------------------------------------------------------------------
def _router_prompt(question: str, df, history: list | None) -> str:
    specs = "\n".join(f"- {name}: args {args}" for name, args in dataqa.TOOL_SPECS.items())
    convo = ""
    if history:
        turns = "\n".join(f"Q: {h['q']}\nA: {h['a'][:200]}" for h in history[-3:])
        convo = f"\nConversation so far (for context on follow-ups):\n{turns}\n"
    return (
        f"Dataset schema:\n{json.dumps(dataqa.schema(df))}\n\n"
        f"Available tools and their arguments:\n{specs}\n{convo}\n"
        "Rules. Use only column names from the schema. For group_stat, agg must be "
        "one of mean median sum count min max std. If the question is general data "
        "science advice and not about this dataset's values, reply {\"tool\": \"none\"}.\n\n"
        f"Question: {question}\n\n"
        "Reply with JSON like {\"tool\": \"group_stat\", \"args\": {\"by\": \"plan\", "
        "\"column\": \"churned\", \"agg\": \"mean\"}} or {\"tool\": \"none\"}."
    )


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def route(question: str, df, history: list | None = None) -> dict | None:
    reply = llm.complete(_ROUTER_SYSTEM, _router_prompt(question, df, history),
                         temperature=0.0, max_tokens=160)
    call = _extract_json(reply)
    if not call:
        return None
    tool = call.get("tool")
    if tool in (None, "none", "") or tool not in dataqa.TOOLS:
        return None
    args = call.get("args") if isinstance(call.get("args"), dict) else {}
    return {"tool": tool, "args": args}


def _format_facts(call: dict, result: dict) -> str:
    tool, args = call["tool"], call.get("args", {})
    if "error" in result:
        return (f"An attempt to compute this from the data with {tool}({json.dumps(args)}) "
                f"did not work: {result['error']}")
    return (f"Computed directly from the dataset with {tool}({json.dumps(args)}). "
            f"{result.get('summary', '')} Values: {json.dumps(result.get('data'))}")


# ---------------------------------------------------------------------------
#  multi-step why analysis
# ---------------------------------------------------------------------------
def multi_why(df, task):
    """Chain several safe analyses to explain a target, then rank the dimensions by
    how much the target moves across their groups. Returns (facts, meta) or (None, None)."""
    target = (task or {}).get("target")
    if not _numeric_target(df, target):
        return None, None
    blocks = []
    for c in _low_card_cats(df, target):
        r = dataqa.group_stat(df, by=c, column=target, agg="mean")
        data = r.get("data") if isinstance(r, dict) else None
        if data:
            vals = list(data.values())
            blocks.append({"label": c, "data": data, "spread": round(max(vals) - min(vals), 4)})
    blocks.sort(key=lambda b: b["spread"], reverse=True)
    corr = dataqa.top_correlations(df, target).get("data", {})
    if not blocks and not corr:
        return None, None

    lines = [f"Driver analysis for {target}, every number computed from the dataset."]
    for b in blocks:
        hi = max(b["data"], key=b["data"].get)
        lo = min(b["data"], key=b["data"].get)
        lines.append(f"By {b['label']}, {target} goes from {b['data'][lo]} for {lo} to "
                     f"{b['data'][hi]} for {hi}, a spread of {b['spread']}.")
    if corr:
        top = list(corr.items())[:4]
        lines.append(f"Numeric features most correlated with {target}: "
                     + ", ".join(f"{k} {v}" for k, v in top) + ".")
    facts = " ".join(lines)
    meta = {"multi": True, "target": target, "blocks": blocks[:4], "correlations": corr}
    return facts, meta


# ---------------------------------------------------------------------------
#  clarify
# ---------------------------------------------------------------------------
def _clarify_bad_column(df, task) -> dict:
    target = (task or {}).get("target")
    cats = _low_card_cats(df, target)
    opts = []
    if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        opts += [f"what is the {target} rate by {c}" for c in cats[:2]]
    opts += [f"show the distribution of {c}" for c in cats[:3]]
    return {"clarify": "I could not find that column. Did you mean one of these?",
            "options": (opts or suggest_questions(df, task))[:5]}


def _clarify_vague(df, task) -> dict:
    return {"clarify": "I am not sure which column you mean. Try one of these.",
            "options": suggest_questions(df, task)}


# ---------------------------------------------------------------------------
#  the single entry point
# ---------------------------------------------------------------------------
def analyze(question: str, df, task=None, history: list | None = None):
    """Return ``(facts, meta)``. ``facts`` grounds the streamed answer. ``meta`` is
    one of: a single tool result, a multi-step driver analysis, or a clarify prompt
    (in which case ``facts`` is None and no answer is streamed)."""
    if df is None or not llm.available():
        return None, None

    # 1. a why question about a numeric target -> multi-step driver analysis
    if _WHY.search(question or "") and _numeric_target(df, (task or {}).get("target")):
        facts, meta = multi_why(df, task)
        if meta:
            return facts, meta

    # 2. route to a single tool
    call = route(question, df, history)
    if call:
        result = dataqa.run_tool(df, call["tool"], call.get("args"))
        if isinstance(result, dict) and "error" in result and "no column" in result["error"].lower():
            return None, _clarify_bad_column(df, task)
        return _format_facts(call, result), {"tool": call["tool"], "args": call.get("args"),
                                             "result": result}

    # 3. looks like a data question we could not map -> clarify, else general -> RAG
    if _DATA_WORDS.search(question or ""):
        return None, _clarify_vague(df, task)
    return None, None
