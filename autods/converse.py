"""The conversational data copilot.

Turns a natural-language question into a real answer about the *loaded dataset*.
The model does not run code. It only routes the question to one whitelisted tool
in :mod:`autods.dataqa` (or to "none" for general questions), we execute that tool
safely, and the computed numbers become grounding for the spoken answer.

Flow:  question -> route (LLM picks a tool + args) -> run tool -> grounded facts
The web layer then streams the answer over those facts plus any retrieved notes.
"""
from __future__ import annotations
import json
import re

from . import llm, dataqa

_ROUTER_SYSTEM = (
    "You decide how to answer a question about a tabular dataset. Choose ONE "
    "analysis tool when the question is about the dataset's actual values, or "
    "answer none for general or conceptual questions. Reply with ONLY a single "
    "JSON object and nothing else."
)


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
    """Pull the first JSON object out of a model reply, tolerating stray prose."""
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
    """Ask the model which tool (if any) answers the question. Returns a validated
    ``{"tool", "args"}`` dict, or ``None`` for a general question or on any failure."""
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


def analyze(question: str, df, history: list | None = None):
    """Route and run one analysis over the dataset. Returns ``(facts, meta)`` where
    ``facts`` is grounding text for the answer (or ``None`` when the question is not
    a data question), and ``meta`` describes the tool call and result for the UI."""
    if df is None or not llm.available():
        return None, None
    call = route(question, df, history)
    if not call:
        return None, None
    result = dataqa.run_tool(df, call["tool"], call.get("args"))
    return _format_facts(call, result), {"tool": call["tool"], "args": call.get("args"),
                                         "result": result}
