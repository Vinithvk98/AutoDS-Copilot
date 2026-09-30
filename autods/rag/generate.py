"""Grounded LLM generation for the copilot.

Every generator here follows the same contract: the model may use ONLY the run
facts and retrieved reference notes it is handed. It must not invent numbers,
column names, metrics, or sources, and it must say when the context does not
cover something. If no LLM is configured, each function returns ``None`` and the
caller uses its rule-based text instead, so output is always grounded and always
works offline.

This module is the single place that turns retrieved RAG material plus verified
run facts into prose. It never talks to a provider directly; it composes prompts
and delegates to :mod:`autods.llm`.
"""
from __future__ import annotations

from .. import llm

# The grounding contract, shared by every generator.
_SYSTEM = (
    "You are the analytical copilot inside AutoDS, a data-science tool. "
    "Answer using ONLY the run facts and retrieved reference notes you are given. "
    "Never invent numbers, column names, metrics, model names, or sources. If the "
    "context does not cover something, say so plainly rather than guessing. Be "
    "concrete, precise, and useful to a working data scientist. Write in clear "
    "plain English. Do not use colons inside headings, and do not use long dashes."
)


def _passages_block(passages: list[dict] | None) -> str:
    if not passages:
        return "(no reference notes were retrieved)"
    return "\n".join(f"[{p.get('title', 'note')}] {p.get('text', '')}"
                     for p in passages)


# ---------------------------------------------------------------------------
#  Ask panel — a detailed, grounded answer
# ---------------------------------------------------------------------------
def _answer_prompt(question: str, passages: list[dict] | None,
                   run_facts: str | None = None) -> str:
    facts = f"\n\nVerified run facts you may rely on:\n{run_facts}" if run_facts else ""
    return (
        f"Question: {question}\n\n"
        f"Reference notes (knowledge base and past runs):\n{_passages_block(passages)}"
        f"{facts}\n\n"
        "Write a clear, well-structured answer grounded strictly in the notes and "
        "facts above. Explain the reasoning, not just the conclusion, and name the "
        "note titles you drew on. If the material is not enough to answer fully, "
        "say what is missing."
    )


def answer(question: str, passages: list[dict] | None,
           run_facts: str | None = None) -> str | None:
    """Generate a detailed answer grounded in the retrieved passages (which
    already include current-run facts when a session is supplied). Returns the
    text, or ``None`` if no LLM is available."""
    return llm.complete(_SYSTEM, _answer_prompt(question, passages, run_facts),
                        max_tokens=800)


def answer_stream(question: str, passages: list[dict] | None,
                  run_facts: str | None = None):
    """Same grounded answer as :func:`answer`, but yielded in chunks as the model
    generates them. Returns a generator, or ``None`` when no LLM is available so
    the caller can fall back to the non-streaming path."""
    if not llm.available():
        return None
    return llm.stream(_SYSTEM, _answer_prompt(question, passages, run_facts),
                      max_tokens=800)


# ---------------------------------------------------------------------------
#  Insight bullets — before and after modelling
# ---------------------------------------------------------------------------
def insights(kind: str, facts: list[str], passages: list[dict] | None,
             subject: str = "") -> list[str] | None:
    """Rewrite verified findings into sharper, grounded insight bullets.

    ``kind`` is 'pre' or 'post'. ``facts`` are ground-truth findings computed
    from the real profile, task, or metrics of this run. ``passages`` are the
    retrieved best-practice or past-run notes. Returns a list of bullet strings,
    or ``None`` when no LLM is available.
    """
    if not facts:
        return None
    stage = ("before modelling, focused on the data and what to watch for"
             if kind == "pre" else
             "after evaluation, focused on what the results mean and what to do next")
    facts_block = "\n".join(f"- {f}" for f in facts)
    user = (
        f"Verified findings about {subject or 'this run'} ({stage}):\n{facts_block}\n\n"
        f"Reference notes (best practice and similar past runs):\n"
        f"{_passages_block(passages)}\n\n"
        "Write 4 to 6 sharp insight bullets for a data scientist. Every number, "
        "column name, and metric must come from the verified findings above, never "
        "invented. Where a reference note applies, use its guidance and name the "
        "source. Return one bullet per line with no numbering and no heading."
    )
    out = llm.complete(_SYSTEM, user, max_tokens=650)
    if not out:
        return None
    bullets = [ln.lstrip("-*• ").strip() for ln in out.splitlines() if ln.strip()]
    return bullets or None


# ---------------------------------------------------------------------------
#  Recommendation rationale — a grounded "why"
# ---------------------------------------------------------------------------
def executive_summary(facts: list[str], passages: list[dict] | None = None) -> str | None:
    """A short stakeholder summary of the finished run, grounded in verified facts.
    Returns text, or None when no LLM is available."""
    if not llm.available() or not facts:
        return None
    block = "\n".join(f"- {f}" for f in facts)
    notes = f"\n\nReference notes:\n{_passages_block(passages)}" if passages else ""
    user = (
        f"Verified facts about the finished run:\n{block}{notes}\n\n"
        "Write a 3 to 5 sentence executive summary for a non-technical stakeholder. "
        "Use only the facts above, invent no numbers. Cover what was predicted, how well "
        "it did, how class imbalance was handled if it was, and one honest caution. Plain "
        "English, no colons in headings, no long dashes."
    )
    return llm.complete(_SYSTEM, user, max_tokens=380)


def compare_models(rows: list[dict], metric: str, passages: list[dict] | None = None) -> str | None:
    """Grounded reasoning over a leaderboard: why the top model leads and the
    trade-offs against the alternatives. Returns text, or None with no LLM."""
    if not llm.available() or not rows:
        return None
    table = "\n".join(f"- {r.get('name')}: {metric} {r.get('score')}" for r in rows[:8]
                      if r.get("score") is not None)
    notes = f"\n\nReference notes:\n{_passages_block(passages)}" if passages else ""
    user = (
        f"Leaderboard scored by {metric} (higher is better):\n{table}{notes}\n\n"
        "In 3 to 4 sentences explain which model leads and why it is a reasonable pick, "
        "and the trade-off against a simpler or a runner-up model (accuracy versus speed "
        "or interpretability). Use only the scores above, invent nothing."
    )
    return llm.complete(_SYSTEM, user, max_tokens=320)


def rationale(decision: str, snippet_text: str, snippet_source: str) -> str | None:
    """Explain a recommendation in two or three sentences, grounded in one
    retrieved guidance snippet. Returns text, or ``None`` if no LLM available."""
    user = (
        f"Decision to explain: {decision}\n\n"
        f"Retrieved guidance [{snippet_source}]: {snippet_text}\n\n"
        "In two or three sentences, explain the recommendation grounded in that "
        "guidance. Add no facts beyond it. Name the source once."
    )
    return llm.complete(_SYSTEM, user, max_tokens=220)
