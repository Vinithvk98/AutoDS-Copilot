"""The agentic planner.

Given a loaded dataset, the copilot proposes a full run in one shot: which target
to predict, which columns to drop, and which model to train, each with a short
grounded reason. The user approves the plan before anything runs, so the agent
drives and the human stays in control.

Model choice is made by analysis, not a default. The candidates are
cross-validated first (a real leaderboard), and the agent picks the best
performer and explains it with the actual scores. So on one dataset it may pick
gradient boosting and on another a naive Bayes, whichever measures best here. The
choice is then validated against the real model registry, so the agent can never
name a model that does not exist. With no LLM the top-scoring model and the
recommended cleaning are used with plain reasons, so auto-pilot still works
offline.
"""
from __future__ import annotations

from . import llm
from .converse import _extract_json


def _model_reason(scored, metric):
    """A grounded reason for the model pick, referencing the leaderboard scores."""
    if not scored:
        return None
    top = scored[0]
    msg = (f"After cross-validating the candidates on {metric}, {top['name']} scored "
           f"best at {top['score']}")
    if len(scored) > 1:
        msg += f", ahead of {scored[1]['name']} at {scored[1]['score']}"
    return msg + "."


def _rule_reasons(target, task_type, drop_columns, model, scored, metric):
    drop_txt = ", ".join(drop_columns) if drop_columns else "none"
    return {
        "target": f"{target} is the target for a {task_type} task.",
        "cleaning": (f"Dropping {drop_txt} to remove leakage or noise."
                     if drop_columns else "No columns need dropping."),
        "model": (_model_reason(scored, metric)
                  or f"{model} is the recommended default for {task_type} on tabular data."),
    }


def _prompt(df, task, plan, scored, metric):
    schema = ", ".join(f"{c} ({'num' if str(df[c].dtype).startswith(('int', 'float')) else 'cat'})"
                       for c in df.columns)
    board = ("\n".join(f"  {r['name']}: {metric} {r['score']}" for r in scored[:10])
             if scored else "not available")
    notes = "; ".join(plan.get("notes", [])) or "no cleaning issues flagged"
    suggested_drop = ", ".join(plan.get("drop_columns", [])) or "none"
    return (
        f"Dataset columns: {schema}\n"
        f"Target: {task.get('target')}   Task: {task.get('task_type')}\n"
        f"Cleaning notes from the data checks: {notes}\n"
        f"Columns the checks suggest dropping: {suggested_drop}\n"
        f"Cross-validated model scores (higher is better):\n{board}\n\n"
        "Choose the best model for THIS dataset based on the scores above, not a generic "
        "default. Prefer the top scorer, unless a clearly simpler model is within about a "
        "point of it, in which case you may prefer the simpler one and say why. Also choose "
        "the columns to drop (only from the suggested ones, or an empty list). Give one "
        "short reason for each choice, and mention the winning score in the model reason. "
        'Reply with ONLY this JSON:\n'
        '{"model": "<name>", "drop_columns": ["..."], "reason_target": "...", '
        '"reason_cleaning": "...", "reason_model": "..."}'
    )


def propose(df, profile, task, plan, candidates, board=None) -> dict:
    """Return an approvable plan: {target, task_type, model, drop_columns,
    drop_duplicates, why:{target,cleaning,model}}. Model chosen from the
    cross-validated leaderboard when supplied. Validated, heuristic fallback."""
    names = [c["name"] for c in candidates]
    metric = board.get("metric") if board else None
    scored = [r for r in board.get("rows", []) if r.get("score") is not None] if board else []

    default_model = (scored[0]["name"] if scored
                     else next((c["name"] for c in candidates if c.get("recommended")),
                               names[0] if names else ""))
    feature_cols = [c for c in df.columns if c != task.get("target")]
    suggested_drop = [c for c in plan.get("drop_columns", []) if c in feature_cols]

    out = {"target": task.get("target"), "task_type": task.get("task_type"),
           "model": default_model, "drop_columns": suggested_drop,
           "drop_duplicates": bool(plan.get("drop_duplicates", False)),
           "score": (scored[0]["score"] if scored else None), "metric": metric, "why": {}}

    if not llm.available():
        out["why"] = _rule_reasons(out["target"], out["task_type"], out["drop_columns"],
                                   out["model"], scored, metric)
        return out

    reply = llm.complete(
        "You are the planning agent inside AutoDS. Pick a model based on the given "
        "cross-validated scores and choose cleaning from the given options only, then "
        "explain briefly. Reply with one JSON object, nothing else.",
        _prompt(df, task, plan, scored, metric), temperature=0.0, max_tokens=320)
    call = _extract_json(reply) or {}

    model = call.get("model")
    if model in names:
        out["model"] = model
        # keep the score aligned with whichever model was chosen
        pick = next((r for r in scored if r["name"] == model), None)
        out["score"] = pick["score"] if pick else out["score"]
    drops = call.get("drop_columns")
    if isinstance(drops, list):
        out["drop_columns"] = [c for c in drops if c in feature_cols]

    rule = _rule_reasons(out["target"], out["task_type"], out["drop_columns"],
                         out["model"], scored, metric)
    out["why"] = {
        "target": call.get("reason_target") or rule["target"],
        "cleaning": call.get("reason_cleaning") or rule["cleaning"],
        "model": call.get("reason_model") or rule["model"],
    }
    return out
