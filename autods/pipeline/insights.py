"""Stages 2 & 8 — Narrative insights before modelling and after evaluation.

The findings themselves are computed by rule from the real profile, task, and
metrics, so the numbers are always ground truth. When an LLM is configured (see
autods/llm.py) those verified findings, plus retrieved best-practice notes, are
handed to the model to write sharper, grounded prose. With no model, the solid
rule-based bullets are used directly. Either way, retrieved guidance is appended
as citation lines (prefixed with a book glyph) so sources stay visible.
"""
from __future__ import annotations
import pandas as pd
from .. import llm
from ..rag import get_retriever
from ..rag import generate


def _hits(query: str, k: int = 3) -> list[dict]:
    """Retrieve relevant best-practice / past-run snippets (RAG grounding)."""
    try:
        return get_retriever().query(query, k=k)
    except Exception:
        return []


def _cite(hits: list[dict]) -> list[str]:
    """Format retrieved snippets as visible citation lines."""
    return [f"\U0001F4DA {h['title']} · {h['text']}" for h in hits]


def _compose(kind: str, facts: list[str], hits: list[dict], subject: str,
             use_llm: bool) -> list[str]:
    """Grounded LLM bullets when requested and a model produces them, else the
    verified facts. Retrieved guidance is always appended as citation lines."""
    body = generate.insights(kind, facts, hits, subject=subject) if use_llm else None
    return (body or facts) + _cite(hits)


def pre_insights(df: pd.DataFrame, profile: dict, task: dict,
                 use_llm: bool | None = None) -> list[str]:
    """Insights to show *before* modelling.

    ``use_llm`` controls the grounded LLM rewrite. ``None`` (the default) uses the
    model when one is configured, which suits the CLI agent and MCP. The web layer
    passes ``False`` to render instantly, then re-requests with ``True`` in the
    background so pages never block on generation.
    """
    b = []
    b.append(f"Dataset has {profile['n_rows']} rows and {profile['n_cols']} columns "
             f"({len(profile['types']['numeric'])} numeric, "
             f"{len(profile['types']['categorical'])} categorical).")
    if profile["missing"]:
        worst = max(profile["missing_pct"], key=profile["missing_pct"].get)
        b.append(f"{len(profile['missing'])} column(s) have missing data; "
                 f"'{worst}' is worst at {profile['missing_pct'][worst]}%.")
    else:
        b.append("No missing values detected.")
    if profile["n_duplicates"]:
        b.append(f"{profile['n_duplicates']} duplicate rows present.")
    b.append(task["reason"])
    if task["task_type"] == "classification" and task["target"]:
        vc = df[task["target"]].value_counts(normalize=True)
        if len(vc) and vc.iloc[0] > 0.7:
            b.append(f"Target is imbalanced, '{vc.index[0]}' is {vc.iloc[0]*100:.0f}% "
                     "of rows, so consider class weighting or resampling.")
    # RAG: ground with retrieved best-practice guidance relevant to this dataset
    query = (f"{task['task_type']} task "
             + ("with missing values " if profile['missing'] else "")
             + ("imbalanced classes " if task.get('imbalanced') else "")
             + "exploratory data analysis preprocessing")
    hits = _hits(query, k=2)
    if use_llm is None:
        use_llm = llm.available()
    return _compose("pre", b, hits, f"a {task['task_type']} dataset", use_llm)


def post_insights(task_type: str, model_name: str, evaluation: dict,
                  dataset: str = "dataset", use_llm: bool | None = None) -> list[str]:
    """Insights to show *after* evaluation. See :func:`pre_insights` for ``use_llm``."""
    m = evaluation["metrics"]
    b = []
    if task_type == "classification":
        b.append(f"{model_name} reached {m.get('accuracy', '?')} accuracy "
                 f"(F1 {m.get('f1', '?')}) on the held-out test set.")
        if "roc_auc" in m:
            b.append(f"ROC-AUC of {m['roc_auc']} indicates "
                     f"{'strong' if m['roc_auc'] > 0.8 else 'moderate'} class separation.")
    elif task_type == "regression":
        b.append(f"{model_name} explains {m.get('r2', '?')} of variance (R2) "
                 f"with RMSE {m.get('rmse', '?')}.")
    else:
        if "silhouette" in m:
            b.append(f"{model_name} found {m.get('n_clusters')} clusters with "
                     f"silhouette {m['silhouette']} "
                     f"({'well-separated' if m['silhouette'] > 0.5 else 'loosely separated'}).")
    fi = evaluation.get("feature_importance", [])
    if fi:
        top = ", ".join(f"'{d['feature']}'" for d in fi[:3])
        b.append(f"Most influential features: {top}.")
    # modelling-rigor findings (all computed, not guessed)
    data = evaluation.get("data", {})
    if data.get("balance") and data["balance"] != "none":
        b.append(f"Class imbalance was handled with {data['balance']} applied to the "
                 "training split only, so no test information leaked in.")
    thr = data.get("threshold")
    if thr and thr["tuned"]["threshold"] != 0.5:
        d0, dt = thr["default"], thr["tuned"]
        b.append(f"Moving the decision threshold from 0.5 to {dt['threshold']} lifts F1 "
                 f"from {d0['f1']} to {dt['f1']} and recall from {d0['recall']} to {dt['recall']}.")
    if "brier" in m:
        b.append(f"Brier score is {m['brier']}, a direct check on how honest the "
                 "predicted probabilities are.")
    fair = data.get("fairness")
    if fair and fair[0]["selection_gap"] > 0:
        f0 = fair[0]
        g = f0["groups"]
        b.append(f"Selection rate varies across '{f0['feature']}' by {f0['selection_gap']} "
                 f"({g[0]['value']} at {g[0]['selection_rate']} versus {g[-1]['value']} at "
                 f"{g[-1]['selection_rate']}), which is worth a fairness check.")
    perm = data.get("permutation")
    if perm:
        drivers = ", ".join(f"'{p['feature']}'" for p in perm[:3])
        b.append(f"By permutation importance the model leans most on {drivers}, measured "
                 "by how much the score falls when each column is shuffled.")
    errs = data.get("errors")
    if errs and errs.get("by_group"):
        g0 = errs["by_group"][0]
        hi, lo = g0["rows"][0], g0["rows"][-1]
        b.append(f"Errors are uneven across '{g0['feature']}', {hi['value']} is wrong "
                 f"{int(hi['error_rate']*100)} percent of the time versus "
                 f"{int(lo['error_rate']*100)} percent for {lo['value']}.")
    b.append("Next steps, try hyperparameter tuning, cross-validation, and "
             "compare against the other recommended models.")
    # ground the closing advice with retrieved guidance relevant to the results
    query = f"{task_type} {model_name} evaluation metrics " + " ".join(m.keys())
    hits = _hits(query, k=2)
    if use_llm is None:
        use_llm = llm.available()
    return _compose("post", b, hits, f"{model_name} on {dataset}", use_llm)
