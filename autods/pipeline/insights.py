"""Stages 2 & 8 — Narrative insights before modelling and after evaluation.

Rule-based by default (always works, no API key). If OPENAI_API_KEY is set the
text is upgraded to LLM-written prose via _llm_polish().
"""
from __future__ import annotations
import pandas as pd
from .. import config
from ..rag import get_retriever


def _guidance(query: str, k: int = 3) -> list[str]:
    """Retrieve relevant best-practice / past-run snippets (RAG grounding)."""
    try:
        hits = get_retriever().query(query, k=k)
    except Exception:
        return []
    return [f"📚 {h['title']} · {h['text']}" for h in hits]


def _llm_polish(prompt: str, bullets: list[str]) -> list[str]:
    """Optionally rewrite bullets as sharper prose using an LLM. Falls back
    silently to the rule-based bullets if anything goes wrong."""
    if not config.USE_LLM:
        return bullets
    try:
        from openai import OpenAI
        client = OpenAI()
        joined = "\n".join(f"- {b}" for b in bullets)
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a senior data scientist. "
                 "Rewrite these findings as 4-6 crisp, insightful bullet points."},
                {"role": "user", "content": f"{prompt}\n\n{joined}"},
            ],
            temperature=0.3,
        )
        text = resp.choices[0].message.content.strip()
        return [ln.lstrip("-* ").strip() for ln in text.splitlines() if ln.strip()]
    except Exception:
        return bullets


def pre_insights(df: pd.DataFrame, profile: dict, task: dict) -> list[str]:
    """Insights to show *before* modelling."""
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
            b.append(f"Target is imbalanced — '{vc.index[0]}' is {vc.iloc[0]*100:.0f}% "
                     "of rows; consider class weighting or resampling.")
    # RAG: ground with retrieved best-practice guidance relevant to this dataset
    query = (f"{task['task_type']} task "
             + ("with missing values " if profile['missing'] else "")
             + ("imbalanced classes " if task.get('imbalanced') else "")
             + "exploratory data analysis preprocessing")
    b = _llm_polish("Pre-modelling EDA findings:", b)
    return b + _guidance(query, k=2)


def post_insights(task_type: str, model_name: str, evaluation: dict,
                  dataset: str = "dataset") -> list[str]:
    """Insights to show *after* evaluation."""
    m = evaluation["metrics"]
    b = []
    if task_type == "classification":
        b.append(f"{model_name} reached {m.get('accuracy', '?')} accuracy "
                 f"(F1 {m.get('f1', '?')}) on the held-out test set.")
        if "roc_auc" in m:
            b.append(f"ROC-AUC of {m['roc_auc']} indicates "
                     f"{'strong' if m['roc_auc'] > 0.8 else 'moderate'} class separation.")
    elif task_type == "regression":
        b.append(f"{model_name} explains {m.get('r2', '?')} of variance (R²) "
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
    b.append("Next steps: try hyperparameter tuning, cross-validation, and "
             "compare against the other recommended models.")
    b = _llm_polish(f"Post-modelling results for {model_name}:", b)
    # ground the closing advice with retrieved guidance relevant to the results.
    # (run recording happens in the caller, where the full profile is available.)
    query = f"{task_type} {model_name} evaluation metrics " + " ".join(m.keys())
    return b + _guidance(query, k=2)
