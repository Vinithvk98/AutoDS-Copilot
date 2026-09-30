"""The retrieval layer.

TF-IDF vector store with cosine similarity over the knowledge base and a growing
memory of past runs. Provides:
  - retrieval with citations (which document an answer came from),
  - experience retrieval, finding similar past runs by a numeric fingerprint,
  - run-fact documents so the Ask panel can answer about the current run,
  - grounded answering that only uses retrieved material.

Everything is lexical TF-IDF by default, so it needs no API key and runs offline.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .. import config
from .knowledge import KNOWLEDGE_BASE

RUNS_FILE = config.OUTPUT_DIR / "run_memory.jsonl"


# ----------------------------------------------------------------------------
#  Retriever
# ----------------------------------------------------------------------------
class Retriever:
    """TF-IDF retriever over a list of {id, title, text, kind} documents."""

    def __init__(self, docs: list[dict]):
        self.docs = docs
        self._vectorizer = TfidfVectorizer(stop_words="english")
        corpus = [f"{d['title']}. {d.get('tags','')} {d['text']}" for d in docs]
        self._matrix = self._vectorizer.fit_transform(corpus) if corpus else None

    def query(self, text: str, k: int = 3, min_score: float = 0.04) -> list[dict]:
        if self._matrix is None or not text.strip():
            return []
        qv = self._vectorizer.transform([text])
        sims = cosine_similarity(qv, self._matrix)[0]
        out = []
        for i in sims.argsort()[::-1][:k]:
            if sims[i] >= min_score:
                d = self.docs[i]
                out.append({"id": d.get("id"), "title": d["title"], "text": d["text"],
                            "kind": d.get("kind", "guidance"), "score": round(float(sims[i]), 3)})
        return out


# ----------------------------------------------------------------------------
#  Run memory (experience)
# ----------------------------------------------------------------------------
def fingerprint(profile: dict, task: dict) -> dict:
    """A compact numeric signature of a dataset + task, for similarity search."""
    n_rows = int(profile.get("n_rows", 0)) or 1
    n_cols = int(profile.get("n_cols", 0)) or 1
    n_num = len(profile.get("types", {}).get("numeric", []))
    n_cat = len(profile.get("types", {}).get("categorical", []))
    missing_total = sum(profile.get("missing", {}).values())
    return {
        "task_type": task.get("task_type"),
        "n_rows": n_rows, "n_cols": n_cols,
        "n_numeric": n_num, "n_categorical": n_cat,
        "numeric_ratio": round(n_num / n_cols, 3),
        "missing_frac": round(missing_total / (n_rows * n_cols), 4),
        "imbalanced": bool(task.get("imbalanced")),
    }


def record_run(dataset: str, task: dict, model: str, metrics: dict,
               profile: dict | None = None, note: str = "") -> None:
    """Append a structured run summary to memory so future runs can learn from it."""
    import time
    metric_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
    fp = fingerprint(profile or {}, task) if profile is not None else {"task_type": task.get("task_type")}
    entry = {
        "ts": time.strftime("%Y%m%d-%H%M%S"),
        "dataset": dataset, "model": model, "metrics": metrics,
        "task_type": task.get("task_type"), "target": task.get("target"),
        "fingerprint": fp,
        "title": f"{task.get('task_type')} on {dataset} with {model}",
        "text": (f"A {task.get('task_type')} task on {dataset} trained {model} and "
                 f"scored {metric_str}. {note}").strip(),
    }
    RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_run_memory() -> list[dict]:
    if not RUNS_FILE.exists():
        return []
    runs = []
    for line in RUNS_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            runs.append(json.loads(line))
        except Exception:
            continue
    return runs


def _fp_vector(fp: dict) -> list[float]:
    return [math.log10(max(fp.get("n_rows", 1), 1)),
            math.log10(max(fp.get("n_cols", 1), 1)),
            fp.get("numeric_ratio", 0.0),
            fp.get("missing_frac", 0.0),
            1.0 if fp.get("imbalanced") else 0.0]


def similar_runs(fp: dict, k: int = 3, exclude_ts: str | None = None) -> list[dict]:
    """Return past runs on the most similar datasets (same task), best first."""
    runs = [r for r in load_run_memory()
            if r.get("fingerprint", {}).get("task_type") == fp.get("task_type")
            and r.get("ts") != exclude_ts]
    if not runs:
        return []
    import numpy as np
    cur = np.array(_fp_vector(fp), dtype=float)
    mat = np.array([_fp_vector(r["fingerprint"]) for r in runs], dtype=float)
    rng = mat.max(0) - mat.min(0)
    rng[rng == 0] = 1.0
    curn = (cur - mat.min(0)) / rng
    matn = (mat - mat.min(0)) / rng
    dist = np.sqrt(((matn - curn) ** 2).sum(1))
    out = []
    for i in dist.argsort()[:k]:
        r = runs[int(i)]
        out.append({"dataset": r.get("dataset"), "model": r.get("model"),
                    "metrics": r.get("metrics", {}),
                    "similarity": round(float(1 / (1 + dist[int(i)])), 3)})
    return out


# ----------------------------------------------------------------------------
#  Retriever cache + convenience
# ----------------------------------------------------------------------------
_CACHE: dict = {}


def _run_docs_from_memory() -> list[dict]:
    return [{"id": f"run-{r.get('ts','')}", "title": f"Past run: {r['title']}",
             "text": r["text"], "kind": "run"} for r in load_run_memory()]


def get_retriever(include_memory: bool = True) -> Retriever:
    mem = _run_docs_from_memory() if include_memory else []
    key = len(mem)
    if _CACHE.get("key") != key:
        _CACHE["retriever"] = Retriever(KNOWLEDGE_BASE + mem)
        _CACHE["key"] = key
    return _CACHE["retriever"]


def rationale(query: str, use_llm: bool | None = None) -> dict | None:
    """Return a grounded "why" for a decision as {text, source}, or None.

    ``use_llm`` mirrors the insight functions: ``None`` uses a model when
    configured, the web layer passes ``False`` for an instant rule-based snippet
    and re-requests with ``True`` in the background."""
    hits = [h for h in get_retriever().query(query, k=3) if h["kind"] == "guidance"]
    if not hits:
        return None
    top = hits[0]
    if use_llm is None:
        from .. import llm
        use_llm = llm.available()
    if use_llm:
        from . import generate
        text = generate.rationale(query, top["text"], top["title"])
        if text:
            return {"text": text, "source": top["title"]}
    return {"text": top["text"], "source": top["title"]}


# ----------------------------------------------------------------------------
#  Ask panel, grounded answering over KB + current run facts
# ----------------------------------------------------------------------------
def build_run_docs(session: dict) -> list[dict]:
    """Turn the current run's artifacts into retrievable documents."""
    prof = session.get("profile", {})
    task = session.get("task", {})
    metrics = session.get("evaluation", {}).get("metrics", {})
    fi = session.get("evaluation", {}).get("feature_importance", [])
    checks = session.get("checks", [])
    model = session.get("model_name")
    dataset = Path(session.get("path", "the dataset")).name
    docs = []
    if prof:
        miss = ", ".join(prof.get("missing", {}).keys()) or "none"
        docs.append({"id": "run-data", "title": f"This dataset ({dataset})", "kind": "run",
                     "text": f"The dataset {dataset} has {prof.get('n_rows')} rows and "
                             f"{prof.get('n_cols')} columns. Columns with missing values, {miss}. "
                             f"The task is {task.get('task_type')} on target {task.get('target')}."})
    if metrics:
        ms = ", ".join(f"{k} {v}" for k, v in metrics.items())
        docs.append({"id": "run-model", "title": "The trained model", "kind": "run",
                     "text": f"{model} was trained for a {task.get('task_type')} task and "
                             f"scored {ms} on the held-out test."})
    if fi:
        tops = ", ".join(d["feature"].split("__")[-1].replace("_", " ") for d in fi[:5])
        docs.append({"id": "run-features", "title": "Most influential features", "kind": "run",
                     "text": f"The prediction leans most on {tops}."})
    if checks:
        cs = "; ".join(f"{c['title']}" for c in checks[:6])
        docs.append({"id": "run-checks", "title": "Data quality findings", "kind": "run",
                     "text": f"The data checks flagged, {cs}."})
    return docs


def retrieve_for_answer(question: str, session: dict | None = None, k: int = 3):
    """Run retrieval for a question and return ``(passages, sources)``.

    Split out from :func:`answer` so the web layer can retrieve once and then
    stream a grounded answer over the same passages. ``passages`` carry the text
    for grounding, ``sources`` are the lighter title/kind pairs for citation.
    """
    docs = list(KNOWLEDGE_BASE) + _run_docs_from_memory()
    if session:
        docs = build_run_docs(session) + docs
    hits = Retriever(docs).query(question, k=k)
    passages = [{"title": h["title"], "text": h["text"], "kind": h["kind"]} for h in hits]
    sources = [{"title": h["title"], "kind": h["kind"]} for h in hits]
    return passages, sources


def _extractive(passages: list[dict]) -> str:
    """Fallback answer: stitch the top retrieved passages together."""
    if not passages:
        return ("I could not find anything relevant in the knowledge base or this "
                "run to answer that.")
    lead = passages[0]["text"]
    if len(passages) > 1:
        lead += " " + passages[1]["text"]
    return lead


def answer(question: str, session: dict | None = None, k: int = 3) -> dict:
    """Answer a question using only retrieved material. Returns
    {answer, passages, sources}. Uses the grounded LLM if a model is configured,
    otherwise stitches the top passages together extractively."""
    passages, sources = retrieve_for_answer(question, session, k)
    if not passages:
        return {"answer": _extractive(passages), "passages": [], "sources": []}

    # Grounded LLM answer over the retrieved passages (which already include the
    # current run's facts when a session is supplied). Falls back if no model.
    from . import generate
    text = generate.answer(question, passages)
    return {"answer": text or _extractive(passages),
            "passages": passages, "sources": sources}
