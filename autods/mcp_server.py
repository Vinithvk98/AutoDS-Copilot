"""MCP server — exposes the AutoDS pipeline as tools for any MCP client.

Each pipeline stage becomes a callable tool, so an external agent (Claude, an
IDE, another orchestrator) can drive AutoDS over the Model Context Protocol.

Run it:
    python -m autods.mcp_server        # stdio transport (default)

Register with an MCP client (example config):
    {
      "mcpServers": {
        "autods": {"command": "python", "args": ["-m", "autods.mcp_server"]}
      }
    }
"""
from __future__ import annotations
from pathlib import Path
from mcp.server.mcpserver import MCPServer

from . import config
from .pipeline import (loader, preclean, task_detect, eda, preprocess, model_select,
                       train, evaluate, insights, leaderboard)
from .rag import get_retriever, record_run, answer as rag_answer

server = MCPServer(
    name="autods-copilot",
    instructions="AutoDS Copilot: an AI data-science assistant. Point these tools "
                 "at a tabular dataset (CSV/Excel path) to profile it, run EDA, "
                 "recommend cleaning and models, train, evaluate, and get insights.",
)


def _resolve(path: str) -> str:
    """Resolve a dataset path so relative names work regardless of the client's
    working directory: try as-given, then relative to the project and its data dir."""
    p = Path(path)
    if p.exists():
        return str(p)
    for base in (config.ROOT, config.DATA_DIR):
        cand = base / path
        if cand.exists():
            return str(cand)
    return path  # let the loader raise a clear error


def _df(path: str):
    df, _ = preclean.preclean(loader.load_data(_resolve(path)))
    return df


@server.tool(description="Load a dataset and return its profile (shape, dtypes, missing values, duplicates).")
def profile_dataset(path: str) -> dict:
    return loader.profile(_df(path))


@server.tool(description="Detect the ML task (classification/regression/clustering/timeseries/text), target, and special columns.")
def detect_task(path: str, target: str = "") -> dict:
    return task_detect.detect_task(_df(path), target=target or None)


@server.tool(description="Run EDA and return each chart's title, the code behind it, an insight, and the saved image path.")
def run_eda(path: str, target: str = "") -> dict:
    charts = eda.generate_eda(_df(path), target=target or None)
    return {"charts": [{"title": c["title"], "code": c["code"], "insight": c["insight"],
                        "image": c["path"]} for c in charts]}


@server.tool(description="Recommend a data-cleaning plan (columns to drop, imputation strategies, duplicates).")
def recommend_cleaning(path: str, target: str = "") -> dict:
    df = _df(path)
    task = task_detect.detect_task(df, target=target or None)
    return preprocess.recommend_plan(df, task["target"])


@server.tool(description="Cross-validate all candidate models and return a ranked leaderboard for the detected task.")
def model_leaderboard(path: str, target: str = "") -> dict:
    df = _df(path)
    task = task_detect.detect_task(df, target=target or None)
    return leaderboard.run_leaderboard(df, task["task_type"], task["target"],
                                       time_col=task.get("time_col"), text_col=task.get("text_col"))


@server.tool(description="Train a model (auto-picks the recommended one if unspecified), evaluate it, and return metrics + insights.")
def train_and_evaluate(path: str, target: str = "", model: str = "") -> dict:
    df = _df(path)
    task = task_detect.detect_task(df, target=target or None)
    recs = model_select.recommend_models(task["task_type"])
    model_name = model or next((m["name"] for m in recs if m.get("recommended")), recs[0]["name"])
    trained = train.train_model(df, task["task_type"], model_name, target=task["target"],
                                time_col=task.get("time_col"), text_col=task.get("text_col"),
                                balanced=task.get("imbalanced", False))
    ev = evaluate.evaluate(trained)
    dataset = path.split("/")[-1]
    post = insights.post_insights(task["task_type"], model_name, ev, dataset=dataset)
    # remember this run so future runs can learn from it
    try:
        record_run(dataset, task, model_name, ev["metrics"], profile=loader.profile(df))
    except Exception:
        pass
    return {"task_type": task["task_type"], "model": model_name,
            "metrics": ev["metrics"], "feature_importance": ev["feature_importance"],
            "insights": post}


@server.tool(description="Run the full AutoDS lifecycle end-to-end (profile → task → insights → cleaning → leaderboard → train → evaluate) and return a complete report.")
def full_run(path: str, target: str = "") -> dict:
    df = _df(path)
    profile = loader.profile(df)
    task = task_detect.detect_task(df, target=target or None)
    pre = insights.pre_insights(df, profile, task)
    plan = preprocess.recommend_plan(df, task["target"])
    board = leaderboard.run_leaderboard(df, task["task_type"], task["target"],
                                        time_col=task.get("time_col"), text_col=task.get("text_col"))
    best = next((r["name"] for r in board["rows"] if r.get("best")), None)
    result = train_and_evaluate(path, target=target, model=best or "")
    return {"profile": {k: profile[k] for k in ("n_rows", "n_cols", "missing")},
            "task": task, "pre_insights": pre, "cleaning_plan": plan,
            "leaderboard": board, "result": result}


@server.tool(description="Retrieve relevant data-science best-practice guidance for a free-text query (RAG over the knowledge base + past runs).")
def guidance(query: str, k: int = 3) -> dict:
    return {"results": get_retriever().query(query, k=k)}


@server.tool(description="Ask a data-science question and get an answer grounded only in the knowledge base and past runs, with the sources it used.")
def ask(question: str) -> dict:
    return rag_answer(question)


def main():
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
