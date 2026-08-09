"""The LangGraph state graph that orchestrates the pipeline as agents.

Flow (mirrors the original notebook sketch):

    load -> profile -> eda -> pre_insights -> detect_task -> recommend_clean
      -> [HUMAN: approve cleaning] -> apply_clean -> recommend_models
      -> [HUMAN: choose model] -> train -> evaluate -> post_insights -> END

Run from the CLI:
    python -m autods.agents.graph data/sample_customers.csv --target churned
    python -m autods.agents.graph data/sample_customers.csv --auto
"""
from __future__ import annotations
import uuid
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from .state import AutoDSState
from ..pipeline import (loader, preclean, task_detect, eda, preprocess,
                        model_select, train, evaluate, insights)

# In-memory store for non-serializable artifacts (DataFrames, fitted models).
# The checkpointed graph state only carries a lightweight `run_id` that keys
# into this store, so state stays msgpack-serializable for interrupts/resume.
_ARTIFACTS: dict[str, dict] = {}


def _store(state: AutoDSState) -> dict:
    return _ARTIFACTS.setdefault(state["run_id"], {})


def _log(state: AutoDSState, msg: str) -> list:
    return state.get("log", []) + [msg]


# ---- Agent nodes -----------------------------------------------------------
def load_node(state: AutoDSState) -> dict:
    run_id = state.get("run_id") or str(uuid.uuid4())
    df = loader.load_data(state["data_path"])
    df, changes = preclean.preclean(df)
    _ARTIFACTS.setdefault(run_id, {})["df"] = df
    log = _log(state, f"[Loader] loaded {df.shape[0]}x{df.shape[1]} from {state['data_path']}")
    log = log + [f"[Pre-clean] {c}" for c in changes]
    return {"run_id": run_id, "log": log}


def profile_node(state: AutoDSState) -> dict:
    prof = loader.profile(_store(state)["df"])
    return {"profile": prof, "log": _log(state, "[Profiler] computed dataset profile")}


def eda_node(state: AutoDSState) -> dict:
    charts = eda.generate_eda(_store(state)["df"], target=state.get("target"))
    return {"eda_charts": charts, "log": _log(state, f"[EDA agent] generated {len(charts)} charts")}


def detect_task_node(state: AutoDSState) -> dict:
    task = task_detect.detect_task(_store(state)["df"], target=state.get("target"))
    return {"task": task, "target": task["target"],
            "log": _log(state, f"[Task detector] {task['task_type']} (target={task['target']})")}


def pre_insights_node(state: AutoDSState) -> dict:
    ins = insights.pre_insights(_store(state)["df"], state["profile"], state["task"])
    return {"pre_insights": ins, "log": _log(state, "[Insight agent] wrote pre-modelling insights")}


def recommend_clean_node(state: AutoDSState) -> dict:
    plan = preprocess.recommend_plan(_store(state)["df"], state["task"]["target"])
    return {"plan": plan, "log": _log(state, "[Cleaning agent] proposed a preprocessing plan")}


def approve_clean_node(state: AutoDSState) -> dict:
    """HUMAN-IN-THE-LOOP #1: approve/edit the cleaning plan."""
    if state.get("auto"):
        return {"approved_plan": state["plan"],
                "log": _log(state, "[Human gate] auto-approved cleaning plan")}
    decision = interrupt({"type": "approve_cleaning", "plan": state["plan"]})
    approved = decision if isinstance(decision, dict) else state["plan"]
    return {"approved_plan": approved,
            "log": _log(state, "[Human gate] cleaning plan approved by user")}


def apply_clean_node(state: AutoDSState) -> dict:
    store = _store(state)
    df_clean = preprocess.apply_plan(store["df"], state["approved_plan"])
    store["df"] = df_clean
    return {"log": _log(state, f"[Cleaning agent] applied plan -> {df_clean.shape}")}


def recommend_models_node(state: AutoDSState) -> dict:
    cands = model_select.recommend_models(state["task"]["task_type"])
    return {"model_candidates": cands,
            "log": _log(state, f"[Model recommender] proposed {len(cands)} candidates")}


def choose_model_node(state: AutoDSState) -> dict:
    """HUMAN-IN-THE-LOOP #2: choose the model."""
    default = next((m["name"] for m in state["model_candidates"] if m.get("recommended")),
                   state["model_candidates"][0]["name"])
    if state.get("auto"):
        return {"chosen_model": default,
                "log": _log(state, f"[Human gate] auto-chose {default}")}
    choice = interrupt({"type": "choose_model", "candidates": state["model_candidates"],
                        "default": default})
    chosen = choice if isinstance(choice, str) and choice else default
    return {"chosen_model": chosen,
            "log": _log(state, f"[Human gate] user chose {chosen}")}


def train_node(state: AutoDSState) -> dict:
    store = _store(state)
    t = state["task"]
    trained = train.train_model(store["df"], t["task_type"], state["chosen_model"],
                                target=t["target"], time_col=t.get("time_col"),
                                text_col=t.get("text_col"), balanced=t.get("imbalanced", False))
    store["trained"] = trained
    return {"log": _log(state, f"[Trainer] trained {state['chosen_model']}")}


def evaluate_node(state: AutoDSState) -> dict:
    ev = evaluate.evaluate(_store(state)["trained"])
    return {"evaluation": ev, "log": _log(state, f"[Evaluator] metrics: {ev['metrics']}")}


def post_insights_node(state: AutoDSState) -> dict:
    import os
    ins = insights.post_insights(state["task"]["task_type"], state["chosen_model"],
                                 state["evaluation"], dataset=os.path.basename(state["data_path"]))
    return {"post_insights": ins, "log": _log(state, "[Insight agent] wrote post-modelling insights")}


# ---- Graph wiring ----------------------------------------------------------
def build_graph(checkpointer=None):
    g = StateGraph(AutoDSState)
    for name, fn in [
        ("load", load_node), ("profile", profile_node), ("eda", eda_node),
        ("detect_task", detect_task_node), ("pre_insights", pre_insights_node),
        ("recommend_clean", recommend_clean_node), ("approve_clean", approve_clean_node),
        ("apply_clean", apply_clean_node), ("recommend_models", recommend_models_node),
        ("choose_model", choose_model_node), ("train", train_node),
        ("evaluate", evaluate_node), ("post_insights", post_insights_node),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "load")
    g.add_edge("load", "profile")
    g.add_edge("profile", "eda")
    g.add_edge("eda", "detect_task")
    g.add_edge("detect_task", "pre_insights")
    g.add_edge("pre_insights", "recommend_clean")
    g.add_edge("recommend_clean", "approve_clean")
    g.add_edge("approve_clean", "apply_clean")
    g.add_edge("apply_clean", "recommend_models")
    g.add_edge("recommend_models", "choose_model")
    g.add_edge("choose_model", "train")
    g.add_edge("train", "evaluate")
    g.add_edge("evaluate", "post_insights")
    g.add_edge("post_insights", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())


# ---- CLI runner (demonstrates interrupts + resumption) ---------------------
def _run_cli():
    import argparse, uuid
    ap = argparse.ArgumentParser(description="Run the AutoDS agent graph")
    ap.add_argument("data_path")
    ap.add_argument("--target", default=None)
    ap.add_argument("--auto", action="store_true", help="auto-approve human gates")
    args = ap.parse_args()

    app = build_graph()
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = {"data_path": args.data_path, "target": args.target, "auto": args.auto}

    result = app.invoke(state, cfg)
    # Resolve any human interrupts interactively
    while "__interrupt__" in result:
        intr = result["__interrupt__"][0].value
        if intr["type"] == "approve_cleaning":
            print("\n--- CLEANING PLAN ---")
            for n in intr["plan"]["notes"]:
                print("  •", n)
            input("Press Enter to approve (or Ctrl-C to abort)... ")
            result = app.invoke(Command(resume=intr["plan"]), cfg)
        elif intr["type"] == "choose_model":
            print("\n--- MODEL CANDIDATES ---")
            for i, m in enumerate(intr["candidates"]):
                star = " (recommended)" if m.get("recommended") else ""
                print(f"  [{i}] {m['name']}{star} — {m['why']}")
            sel = input(f"Choose model [default {intr['default']}]: ").strip()
            chosen = intr["candidates"][int(sel)]["name"] if sel.isdigit() else intr["default"]
            result = app.invoke(Command(resume=chosen), cfg)

    print("\n=== RUN COMPLETE ===")
    print("Task:", result["task"]["task_type"], "| Model:", result["chosen_model"])
    print("Metrics:", result["evaluation"]["metrics"])
    print("\nInsights:")
    for b in result["post_insights"]:
        print("  •", b)
    print("\nAgent trace:")
    for line in result["log"]:
        print("  ", line)


if __name__ == "__main__":
    _run_cli()
