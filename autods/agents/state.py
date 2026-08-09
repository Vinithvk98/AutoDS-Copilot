"""Shared state passed between agent nodes in the LangGraph graph."""
from __future__ import annotations
from typing import TypedDict, Optional, Any


class AutoDSState(TypedDict, total=False):
    # inputs
    data_path: str
    target: Optional[str]
    auto: bool               # if True, auto-approve human-in-the-loop steps
    run_id: str              # keys into the non-serializable artifact store

    # produced by nodes
    df: Any                  # loaded / cleaned DataFrame
    profile: dict
    eda_charts: list
    pre_insights: list
    task: dict
    plan: dict               # recommended cleaning plan
    approved_plan: dict      # plan after human review
    model_candidates: list
    chosen_model: str        # model after human choice
    trained: dict
    evaluation: dict
    post_insights: list
    log: list                # human-readable trace of which agent did what
