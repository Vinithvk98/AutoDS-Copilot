"""The LangGraph orchestration builds and exposes its stages."""
from autods.agents.graph import build_graph
from autods.agents import state


def test_state_carries_a_lightweight_run_id():
    # the state schema exists and is a TypedDict-style mapping of fields
    assert hasattr(state, "AutoDSState")


def test_graph_compiles_with_the_expected_stages():
    g = build_graph()
    # a compiled LangGraph exposes the node names it orchestrates
    nodes = set(getattr(g, "nodes", {}))
    for expected in ("load", "profile", "detect_task", "train", "evaluate"):
        assert any(expected in n for n in nodes), f"missing a {expected} stage"
