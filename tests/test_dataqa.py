"""The conversational data copilot: safe analysis tools + LLM routing.
All offline: the router's model call is mocked."""
import json
from autods import dataqa, converse


# ---- safe analysis tools ---------------------------------------------------
def test_schema_lists_columns_and_types(df):
    sch = dataqa.schema(df)
    assert sch["n_rows"] == len(df)
    names = {c["name"]: c["type"] for c in sch["columns"]}
    assert names["income"] == "numeric" and names["plan"] == "categorical"


def test_group_stat_churn_rate_by_plan(df):
    out = dataqa.run_tool(df, "group_stat", {"by": "plan", "column": "churned", "agg": "mean"})
    assert "data" in out and out["data"]
    # every value is a real proportion between 0 and 1
    assert all(0.0 <= v <= 1.0 for v in out["data"].values())


def test_rate_matches_manual(df):
    out = dataqa.run_tool(df, "rate", {"column": "churned"})
    assert abs(out["data"]["rate"] - round(float(df["churned"].mean()), 4)) < 1e-6


def test_correlation_and_column_summary(df):
    c = dataqa.run_tool(df, "correlation", {"column_a": "age", "column_b": "income"})
    assert "correlation" in c["data"]
    s = dataqa.run_tool(df, "column_summary", {"column": "income"})
    assert s["data"]["min"] <= s["data"]["mean"] <= s["data"]["max"]


def test_tools_reject_bad_input(df):
    assert "error" in dataqa.run_tool(df, "column_summary", {"column": "nope"})
    assert "error" in dataqa.run_tool(df, "group_stat", {"by": "plan", "column": "income", "agg": "hack"})
    assert "error" in dataqa.run_tool(df, "made_up_tool", {})


# ---- routing ---------------------------------------------------------------
def test_route_parses_tool_call(df, monkeypatch):
    monkeypatch.setattr("autods.llm.complete",
                        lambda s, u, **k: 'sure: {"tool": "group_stat", "args": {"by": "plan", "column": "churned", "agg": "mean"}}')
    call = converse.route("churn by plan?", df)
    assert call["tool"] == "group_stat" and call["args"]["by"] == "plan"


def test_route_none_for_conceptual(df, monkeypatch):
    monkeypatch.setattr("autods.llm.complete", lambda s, u, **k: '{"tool": "none"}')
    assert converse.route("what is overfitting", df) is None


def test_route_survives_garbage(df, monkeypatch):
    monkeypatch.setattr("autods.llm.complete", lambda s, u, **k: "I cannot help with that")
    assert converse.route("hello", df) is None


def test_analyze_returns_grounded_facts(df, monkeypatch):
    monkeypatch.setattr("autods.llm.available", lambda: True)
    monkeypatch.setattr("autods.llm.complete",
                        lambda s, u, **k: '{"tool": "rate", "args": {"column": "churned"}}')
    facts, meta = converse.analyze("what is the churn rate", df)
    assert facts and "Computed directly from the dataset" in facts
    assert meta["tool"] == "rate" and "rate" in meta["result"]["data"]


def test_analyze_noop_without_model(df, monkeypatch):
    monkeypatch.setattr("autods.llm.available", lambda: False)
    assert converse.analyze("churn by plan", df) == (None, None)


# ---- suggested questions ---------------------------------------------------
def test_suggest_questions_from_schema(df, task):
    qs = converse.suggest_questions(df, task)
    assert qs and any("rate by" in q for q in qs)
    assert any("drives" in q for q in qs) and any("correlate" in q for q in qs)


# ---- multi-step why analysis ----------------------------------------------
def test_multi_why_chains_and_ranks(df, task, monkeypatch):
    monkeypatch.setattr("autods.llm.available", lambda: True)
    facts, meta = converse.analyze("why do customers churn", df, task=task)
    assert meta["multi"] is True and meta["blocks"]
    spreads = [b["spread"] for b in meta["blocks"]]
    assert spreads == sorted(spreads, reverse=True)      # ranked by how much it moves
    assert "Driver analysis" in facts and meta["correlations"]


# ---- clarify ---------------------------------------------------------------
def test_clarify_on_vague_data_question(df, task, monkeypatch):
    monkeypatch.setattr("autods.llm.available", lambda: True)
    monkeypatch.setattr("autods.llm.complete", lambda s, u, **k: '{"tool": "none"}')
    facts, meta = converse.analyze("show the breakdown by region", df, task=task)
    assert facts is None and meta.get("clarify") and meta.get("options")


def test_conceptual_question_is_not_clarified(df, task, monkeypatch):
    monkeypatch.setattr("autods.llm.available", lambda: True)
    monkeypatch.setattr("autods.llm.complete", lambda s, u, **k: '{"tool": "none"}')
    # a general concept with no data words falls through to the knowledge base
    assert converse.analyze("what is overfitting", df, task=task) == (None, None)


# ---- endpoint: data-aware streaming ---------------------------------------
def test_ask_stream_is_data_aware(df, monkeypatch):
    from autods.web import app as webapp
    monkeypatch.setattr("autods.llm.available", lambda: True)
    # router picks a tool; answer streams a token
    monkeypatch.setattr("autods.converse.route",
                        lambda q, d, history=None: {"tool": "group_stat",
                                                    "args": {"by": "plan", "column": "churned", "agg": "mean"}})
    monkeypatch.setattr("autods.llm.stream", lambda s, u, **k: iter(["Standard ", "churns most."]))
    webapp.SESSIONS["dq"] = {"path": "data/sample_customers.csv", "df": df}
    c = webapp.app.test_client()
    body = c.post("/api/ask_stream/dq", data={"question": "which plan churns most"}).get_data(as_text=True)
    assert "event: compute" in body and "group_stat" in body
    assert '"Standard "' in body
    # the turn was recorded for multi-turn context
    assert webapp.SESSIONS["dq"]["chat"][-1]["q"] == "which plan churns most"
