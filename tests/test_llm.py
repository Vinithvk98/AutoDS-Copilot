"""The LLM layer and grounded generation. All offline: the provider calls are
mocked, so no key and no network are ever needed to run these."""
import autods.config as config
import autods.llm as llm
from autods.rag import generate, store
from autods.pipeline import insights


# ---- provider resolution ---------------------------------------------------
def test_defaults_to_none_offline(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_ollama_reachable", lambda: False)
    assert llm.provider() == "none"
    assert llm.available() is False
    assert llm.complete("system", "user") is None


def test_auto_detects_openai_key(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert llm.provider() == "openai"
    assert llm.available() is True


def test_forced_provider_and_model(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "LLM_MODEL", "")
    assert llm.provider() == "anthropic"
    assert llm.model_name() == "claude-3-5-haiku-latest"
    monkeypatch.setattr(config, "LLM_MODEL", "claude-sonnet-4")
    assert llm.model_name() == "claude-sonnet-4"


# ---- grounded generators ---------------------------------------------------
def test_answer_passes_question_and_passages(monkeypatch):
    seen = {}

    def fake(system, user, **kw):
        seen["u"] = user
        return "grounded reply"

    monkeypatch.setattr("autods.llm.complete", fake)
    out = generate.answer("why does accuracy mislead", [{"title": "Metric note", "text": "body"}])
    assert out == "grounded reply"
    assert "why does accuracy mislead" in seen["u"] and "[Metric note]" in seen["u"]


def test_answer_none_without_model(monkeypatch):
    monkeypatch.setattr("autods.llm.complete", lambda *a, **k: None)
    assert generate.answer("q", []) is None


def test_insights_parses_bullets(monkeypatch):
    monkeypatch.setattr("autods.llm.complete", lambda s, u, **k: "- one\n* two\n• three")
    out = generate.insights("pre", ["a verified fact"], [{"title": "T", "text": "B"}])
    assert out == ["one", "two", "three"]


# ---- wiring into insights (fallback + LLM body) ----------------------------
def test_pre_insights_offline_is_rule_based(df, prof, task, monkeypatch):
    monkeypatch.setattr(insights.llm, "available", lambda: False)
    out = insights.pre_insights(df, prof, task)
    assert out and all(isinstance(b, str) for b in out)
    assert any("rows and" in b for b in out)  # the verified finding survived


def test_pre_insights_uses_llm_body(df, prof, task, monkeypatch):
    monkeypatch.setattr(insights.llm, "available", lambda: True)
    monkeypatch.setattr(insights.generate, "insights",
                        lambda kind, facts, hits, subject="": ["LLM bullet one", "LLM bullet two"])
    out = insights.pre_insights(df, prof, task)
    assert "LLM bullet one" in out


# ---- wiring into the Ask answer and rationale ------------------------------
def test_store_answer_prefers_grounded(monkeypatch):
    monkeypatch.setattr("autods.rag.generate.answer",
                        lambda q, p, run_facts=None: "A grounded generated answer.")
    res = store.answer("why can accuracy mislead on imbalanced data")
    assert res["answer"] == "A grounded generated answer."
    assert res["sources"]  # citations still returned


def test_store_answer_falls_back(monkeypatch):
    monkeypatch.setattr("autods.rag.generate.answer", lambda q, p, run_facts=None: None)
    res = store.answer("why can accuracy mislead on imbalanced data")
    assert isinstance(res["answer"], str) and res["answer"].strip()


def test_rationale_grounded_when_llm_available(monkeypatch):
    monkeypatch.setattr("autods.llm.available", lambda: True)
    monkeypatch.setattr("autods.rag.generate.rationale", lambda d, t, s: "Because the guidance says so.")
    r = store.rationale("how should I handle missing values before modelling")
    assert r and r["text"] == "Because the guidance says so." and r["source"]


# ---- streaming ------------------------------------------------------------
def test_stream_yields_nothing_offline(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "none")
    assert list(llm.stream("s", "u")) == []


def test_answer_stream_none_without_model(monkeypatch):
    monkeypatch.setattr("autods.llm.available", lambda: False)
    assert generate.answer_stream("q", [{"title": "T", "text": "B"}]) is None


def test_answer_stream_yields_chunks(monkeypatch):
    monkeypatch.setattr("autods.llm.available", lambda: True)
    monkeypatch.setattr("autods.llm.stream", lambda s, u, **k: iter(["Hel", "lo"]))
    out = list(generate.answer_stream("q", [{"title": "T", "text": "B"}]))
    assert out == ["Hel", "lo"]


def test_retrieve_for_answer_returns_passages_and_sources():
    passages, sources = store.retrieve_for_answer("why can accuracy mislead on imbalanced data")
    assert passages and sources
    assert set(passages[0]) >= {"title", "text", "kind"}
    assert set(sources[0]) >= {"title", "kind"}


# ---- SSE endpoint (full plumbing through Flask) ---------------------------
def test_ask_stream_endpoint_offline_falls_back():
    from autods.web import app as webapp
    webapp.SESSIONS["t_offline"] = {"path": "data/sample_customers.csv"}
    client = webapp.app.test_client()
    r = client.post("/api/ask_stream/t_offline",
                    data={"question": "why can accuracy mislead on imbalanced data"})
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "event: sources" in body and "event: token" in body and "event: done" in body


def test_ask_stream_endpoint_streams_tokens(monkeypatch):
    from autods.web import app as webapp
    monkeypatch.setattr("autods.llm.available", lambda: True)
    monkeypatch.setattr("autods.llm.stream", lambda s, u, **k: iter(["Grow ", " up."]))
    webapp.SESSIONS["t_stream"] = {"path": "data/sample_customers.csv"}
    client = webapp.app.test_client()
    r = client.post("/api/ask_stream/t_stream", data={"question": "handle missing values"})
    body = r.get_data(as_text=True)
    assert '"Grow "' in body and '" up."' in body


# ---- instant pages + background enhancement --------------------------------
def test_insights_instant_vs_enhanced(df, prof, task, monkeypatch):
    # instant render must never call the model, even when one is available
    monkeypatch.setattr(insights.llm, "available", lambda: True)
    calls = []
    monkeypatch.setattr(insights.generate, "insights",
                        lambda *a, **k: calls.append(1) or ["AI bullet"])
    fast = insights.pre_insights(df, prof, task, use_llm=False)
    assert calls == [] and any("rows and" in b for b in fast)
    enhanced = insights.pre_insights(df, prof, task, use_llm=True)
    assert "AI bullet" in enhanced and calls  # the model was used this time


def test_enhance_endpoint_noop_offline(monkeypatch):
    from autods.web import app as webapp
    monkeypatch.setattr("autods.llm.available", lambda: False)
    webapp.SESSIONS["e_off"] = {"path": "data/sample_customers.csv"}
    r = webapp.app.test_client().get("/api/enhance/e_off/pre")
    assert r.status_code == 200 and r.get_json()["html"] is None


def test_enhance_endpoint_returns_and_caches(df, prof, task, monkeypatch):
    from autods.web import app as webapp
    monkeypatch.setattr("autods.llm.available", lambda: True)
    monkeypatch.setattr(insights.generate, "insights",
                        lambda *a, **k: ["Enhanced insight bullet"])
    webapp.SESSIONS["e_on"] = {"path": "data/sample_customers.csv",
                               "df": df, "profile": prof, "task": task}
    c = webapp.app.test_client()
    r = c.get("/api/enhance/e_on/pre")
    html = r.get_json()["html"]
    assert html and "Enhanced insight bullet" in html
    assert webapp.SESSIONS["e_on"]["_enh"]["pre"] == html  # cached for next visit
