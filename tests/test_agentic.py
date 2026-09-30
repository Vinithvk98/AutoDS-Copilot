"""The agentic planner and auto-pilot. Offline; the model call is mocked."""
from autods import planner
from autods.pipeline import task_detect, preprocess, model_select


def _ctx(df):
    task = task_detect.detect_task(df, target="churned")
    plan = preprocess.recommend_plan(df, "churned", task=task)
    candidates = model_select.recommend_models(task["task_type"])
    return task, plan, candidates


def test_plan_offline_uses_heuristics(df, monkeypatch):
    monkeypatch.setattr("autods.llm.available", lambda: False)
    task, plan, cands = _ctx(df)
    p = planner.propose(df, {}, task, plan, cands)
    assert p["target"] == "churned"
    assert p["model"] in [c["name"] for c in cands]
    assert all(c in df.columns for c in p["drop_columns"])
    assert p["why"]["model"]  # a reason is present


def test_plan_llm_choices_are_validated(df, monkeypatch):
    monkeypatch.setattr("autods.llm.available", lambda: True)
    # the model tries to pick a bogus model and a non-existent column
    monkeypatch.setattr("autods.llm.complete",
                        lambda s, u, **k: '{"model": "NotARealModel", "drop_columns": ["ghost"], '
                                          '"reason_model": "because", "reason_cleaning": "tidy", '
                                          '"reason_target": "obvious"}')
    task, plan, cands = _ctx(df)
    p = planner.propose(df, {}, task, plan, cands)
    assert p["model"] in [c["name"] for c in cands]     # bogus model rejected
    assert "ghost" not in p["drop_columns"]              # bogus column rejected
    assert p["why"]["model"] == "because"                # the grounded reason kept


def test_plan_picks_best_by_leaderboard(df, monkeypatch):
    monkeypatch.setattr("autods.llm.available", lambda: False)
    task, plan, cands = _ctx(df)
    board = {"metric": "f1_weighted", "rows": [
        {"name": "GaussianNB", "score": 0.81},
        {"name": "GradientBoostingClassifier", "score": 0.79}]}
    p = planner.propose(df, {}, task, plan, cands, board=board)
    assert p["model"] == "GaussianNB" and p["score"] == 0.81
    assert "GaussianNB" in p["why"]["model"] and "0.81" in p["why"]["model"]


def test_plan_llm_valid_choice_is_used(df, monkeypatch):
    task, plan, cands = _ctx(df)
    pick = cands[-1]["name"]
    monkeypatch.setattr("autods.llm.available", lambda: True)
    monkeypatch.setattr("autods.llm.complete",
                        lambda s, u, **k: '{"model": "%s", "drop_columns": []}' % pick)
    p = planner.propose(df, {}, task, plan, cands)
    assert p["model"] == pick


# ---- endpoint + full auto-pilot run ---------------------------------------
def test_autopilot_endpoint_and_run(df, monkeypatch):
    monkeypatch.setattr("autods.llm.available", lambda: False)
    from autods.web import app as webapp
    c = webapp.app.test_client()
    sid = c.post("/api/upload", data={"sample": "sample_customers.csv"}).get_json()["sid"]
    c.get(f"/api/explore/{sid}")
    # agent proposes a plan
    ap = c.get(f"/api/autopilot/{sid}")
    html = ap.get_json()["html"]
    assert "The copilot's plan" in html and "autoplan" in html
    plan = webapp.SESSIONS[sid]["autoplan"]
    assert plan["model"] and plan["target"] == "churned"
    # approve-and-run drives the real endpoints: apply plan then train
    fd1 = {"drop": plan["drop_columns"], "drop_duplicates": "on" if plan["drop_duplicates"] else ""}
    c.post(f"/api/models/{sid}", data=fd1)
    res = c.post(f"/api/train/{sid}", data={"model": plan["model"]})
    assert res.status_code == 200 and "insights" in res.get_json()["html"].lower()
