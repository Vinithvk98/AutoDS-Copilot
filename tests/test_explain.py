"""Explainability and trust: permutation importance, error analysis, executive
summary. Offline; the LLM is mocked."""
from autods.pipeline import loader, train, evaluate, explain
from autods.rag import generate


def _trained(df, model="RandomForestClassifier"):
    return train.train_model(df, "classification", model, target="churned")


# ---- permutation importance (model-agnostic) ------------------------------
def test_permutation_importance_names_original_columns(df):
    tr = _trained(df)
    perm = explain.permutation_importance(tr, n_repeats=3)
    assert perm and set(perm[0]) >= {"feature", "importance", "std"}
    # features are the ORIGINAL columns, not one-hot encoded names
    assert all("__" not in p["feature"] for p in perm)
    assert all(c in df.columns for c in [p["feature"] for p in perm])


def test_permutation_works_for_model_without_native_importance(df):
    tr = _trained(df, model="KNeighborsClassifier")  # no coef_ or feature_importances_
    assert explain.permutation_importance(tr, n_repeats=3)  # still explains it


# ---- error analysis --------------------------------------------------------
def test_error_analysis_classification(df):
    tr = _trained(df)
    errs = explain.error_analysis(tr)
    assert 0.0 <= errs["error_rate"] <= 1.0
    assert errs["n_errors"] <= errs["n_test"]
    if errs["by_group"]:
        assert "feature" in errs["by_group"][0] and "rows" in errs["by_group"][0]


def test_error_analysis_regression_worst_residuals():
    import pandas as pd
    d = pd.DataFrame({"x": range(200), "y": [i * 2.0 + (i % 7) for i in range(200)]})
    tr = train.train_model(d, "regression", "LinearRegression", target="y")
    errs = explain.error_analysis(tr)
    assert "worst" in errs and errs["worst"]
    assert errs["worst"][0]["abs_error"] >= errs["worst"][-1]["abs_error"]


# ---- evaluate wires them in -----------------------------------------------
def test_evaluate_includes_explainability(df):
    tr = _trained(df)
    ev = evaluate.evaluate(tr)
    assert "permutation" in ev["data"] and "errors" in ev["data"]


# ---- executive summary + comparison (grounded) ----------------------------
def test_executive_summary_none_offline(monkeypatch):
    monkeypatch.setattr("autods.llm.available", lambda: False)
    assert generate.executive_summary(["a fact"]) is None


def test_executive_summary_grounded(monkeypatch):
    monkeypatch.setattr("autods.llm.available", lambda: True)
    seen = {}

    def fake(system, user, **kw):
        seen["u"] = user
        return "This model predicts churn well."

    monkeypatch.setattr("autods.llm.complete", fake)
    out = generate.executive_summary(["Model RandomForest.", "accuracy 0.87"])
    assert out == "This model predicts churn well."
    assert "accuracy 0.87" in seen["u"]


def test_enhance_summary_endpoint(df, monkeypatch):
    from autods.web import app as webapp
    from autods.pipeline import task_detect
    monkeypatch.setattr("autods.llm.available", lambda: True)
    monkeypatch.setattr("autods.rag.generate.executive_summary",
                        lambda facts, passages=None: "A grounded summary.")
    task = task_detect.detect_task(df, target="churned")
    webapp.SESSIONS["sum1"] = {"path": "data/sample_customers.csv", "task": task,
                               "model_name": "RandomForestClassifier",
                               "evaluation": {"metrics": {"accuracy": 0.87}, "data": {}}}
    r = webapp.app.test_client().get("/api/enhance/sum1/summary")
    html = r.get_json()["html"]
    assert html and "A grounded summary." in html
