"""Modelling rigor: resampling, threshold tuning, calibration, fairness.
Runs on the bundled sample, offline, deterministic."""
import numpy as np
import pandas as pd

from autods.pipeline import rigor, train, evaluate


# ---- resampling ------------------------------------------------------------
def test_oversample_balances_classes():
    X = pd.DataFrame({"f": range(100)})
    y = pd.Series([0] * 90 + [1] * 10)
    Xr, yr = rigor.oversample(X, y)
    counts = yr.value_counts()
    assert counts[0] == counts[1]           # perfectly balanced
    assert len(Xr) == len(yr) == counts.sum()


def test_resolve_balance_strategy():
    assert rigor.resolve_balance(False, None) == "none"
    assert rigor.resolve_balance(True, None) == "class_weight"
    assert rigor.resolve_balance(False, "smote") == "smote"


def test_train_records_strategy_and_no_leak(df):
    tr = train.train_model(df, "classification", "LogisticRegression",
                           target="churned", balance="oversample")
    assert tr["balance"] == "oversample"
    # oversampling touched only the training split, test split is untouched size
    assert len(tr["X_test"]) == round(len(df.dropna(subset=["churned"])) * 0.2)


def test_smote_or_fallback_trains(df):
    tr = train.train_model(df, "classification", "RandomForestClassifier",
                           target="churned", balance="smote")
    # smote when imbalanced-learn is present, else the oversampler fallback
    assert tr["balance"] in ("smote", "oversample")
    assert tr["pipeline"].predict(tr["X_test"]) is not None


# ---- threshold tuning ------------------------------------------------------
def test_tune_threshold_maximises_f1():
    y = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.45, 0.4, 0.8, 0.9])
    out = rigor.tune_threshold(y, proba, pos_label=1)
    assert out["tuned"]["f1"] >= out["default"]["f1"]
    assert 0.0 < out["tuned"]["threshold"] < 1.0


# ---- calibration -----------------------------------------------------------
def test_calibration_reports_brier_and_curve():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    cal = rigor.calibration(y, proba, pos_label=1)
    assert 0.0 <= cal["brier"] <= 1.0 and cal["curve"]


# ---- fairness --------------------------------------------------------------
def test_fairness_reports_group_gap():
    X = pd.DataFrame({"group": ["A"] * 10 + ["B"] * 10})
    y_true = np.array([1] * 5 + [0] * 5 + [1] * 5 + [0] * 5)
    y_pred = np.array([1] * 10 + [0] * 10)   # group A always positive, B never
    rep = rigor.fairness(X, y_true, y_pred, pos_label=1)
    assert rep and rep[0]["feature"] == "group"
    assert rep[0]["selection_gap"] == 1.0


# ---- the visible balance step ---------------------------------------------
def test_balance_preview_evens_train_keeps_test(df):
    rep = train.balance_preview(df, "churned", "smote")
    assert rep["changed"] is True
    before = list(rep["before"].values())
    after = list(rep["after"].values())
    test = list(rep["test"].values())
    assert max(before) != min(before)   # training split started imbalanced
    assert max(after) == min(after)      # after resampling the classes are even
    assert max(test) != min(test)        # test split left at its real ratio


def test_balance_endpoint_renders_proof(df):
    from autods.web import app as webapp
    from autods.pipeline import task_detect, model_select
    task = task_detect.detect_task(df, target="churned")
    webapp.SESSIONS["bstep"] = {"path": "data/sample_customers.csv", "df_clean": df,
                                "task": task,
                                "candidates": model_select.recommend_models(task["task_type"])}
    c = webapp.app.test_client()
    r = c.post("/api/balance/bstep", data={"model": "LogisticRegression"})
    assert r.status_code == 200
    html = r.get_json()["html"]
    assert "Balancing the classes" in html and "before" in html and "after" in html


# ---- end to end through evaluate ------------------------------------------
def test_evaluate_emits_rigor_diagnostics(df):
    tr = train.train_model(df, "classification", "RandomForestClassifier",
                           target="churned", balance="smote")
    ev = evaluate.evaluate(tr)
    assert "brier" in ev["metrics"] and "best_threshold" in ev["metrics"]
    assert "threshold" in ev["data"] and "calibration" in ev["data"]
    assert ev["data"]["balance"] in ("smote", "oversample")
