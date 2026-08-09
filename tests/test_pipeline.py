"""The ML engine. Offline, deterministic, no API key."""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from autods.pipeline import loader, preclean, task_detect, quality, preprocess
from autods.pipeline import model_select, train, evaluate, leaderboard


# ---- load + profile -----------------------------------------------------
def test_profile_shape(df, prof):
    assert prof["n_rows"] == 600 and prof["n_cols"] == 7
    assert "churned" in prof["columns"]
    assert prof["preview"]  # JSON-safe preview present


# ---- pre-clean is mechanical only, never imputes ------------------------
def test_preclean_fixes_representation_not_meaning():
    raw = pd.DataFrame({
        "name": [" Ann ", "Bob", "?", "Ann", "Cat"],        # whitespace + placeholder
        "amount": ["1,200", "300", "unknown", "1,200", "4,500"],  # 3 distinct -> numeric
        "real_missing": [1.0, np.nan, 3.0, 1.0, 5.0],
    })
    out, changes = preclean.preclean(raw)
    assert out["name"].iloc[0] == "Ann"             # trimmed
    assert pd.isna(out["name"].iloc[2])             # "?" -> missing
    assert pd.api.types.is_numeric_dtype(out["amount"])   # "1,200" -> 1200
    assert pd.isna(out["real_missing"]).sum() == 1  # genuine NaN left untouched
    assert len(out) == 4                            # one exact duplicate dropped
    assert any("duplicate" in c for c in changes)


# ---- task detection -----------------------------------------------------
def test_detects_classification(task):
    assert task["task_type"] == "classification"
    assert task["target"] == "churned"


def test_detects_regression(df):
    t = task_detect.detect_task(df, target="income")   # numeric, many values
    assert t["task_type"] == "regression"


# ---- quality guardrails catch a leak ------------------------------------
def test_quality_flags_target_leakage(df, task):
    leaky = df.copy()
    # a numeric feature that moves almost perfectly with the target (with >2 values)
    leaky["leak"] = leaky["churned"] * 100.0 + np.arange(len(leaky)) * 1e-6
    flags = quality.run_checks(leaky, {**task, "target": "churned"})
    joined = " ".join(str(f).lower() for f in flags)
    assert "leak" in joined and any(f.get("level") == "risk" for f in flags)


# ---- cleaning plan builds and applies -----------------------------------
def test_recommend_and_apply_plan(df, task):
    plan = preprocess.recommend_plan(df, target="churned", task=task)
    assert isinstance(plan, dict)
    out = preprocess.apply_plan(df, plan)
    assert isinstance(out, pd.DataFrame) and len(out) > 0


# ---- the model hub ------------------------------------------------------
def test_model_hub_counts_and_recommendations():
    counts = model_select.count()
    assert sum(counts.values()) >= 30
    assert len(model_select.recommend_models("classification")) >= 5


# ---- train is leakage-safe, evaluate scores -----------------------------
def test_train_pipeline_is_leakage_safe_and_scores(df, task):
    name = model_select.recommend_models("classification")[0]["name"]
    trained = train.train_model(df, "classification", name, target="churned")
    pipe = trained["pipeline"]
    # the correctness guarantee: preprocessing lives INSIDE the sklearn Pipeline,
    # so it is fit on the train split only, never the whole dataset
    assert isinstance(pipe, Pipeline)
    assert set(pipe.named_steps) == {"pre", "model"}
    res = evaluate.evaluate(trained)
    assert 0.0 <= res["metrics"]["accuracy"] <= 1.0
    assert "f1" in res["metrics"]


# ---- the cross-validated leaderboard ranks ------------------------------
def test_leaderboard_ranks_best_first(df):
    board = leaderboard.run_leaderboard(df, "classification", "churned", cv=3)
    scored = [r for r in board["rows"] if r.get("score") is not None]
    assert len(scored) >= 3
    assert scored[0].get("best") is True
    assert scored[0]["score"] >= scored[-1]["score"]   # sorted best-first


# ---- model export produces a loadable bundle ----------------------------
def test_export_model(df, tmp_path):
    name = model_select.recommend_models("classification")[0]["name"]
    trained = train.train_model(df, "classification", name, target="churned")
    path = leaderboard.export_model(trained["pipeline"], filename="test_model.joblib")
    import os, joblib
    assert os.path.exists(path)
    assert joblib.load(path) is not None
