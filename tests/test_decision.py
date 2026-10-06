"""Decision Studio analytics engine. Pure and grounded, no model needed."""
import numpy as np
import pandas as pd

from autods import decision


def test_roles_detects_measures_and_dimensions(df):
    r = decision.roles(df)
    assert "income" in r["measures"] and "churned" in r["measures"]
    assert "plan" in r["dimensions"] and "city" in r["dimensions"]
    assert r["time"] is None   # the sample has no date column


def test_overview_has_all_sections(df):
    ov = decision.build_overview(df)
    assert ov["kpis"] and ov["breakdowns"] and ov["findings"]
    assert ov["recommendation"] and ov["recommendation"]["headline"]
    # a rate breakdown of churn by plan, values are percentages
    b = ov["breakdowns"][0]
    assert all(0 <= v <= 100 for v in b["data"].values())


def test_continuous_float_is_a_measure_not_an_id():
    d = pd.DataFrame({"g": ["a", "b"] * 50, "sales": np.random.RandomState(0).gamma(2, 100, 100)})
    r = decision.roles(d)
    assert "sales" in r["measures"]   # all-unique floats are measures, not ids


def test_time_overview_has_trend_and_change():
    rng = np.random.RandomState(1); n = 300
    d = pd.DataFrame({
        "order_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.randint(0, 730, n), unit="D"),
        "region": rng.choice(["East", "West", "North"], n),
        "sales": rng.gamma(2, 150, n).round(2),
    })
    ov = decision.build_overview(d)
    assert ov["time"] == "order_date"
    assert ov["trend"] and len(ov["trend"]["labels"]) >= 2
    kpi = next(k for k in ov["kpis"] if k["label"] == "sales")
    assert "change_pct" in kpi and kpi["direction"] in ("up", "down")


def test_empty_of_measures_is_graceful():
    d = pd.DataFrame({"city": ["a", "b", "c"], "plan": ["x", "y", "z"]})
    ov = decision.build_overview(d)
    assert ov["kpis"] == [] and ov["findings"]


def test_studio_endpoint_returns_overview(df):
    from autods.web import app as webapp
    webapp.SESSIONS["ds"] = {"path": "data/sample_customers.csv", "df": df}
    ov = webapp.app.test_client().get("/api/studio/ds").get_json()
    assert ov["kpis"] and ov["recommendation"] and "suggestions" in ov
