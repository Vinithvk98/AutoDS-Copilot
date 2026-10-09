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


# ---- Phase 2, what if and goal seeking --------------------------------------

def _overall_pct(d, col):
    return pd.to_numeric(d[col], errors="coerce").mean() * 100


def _shift_group(d, measure, dim, group, new_pct):
    """Shift every row of a group so its rate becomes new_pct, the ground truth
    the estimate should match."""
    d = d.copy()
    d[measure] = pd.to_numeric(d[measure], errors="coerce").astype(float)
    mask = d[dim].astype(str).eq(group) & d[dim].notna() & d[measure].notna()
    d.loc[mask, measure] += new_pct / 100 - d.loc[mask, measure].mean()
    return d


def test_estimate_effect_matches_recomputed_overall(df):
    eff = decision.estimate_effect(df, "churned", "plan", "Basic", 10)
    truth = _overall_pct(_shift_group(df, "churned", "plan", "Basic", 10), "churned")
    assert abs(eff["new"] - truth) < 1e-3
    assert eff["current"] > eff["new"] and eff["change"] < 0
    assert "Estimate only" in eff["math"] and eff["unit"] == "percent"


def test_estimate_effects_add_up(df):
    a = decision.estimate_effect(df, "churned", "city", "Denver", 10)["change"]
    b = decision.estimate_effect(df, "churned", "city", "Boston", 12)["change"]
    both = decision.estimate_effects(df, "churned", "city", {"Denver": 10, "Boston": 12})
    assert abs(both["change"] - (a + b)) < 1e-3
    assert len(both["groups"]) == 2


def test_estimate_effect_on_a_total():
    d = pd.DataFrame({"region": ["East", "West", "East", "West"], "sales": [10.0, 20.0, 30.0, 40.0]})
    eff = decision.estimate_effect(d, "sales", "region", "East", 60)
    assert eff["agg"] == "sum" and eff["current"] == 100 and eff["new"] == 120


def test_estimate_effect_clamps_rates_and_rejects_unknown_groups(df):
    eff = decision.estimate_effect(df, "churned", "plan", "Basic", 250)
    assert eff["group_to"] == 100
    import pytest
    with pytest.raises(ValueError):
        decision.estimate_effect(df, "churned", "plan", "Gold", 5)


def test_recommendation_targets_the_lagging_group(df):
    # churn is a lower is better measure, so the lever is the highest churn plan
    rec = decision.build_overview(df)["recommendation"]
    assert rec["group"] == "Basic" and rec["lower_is_better"]
    assert rec["move"].startswith("Bring down churned for Basic")
    eff = decision.estimate_effect(df, "churned", "plan", "Basic", 13.3333)
    assert str(round(abs(eff["change"]), 2)) in rec["estimate"]


def test_recommendation_lifts_a_higher_is_better_measure():
    d = pd.DataFrame({"team": ["a"] * 10 + ["b"] * 10 + ["c"] * 10,
                      "on_time_rate": [1] * 9 + [0] + [1] * 5 + [0] * 5 + [1] * 8 + [0] * 2})
    rec = decision.build_overview(d)["recommendation"]
    assert rec["group"] == "b" and rec["move"].startswith("Lift")


def test_goal_seek_hits_a_feasible_target(df):
    g = decision.goal_seek(df, "churned", 11)
    assert g["feasible"] and abs(g["result"] - 11) < 1e-3
    # applying the recommended changes to the data reproduces the target
    d = df
    for c in g["changes"]:
        d = _shift_group(d, "churned", g["dimension"], c["group"], c["to"])
    assert abs(_overall_pct(d, "churned") - 11) < 0.05
    # never past the best level a group already reaches
    assert all(c["to"] >= c["benchmark"] - 1e-6 for c in g["changes"])
    assert g["dimension"] == "plan" and len(g["changes"]) == 1   # fewest groups wins


def test_goal_seek_reports_an_out_of_reach_target(df):
    g = decision.goal_seek(df, "churned", 1)
    assert not g["feasible"] and g["result"] > 1
    assert "beyond what the data supports" in g["text"]


def test_goal_seek_target_already_met(df):
    cur = decision.goal_seek(df, "churned", 50)["current"]
    g = decision.goal_seek(df, None, cur)
    assert g["feasible"] and g["changes"] == []


def test_goal_seek_on_a_total():
    d = pd.DataFrame({"region": ["East"] * 5 + ["West"] * 5,
                      "sales": [10.0] * 5 + [20.0] * 5})
    g = decision.goal_seek(d, "sales", 180)
    assert g["feasible"] and g["changes"][0]["group"] == "East"
    assert abs(g["changes"][0]["to"] - 80) < 1e-6


def _house_style(text):
    return not any(ch in text for ch in (":", ";", "—", "–", "‘", "’", "“", "”"))


def test_phase2_copy_is_in_house_style(df):
    eff = decision.estimate_effect(df, "churned", "city", "Denver", 10)
    g1 = decision.goal_seek(df, "churned", 11)
    g2 = decision.goal_seek(df, "churned", 1)
    for t in (eff["text"], eff["math"], eff["note"], g1["text"], g1["math"], g2["text"]):
        assert _house_style(t), t


def test_whatif_and_goal_endpoints(df):
    from autods.web import app as webapp
    webapp.SESSIONS["ds2"] = {"path": "data/sample_customers.csv", "df": df}
    c = webapp.app.test_client()
    w = c.post("/api/studio/ds2/whatif", json={"dimension": "plan", "changes": {"Basic": 10}}).get_json()
    assert w["measure"] == "churned" and w["change"] < 0
    w1 = c.get("/api/studio/ds2/whatif?dimension=plan&group=Basic&value=10").get_json()
    assert w1["new"] == w["new"]
    assert c.post("/api/studio/ds2/whatif", json={"dimension": "plan", "changes": {"Gold": 1}}).status_code == 400
    g = c.post("/api/studio/ds2/goal", json={"target": 11}).get_json()
    assert g["feasible"] and g["changes"]
    assert c.post("/api/studio/ds2/goal", json={"target": "lots"}).status_code == 400


def test_studio_report_page(df):
    from autods.web import app as webapp
    webapp.SESSIONS["ds3"] = {"path": "data/sample_customers.csv", "df": df}
    c = webapp.app.test_client()
    r = c.get("/studio/ds3/report?target=11")
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "Save as PDF" in html and "Key findings" in html and "Focus on Basic" in html
    assert "Basic" in html   # the goal section
    assert c.get("/studio/nope/report").status_code == 404


def test_studio_report_with_a_trend_and_a_total():
    from autods.web import app as webapp
    rng = np.random.RandomState(1); n = 300
    d = pd.DataFrame({
        "order_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.randint(0, 730, n), unit="D"),
        "region": rng.choice(["East", "West", "North"], n),
        "sales": rng.gamma(2, 150, n).round(2),
    })
    webapp.SESSIONS["ds4"] = {"path": "sales.csv", "df": d}
    c = webapp.app.test_client()
    r = c.get("/studio/ds4/report?target=100000")
    html = r.get_data(as_text=True)
    assert r.status_code == 200 and "sales over time" in html and 'id="trend"' in html
    w = c.post("/api/studio/ds4/whatif", json={"dimension": "region", "changes": {"East": 40000}}).get_json()
    assert w["agg"] == "sum" and w["change"] > 0
