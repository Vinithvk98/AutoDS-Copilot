"""Compose a readable analysis report from a completed run.

This is the signature output. Instead of a dashboard of numbers, it writes the
analysis the way a person would, in plain sentences, from the same artifacts the
pipeline already produced. Returns a list of sections the templates render as an
editorial document.

Copy is written to read like a human wrote it. No colons, semicolons, or long
dashes in the generated prose.
"""
from __future__ import annotations
from pathlib import Path


def _clean_feat(name: str) -> str:
    """Turn a pipeline feature name like 'num__income' or 'cat__plan_Basic'
    into something readable."""
    n = name.split("__", 1)[-1]
    return n.replace("_", " ")


def _pct(x) -> str:
    try:
        return f"{round(float(x) * 100)} percent"
    except Exception:
        return str(x)


def _findings(pre_insights: list[str]) -> list[str]:
    """The rule-based EDA findings, without the retrieved guidance lines or the
    technical task-detection string (which has an arrow in it)."""
    return [b for b in pre_insights if not b.startswith("📚") and "→" not in b]


def _guidance(insights: list[str]) -> list[str]:
    return [b[1:].strip() if b.startswith("📚") else b
            for b in insights if b.startswith("📚")]


def build_report(s: dict) -> dict:
    """Return {title, subtitle, sections:[...]} for the run in session `s`."""
    prof = s.get("profile", {})
    task = s.get("task", {})
    metrics = s.get("evaluation", {}).get("metrics", {})
    fi = s.get("evaluation", {}).get("feature_importance", [])
    pre = s.get("pre_insights", [])
    post = s.get("post_insights", [])
    preclean = s.get("preclean", [])
    board = s.get("board")
    model = s.get("model_name", "the model")
    ttype = task.get("task_type", "analysis")
    target = task.get("target")
    dataset = Path(s.get("path", "your data")).name

    sections: list[dict] = []

    # --- 1. Overview -----------------------------------------------------
    n_rows, n_cols = prof.get("n_rows", "?"), prof.get("n_cols", "?")
    nnum = len(prof.get("types", {}).get("numeric", []))
    ncat = len(prof.get("types", {}).get("categorical", []))
    overview = [
        f"This is an analysis of {dataset}. The file has {n_rows} rows and "
        f"{n_cols} columns, {nnum} of them numeric and {ncat} holding categories."
    ]
    if target:
        line = (f"The question is about {_clean_feat(target)}, and AutoDS treated "
                f"it as a {ttype} problem.")
        if task.get("imbalanced"):
            line += (" The outcomes are lopsided, one is much rarer than the other, "
                     "so the scores below are chosen to stay honest when that happens.")
        overview.append(line)
    else:
        overview.append(
            "There was no column to predict, so AutoDS grouped the rows into "
            "natural clusters instead.")
    sections.append({"heading": "In short", "paras": overview})

    # --- 2. The data itself ---------------------------------------------
    miss = prof.get("missing", {})
    data_paras = []
    if miss:
        worst = max(prof.get("missing_pct", {}), key=prof["missing_pct"].get)
        data_paras.append(
            f"{len(miss)} column{'s' if len(miss) != 1 else ''} had gaps in them. "
            f"The emptiest was {_clean_feat(worst)}, missing about "
            f"{prof['missing_pct'][worst]} percent of its values. "
            "Rather than quietly fill those gaps, AutoDS left them visible while "
            "exploring, so the charts show the real picture, and handled them "
            "safely later inside the model.")
    else:
        data_paras.append("No values were missing, which is a good start.")
    # cleaning
    if preclean and not (len(preclean) == 1 and preclean[0].startswith("no mechanical")):
        acts = [c[0].lower() + c[1:] for c in preclean]
        data_paras.append("Before looking at anything, AutoDS tidied the raw file. "
                          "It " + ", then ".join(acts) + ".")
    else:
        data_paras.append("The file arrived clean, so nothing mechanical needed "
                          "fixing before the analysis.")
    sections.append({"heading": "What the data looks like", "paras": data_paras,
                     "charts": ["Missing values by column"]})

    # --- 3. What exploring it showed ------------------------------------
    findings = _findings(pre)
    explore_paras = []
    if findings:
        explore_paras.append("A few things stood out while exploring the data.")
    sections.append({"heading": "What we found", "paras": explore_paras,
                     "bullets": findings[1:] if len(findings) > 1 else findings,
                     "charts": ["Target", "Correlation heatmap"]})

    # --- 4. Choosing a model --------------------------------------------
    model_paras = []
    if board and board.get("rows"):
        ranked = [r for r in board["rows"] if r.get("score") is not None]
        if ranked:
            best = ranked[0]
            names = " and ".join(_pretty_model(r["name"]) for r in ranked[1:3])
            model_paras.append(
                f"AutoDS did not settle for one guess. It cross checked every "
                f"candidate on your data and ranked them by {board['metric']}. "
                f"{_pretty_model(best['name'])} came out on top" +
                (f", ahead of {names}." if names else "."))
            if model and _pretty_model(model) != _pretty_model(best["name"]):
                model_paras.append(
                    f"You chose to train {_pretty_model(model)} instead, which is "
                    "the model the rest of this report looks at.")
    if not model_paras:
        model_paras.append(
            f"AutoDS recommended {_pretty_model(model)} for this kind of problem "
            "and trained it on a portion of the data, holding the rest back to "
            "test it fairly.")
    sections.append({"heading": "How we chose a model", "paras": model_paras,
                     "board": board})

    # --- 5. How well it does --------------------------------------------
    perf_paras = [_performance_sentence(ttype, model, metrics)]
    if fi:
        top = [_clean_feat(d["feature"]) for d in fi[:3]]
        if len(top) >= 2:
            perf_paras.append(
                f"When it makes a call, it leans most on {top[0]}, "
                f"{top[1]}" + (f", and {top[2]}" if len(top) > 2 else "") +
                ". Those are the levers worth paying attention to.")
    sections.append({"heading": "How well it does", "paras": perf_paras,
                     "metrics": metrics, "features": fi[:8]})

    # --- 6. What it means -----------------------------------------------
    conclusion = [_conclusion_sentence(ttype, metrics, target)]
    conclusion.append(
        "From here it is worth trying the other strong models on the leaderboard, "
        "tuning the winner, and scoring fresh data to see how it holds up.")
    guide = _guidance(post) or _guidance(pre)
    sections.append({"heading": "What this means, and what to do next",
                     "paras": conclusion,
                     "note": guide[0] if guide else None})

    subtitle = (f"A {ttype} analysis of {dataset}"
                + (f", predicting {_clean_feat(target)}." if target else "."))
    return {"title": "The analysis", "subtitle": subtitle, "sections": sections}


def _pretty_model(name: str) -> str:
    import re
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name or "")


def _performance_sentence(ttype: str, model: str, m: dict) -> str:
    mp = _pretty_model(model)
    if ttype in ("classification", "text"):
        acc = m.get("accuracy")
        s = (f"On data it had never seen, {mp} got the answer right about "
             f"{_pct(acc)} of the time")
        if "f1" in m:
            s += f", with a balanced score of {m['f1']} that accounts for both "\
                 "kinds of mistake"
        s += "."
        if "roc_auc" in m:
            strength = ("does a strong job" if m["roc_auc"] > 0.8
                        else "does a fair job" if m["roc_auc"] > 0.65 else "struggles")
            s += (f" It {strength} of separating the two outcomes, with a ranking "
                  f"score of {m['roc_auc']}.")
        return s
    if ttype in ("regression", "timeseries"):
        r2 = m.get("r2")
        if r2 is not None and r2 < 0:
            return (f"{mp} did not find enough signal here. It actually does worse "
                    "than simply guessing the average, which usually means the "
                    "columns available do not explain the target very well.")
        s = f"{mp} explains about {_pct(r2)} of the variation in the target"
        if "rmse" in m:
            s += f", and its predictions are off by roughly {m['rmse']} on average"
        s += "."
        return s
    if ttype == "clustering":
        n = m.get("n_clusters")
        sil = m.get("silhouette")
        s = f"{mp} split the rows into {n} groups."
        if sil is not None:
            s += (" The groups are well separated." if sil > 0.5
                  else " The groups overlap a fair amount, so treat them as loose.")
        return s
    return f"{mp} was trained and evaluated."


def _conclusion_sentence(ttype: str, m: dict, target) -> str:
    if ttype in ("classification", "text"):
        acc = m.get("accuracy", 0) or 0
        if acc >= 0.85:
            return ("The model is accurate enough to be genuinely useful for "
                    "flagging cases worth attention.")
        if acc >= 0.7:
            return ("The model is a solid first cut. It is right more often than "
                    "not, and a round of tuning would likely push it further.")
        return ("The model is a starting point. The signal is weak so far, and it "
                "would benefit from more or better features before you rely on it.")
    if ttype in ("regression", "timeseries"):
        r2 = m.get("r2", 0) or 0
        if r2 >= 0.6:
            return "The model captures the main drivers well enough to plan around."
        if r2 >= 0.3:
            return ("The model sees part of the story. It is useful for direction, "
                    "less so for precise numbers.")
        return ("The model does not yet explain much. The honest read is that the "
                "current columns do not carry enough signal.")
    return ("The groups give you a first map of the data. Look at what each one has "
            "in common to decide whether the split is useful.")
