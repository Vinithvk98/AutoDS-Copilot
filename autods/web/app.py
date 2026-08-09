"""Flask app driving the pipeline with a human-in-the-loop at each decision.

Steps: upload -> explore (EDA + insights + task) -> clean (approve plan)
       -> models (choose model) -> results (metrics + insights).

The pipeline package does the ML work; this app is the "Recommend + you approve"
interface. Run with:  python run.py   (or  flask --app autods.web.app run)
"""
from __future__ import annotations
import uuid
from pathlib import Path
from flask import (Flask, render_template, request, redirect, url_for,
                   send_from_directory, abort, jsonify)

from .. import config
from ..pipeline import (loader, preclean, task_detect, eda, preprocess, model_select,
                        train, evaluate, insights, leaderboard, report, quality)
from .. import rag

app = Flask(__name__)
app.config["UPLOAD_DIR"] = config.OUTPUT_DIR / "uploads"
app.config["UPLOAD_DIR"].mkdir(parents=True, exist_ok=True)

# Simple in-memory session store: sid -> dict of artifacts for that run.
SESSIONS: dict[str, dict] = {}


def _rel(path: str) -> str:
    """Turn an absolute output path into a URL the browser can request."""
    p = Path(path).resolve()
    return str(p.relative_to(config.OUTPUT_DIR.resolve()))


@app.route("/outputs/<path:relpath>")
def outputs(relpath):
    return send_from_directory(config.OUTPUT_DIR, relpath)


@app.route("/")
def index():
    """Expressive landing page."""
    return render_template("landing.html")


@app.route("/app")
def console():
    """Single-page console experience (living rail + agent console)."""
    samples = [p.name for p in config.DATA_DIR.glob("*.csv")]
    return render_template("console.html", samples=samples)


@app.route("/classic")
def classic():
    """The original multi-page flow (kept as a fallback)."""
    samples = [p.name for p in config.DATA_DIR.glob("*.csv")]
    return render_template("index.html", samples=samples)


def _algorithms_data():
    tasks = {"classification": "Classification", "regression": "Regression",
             "clustering": "Clustering"}
    groups = {label: model_select.recommend_grouped(t) for t, label in tasks.items()}
    counts = model_select.count()
    return {"groups": groups, "counts": counts, "total": sum(counts.values())}


@app.route("/api/algorithms")
def api_algorithms():
    return jsonify(_algorithms_data())


@app.route("/algorithms")
def algorithms_page():
    return render_template("algorithms.html", d=_algorithms_data())


@app.route("/about")
def about_page():
    return render_template("about.html")


@app.route("/upload", methods=["POST"])
def upload():
    target = request.form.get("target", "").strip() or None
    sample = request.form.get("sample", "").strip()

    if sample:
        path = config.DATA_DIR / sample
    else:
        f = request.files.get("file")
        if not f or not f.filename:
            return redirect(url_for("index"))
        path = app.config["UPLOAD_DIR"] / f.filename
        f.save(path)

    try:
        df = loader.load_data(path)
    except Exception as e:
        return render_template("index.html",
                               samples=[p.name for p in config.DATA_DIR.glob("*.csv")],
                               error=str(e))

    df, _changes = preclean.preclean(df)
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {"df": df, "target": target, "path": str(path), "preclean": _changes}
    return redirect(url_for("explore", sid=sid))


@app.route("/explore/<sid>", methods=["GET", "POST"])
def explore(sid):
    s = SESSIONS.get(sid) or abort(404)
    df = s["df"]
    # allow the user to (re)choose the target from a dropdown
    if request.method == "POST":
        s["target"] = request.form.get("target") or None

    s["profile"] = loader.profile(df)
    s["task"] = task_detect.detect_task(df, target=s.get("target"))
    s["eda"] = eda.generate_eda(df, target=s["task"]["target"])
    s["pre_insights"] = insights.pre_insights(df, s["profile"], s["task"])
    charts = [{**c, "url": url_for("outputs", relpath=_rel(c["path"]))} for c in s["eda"]]
    return render_template("explore.html", sid=sid, profile=s["profile"], task=s["task"],
                           charts=charts, pre_insights=s["pre_insights"],
                           columns=list(df.columns), preclean=s.get("preclean"),
                           checks=quality.run_checks(df, s["task"]))


@app.route("/clean/<sid>")
def clean(sid):
    s = SESSIONS.get(sid) or abort(404)
    s["plan"] = preprocess.recommend_plan(s["df"], s["task"]["target"], task=s["task"])
    return render_template("clean.html", sid=sid, plan=s["plan"])


@app.route("/models/<sid>", methods=["GET", "POST"])
def models(sid):
    s = SESSIONS.get(sid) or abort(404)
    if request.method == "POST":
        # apply the (approved) plan — the form's checked "drop" boxes are the
        # columns the user confirmed for dropping.
        plan = dict(s["plan"])
        plan["drop_columns"] = request.form.getlist("drop")
        plan["drop_duplicates"] = bool(request.form.get("drop_duplicates"))
        s["approved_plan"] = plan
        s["df_clean"] = preprocess.apply_plan(s["df"], plan)
        s["candidates"] = model_select.recommend_models(s["task"]["task_type"])
    if "candidates" not in s:
        abort(400)
    grouped = model_select.recommend_grouped(s["task"]["task_type"])
    return render_template("models.html", sid=sid, grouped=grouped, task=s["task"])


@app.route("/leaderboard/<sid>")
def leaderboard_route(sid):
    s = SESSIONS.get(sid) or abort(404)
    if "df_clean" not in s:
        s["df_clean"] = s["df"]
    board = leaderboard.run_leaderboard(
        s["df_clean"], s["task"]["task_type"], s["task"]["target"],
        time_col=s["task"].get("time_col"), text_col=s["task"].get("text_col"))
    s["candidates"] = model_select.recommend_models(s["task"]["task_type"])
    return render_template("leaderboard.html", sid=sid, board=board, task=s["task"])


@app.route("/train/<sid>", methods=["POST"])
def train_route(sid):
    s = SESSIONS.get(sid) or abort(404)
    model_name = request.form.get("model") or next(
        (c["name"] for c in s["candidates"] if c.get("recommended")),
        s["candidates"][0]["name"])
    t = s["task"]
    trained = train.train_model(s["df_clean"], t["task_type"], model_name,
                                target=t["target"], time_col=t.get("time_col"),
                                text_col=t.get("text_col"), balanced=t.get("imbalanced", False))
    ev = evaluate.evaluate(trained)
    post = insights.post_insights(t["task_type"], model_name, ev,
                                  dataset=Path(s["path"]).name)
    plots = [url_for("outputs", relpath=_rel(p)) for p in ev["plots"]]
    # export the fitted model so the user can download it
    s["trained"] = trained
    s["model_name"] = model_name
    s["evaluation"] = ev
    s["post_insights"] = post
    rag.record_run(Path(s["path"]).name, s["task"], model_name, ev["metrics"],
                   profile=s.get("profile"))
    can_tune = (model_name in leaderboard.PARAM_GRIDS
                and t["task_type"] not in ("text", "timeseries", "clustering"))
    snippet = (None if s["task"]["task_type"] == "clustering"
               else _usage_snippet(model_name, trained["feature_cols"], s["task"]["task_type"]))
    return render_template("results.html", sid=sid, model_name=model_name, task=s["task"],
                           metrics=ev["metrics"], plots=plots, can_tune=can_tune,
                           feature_importance=ev["feature_importance"], post_insights=post,
                           snippet=snippet)


def _usage_snippet(model_name, features, task_type):
    cols = ", ".join(repr(c) for c in features)
    proba = ("\n# for probabilities: model.predict_proba(new[features])"
             if task_type in ("classification", "text") else "")
    return (f"import joblib\n"
            f"import pandas as pd\n\n"
            f"# load the trained pipeline\n"
            f'model = joblib.load("{model_name}.joblib")\n\n'
            f"# your new data needs these columns\n"
            f"features = [{cols}]\n\n"
            f"# score new rows from a CSV\n"
            f'new = pd.read_csv("new_data.csv")\n'
            f"predictions = model.predict(new[features]){proba}\n"
            f"print(predictions)\n")


@app.route("/download/<sid>")
def download_route(sid):
    s = SESSIONS.get(sid) or abort(404)
    if s.get("task", {}).get("task_type") == "clustering" or "trained" not in s:
        abort(400)
    pipe = s["trained"].get("pipeline")
    if pipe is None:
        abort(400)
    import zipfile
    outdir = config.OUTPUT_DIR / "models"
    outdir.mkdir(parents=True, exist_ok=True)
    model_name = s["model_name"]
    features = s["trained"]["feature_cols"]
    model_file = leaderboard.export_model(pipe, f"{model_name}.joblib")
    zip_name = f"{model_name}_{sid[:8]}.zip"
    with zipfile.ZipFile(outdir / zip_name, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(model_file, arcname=f"{model_name}.joblib")
        z.writestr("use_model.py", _usage_snippet(model_name, features, s["task"]["task_type"]))
        z.writestr("features.txt", "\n".join(features))
        z.writestr("README.txt",
                   "AutoDS model export\n\n"
                   "Load the .joblib with joblib.load and call .predict on a pandas\n"
                   "DataFrame that has the columns listed in features.txt.\n"
                   "See use_model.py for a working example.\n")
    return send_from_directory(outdir, zip_name, as_attachment=True)


@app.route("/tune/<sid>", methods=["POST"])
def tune_route(sid):
    s = SESSIONS.get(sid) or abort(404)
    result = leaderboard.tune_model(s["df_clean"], s["task"]["task_type"],
                                    s["task"]["target"], s["model_name"])
    return render_template("tune.html", sid=sid, result=result, model_name=s["model_name"])


# =====================================================================
#  JSON API — drives the single-page console (partial HTML + agent log
#  lines + "rail" artifact data the sidebar renders live).
# =====================================================================

def _spark(df, task) -> list:
    """Small numeric series for the rail sparkline (target shape at a glance)."""
    import numpy as np
    target = task.get("target")
    try:
        if (target and task["task_type"] in ("classification", "text")
                and df[target].nunique() >= 4):
            vc = df[target].value_counts().head(10)
            return [int(v) for v in vc.tolist()]
        col = target if (target and df[target].dtype.kind in "if"
                         and df[target].nunique() > 3) else None
        if col is None:
            num = df.select_dtypes(include="number").columns
            if not len(num):
                return []
            col = num[0]
        counts, _ = np.histogram(df[col].dropna(), bins=12)
        return [int(v) for v in counts.tolist()]
    except Exception:
        return []


def _api(stage, html, log, rail):
    return jsonify({"stage": stage, "html": html, "log": log, "rail": rail})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    target = request.form.get("target", "").strip() or None
    sample = request.form.get("sample", "").strip()
    if sample:
        path = config.DATA_DIR / sample
    else:
        f = request.files.get("file")
        if not f or not f.filename:
            return jsonify({"error": "No file selected"}), 400
        path = app.config["UPLOAD_DIR"] / f.filename
        f.save(path)
    try:
        df = loader.load_data(path)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    log = [f"[Loader] loaded {df.shape[0]}×{df.shape[1]} from {Path(path).name}"]
    df, changes = preclean.preclean(df)   # mechanical fixes only, before EDA
    log += [f"[Pre-clean] {ch}" for ch in changes]
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {"df": df, "target": target, "path": str(path), "preclean": changes}
    return jsonify({"sid": sid, "log": log})


@app.route("/api/explore/<sid>")
def api_explore(sid):
    s = SESSIONS.get(sid) or abort(404)
    df = s["df"]
    target = request.args.get("target")
    if target is not None:
        s["target"] = target or None
    s["profile"] = loader.profile(df)
    s["task"] = task_detect.detect_task(df, target=s.get("target"))
    s["eda"] = eda.generate_eda(df, target=s["task"]["target"])
    s["pre_insights"] = insights.pre_insights(df, s["profile"], s["task"])
    s["checks"] = quality.run_checks(df, s["task"])
    charts = [{**c, "url": url_for("outputs", relpath=_rel(c["path"]))} for c in s["eda"]]
    html = render_template("partials/explore.html", sid=sid, profile=s["profile"],
                           task=s["task"], charts=charts, pre_insights=s["pre_insights"],
                           columns=list(df.columns), preclean=s.get("preclean"),
                           checks=s["checks"])
    log = [f"[Profiler] {s['profile']['n_rows']} rows · {len(s['profile']['missing'])} cols with gaps",
           f"[EDA agent] generated {len(charts)} charts",
           f"[Task detector] {s['task']['task_type']} (target={s['task']['target']})",
           "[Insight agent] wrote pre-modelling insights"]
    rail = {"active": 2, "arts": {"2": {"spark": _spark(df, s["task"]),
                                        "badge": s["task"]["task_type"]}}}
    return _api("explore", html, log, rail)


@app.route("/api/clean/<sid>")
def api_clean(sid):
    s = SESSIONS.get(sid) or abort(404)
    s["plan"] = preprocess.recommend_plan(s["df"], s["task"]["target"], task=s["task"])
    q = "handling missing values and messy columns"
    if any("leak" in n.lower() for n in s["plan"].get("notes", [])):
        q = "data leakage dropping columns"
    why = rag.rationale(q)
    html = render_template("partials/clean.html", sid=sid, plan=s["plan"], why=why)
    n = len(s["plan"].get("impute", {}))
    log = [f"[Cleaning agent] proposed plan · {n} column(s) to impute"]
    return _api("clean", html, log, {"active": 3})


@app.route("/api/models/<sid>", methods=["GET", "POST"])
def api_models(sid):
    s = SESSIONS.get(sid) or abort(404)
    log = []
    if request.method == "POST":
        plan = dict(s.get("plan", {}))
        plan["drop_columns"] = request.form.getlist("drop")
        plan["drop_duplicates"] = bool(request.form.get("drop_duplicates"))
        s["approved_plan"] = plan
        s["df_clean"] = preprocess.apply_plan(s["df"], plan)
        s["candidates"] = model_select.recommend_models(s["task"]["task_type"])
        log = [f"[Cleaning agent] applied plan → {s['df_clean'].shape[0]}×{s['df_clean'].shape[1]}",
               f"[Model recommender] proposed {len(s['candidates'])} candidates"]
    if "candidates" not in s:
        abort(400)
    grouped = model_select.recommend_grouped(s["task"]["task_type"])
    # experience retrieval + a grounded reason for the recommendation
    fp = rag.fingerprint(s.get("profile", {}), s["task"])
    sims = rag.similar_runs(fp)
    why = rag.rationale(f"choosing a model for {s['task']['task_type']} tabular data"
                        + (" with imbalanced classes" if s["task"].get("imbalanced") else ""))
    html = render_template("partials/models.html", sid=sid, grouped=grouped, task=s["task"],
                           sims=sims, why=why)
    return _api("models", html, log, {"active": 4})


@app.route("/api/leaderboard/<sid>")
def api_leaderboard(sid):
    s = SESSIONS.get(sid) or abort(404)
    if "df_clean" not in s:
        s["df_clean"] = s["df"]
    t = s["task"]
    board = leaderboard.run_leaderboard(s["df_clean"], t["task_type"], t["target"],
                                        time_col=t.get("time_col"), text_col=t.get("text_col"))
    s["candidates"] = model_select.recommend_models(t["task_type"])
    s["board"] = board
    html = render_template("partials/leaderboard.html", sid=sid, board=board, task=t)
    best = next((r for r in board["rows"] if r.get("best")), None)
    log = [f"[Leaderboard] cross-validated {len(board['rows'])} models on {board['metric']}"]
    if best:
        log.append(f"[Leaderboard] best: {best['name']} = {best['score']}")
    rail = {"active": 4}
    if best:
        rail["arts"] = {"4": {"badge": best["name"], "sub": f"{board['metric']} {best['score']}"}}
    return _api("leaderboard", html, log, rail)


@app.route("/api/train/<sid>", methods=["POST"])
def api_train(sid):
    s = SESSIONS.get(sid) or abort(404)
    t = s["task"]
    model_name = request.form.get("model") or next(
        (c["name"] for c in s["candidates"] if c.get("recommended")), s["candidates"][0]["name"])
    try:
        trained = train.train_model(s["df_clean"], t["task_type"], model_name, target=t["target"],
                                    time_col=t.get("time_col"), text_col=t.get("text_col"),
                                    balanced=t.get("imbalanced", False))
        ev = evaluate.evaluate(trained)
    except Exception as e:
        return jsonify({"error": f"{model_name} could not train on this data ({e})"}), 400
    post = insights.post_insights(t["task_type"], model_name, ev, dataset=Path(s["path"]).name)
    plots = [url_for("outputs", relpath=_rel(p)) for p in ev["plots"]]
    s["trained"] = trained
    s["model_name"] = model_name
    s["evaluation"] = ev
    s["post_insights"] = post
    rag.record_run(Path(s["path"]).name, s["task"], model_name, ev["metrics"],
                   profile=s.get("profile"))
    can_tune = (model_name in leaderboard.PARAM_GRIDS
                and t["task_type"] not in ("text", "timeseries", "clustering"))
    snippet = (None if t["task_type"] == "clustering"
               else _usage_snippet(model_name, trained["feature_cols"], t["task_type"]))
    cv = leaderboard.cross_validate_model(s["df_clean"], t["task_type"], t["target"],
                                          model_name, time_col=t.get("time_col"),
                                          text_col=t.get("text_col"))
    s["cv"] = cv
    html = render_template("partials/results.html", sid=sid, model_name=model_name, task=t,
                           metrics=ev["metrics"], plots=plots, can_tune=can_tune,
                           feature_importance=ev["feature_importance"], post_insights=post,
                           snippet=snippet, cv=cv)
    # headline metric for the rail
    order = ["accuracy", "r2", "f1", "silhouette"]
    key = next((k for k in order if k in ev["metrics"]), next(iter(ev["metrics"]), None))
    log = [f"[Trainer] trained {model_name}",
           f"[Evaluator] {', '.join(f'{k}={v}' for k, v in list(ev['metrics'].items())[:3])}",
           "[Insight agent] wrote post-modelling insights (RAG-grounded)"]
    rail = {"active": 5, "arts": {
        "4": {"badge": model_name},
        "5": {"badge": str(ev["metrics"].get(key, "")), "sub": key or ""}}}
    return _api("results", html, log, rail)


@app.route("/api/predict/<sid>", methods=["GET", "POST"])
def api_predict(sid):
    s = SESSIONS.get(sid) or abort(404)
    if "trained" not in s or s["task"]["task_type"] in ("clustering", "timeseries"):
        return jsonify({"error": "Prediction works after training a classification, "
                                 "regression, or text model."}), 400
    trained = s["trained"]
    feats = trained["feature_cols"]
    pipe = trained.get("pipeline")

    if request.method == "GET":
        html = render_template("partials/predict.html", sid=sid, predictions=None,
                               features=feats, model_name=s["model_name"])
        return _api("predict", html, ["[Predictor] awaiting new data"], {"active": 5})

    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "No file selected"}), 400
    path = app.config["UPLOAD_DIR"] / f.filename
    f.save(path)
    try:
        newdf = loader.load_data(path)
        newdf, _ = preclean.preclean(newdf)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    missing = [c for c in feats if c not in newdf.columns]
    if missing:
        return jsonify({"error": "Missing required columns: " + ", ".join(missing)}), 400

    X = newdf[feats]
    try:
        preds = pipe.predict(X)
    except Exception as e:
        return jsonify({"error": f"Prediction failed ({e})"}), 400
    proba = None
    if s["task"]["task_type"] in ("classification", "text"):
        try:
            proba = pipe.predict_proba(X).max(axis=1)
        except Exception:
            proba = None

    shown = min(50, len(newdf))
    def _fmt(v):
        return "" if v is None or (isinstance(v, float) and v != v) else v
    def _pred(v):
        try:
            fv = float(v)
            return int(fv) if fv == int(fv) else round(fv, 3)
        except (ValueError, TypeError):
            return v
    rows = [{"vals": [_fmt(newdf.iloc[i][c]) for c in feats],
             "pred": _pred(preds[i]),
             "prob": (round(float(proba[i]), 3) if proba is not None else None)}
            for i in range(shown)]
    # keep the full result so it can be downloaded as CSV
    out_df = newdf.copy()
    out_df["prediction"] = preds
    if proba is not None:
        out_df["confidence"] = [round(float(x), 3) for x in proba]
    s["pred_df"] = out_df
    html = render_template("partials/predict.html", sid=sid, predictions=rows, cols=feats,
                           n=len(newdf), shown=shown, model_name=s["model_name"],
                           has_prob=proba is not None)
    return _api("predict", html, [f"[Predictor] scored {len(newdf)} rows with {s['model_name']}"],
                {"active": 5})


@app.route("/api/ensemble/<sid>", methods=["POST"])
def api_ensemble(sid):
    s = SESSIONS.get(sid) or abort(404)
    t = s["task"]
    if t["task_type"] == "clustering":
        return jsonify({"error": "Ensembles apply to supervised tasks, not clustering."}), 400
    board = _ensure_board(s)
    names = [r["name"] for r in board["rows"] if r.get("score") is not None]
    n = max(2, min(int(request.form.get("n", 3)), len(names)))
    kind = request.form.get("kind", "voting")
    members = names[:n]
    try:
        est = leaderboard.build_ensemble(t["task_type"], members, kind)
        model_name = f"{kind.capitalize()} ensemble"
        trained = train.train_model(s["df_clean"], t["task_type"], model_name, target=t["target"],
                                    time_col=t.get("time_col"), text_col=t.get("text_col"),
                                    balanced=t.get("imbalanced", False), estimator=est)
        ev = evaluate.evaluate(trained)
    except Exception as e:
        return jsonify({"error": f"Could not build the ensemble ({e})"}), 400
    post = insights.post_insights(t["task_type"], model_name, ev, dataset=Path(s["path"]).name)
    plots = [url_for("outputs", relpath=_rel(p)) for p in ev["plots"]]
    s["trained"] = trained
    s["model_name"] = model_name
    s["evaluation"] = ev
    s["post_insights"] = post
    s["cv"] = None
    rag.record_run(Path(s["path"]).name, t, model_name, ev["metrics"], profile=s.get("profile"))
    snippet = _usage_snippet(model_name, trained["feature_cols"], t["task_type"])
    html = render_template("partials/results.html", sid=sid, model_name=model_name, task=t,
                           metrics=ev["metrics"], plots=plots, can_tune=False,
                           feature_importance=ev["feature_importance"], post_insights=post,
                           snippet=snippet, cv=None)
    log = [f"[Ensemble] combined {n} models into a {kind} ensemble",
           f"[Evaluator] {', '.join(f'{k}={v}' for k, v in list(ev['metrics'].items())[:3])}"]
    order = ["accuracy", "r2", "f1", "silhouette"]
    key = next((k for k in order if k in ev["metrics"]), next(iter(ev["metrics"]), ""))
    rail = {"active": 5, "arts": {"4": {"badge": model_name},
            "5": {"badge": str(ev["metrics"].get(key, "")), "sub": key}}}
    return _api("results", html, log, rail)


@app.route("/predict_download/<sid>")
def predict_download(sid):
    s = SESSIONS.get(sid) or abort(404)
    df = s.get("pred_df")
    if df is None:
        abort(400)
    outdir = config.OUTPUT_DIR / "predictions"
    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"predictions_{sid[:8]}.csv"
    df.to_csv(outdir / fname, index=False)
    return send_from_directory(outdir, fname, as_attachment=True)


@app.route("/api/tune/<sid>", methods=["POST"])
def api_tune(sid):
    s = SESSIONS.get(sid) or abort(404)
    t = s["task"]
    model_name = s["model_name"]
    result = leaderboard.tune_model(s["df_clean"], t["task_type"], t["target"], model_name)
    log = ["[Tuner] grid-searched hyperparameters"]
    before = (s.get("evaluation") or {}).get("metrics", {})
    after = None
    if result:
        log.append(f"[Tuner] best {result['metric']} = {result['best_score']}")
        # actually apply the winning settings, retrain, and evaluate fairly
        try:
            tuned = train.train_model(s["df_clean"], t["task_type"], model_name,
                                      target=t["target"], balanced=t.get("imbalanced", False),
                                      params=result["best_params"])
            ev = evaluate.evaluate(tuned)
            after = ev["metrics"]
            # make the tuned model the active one for download / predict / report
            s["trained"] = tuned
            s["evaluation"] = ev
            s["post_insights"] = insights.post_insights(t["task_type"], model_name, ev,
                                                        dataset=Path(s["path"]).name)
            log.append("[Tuner] retrained with the best settings and kept the improved model")
        except Exception as e:
            log.append(f"[Tuner] could not retrain ({e})")
    lower_better = {"rmse", "mae"}
    improved = {}
    if after and before:
        for k, v in after.items():
            b = before.get(k)
            if b is None:
                continue
            improved[k] = (v < b) if k in lower_better else (v > b)
    html = render_template("partials/tune.html", sid=sid, result=result, model_name=model_name,
                           before=before, after=after, improved=improved, task=t)
    return _api("tune", html, log, {"active": 5})


def _ensure_board(s):
    if "board" not in s:
        t = s["task"]
        s["board"] = leaderboard.run_leaderboard(
            s.get("df_clean", s["df"]), t["task_type"], t["target"],
            time_col=t.get("time_col"), text_col=t.get("text_col"))
    return s["board"]


def _attach_charts(rep, s):
    eda_list = s.get("eda", [])
    def url_for_title(kw):
        for c in eda_list:
            if kw.lower() in c["title"].lower():
                return url_for("outputs", relpath=_rel(c["path"]))
        return None
    for sec in rep["sections"]:
        urls = []
        for kw in sec.get("charts", []):
            u = url_for_title(kw)
            if u and u not in urls:
                urls.append(u)
        sec["chart_urls"] = urls
    return rep


@app.route("/api/report/<sid>")
def api_report(sid):
    s = SESSIONS.get(sid) or abort(404)
    if "evaluation" not in s:
        abort(400)
    _ensure_board(s)
    rep = _attach_charts(report.build_report(s), s)
    html = render_template("partials/report.html", sid=sid, rep=rep)
    return _api("report", html, ["[Analyst] wrote up the analysis"], {"active": 5})


@app.route("/api/ask/<sid>", methods=["GET", "POST"])
def api_ask(sid):
    s = SESSIONS.get(sid) or abort(404)
    question = (request.form.get("question") or "").strip() if request.method == "POST" else ""
    result = rag.answer(question, session=s) if question else None
    html = render_template("partials/ask.html", sid=sid, result=result, question=question)
    log = [f"[Ask] answered '{question[:40]}' from {len(result['sources'])} sources"] if result else []
    return _api("ask", html, log, {"active": 5})


@app.route("/report/<sid>")
def report_page(sid):
    s = SESSIONS.get(sid) or abort(404)
    if "evaluation" not in s:
        abort(400)
    _ensure_board(s)
    rep = _attach_charts(report.build_report(s), s)
    return render_template("report_page.html", rep=rep)


def _dashboard_data(s):
    import json as _json
    t = s["task"]; ttype = t["task_type"]
    ev = s.get("evaluation", {}); m = ev.get("metrics", {}); data = ev.get("data", {})
    prof = s.get("profile", {})
    board = _ensure_board(s)
    fi = ev.get("feature_importance", [])
    df = s.get("df_clean", s["df"])
    target = t.get("target")

    # sample the data for client-side charts, filtering, and the table
    sample = df.sample(min(len(df), 1500), random_state=42) if len(df) > 1500 else df
    columns = [{"name": c, "type": "numeric" if df[c].dtype.kind in "if" else "category",
                "is_target": c == target} for c in df.columns]
    rows = _json.loads(sample.to_json(orient="values"))

    # KPI tiles
    kpis = []
    if ttype in ("classification", "text"):
        kpis += [{"label": "Accuracy", "value": m.get("accuracy")},
                 {"label": "F1 score", "value": m.get("f1")}]
        if "roc_auc" in m:
            kpis.append({"label": "ROC AUC", "value": m["roc_auc"]})
    elif ttype in ("regression", "timeseries"):
        kpis += [{"label": "R squared", "value": m.get("r2")},
                 {"label": "RMSE", "value": m.get("rmse")}, {"label": "MAE", "value": m.get("mae")}]
    elif ttype == "clustering":
        kpis.append({"label": "Clusters", "value": m.get("n_clusters")})
        if "silhouette" in m:
            kpis.append({"label": "Silhouette", "value": m["silhouette"]})
    kpis += [{"label": "Rows", "value": prof.get("n_rows")},
             {"label": "Features", "value": len(s.get("trained", {}).get("feature_cols", []))}]
    if board and board.get("rows"):
        kpis.append({"label": "Models compared",
                     "value": len([r for r in board["rows"] if r.get("score") is not None])})

    missing = prof.get("missing", {})
    n_cells = max(prof.get("n_rows", 1) * prof.get("n_cols", 1), 1)
    completeness = round(100 * (1 - sum(missing.values()) / n_cells), 1)

    allins = list(s.get("pre_insights", [])) + list(s.get("post_insights", []))
    seen, findings = set(), []
    for b in allins:
        if not b.startswith("📚") and b not in seen:
            seen.add(b); findings.append(b)
    guidance = list(dict.fromkeys(b[1:].strip() for b in allins if b.startswith("📚")))

    return {
        "dataset": Path(s["path"]).name, "model": s.get("model_name"),
        "task": ttype, "target": target,
        "columns": columns, "rows": rows,
        "kpis": kpis,
        "profile": {"n_rows": prof.get("n_rows"), "n_cols": prof.get("n_cols"),
                    "duplicates": prof.get("n_duplicates"), "memory_kb": prof.get("memory_kb"),
                    "completeness": completeness,
                    "type_split": {"numeric": len(prof.get("types", {}).get("numeric", [])),
                                   "categorical": len(prof.get("types", {}).get("categorical", []))},
                    "missing": {"labels": list(missing.keys()), "values": list(missing.values())}},
        "checks": s.get("checks", []),
        "insights": {"findings": findings, "guidance": guidance},
        "leaderboard": [{"name": r["name"], "score": r["score"]}
                        for r in (board["rows"] if board else []) if r.get("score") is not None][:12],
        "leaderboard_metric": board["metric"] if board else "",
        "features": [{"name": d["feature"].split("__")[-1].replace("_", " "),
                      "value": round(float(d["importance"]), 4)} for d in fi[:12]],
        "metrics": m,
        "confusion": data.get("confusion"), "class_dist": data.get("class_dist"),
        "per_class": data.get("per_class"), "roc": data.get("roc"), "pr": data.get("pr"),
        "scatter": data.get("scatter"), "series": data.get("series"),
        "cluster_sizes": data.get("cluster_sizes"), "cv": s.get("cv"),
    }


@app.route("/dashboard/<sid>")
def dashboard(sid):
    s = SESSIONS.get(sid) or abort(404)
    if "evaluation" not in s:
        abort(400)
    return render_template("dashboard.html", d=_dashboard_data(s))


if __name__ == "__main__":
    import os
    app.run(debug=True, port=int(os.environ.get("PORT", 5050)))
