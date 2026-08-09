"""Model leaderboard, hyperparameter tuning, and export.

Cross-validates every candidate model for a task and ranks them, so the user
picks from evidence rather than a single guess. Also tunes a chosen model and
exports the fitted pipeline for reuse.
"""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn import metrics
from .. import config
from .train import build_preprocessor, make_timeseries_features
from .model_select import _REGISTRY, build_estimator, NO_K_CLUSTERERS

# Primary scoring metric per task (higher = better for all of these).
PRIMARY = {"classification": "f1_weighted", "regression": "r2", "text": "f1_weighted",
           "timeseries": "r2"}

# Small, fast hyperparameter grids keyed by model name.
PARAM_GRIDS = {
    # classification
    "LogisticRegression": {"C": [0.1, 1.0, 10.0]},
    "KNeighborsClassifier": {"n_neighbors": [3, 5, 11]},
    "DecisionTreeClassifier": {"max_depth": [None, 5, 10]},
    "RandomForestClassifier": {"n_estimators": [100, 300], "max_depth": [None, 6, 12]},
    "ExtraTreesClassifier": {"n_estimators": [100, 300]},
    "AdaBoostClassifier": {"n_estimators": [50, 100]},
    "GradientBoostingClassifier": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
    "SVC": {"C": [0.1, 1.0, 10.0]},
    "XGBClassifier": {"n_estimators": [200, 400], "learning_rate": [0.05, 0.1]},
    "LGBMClassifier": {"n_estimators": [200, 400], "learning_rate": [0.05, 0.1]},
    # regression
    "Ridge": {"alpha": [0.1, 1.0, 10.0]},
    "Lasso": {"alpha": [0.01, 0.1, 1.0]},
    "ElasticNet": {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]},
    "KNeighborsRegressor": {"n_neighbors": [3, 5, 11]},
    "DecisionTreeRegressor": {"max_depth": [None, 5, 10]},
    "RandomForestRegressor": {"n_estimators": [100, 300], "max_depth": [None, 6, 12]},
    "ExtraTreesRegressor": {"n_estimators": [100, 300]},
    "AdaBoostRegressor": {"n_estimators": [50, 100]},
    "GradientBoostingRegressor": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
    "SVR": {"C": [0.1, 1.0, 10.0]},
    "XGBRegressor": {"n_estimators": [200, 400], "learning_rate": [0.05, 0.1]},
    "LGBMRegressor": {"n_estimators": [200, 400], "learning_rate": [0.05, 0.1]},
}


def _supervised_frame(df: pd.DataFrame, target: str):
    df = df.dropna(subset=[target]).reset_index(drop=True)
    features = [c for c in df.columns if c != target]
    return df, features, df[features], df[target]


def _rank(rows: list[dict]) -> list[dict]:
    rows.sort(key=lambda r: (r["score"] is not None, r["score"] if r["score"] is not None else -1e9),
              reverse=True)
    if rows and rows[0]["score"] is not None:
        rows[0]["best"] = True
    return rows


def run_leaderboard(df: pd.DataFrame, task_type: str, target: str | None, cv: int = 5,
                    time_col: str | None = None, text_col: str | None = None) -> dict:
    """Return {metric, rows:[{name, score, std, best}]} ranked best-first."""
    if task_type == "clustering":
        return _clustering_leaderboard(df, target)
    if task_type == "text":
        return _text_leaderboard(df, target, text_col)
    if task_type == "timeseries":
        return _timeseries_leaderboard(df, target, time_col)

    df, features, X, y = _supervised_frame(df, target)
    scoring = PRIMARY.get(task_type, "f1_weighted")
    n_splits = max(2, min(cv, int(y.value_counts().min()))) if task_type == "classification" else cv
    rows = []
    for m in _REGISTRY.get(task_type, []):
        pipe = Pipeline([("pre", build_preprocessor(df, features)), ("model", m["factory"]())])
        try:
            scores = cross_val_score(pipe, X, y, cv=n_splits, scoring=scoring)
            rows.append({"name": m["name"], "score": round(float(scores.mean()), 4),
                         "std": round(float(scores.std()), 4)})
        except Exception as e:
            rows.append({"name": m["name"], "score": None, "std": None, "error": str(e)[:120]})
    return {"metric": scoring, "rows": _rank(rows)}


def _text_leaderboard(df: pd.DataFrame, target: str, text_col: str) -> dict:
    df, _, _, y = _supervised_frame(df, target)
    X = df[[text_col]].astype(str)
    n_splits = max(2, min(5, int(y.value_counts().min())))
    rows = []
    for m in _REGISTRY["classification"]:
        pre = ColumnTransformer([("tfidf", TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), stop_words="english"), text_col)])
        pipe = Pipeline([("pre", pre), ("model", m["factory"]())])
        try:
            scores = cross_val_score(pipe, X, y, cv=n_splits, scoring="f1_weighted")
            rows.append({"name": m["name"], "score": round(float(scores.mean()), 4),
                         "std": round(float(scores.std()), 4)})
        except Exception as e:
            rows.append({"name": m["name"], "score": None, "std": None, "error": str(e)[:120]})
    return {"metric": "f1_weighted", "rows": _rank(rows)}


def _timeseries_leaderboard(df: pd.DataFrame, target: str, time_col: str) -> dict:
    from .train import timeseries_fit
    d, feat = make_timeseries_features(df, target, time_col)
    split = int(len(d) * (1 - config.TEST_SIZE))
    rows = []
    for m in _REGISTRY["regression"]:
        try:
            _, y_true, y_pred = timeseries_fit(d, feat, target, m["factory"](), split)
            rows.append({"name": m["name"],
                         "score": round(float(metrics.r2_score(y_true, y_pred)), 4), "std": None})
        except Exception as e:
            rows.append({"name": m["name"], "score": None, "std": None, "error": str(e)[:120]})
    return {"metric": "r2", "rows": _rank(rows)}


def _clustering_leaderboard(df: pd.DataFrame, target: str | None) -> dict:
    features = [c for c in df.columns if c != target]
    pre = build_preprocessor(df, features)
    X = pre.fit_transform(df[features])
    rows = []
    for m in _REGISTRY["clustering"]:
        try:
            est = m["factory"]() if m["name"] in NO_K_CLUSTERERS else m["factory"](k=3)
            labels = est.fit_predict(X)
            n = len(set(labels)) - (1 if -1 in labels else 0)
            sil = round(float(metrics.silhouette_score(X, labels)), 4) if n > 1 else None
            rows.append({"name": m["name"], "score": sil, "std": None, "n_clusters": int(n)})
        except Exception as e:
            rows.append({"name": m["name"], "score": None, "std": None, "error": str(e)[:120]})
    rows.sort(key=lambda r: (r["score"] is not None, r["score"] if r["score"] is not None else -1e9),
              reverse=True)
    if rows and rows[0]["score"] is not None:
        rows[0]["best"] = True
    return {"metric": "silhouette", "rows": rows}


def cross_validate_model(df: pd.DataFrame, task_type: str, target: str, model_name: str,
                         time_col: str | None = None, text_col: str | None = None,
                         cv: int = 5) -> dict | None:
    """Cross-validate a single chosen model, giving an honest generalisation
    estimate (mean and spread) beyond the one train/test split. Returns
    {metric, mean, std, folds} or None for tasks where it does not apply."""
    if task_type == "clustering":
        return None
    scoring = PRIMARY.get(task_type, "f1_weighted")
    try:
        if task_type == "text":
            d, _, _, y = _supervised_frame(df, target)
            X = d[[text_col]].astype(str)
            pre = ColumnTransformer([("tfidf", TfidfVectorizer(
                max_features=5000, ngram_range=(1, 2), stop_words="english"), text_col)])
            est = build_estimator("classification", model_name)
        elif task_type == "timeseries":
            from sklearn.model_selection import TimeSeriesSplit
            d, feat = make_timeseries_features(df, target, time_col)
            X, y = d[feat], d[target]
            pre = build_preprocessor(d, feat)
            est = build_estimator("timeseries", model_name)
            cv = TimeSeriesSplit(n_splits=min(4, max(2, len(d) // 20)))
        else:
            d, feat, X, y = _supervised_frame(df, target)
            pre = build_preprocessor(d, feat)
            est = build_estimator(task_type, model_name)
            if task_type == "classification":
                cv = max(2, min(cv, int(y.value_counts().min())))
        pipe = Pipeline([("pre", pre), ("model", est)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
        return {"metric": scoring, "mean": round(float(scores.mean()), 4),
                "std": round(float(scores.std()), 4), "folds": len(scores),
                "scores": [round(float(x), 4) for x in scores]}
    except Exception:
        return None


def tune_model(df: pd.DataFrame, task_type: str, target: str, model_name: str, cv: int = 3) -> dict | None:
    """Grid-search a model's hyperparameters. Returns best params + score, or
    None if no grid is defined for that model."""
    grid = PARAM_GRIDS.get(model_name)
    if not grid or task_type in ("text", "timeseries", "clustering"):
        return None
    df, features, X, y = _supervised_frame(df, target)
    scoring = PRIMARY.get(task_type, "f1_weighted")
    base = "classification" if task_type == "text" else "regression" if task_type == "timeseries" else task_type
    pipe = Pipeline([("pre", build_preprocessor(df, features)),
                     ("model", build_estimator(base, model_name))])
    search = GridSearchCV(pipe, {f"model__{k}": v for k, v in grid.items()},
                          cv=cv, scoring=scoring, n_jobs=-1)
    search.fit(X, y)
    return {
        "model_name": model_name,
        "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        "best_score": round(float(search.best_score_), 4),
        "metric": scoring,
        "estimator": search.best_estimator_,
    }


def build_ensemble(task_type: str, member_names: list[str], kind: str = "voting"):
    """Build a voting or stacking ensemble estimator from named base models.
    Returns an unfitted estimator to plug into the training pipeline."""
    from sklearn.ensemble import (VotingClassifier, VotingRegressor,
                                  StackingClassifier, StackingRegressor)
    from sklearn.linear_model import LogisticRegression, Ridge
    base = "classification" if task_type in ("classification", "text") else "regression"
    members = [(n, build_estimator(base, n)) for n in member_names]
    if base == "classification":
        if kind == "stacking":
            return StackingClassifier(members, final_estimator=LogisticRegression(max_iter=1000), cv=3)
        soft = all(hasattr(m, "predict_proba") for _, m in members)
        return VotingClassifier(members, voting="soft" if soft else "hard")
    if kind == "stacking":
        return StackingRegressor(members, final_estimator=Ridge(), cv=3)
    return VotingRegressor(members)


def export_model(fitted_pipeline, filename: str = "model.joblib") -> str:
    """Persist a fitted pipeline to outputs/ and return the path."""
    outdir = config.OUTPUT_DIR / "models"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    joblib.dump(fitted_pipeline, path)
    return str(path)
