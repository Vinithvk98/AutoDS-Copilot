"""Explainability and error analysis.

Two model-agnostic trust tools that work for any estimator, not just trees:

  permutation_importance  which original columns the model actually relies on,
                          measured by how much the score drops when each column
                          is shuffled on the held-out test set.
  error_analysis          where the model is getting things wrong, by group for
                          classification and by largest residual for regression.

Both read the fitted pipeline and the untouched test split, so they describe the
real model on real held-out data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .. import config


def _norm_task(task_type: str) -> str:
    return {"text": "classification", "timeseries": "regression"}.get(task_type, task_type)


def permutation_importance(trained: dict, n_repeats: int = 6, top: int = 12) -> list[dict]:
    """Model-agnostic importance per ORIGINAL feature, from the drop in score when
    each column is shuffled on the test set. Works for models with no built-in
    importances (SVM, KNN, neural nets). Returns [] when not applicable."""
    task = _norm_task(trained.get("task_type", ""))
    pipe = trained.get("pipeline")
    if task not in ("classification", "regression") or pipe is None:
        return []
    X_test = trained.get("X_test")
    y_test = trained.get("y_test")
    if X_test is None or y_test is None or not hasattr(X_test, "columns"):
        return []
    try:
        from sklearn.inspection import permutation_importance as _pi
        r = _pi(pipe, X_test, y_test, n_repeats=n_repeats,
                random_state=config.RANDOM_STATE, n_jobs=1)
    except Exception:
        return []
    cols = list(X_test.columns)
    out = [{"feature": str(cols[i]),
            "importance": round(float(r.importances_mean[i]), 4),
            "std": round(float(r.importances_std[i]), 4)}
           for i in range(len(cols))]
    out = [d for d in out if d["importance"] > 0]
    out.sort(key=lambda d: d["importance"], reverse=True)
    return out[:top]


def error_analysis(trained: dict, max_card: int = 8, examples: int = 5) -> dict:
    """Where the model errs. For classification: overall error rate plus the
    categorical groups it struggles with most. For regression: the largest
    residuals. Reads the held-out test split."""
    task = _norm_task(trained.get("task_type", ""))
    pipe = trained.get("pipeline")
    X_test = trained.get("X_test")
    y_test = trained.get("y_test")
    if pipe is None or X_test is None or y_test is None:
        return {}

    if task == "classification":
        y_true = np.asarray(y_test)
        y_pred = np.asarray(trained.get("y_pred")) if trained.get("y_pred") is not None \
            else pipe.predict(X_test)
        wrong = y_pred != y_true
        out = {"n_test": int(len(y_true)), "n_errors": int(wrong.sum()),
               "error_rate": round(float(wrong.mean()), 3), "by_group": []}
        if hasattr(X_test, "columns"):
            groups = []
            for col in X_test.columns:
                s = X_test[col]
                if pd.api.types.is_numeric_dtype(s) or s.nunique() > max_card:
                    continue
                rows = []
                for val, idx in s.groupby(s).groups.items():
                    m = X_test.index.isin(idx)
                    if m.sum() < 5:
                        continue
                    rows.append({"value": str(val), "n": int(m.sum()),
                                 "error_rate": round(float((y_pred[m] != y_true[m]).mean()), 3)})
                if len(rows) >= 2:
                    rows.sort(key=lambda r: r["error_rate"], reverse=True)
                    groups.append({"feature": col, "rows": rows,
                                   "gap": round(rows[0]["error_rate"] - rows[-1]["error_rate"], 3)})
            groups.sort(key=lambda g: g["gap"], reverse=True)
            out["by_group"] = groups[:3]
        return out

    if task == "regression":
        y_true = np.asarray(y_test, dtype=float)
        y_pred = np.asarray(trained.get("y_pred")) if trained.get("y_pred") is not None \
            else np.asarray(pipe.predict(X_test), dtype=float)
        resid = np.abs(y_true - y_pred)
        order = np.argsort(resid)[::-1][:examples]
        worst = [{"actual": round(float(y_true[i]), 3), "predicted": round(float(y_pred[i]), 3),
                  "abs_error": round(float(resid[i]), 3)} for i in order]
        return {"n_test": int(len(y_true)), "mean_abs_error": round(float(resid.mean()), 3),
                "worst": worst}
    return {}
