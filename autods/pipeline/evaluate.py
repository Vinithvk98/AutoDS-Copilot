"""Stage 7 — Evaluate the trained model and produce metrics + diagnostic plots."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import metrics
from .. import config


def _save(fig, name: str, outdir: Path) -> str:
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / name
    fig.savefig(p, bbox_inches="tight", dpi=110)
    plt.close(fig)
    return str(p)


def feature_importance(trained: dict) -> list[dict]:
    """Extract feature importances / coefficients when available."""
    pipe = trained.get("pipeline")
    if pipe is None:
        return []
    model = pipe.named_steps["model"]
    try:
        names = pipe.named_steps["pre"].get_feature_names_out()
    except Exception:
        return []
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        vals = np.abs(np.ravel(model.coef_))
    else:
        return []
    order = np.argsort(vals)[::-1][:12]
    return [{"feature": str(names[i]), "importance": float(vals[i])} for i in order]


def evaluate(trained: dict, outdir: Path | None = None) -> dict:
    """Return {metrics: {...}, plots: [paths], feature_importance: [...]}"""
    outdir = Path(outdir or config.OUTPUT_DIR / "eval")
    raw_task = trained["task_type"]
    # text is scored like classification; timeseries like regression
    task = {"text": "classification", "timeseries": "regression"}.get(raw_task, raw_task)
    result = {"metrics": {}, "plots": [], "feature_importance": [], "data": {}}

    if task == "clustering":
        X, labels = trained["X"], trained["labels"]
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        result["metrics"]["n_clusters"] = int(n_clusters)
        if n_clusters > 1:
            result["metrics"]["silhouette"] = round(
                float(metrics.silhouette_score(X, labels)), 3)
        uniq, counts = np.unique(labels, return_counts=True)
        result["data"]["cluster_sizes"] = {("noise" if int(u) == -1 else f"cluster {int(u)}"): int(c)
                                           for u, c in zip(uniq, counts)}
        # 2D PCA scatter coloured by cluster
        try:
            from sklearn.decomposition import PCA
            pcs = PCA(n_components=2).fit_transform(X)
            fig, ax = plt.subplots(figsize=(5.5, 4))
            sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=labels, cmap="tab10", s=15)
            ax.set_title("Clusters (PCA projection)")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            result["plots"].append(_save(fig, "clusters.png", outdir))
        except Exception:
            pass
        return result

    pipe = trained["pipeline"]
    X_test, y_test = trained["X_test"], trained["y_test"]
    y_test = np.asarray(y_test)
    # timeseries supplies reconstructed level predictions; others predict here
    y_pred = trained.get("y_pred")
    y_pred = np.asarray(y_pred) if y_pred is not None else pipe.predict(X_test)

    if task == "classification":
        result["metrics"] = {
            "accuracy": round(float(metrics.accuracy_score(y_test, y_pred)), 3),
            "precision": round(float(metrics.precision_score(y_test, y_pred, average="weighted", zero_division=0)), 3),
            "recall": round(float(metrics.recall_score(y_test, y_pred, average="weighted", zero_division=0)), 3),
            "f1": round(float(metrics.f1_score(y_test, y_pred, average="weighted", zero_division=0)), 3),
        }
        # ROC-AUC for binary
        try:
            if len(np.unique(y_test)) == 2 and hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(X_test)[:, 1]
                result["metrics"]["roc_auc"] = round(float(metrics.roc_auc_score(y_test, proba)), 3)
        except Exception:
            pass
        # Confusion matrix
        cm = metrics.confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title("Confusion matrix")
        fig.colorbar(im, ax=ax, fraction=0.046)
        result["plots"].append(_save(fig, "confusion.png", outdir))
        # raw numbers for the interactive dashboard
        labels = sorted(set(y_test.tolist()) | set(np.asarray(y_pred).tolist()), key=str)
        cm2 = metrics.confusion_matrix(y_test, y_pred, labels=labels)
        vc = pd.Series(y_test).value_counts()
        result["data"]["confusion"] = {"labels": [str(l) for l in labels],
                                       "matrix": cm2.tolist()}
        result["data"]["class_dist"] = {str(k): int(v) for k, v in vc.items()}
        # per-class precision / recall / f1
        try:
            rep = metrics.classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            result["data"]["per_class"] = [
                {"label": str(k), "precision": round(v["precision"], 3),
                 "recall": round(v["recall"], 3), "f1": round(v["f1-score"], 3),
                 "support": int(v["support"])}
                for k, v in rep.items()
                if isinstance(v, dict) and k not in ("accuracy", "macro avg", "weighted avg")]
        except Exception:
            pass
        # ROC and precision-recall curves for binary problems
        try:
            if len(np.unique(y_test)) == 2 and hasattr(pipe, "predict_proba"):
                pr_pos = pipe.predict_proba(X_test)[:, 1]
                def _ds(a, n=60):
                    i = np.linspace(0, len(a) - 1, min(n, len(a))).astype(int)
                    return [round(float(a[j]), 4) for j in i]
                fpr, tpr, _ = metrics.roc_curve(y_test, pr_pos)
                result["data"]["roc"] = {"fpr": _ds(fpr), "tpr": _ds(tpr)}
                prec, rec, _ = metrics.precision_recall_curve(y_test, pr_pos)
                result["data"]["pr"] = {"precision": _ds(prec), "recall": _ds(rec)}
        except Exception:
            pass

    else:  # regression
        result["metrics"] = {
            "r2": round(float(metrics.r2_score(y_test, y_pred)), 3),
            "mae": round(float(metrics.mean_absolute_error(y_test, y_pred)), 3),
            "rmse": round(float(np.sqrt(metrics.mean_squared_error(y_test, y_pred))), 3),
        }
        # raw numbers for the interactive dashboard
        n = min(300, len(y_test))
        idx = np.linspace(0, len(y_test) - 1, n).astype(int) if len(y_test) else []
        result["data"]["scatter"] = [[round(float(y_test[i]), 3), round(float(y_pred[i]), 3)]
                                     for i in idx]
        if raw_task == "timeseries" and trained.get("time_index"):
            result["data"]["series"] = {
                "actual": [round(float(v), 3) for v in y_test],
                "pred": [round(float(v), 3) for v in y_pred]}
        if raw_task == "timeseries" and trained.get("time_index"):
            # time-ordered line plot of actual vs forecast
            fig, ax = plt.subplots(figsize=(7, 3.8))
            idx = range(len(y_test))
            ax.plot(idx, list(y_test), label="actual", color="#4f81bd")
            ax.plot(idx, list(y_pred), label="forecast", color="#f6ad55")
            ax.set_title("Forecast vs actual (test period)")
            ax.set_xlabel("time step"); ax.legend()
            result["plots"].append(_save(fig, "forecast.png", outdir))
        else:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(y_test, y_pred, alpha=0.5, s=18, color="#4f81bd")
            lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            ax.plot([lo, hi], [lo, hi], "r--", lw=1)
            ax.set_xlabel("actual"); ax.set_ylabel("predicted"); ax.set_title("Actual vs predicted")
            result["plots"].append(_save(fig, "actual_vs_pred.png", outdir))

    result["feature_importance"] = feature_importance(trained)
    return result
