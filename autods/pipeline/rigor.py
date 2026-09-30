"""Modelling-rigor helpers: class-imbalance resampling, decision-threshold
tuning, probability-calibration diagnostics, and simple fairness checks.

All of these operate only on the training split (for resampling) or on held-out
predictions (for the diagnostics), so nothing here leaks test information. SMOTE
is used when imbalanced-learn is installed; otherwise a dependency-free random
oversampler gives the same balancing effect with plain pandas.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import metrics

from .. import config

RS = config.RANDOM_STATE


# ---------------------------------------------------------------------------
#  Class imbalance
# ---------------------------------------------------------------------------
def smote_available() -> bool:
    try:
        import imblearn  # noqa: F401
        return True
    except Exception:
        return False


def positive_label(classes) -> object:
    """The class treated as 'positive' for binary metrics: 1 if present, else max."""
    classes = list(classes)
    if 1 in classes:
        return 1
    if "1" in [str(c) for c in classes]:
        return next(c for c in classes if str(c) == "1")
    return max(classes, key=str)


def oversample(X_train: pd.DataFrame, y_train, random_state: int = RS):
    """Random oversampling of minority classes up to the majority count. Works on
    the raw training rows, so it needs no extra dependency and stays leakage-safe."""
    X = X_train.reset_index(drop=True)
    y = pd.Series(list(y_train)).reset_index(drop=True)
    counts = y.value_counts()
    target_n = int(counts.max())
    picks: list[int] = []
    for cls, n in counts.items():
        idx = y.index[y == cls].tolist()
        picks.extend(idx)
        if n < target_n:
            extra = pd.Series(idx).sample(target_n - n, replace=True,
                                          random_state=random_state).tolist()
            picks.extend(extra)
    rng = np.random.RandomState(random_state)
    picks = np.array(picks)
    rng.shuffle(picks)
    return X.iloc[picks].reset_index(drop=True), y.iloc[picks].reset_index(drop=True)


def resolve_balance(balanced: bool, balance: str | None) -> str:
    """Turn the legacy ``balanced`` flag and optional ``balance`` choice into a
    single strategy: none | class_weight | oversample | smote."""
    if balance:
        return balance
    return "class_weight" if balanced else "none"


# ---------------------------------------------------------------------------
#  Decision-threshold tuning (binary)
# ---------------------------------------------------------------------------
def tune_threshold(y_true, proba, pos_label) -> dict:
    """Find the probability threshold that maximises F1 for the positive class,
    and report metrics at both the default 0.5 and the tuned threshold."""
    y_bin = (np.asarray(y_true) == pos_label).astype(int)
    p = np.asarray(proba, dtype=float)

    def at(thresh):
        pred = (p >= thresh).astype(int)
        return {"threshold": round(float(thresh), 3),
                "precision": round(float(metrics.precision_score(y_bin, pred, zero_division=0)), 3),
                "recall": round(float(metrics.recall_score(y_bin, pred, zero_division=0)), 3),
                "f1": round(float(metrics.f1_score(y_bin, pred, zero_division=0)), 3)}

    best, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        f1 = metrics.f1_score(y_bin, (p >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best = f1, t
    return {"default": at(0.5), "tuned": at(best)}


# ---------------------------------------------------------------------------
#  Probability calibration diagnostics (binary)
# ---------------------------------------------------------------------------
def calibration(y_true, proba, pos_label, n_bins: int = 10) -> dict:
    """Brier score plus a reliability curve, so we can see whether the predicted
    probabilities actually mean what they say."""
    y_bin = (np.asarray(y_true) == pos_label).astype(int)
    p = np.asarray(proba, dtype=float)
    brier = float(metrics.brier_score_loss(y_bin, p))
    edges = np.linspace(0, 1, n_bins + 1)
    curve = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            curve.append({"mean_pred": round(float(p[m].mean()), 3),
                          "frac_pos": round(float(y_bin[m].mean()), 3),
                          "count": int(m.sum())})
    return {"brier": round(brier, 4), "curve": curve}


# ---------------------------------------------------------------------------
#  Fairness across groups (classification)
# ---------------------------------------------------------------------------
def fairness(X_test: pd.DataFrame, y_true, y_pred, pos_label,
             max_cardinality: int = 8) -> list[dict]:
    """Per-group selection rate and accuracy for each low-cardinality categorical
    feature, with the largest gap flagged. A large selection-rate gap is a signal
    to inspect, not a verdict."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    reports = []
    for col in X_test.columns:
        s = X_test[col]
        if pd.api.types.is_numeric_dtype(s) or s.nunique() > max_cardinality:
            continue
        groups = []
        for val, m in s.groupby(s).groups.items():
            mask = X_test.index.isin(m)
            if mask.sum() < 5:
                continue
            sel = float((y_pred[mask] == pos_label).mean())
            acc = float((y_pred[mask] == y_true[mask]).mean())
            groups.append({"value": str(val), "n": int(mask.sum()),
                           "selection_rate": round(sel, 3), "accuracy": round(acc, 3)})
        if len(groups) < 2:
            continue
        rates = [g["selection_rate"] for g in groups]
        reports.append({
            "feature": col,
            "groups": sorted(groups, key=lambda g: g["selection_rate"], reverse=True),
            "selection_gap": round(max(rates) - min(rates), 3),
        })
    reports.sort(key=lambda r: r["selection_gap"], reverse=True)
    return reports
