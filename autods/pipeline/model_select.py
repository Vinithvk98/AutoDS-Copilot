"""Stage 5 — the algorithm hub. Recommend candidate models for a task.

A broad roster of scikit-learn estimators across classification, regression, and
clustering, plus XGBoost and LightGBM when they are installed. Factories are kept
internal; use build_estimator(name) to instantiate.
"""
from __future__ import annotations
from sklearn.linear_model import (LogisticRegression, RidgeClassifier,
                                  LinearRegression, Ridge, Lasso, ElasticNet)
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              AdaBoostClassifier, AdaBoostRegressor)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import (KMeans, MiniBatchKMeans, AgglomerativeClustering,
                             DBSCAN, MeanShift, SpectralClustering, Birch, OPTICS)
from sklearn.mixture import GaussianMixture

RS = 42

# Clustering algorithms that do NOT take a number-of-clusters argument.
NO_K_CLUSTERERS = {"DBSCAN", "MeanShift", "OPTICS"}

_REGISTRY: dict[str, list[dict]] = {
    "classification": [
        {"name": "LogisticRegression", "why": "Fast, interpretable linear baseline.",
         "factory": lambda: LogisticRegression(max_iter=1000)},
        {"name": "RidgeClassifier", "why": "Linear classifier with L2 regularisation.",
         "factory": lambda: RidgeClassifier()},
        {"name": "GaussianNB", "why": "Naive Bayes, a very fast probabilistic baseline.",
         "factory": lambda: GaussianNB()},
        {"name": "KNeighborsClassifier", "why": "Instance based, captures local structure.",
         "factory": lambda: KNeighborsClassifier()},
        {"name": "DecisionTreeClassifier", "why": "A single tree with readable rules.",
         "factory": lambda: DecisionTreeClassifier(random_state=RS)},
        {"name": "RandomForestClassifier", "why": "Strong default, handles mixed types.",
         "factory": lambda: RandomForestClassifier(n_estimators=200, random_state=RS)},
        {"name": "ExtraTreesClassifier", "why": "Randomised forest, fast and robust.",
         "factory": lambda: ExtraTreesClassifier(n_estimators=200, random_state=RS)},
        {"name": "AdaBoostClassifier", "why": "Boosts weak learners into a strong one.",
         "factory": lambda: AdaBoostClassifier(random_state=RS)},
        {"name": "GradientBoostingClassifier", "why": "Often top accuracy on tabular data.",
         "factory": lambda: GradientBoostingClassifier(random_state=RS)},
        {"name": "SVC", "why": "Support vector machine, strong on clean scaled data.",
         "factory": lambda: SVC()},
        {"name": "MLPClassifier", "why": "Small neural net for non-linear patterns.",
         "factory": lambda: MLPClassifier(max_iter=300, random_state=RS)},
    ],
    "regression": [
        {"name": "LinearRegression", "why": "Simple, interpretable baseline.",
         "factory": lambda: LinearRegression()},
        {"name": "Ridge", "why": "Linear with L2, robust to collinearity.",
         "factory": lambda: Ridge()},
        {"name": "Lasso", "why": "L1 regularisation, does feature selection.",
         "factory": lambda: Lasso()},
        {"name": "ElasticNet", "why": "Blend of L1 and L2 penalties.",
         "factory": lambda: ElasticNet()},
        {"name": "KNeighborsRegressor", "why": "Instance-based local prediction.",
         "factory": lambda: KNeighborsRegressor()},
        {"name": "DecisionTreeRegressor", "why": "A single tree, captures non-linearity.",
         "factory": lambda: DecisionTreeRegressor(random_state=RS)},
        {"name": "RandomForestRegressor", "why": "Reliable non-linear default.",
         "factory": lambda: RandomForestRegressor(n_estimators=200, random_state=RS)},
        {"name": "ExtraTreesRegressor", "why": "Randomised forest, fast and robust.",
         "factory": lambda: ExtraTreesRegressor(n_estimators=200, random_state=RS)},
        {"name": "AdaBoostRegressor", "why": "Boosted ensemble of weak learners.",
         "factory": lambda: AdaBoostRegressor(random_state=RS)},
        {"name": "GradientBoostingRegressor", "why": "Usually best-in-class on tabular.",
         "factory": lambda: GradientBoostingRegressor(random_state=RS)},
        {"name": "SVR", "why": "Support-vector regression on scaled data.",
         "factory": lambda: SVR()},
        {"name": "MLPRegressor", "why": "Small neural net for non-linear targets.",
         "factory": lambda: MLPRegressor(max_iter=300, random_state=RS)},
    ],
    "clustering": [
        {"name": "KMeans", "why": "Fast, good for roughly spherical clusters.",
         "factory": lambda k=3: KMeans(n_clusters=k, n_init=10, random_state=RS)},
        {"name": "MiniBatchKMeans", "why": "KMeans for larger data, quicker.",
         "factory": lambda k=3: MiniBatchKMeans(n_clusters=k, n_init=10, random_state=RS)},
        {"name": "AgglomerativeClustering", "why": "Hierarchical, no shape assumption.",
         "factory": lambda k=3: AgglomerativeClustering(n_clusters=k)},
        {"name": "SpectralClustering", "why": "Graph based, finds non convex groups.",
         "factory": lambda k=3: SpectralClustering(n_clusters=k, random_state=RS,
                                                   assign_labels="discretize")},
        {"name": "Birch", "why": "Efficient for large datasets.",
         "factory": lambda k=3: Birch(n_clusters=k)},
        {"name": "GaussianMixture", "why": "Soft, probabilistic clusters.",
         "factory": lambda k=3: GaussianMixture(n_components=k, random_state=RS)},
        {"name": "DBSCAN", "why": "Density based, finds shapes and outliers.",
         "factory": lambda: DBSCAN()},
        {"name": "MeanShift", "why": "Finds clusters without picking k.",
         "factory": lambda: MeanShift()},
        {"name": "OPTICS", "why": "Density based, varying density support.",
         "factory": lambda: OPTICS(min_samples=5)},
    ],
}

# Optional gradient-boosting libraries — added only if installed.
try:
    from xgboost import XGBClassifier, XGBRegressor
    _REGISTRY["classification"].append(
        {"name": "XGBClassifier", "why": "XGBoost, powerful gradient boosting.",
         "factory": lambda: XGBClassifier(n_estimators=300, verbosity=0, random_state=RS)})
    _REGISTRY["regression"].append(
        {"name": "XGBRegressor", "why": "XGBoost, powerful gradient boosting.",
         "factory": lambda: XGBRegressor(n_estimators=300, verbosity=0, random_state=RS)})
except Exception:
    pass
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _REGISTRY["classification"].append(
        {"name": "LGBMClassifier", "why": "LightGBM, fast and accurate boosting.",
         "factory": lambda: LGBMClassifier(n_estimators=300, verbose=-1, random_state=RS)})
    _REGISTRY["regression"].append(
        {"name": "LGBMRegressor", "why": "LightGBM, fast and accurate boosting.",
         "factory": lambda: LGBMRegressor(n_estimators=300, verbose=-1, random_state=RS)})
except Exception:
    pass

# Some task types reuse another task's estimator family.
_BASE_TASK = {"timeseries": "regression", "text": "classification"}

# The default the UI pre-selects (falls back to the last supervised learner).
_DEFAULTS = {"classification": "GradientBoostingClassifier",
             "regression": "GradientBoostingRegressor",
             "clustering": "KMeans", "timeseries": "RandomForestRegressor",
             "text": "LogisticRegression"}


# Group each model into a family for a scannable picker.
_FAMILY = {
    "LogisticRegression": "Linear", "RidgeClassifier": "Linear", "LinearRegression": "Linear",
    "Ridge": "Linear", "Lasso": "Linear", "ElasticNet": "Linear",
    "GaussianNB": "Naive Bayes",
    "KNeighborsClassifier": "Neighbors", "KNeighborsRegressor": "Neighbors",
    "DecisionTreeClassifier": "Tree", "DecisionTreeRegressor": "Tree",
    "RandomForestClassifier": "Ensemble", "RandomForestRegressor": "Ensemble",
    "ExtraTreesClassifier": "Ensemble", "ExtraTreesRegressor": "Ensemble",
    "AdaBoostClassifier": "Boosting", "AdaBoostRegressor": "Boosting",
    "GradientBoostingClassifier": "Boosting", "GradientBoostingRegressor": "Boosting",
    "XGBClassifier": "Boosting", "XGBRegressor": "Boosting",
    "LGBMClassifier": "Boosting", "LGBMRegressor": "Boosting",
    "SVC": "SVM", "SVR": "SVM",
    "MLPClassifier": "Neural net", "MLPRegressor": "Neural net",
    "KMeans": "Centroid", "MiniBatchKMeans": "Centroid",
    "AgglomerativeClustering": "Hierarchical", "Birch": "Hierarchical",
    "DBSCAN": "Density", "OPTICS": "Density", "MeanShift": "Density",
    "SpectralClustering": "Graph", "GaussianMixture": "Probabilistic",
}
_FAMILY_ORDER = ["Linear", "Naive Bayes", "Neighbors", "Tree", "Ensemble", "Boosting",
                 "SVM", "Neural net", "Centroid", "Hierarchical", "Density", "Graph",
                 "Probabilistic", "Other"]


def base_task(task_type: str) -> str:
    return _BASE_TASK.get(task_type, task_type)


def recommend_models(task_type: str) -> list[dict]:
    """Return serializable candidate dicts (recommended default flagged)."""
    bt = base_task(task_type)
    models = _REGISTRY.get(bt, [])
    default = _DEFAULTS.get(task_type, models[-1]["name"] if models else "")
    return [{"name": m["name"], "why": m["why"], "recommended": m["name"] == default,
             "family": _FAMILY.get(m["name"], "Other")} for m in models]


def recommend_grouped(task_type: str) -> list[dict]:
    """Same candidates, grouped by family in a sensible order for the UI."""
    models = recommend_models(task_type)
    buckets: dict[str, list] = {}
    for m in models:
        buckets.setdefault(m["family"], []).append(m)
    return [{"family": fam, "models": buckets[fam]}
            for fam in _FAMILY_ORDER if fam in buckets]


def build_estimator(task_type: str, name: str, **kwargs):
    """Instantiate an estimator by name for the given task."""
    for m in _REGISTRY.get(base_task(task_type), []):
        if m["name"] == name:
            return m["factory"](**kwargs) if kwargs else m["factory"]()
    raise ValueError(f"Unknown model '{name}' for task '{task_type}'")


def count() -> dict:
    """Handy summary of how many algorithms the hub exposes."""
    return {t: len(v) for t, v in _REGISTRY.items()}
