"""Stage 3 — Exploratory Data Analysis.

Produces a varied, readable set of charts (seaborn-styled) with the code that
made each one and a short insight: KDE histograms, box plots for outliers, pie
charts for categories, an annotated correlation heatmap, a scatter of the most
correlated pair, and target-relationship plots.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")  # headless backend for the web app
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from .. import config
from .loader import column_types

sns.set_theme(style="whitegrid", palette="deep")
_PALETTE = sns.color_palette("crest", as_cmap=False)


def _save(fig, outdir: Path, name: str) -> str:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=110)
    plt.close(fig)
    return str(path)


def generate_eda(df: pd.DataFrame, target: str | None = None,
                 outdir: Path | None = None) -> list[dict]:
    """Return a list of EDA charts, each as {title, path, code, insight}."""
    outdir = Path(outdir or config.OUTPUT_DIR / "eda")
    types = column_types(df)
    numeric = [c for c in types["numeric"]]
    categorical = types["categorical"]
    charts: list[dict] = []

    # 1) Missing-values overview -------------------------------------------
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss):
        fig, ax = plt.subplots(figsize=(6, 3.6))
        sns.barplot(x=miss.values, y=miss.index, hue=miss.index, legend=False,
                    palette="rocket", ax=ax)
        ax.set_xlabel("missing count"); ax.set_ylabel(""); ax.set_title("Missing values by column")
        charts.append({
            "title": "Missing values by column",
            "path": _save(fig, outdir, "missing.png"),
            "code": "sns.barplot(x=df.isna().sum().values, y=df.columns)",
            "insight": f"{len(miss)} column(s) have gaps; '{miss.index[0]}' is the "
                       f"emptiest with {int(miss.iloc[0])} missing.",
        })

    # 2) Numeric distributions (histogram + KDE) ---------------------------
    for col in [c for c in numeric if c != target][:5]:
        if df[col].nunique() < 2:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.6))
        sns.histplot(df[col].dropna(), kde=True, color="#2a9d8f", edgecolor="white", ax=ax)
        ax.set_title(f"Distribution of {col}")
        skew = float(df[col].skew())
        shape = "right-skewed" if skew > 1 else "left-skewed" if skew < -1 else "roughly symmetric"
        charts.append({
            "title": f"Distribution of {col}",
            "path": _save(fig, outdir, f"hist_{col}.png"),
            "code": f"sns.histplot(df['{col}'], kde=True)",
            "insight": f"'{col}' spans {df[col].min():.2f} to {df[col].max():.2f}, "
                       f"mean {df[col].mean():.2f}, and looks {shape} (skew {skew:.2f}).",
        })

    # 3) Box plots for outliers (standardised so scales are comparable) ----
    if len(numeric) >= 2:
        z = (df[numeric] - df[numeric].mean()) / df[numeric].std(ddof=0)
        long = z.melt(var_name="feature", value_name="z-score").dropna()
        fig, ax = plt.subplots(figsize=(6, 3.8))
        sns.boxplot(data=long, x="z-score", y="feature", hue="feature", legend=False,
                    palette="crest", ax=ax)
        ax.set_title("Spread & outliers (standardised)"); ax.set_ylabel("")
        charts.append({
            "title": "Spread & outliers",
            "path": _save(fig, outdir, "box.png"),
            "code": "sns.boxplot(data=(df[num]-df[num].mean())/df[num].std())",
            "insight": "Points far past the whiskers are outliers. Standardising lets "
                       "you compare spread across features on one scale.",
        })

    # 4) Categorical counts + pie charts -----------------------------------
    cat_small = [c for c in categorical if c != target
                 and df[c].nunique() <= config.MAX_CATEGORICAL_CARDINALITY]
    for col in cat_small[:2]:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        order = df[col].value_counts().head(12).index
        sns.countplot(data=df, y=col, order=order, hue=col, legend=False,
                      palette="mako", ax=ax)
        ax.set_title(f"Counts of {col}"); ax.set_ylabel("")
        top = df[col].value_counts().idxmax()
        charts.append({
            "title": f"Counts of {col}",
            "path": _save(fig, outdir, f"count_{col}.png"),
            "code": f"sns.countplot(data=df, y='{col}')",
            "insight": f"'{col}' has {df[col].nunique()} categories; '{top}' is the most common.",
        })
    # pie for the lowest-cardinality categorical(s)
    for col in sorted(cat_small, key=lambda c: df[c].nunique())[:2]:
        if df[col].nunique() > 6:
            continue
        vc = df[col].value_counts()
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        ax.pie(vc.values, labels=vc.index, autopct="%1.0f%%", startangle=90,
               colors=sns.color_palette("crest", len(vc)),
               wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        ax.set_title(f"Share of {col}")
        charts.append({
            "title": f"Share of {col}",
            "path": _save(fig, outdir, f"pie_{col}.png"),
            "code": f"plt.pie(df['{col}'].value_counts(), autopct='%1.0f%%')",
            "insight": f"'{vc.index[0]}' makes up {vc.iloc[0]/vc.sum()*100:.0f}% of '{col}'.",
        })

    # 4b) Free-text columns: length distribution + top terms --------------
    text_cols = []
    for col in df.select_dtypes(include="object").columns:
        if col == target:
            continue
        s = df[col].dropna().astype(str)
        if len(s) and s.str.len().mean() >= 20 and s.str.split().str.len().mean() >= 3:
            text_cols.append(col)
    for col in text_cols[:1]:
        s = df[col].dropna().astype(str)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        sns.histplot(s.str.len(), kde=True, color="#e76f51", edgecolor="white", ax=ax)
        ax.set_title(f"Text length: {col}"); ax.set_xlabel("characters")
        charts.append({
            "title": f"Text length of {col}",
            "path": _save(fig, outdir, f"txtlen_{col}.png"),
            "code": f"sns.histplot(df['{col}'].str.len(), kde=True)",
            "insight": f"'{col}' averages {s.str.len().mean():.0f} characters per entry.",
        })
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            cv = CountVectorizer(stop_words="english", max_features=12)
            counts = np.asarray(cv.fit_transform(s).sum(axis=0)).ravel()
            terms = cv.get_feature_names_out()
            order = counts.argsort()[::-1]
            top = pd.Series(counts[order], index=terms[order])
            fig, ax = plt.subplots(figsize=(6, 3.8))
            sns.barplot(x=top.values, y=top.index, hue=top.index, legend=False,
                        palette="flare", ax=ax)
            ax.set_title(f"Top words in {col}"); ax.set_xlabel("count"); ax.set_ylabel("")
            charts.append({
                "title": f"Top words in {col}",
                "path": _save(fig, outdir, f"terms_{col}.png"),
                "code": "CountVectorizer(stop_words='english').fit_transform(text)",
                "insight": f"Most frequent term is '{top.index[0]}'. These drive a "
                           "TF-IDF text model.",
            })
        except Exception:
            pass

    # 5) Correlation heatmap (annotated when small) ------------------------
    strongest = None
    if len(numeric) >= 2:
        corr = df[numeric].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(5.6, 4.6))
        sns.heatmap(corr, annot=len(numeric) <= 8, fmt=".2f", cmap="coolwarm",
                    vmin=-1, vmax=1, center=0, linewidths=.5, square=False,
                    cbar_kws={"shrink": .8}, ax=ax)
        ax.set_title("Correlation heatmap")
        ac = corr.abs(); np.fill_diagonal(ac.values, 0.0)
        pair_note = "No strong pairs stand out."
        if ac.values.size and ac.values.max() > 0:
            i, j = divmod(int(ac.values.argmax()), ac.shape[1])
            a, b = ac.index[i], ac.columns[j]
            strongest = (a, b)
            pair_note = f"Strongest link is '{a}' and '{b}' (r = {corr.loc[a, b]:.2f})."
        charts.append({
            "title": "Correlation heatmap",
            "path": _save(fig, outdir, "corr.png"),
            "code": "sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')",
            "insight": pair_note + " Strong links can mean redundant or predictive features.",
        })

    # 6) Scatter of the most correlated pair (coloured by target if useful)
    if strongest:
        a, b = strongest
        hue = target if (target in df.columns and df[target].nunique() <= 10) else None
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x=a, y=b, hue=hue, palette="viridis", s=28,
                        alpha=.7, edgecolor="none", ax=ax)
        ax.set_title(f"{a} vs {b}")
        charts.append({
            "title": f"{a} vs {b}",
            "path": _save(fig, outdir, "scatter.png"),
            "code": f"sns.scatterplot(data=df, x='{a}', y='{b}'"
                    + (f", hue='{target}')" if hue else ")"),
            "insight": f"How '{a}' and '{b}' move together"
                       + (f", split by '{target}'." if hue else "."),
        })

    # 7) Target relationship ----------------------------------------------
    if target and target in df.columns:
        is_class = not (pd.api.types.is_numeric_dtype(df[target])
                        and df[target].nunique() > config.CLASSIFICATION_MAX_UNIQUE)
        if is_class:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            order = df[target].value_counts().index
            sns.countplot(data=df, x=target, order=order, hue=target, legend=False,
                          palette="flare", ax=ax)
            ax.set_title(f"Target balance: {target}")
            charts.append({
                "title": f"Target: {target}",
                "path": _save(fig, outdir, "target.png"),
                "code": f"sns.countplot(data=df, x='{target}')",
                "insight": f"Check '{target}' for class imbalance before modelling.",
            })
            # violin of the top numeric feature by class
            feat = None
            if strongest:
                feat = next((c for c in strongest if c in numeric and c != target), None)
            feat = feat or (numeric[0] if numeric else None)
            if feat and feat != target:
                fig, ax = plt.subplots(figsize=(6, 3.8))
                sns.violinplot(data=df, x=target, y=feat, hue=target, legend=False,
                               palette="flare", inner="box", ax=ax)
                ax.set_title(f"{feat} by {target}")
                charts.append({
                    "title": f"{feat} by {target}",
                    "path": _save(fig, outdir, "violin.png"),
                    "code": f"sns.violinplot(data=df, x='{target}', y='{feat}')",
                    "insight": f"If the '{feat}' shapes differ across classes, it likely "
                               "helps predict the target.",
                })
        else:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            sns.histplot(df[target].dropna(), kde=True, color="#8064a2",
                         edgecolor="white", ax=ax)
            ax.set_title(f"Target distribution: {target}")
            charts.append({
                "title": f"Target: {target}",
                "path": _save(fig, outdir, "target.png"),
                "code": f"sns.histplot(df['{target}'], kde=True)",
                "insight": f"Inspect '{target}' for skew or a long tail before regression.",
            })

    return charts
