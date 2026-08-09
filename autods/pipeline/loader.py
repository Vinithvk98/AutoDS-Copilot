"""Stage 1 — Load a dataset and profile it."""
from __future__ import annotations
import json
import pandas as pd
from pathlib import Path


def load_data(path: str | Path) -> pd.DataFrame:
    """Load CSV / Excel / TSV / JSON into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".tsv"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def column_types(df: pd.DataFrame) -> dict:
    """Split columns into numeric / categorical / datetime buckets."""
    numeric, categorical, datetime = [], [], []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            numeric.append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            datetime.append(col)
        else:
            categorical.append(col)
    return {"numeric": numeric, "categorical": categorical, "datetime": datetime}


def profile(df: pd.DataFrame) -> dict:
    """Return a compact profile of the dataset used across the app."""
    types = column_types(df)
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "types": types,
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(v) for c, v in missing.items()},
        "missing_pct": {c: round(float(100 * v / len(df)), 1) for c, v in missing.items()},
        "n_duplicates": int(df.duplicated().sum()),
        "memory_kb": round(float(df.memory_usage(deep=True).sum() / 1024), 1),
        # JSON round-trip converts numpy scalars -> native Python and NaN -> None,
        # keeping the profile safe to serialize into graph state / a web response.
        "preview": json.loads(df.head(5).to_json(orient="records")),
    }
