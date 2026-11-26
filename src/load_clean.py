"""Read raw CSV, handle missing/duplicate/constant/extreme values, write clean.csv + log.

Every step is a pure function: DataFrame in -> (DataFrame, log entry) out, so nothing is
removed without being recorded. The log feeds the Dataset section of the paper.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "critical_temp"
Z_THRESHOLD = 4.0

ROOT = Path(__file__).resolve().parents[1]


def _entry(step, rows_removed=0, cols_removed=0, detail=""):
    return {"step": step, "rows_removed": rows_removed, "cols_removed": cols_removed, "detail": detail}


def load_raw(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"target column '{TARGET}' missing from {path}")
    non_numeric = df.columns[~df.dtypes.map(pd.api.types.is_numeric_dtype)].tolist()
    if non_numeric:
        raise ValueError(f"non-numeric columns: {non_numeric}")
    return df


def drop_missing(df):
    """Drop rows with any NaN; the per-column counts go into the log."""
    per_col = df.isna().sum()
    out = df.dropna()
    detail = "; ".join(f"{c}={n}" for c, n in per_col[per_col > 0].items()) or "no missing values"
    return out, _entry("drop_missing", len(df) - len(out), 0, detail)


def drop_duplicates(df):
    out = df.drop_duplicates()
    return out, _entry("drop_duplicates", len(df) - len(out), 0, "exact duplicate rows")


def drop_constant_columns(df):
    const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1 and c != TARGET]
    out = df.drop(columns=const)
    return out, _entry("drop_constant_columns", 0, len(const), ", ".join(const) or "none")


def flag_extremes(df, z=Z_THRESHOLD):
    """Count |z| > threshold per feature. Flags only: these are real measurements, kept."""
    feats = df.drop(columns=TARGET)
    std = feats.std(ddof=0).replace(0, np.nan)  # constant columns would divide by zero
    counts = ((feats - feats.mean()).abs() / std > z).sum()
    flagged = counts[counts > 0].sort_values(ascending=False)
    detail = f"{int(flagged.sum())} values in {len(flagged)} columns kept; top: " + \
        ", ".join(f"{c}={n}" for c, n in flagged.head(5).items())
    return df, _entry(f"flag_extremes(|z|>{z})", 0, 0, detail)


def clean(df):
    """Run every step in order. Returns (clean DataFrame, log DataFrame)."""
    log = [_entry("load_raw", 0, 0, f"{df.shape[0]} rows x {df.shape[1]} columns")]
    for step in (drop_missing, drop_duplicates, drop_constant_columns, flag_extremes):
        df, entry = step(df)
        log.append(entry)
    log.append(_entry("result", 0, 0, f"{df.shape[0]} rows x {df.shape[1]} columns"))
    return df.reset_index(drop=True), pd.DataFrame(log)


def main(raw_path, clean_path, log_path):
    df, log = clean(load_raw(raw_path))
    for p in (clean_path, log_path):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(clean_path, index=False)
    log.to_csv(log_path, index=False)
    print(log.to_string(index=False))


if __name__ == "__main__":
    args = sys.argv[1:] or [ROOT / "data/raw/train.csv",
                            ROOT / "results/data/clean.csv",
                            ROOT / "results/logs/cleaning_log.csv"]
    main(*args)
