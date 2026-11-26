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


def merge_repeated_materials(df, agg="median"):
    """Rows with identical features are the same material measured several times:
    keep one row per material with the median target, so each material counts once."""
    feats = [c for c in df.columns if c != TARGET]
    out = df.groupby(feats, sort=False, as_index=False)[TARGET].agg(agg)[df.columns]
    repeated = df.duplicated(subset=feats, keep=False).sum()
    detail = f"{repeated} rows shared features with another row; merged into {repeated - (len(df) - len(out))} materials ({agg} {TARGET})"
    return out, _entry("merge_repeated_materials", len(df) - len(out), 0, detail)


def drop_constant_columns(df):
    const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1 and c != TARGET]
    out = df.drop(columns=const)
    return out, _entry("drop_constant_columns", 0, len(const), ", ".join(const) or "none")


def extreme_counts(df, z=Z_THRESHOLD):
    """Number of |z| > threshold values per feature (only columns with at least one)."""
    feats = df.drop(columns=TARGET)
    std = feats.std(ddof=0).replace(0, np.nan)  # constant columns would divide by zero
    counts = ((feats - feats.mean()).abs() / std > z).sum()
    return counts[counts > 0].sort_values(ascending=False)


def flag_extremes(df, z=Z_THRESHOLD):
    """Count |z| > threshold per feature. Flags only: these are real measurements, kept."""
    flagged = extreme_counts(df, z)
    detail = f"{int(flagged.sum())} values in {len(flagged)} columns kept; top: " + \
        ", ".join(f"{c}={n}" for c, n in flagged.head(5).items())
    return df, _entry(f"flag_extremes(|z|>{z})", 0, 0, detail)


def clean(df):
    """Run every step in order. Returns (clean DataFrame, log DataFrame)."""
    log = [_entry("load_raw", 0, 0, f"{df.shape[0]} rows x {df.shape[1]} columns")]
    for step in (drop_missing, drop_duplicates, merge_repeated_materials, drop_constant_columns, flag_extremes):
        df, entry = step(df)
        log.append(entry)
    log.append(_entry("result", 0, 0, f"{df.shape[0]} rows x {df.shape[1]} columns"))
    return df.reset_index(drop=True), pd.DataFrame(log)


def sensitivity(raw):
    """Do the judgment calls change the data? Compares median vs mean for merging repeated
    materials, and how many values each |z| threshold would flag (flag only, nothing removed)."""
    base = drop_duplicates(drop_missing(raw)[0])[0]
    feats = [c for c in base.columns if c != TARGET]
    rows = []
    for agg in ("median", "mean"):
        y = merge_repeated_materials(base, agg)[0][TARGET]
        rows.append({"check": f"merge with {agg}", "n_rows": len(y), "tc_mean": y.mean(),
                     "tc_median": y.median(), "tc_max": y.max(), "tc_skew": y.skew(),
                     "share_below_10K": (y < 10).mean(), "share_70_100K": y.between(70, 100).mean()})
    spread = base[base.duplicated(subset=feats, keep=False)].groupby(feats)[TARGET].agg(["mean", "median"])
    rows.append({"check": "repeated materials where |mean - median| > 5 K",
                 "n_rows": int(((spread["mean"] - spread["median"]).abs() > 5).sum()),
                 "note": f"out of {len(spread)} repeated materials"})
    merged = merge_repeated_materials(base)[0]
    for z in (3.0, 4.0, 5.0):
        rows.append({"check": f"values flagged at |z|>{z:g}", "n_rows": int(extreme_counts(merged, z).sum()),
                     "note": "flag only, nothing removed"})
    return pd.DataFrame(rows)


def main(raw_path, clean_path, log_path):
    raw = load_raw(raw_path)
    df, log = clean(raw)
    for p in (clean_path, log_path):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(clean_path, index=False)
    log.to_csv(log_path, index=False)
    sens = sensitivity(raw)
    sens.to_csv(Path(log_path).with_name("sensitivity.csv"), index=False)
    print(log.to_string(index=False))
    print(sens.round(3).to_string(index=False))


if __name__ == "__main__":
    args = sys.argv[1:] or [ROOT / "data/raw/train.csv",
                            ROOT / "results/data/clean.csv",
                            ROOT / "results/logs/cleaning_log.csv"]
    main(*args)
