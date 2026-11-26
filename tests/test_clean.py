"""Tests for src/load_clean.py."""
import numpy as np
import pandas as pd
import pytest

from src.load_clean import (TARGET, clean, drop_constant_columns, drop_duplicates, flag_extremes, load_raw,
                            merge_repeated_materials, sensitivity)


@pytest.fixture
def raw():
    return pd.DataFrame({
        "a": [1.0, 2.0, 2.0, 3.0, np.nan],
        "b": [5.0, 6.0, 6.0, 7.0, 8.0],
        "const": [1.0] * 5,
        TARGET: [10.0, 20.0, 20.0, 30.0, 40.0],
    })


def test_duplicates_removed_columns_kept(raw):
    out, entry = drop_duplicates(raw)
    assert not out.duplicated().any()
    assert list(out.columns) == list(raw.columns)
    assert entry["rows_removed"] == 1


def test_constant_column_dropped_and_logged(raw):
    out, entry = drop_constant_columns(raw)
    assert "const" not in out.columns
    assert TARGET in out.columns
    assert "const" in entry["detail"]


def test_no_silent_row_loss(raw):
    out, log = clean(raw)
    assert len(raw) - len(out) == log["rows_removed"].sum()
    assert len(out) == 3  # one NaN row, one duplicate


def test_repeated_materials_merged_with_median():
    df = pd.DataFrame({"a": [1.0, 1.0, 1.0, 2.0], TARGET: [10.0, 20.0, 60.0, 5.0]})
    out, entry = merge_repeated_materials(df)
    assert len(out) == 2
    assert out.loc[out["a"] == 1.0, TARGET].item() == 20.0
    assert list(out.columns) == list(df.columns)
    assert entry["rows_removed"] == 2


def test_extremes_flagged_not_removed():
    df = pd.DataFrame({"x": [0.0] * 50 + [1000.0], TARGET: range(51)})
    out, entry = flag_extremes(df)
    assert len(out) == len(df)
    assert entry["detail"].startswith("1 values")


def test_load_raw_requires_target(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_raw(p)


def test_sensitivity_compares_median_and_mean():
    df = pd.DataFrame({"a": [1.0, 1.0, 1.0, 2.0], TARGET: [10.0, 20.0, 60.0, 5.0]})
    sens = sensitivity(df).set_index("check")
    assert sens.loc["merge with median", "tc_max"] == 20.0
    assert sens.loc["merge with mean", "tc_max"] == 30.0
    assert sens.loc["repeated materials where |mean - median| > 5 K", "n_rows"] == 1
    assert len(df) == 4  # input is not modified
