"""Tests for src/eda.py."""
import numpy as np
import pandas as pd
import pytest

from src.eda import (PROPERTIES, plot_correlation, plot_feature_distributions, plot_feature_scales,
                     plot_target, summary)
from src.load_clean import TARGET


@pytest.fixture
def df():
    """Small dataset shaped like the real one: one mean_* and one std_* column per property."""
    rng = np.random.default_rng(0)
    cols = {f"{stat}_{p}": rng.normal(size=100) for p in PROPERTIES for stat in ("mean", "std")}
    out = pd.DataFrame(cols)
    out[TARGET] = rng.uniform(0, 150, size=100)
    return out


def test_summary_known_values():
    # a and b perfectly correlated, c independent -> 1 of 3 pairs has |r| > 0.8
    df = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [10.0, 20, 30, 40], "c": [1.0, -1, -1, 1],
                       TARGET: [1.0, 2, 3, 100]})
    s = summary(df)
    assert s["n_features"] == 3
    assert s["share_pairs_abs_r_gt_0.8"] == pytest.approx(1 / 3)
    assert s["feature_std_ratio"] == pytest.approx(df["b"].std() / df["c"].std())
    assert s["target_median"] == 2.5
    assert s["target_skew"] > 0  # long right tail


def test_summary_excludes_target(df):
    assert summary(df)["n_features"] == df.shape[1] - 1


@pytest.mark.parametrize("plot", [plot_target, plot_feature_scales, plot_feature_distributions,
                                  plot_correlation])
def test_plots_write_pdf(df, tmp_path, plot):
    out = tmp_path / "fig.pdf"
    plot(df, out)
    assert out.read_bytes()[:4] == b"%PDF"
