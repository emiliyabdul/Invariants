"""Tests for src/preprocess.py."""
import numpy as np
import pandas as pd
import pytest

from src.load_clean import TARGET
from src.preprocess import fit_pca, scale, split_features_target


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(200, 1))
    X = np.hstack([base * 1000, base * 0.01 + rng.normal(scale=0.001, size=(200, 1)),
                   rng.normal(size=(200, 3)) * [1, 50, 0.2]])
    out = pd.DataFrame(X, columns=list("abcde"))
    out[TARGET] = rng.uniform(0, 150, size=200)
    return out


def test_target_absent_from_clustering_input(df):
    X, y = split_features_target(df)
    assert TARGET not in X.columns
    assert y.name == TARGET and len(y) == len(X)


def test_scaled_mean_zero_std_one(df):
    X_scaled, _ = scale(split_features_target(df)[0])
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-8)


def test_pca_meets_variance_threshold(df):
    X_scaled, _ = scale(split_features_target(df)[0])
    X_pca, pca, cum = fit_pca(X_scaled, var_threshold=0.9)
    n = pca.n_components_
    assert X_pca.shape == (len(df), n)
    assert cum[n - 1] >= 0.9
    assert n == 1 or cum[n - 2] < 0.9  # smallest n that reaches the threshold


def test_pca_is_reproducible(df):
    X_scaled, _ = scale(split_features_target(df)[0])
    assert np.array_equal(fit_pca(X_scaled)[0], fit_pca(X_scaled)[0])
