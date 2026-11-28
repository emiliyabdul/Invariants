"""Standardise features (target excluded), optional PCA, fixed seeds.

Outputs, all in the same row order as clean.csv:
  X_scaled.npy  standardised features  -> clustering option K1
  X_pca.npy     PCA scores             -> clustering option K2 (default)
  y.npy         target, kept apart     -> only used by stats_tests
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.load_clean import TARGET

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
VAR_THRESHOLD = 0.90  # fixed before looking at the data: smallest n explaining >= 90%


def split_features_target(df, target=TARGET):
    X, y = df.drop(columns=target), df[target]
    assert target not in X.columns  # circularity guard: target must never be clustered on
    return X, y


def scale(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def fit_pca(X_scaled, var_threshold=VAR_THRESHOLD, seed=SEED):
    """Smallest number of components whose cumulative explained variance >= threshold."""
    full = PCA(random_state=seed).fit(X_scaled)
    cum = np.cumsum(full.explained_variance_ratio_)
    n = int(np.searchsorted(cum, var_threshold) + 1)
    pca = PCA(n_components=n, random_state=seed).fit(X_scaled)
    return pca.transform(X_scaled), pca, cum


def plot_variance(cum, n, var_threshold, out):
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    k = np.arange(1, len(cum) + 1)
    ax.plot(k, cum, color="#2a6fb0", lw=2, marker="o", ms=3)
    ax.axhline(var_threshold, color="#888888", lw=1, ls="--")
    ax.axvline(n, color="#888888", lw=1, ls="--")
    ax.annotate(f"{n} components = {cum[n - 1]:.1%}", (n, cum[n - 1]),
                textcoords="offset points", xytext=(8, -14), fontsize=8)
    ax.set(xlabel="number of principal components", ylabel="cumulative explained variance",
           title="PCA explained variance (standardised features)", ylim=(0, 1.02), xlim=(0, len(cum) + 1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e5e5e5", lw=0.6)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(clean_path, data_dir, fig_dir):
    df = pd.read_csv(clean_path)
    X, y = split_features_target(df)
    X_scaled, _ = scale(X)
    X_pca, pca, cum = fit_pca(X_scaled)
    n = pca.n_components_

    data_dir, fig_dir = Path(data_dir), Path(fig_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "X_scaled.npy", X_scaled)
    np.save(data_dir / "X_pca.npy", X_pca)
    np.save(data_dir / "y.npy", y.to_numpy())
    plot_variance(cum, n, VAR_THRESHOLD, fig_dir / "pca_variance.pdf")

    info = {
        "n_rows": len(df), "n_features": X.shape[1], "seed": SEED,
        "var_threshold": VAR_THRESHOLD, "n_components": int(n),
        "variance_explained": round(float(cum[n - 1]), 4),
        "components_for_80_90_95": [int(np.searchsorted(cum, t) + 1) for t in (0.8, 0.9, 0.95)],
        "pc1_pc2_variance": round(float(cum[1]), 4),
        # top features per component, to help interpret clusters later
        "top_loadings": {f"PC{i + 1}": X.columns[np.argsort(-np.abs(c))[:5]].tolist()
                         for i, c in enumerate(pca.components_[:3])},
    }
    (fig_dir / "pca_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    args = sys.argv[1:] or [ROOT / "results/data/clean.csv", ROOT / "results/data",
                            ROOT / "results/preprocess"]
    main(*args)
