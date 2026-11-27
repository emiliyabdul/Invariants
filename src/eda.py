"""Target/feature distributions, feature scales, correlation matrix.

Each figure supports one method decision (written in PAPER.md, section 3.2):
target shape -> parametric vs rank-based tests, feature scales -> standardisation,
correlations -> PCA.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.load_clean import TARGET

ROOT = Path(__file__).resolve().parents[1]
PROPERTIES = ["atomic_mass", "fie", "atomic_radius", "Density",
              "ElectronAffinity", "FusionHeat", "ThermalConductivity", "Valence"]
COLOR = "#2a6fb0"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "font.size": 9})


def features(df):
    return df.drop(columns=TARGET)


def plot_target(df, out):
    """Histogram (linear and log scale) and QQ plot of the target."""
    y = df[TARGET]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].hist(y, bins=60, color=COLOR)
    ax[0].set(xlabel="critical temperature (K)", ylabel="materials", title="Target distribution")
    ax[1].hist(np.log1p(y), bins=60, color=COLOR)
    ax[1].set(xlabel="log(1 + critical temperature)", title="Log scale")
    (osm, osr), (slope, intercept, _) = stats.probplot(y)
    ax[2].scatter(osm, osr, s=3, color=COLOR, rasterized=True)
    ax[2].plot(osm, slope * osm + intercept, color="#555555", lw=1.2)
    ax[2].set(xlabel="theoretical normal quantiles", ylabel="observed (K)", title="QQ plot vs normal")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_feature_scales(df, out):
    """Standard deviation of every feature on a log axis: shows the need for scaling."""
    sd = features(df).std().sort_values()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.scatter(range(len(sd)), sd.values, s=12, color=COLOR)
    ax.set_yscale("log")
    ax.set(xlabel=f"features sorted by standard deviation (n = {len(sd)})",
           ylabel="standard deviation (log scale)",
           title=f"Feature scales span {sd.max() / sd.min():,.0f}x "
                 f"(smallest: {sd.index[0]}, largest: {sd.index[-1]})")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_feature_distributions(df, out):
    """One histogram per property family (its plain mean statistic)."""
    fig, axes = plt.subplots(2, 4, figsize=(11, 4.8))
    for ax, prop in zip(axes.flat, PROPERTIES):
        ax.hist(df[f"mean_{prop}"], bins=50, color=COLOR)
        ax.set_title(f"mean_{prop}", fontsize=9)
    fig.supylabel("materials")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_correlation(df, out):
    """Feature correlation heatmap, columns in dataset order (grouped by property)."""
    corr = features(df).corr()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    # one tick per property family, at the centre of its block of columns
    cols = corr.columns
    ticks = [np.mean([i for i, c in enumerate(cols) if c.endswith(p)]) for p in PROPERTIES]
    ax.set_xticks(ticks, PROPERTIES, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(ticks, PROPERTIES, fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def summary(df):
    """Numbers quoted in the paper: target shape, scale spread, redundancy."""
    X, y = features(df), df[TARGET]
    corr = X.corr().to_numpy()
    upper = np.abs(corr[np.triu_indices_from(corr, k=1)])
    sd = X.std()
    return pd.Series({
        "n_rows": len(df), "n_features": X.shape[1],
        "target_mean": y.mean(), "target_median": y.median(), "target_skew": stats.skew(y),
        "target_min": y.min(), "target_max": y.max(),
        "feature_std_min": sd.min(), "feature_std_max": sd.max(),
        "feature_std_ratio": sd.max() / sd.min(),
        "share_pairs_abs_r_gt_0.8": (upper > 0.8).mean(),
        "share_pairs_abs_r_gt_0.9": (upper > 0.9).mean(),
    })


def main(clean_path, out_dir):
    df = pd.read_csv(clean_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_target(df, out / "target_distribution.pdf")
    plot_feature_scales(df, out / "feature_scales.pdf")
    plot_feature_distributions(df, out / "feature_distributions.pdf")
    plot_correlation(df, out / "correlation_matrix.pdf")
    s = summary(df)
    s.to_csv(out / "eda_summary.csv", header=["value"])
    features(df).describe().T.to_csv(out / "feature_describe.csv")
    print(s.to_string())


if __name__ == "__main__":
    args = sys.argv[1:] or [ROOT / "results/data/clean.csv", ROOT / "results/eda"]
    main(*args)
