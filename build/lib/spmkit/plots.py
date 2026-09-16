from importlib.resources import files
from typing import Literal

import matplotlib.colors as cl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np


def _init():
    arial_path = files("spmkit.assets") / "Arial.ttf"
    arial = fm.FontProperties(fname=arial_path)
    plt.rcParams["font.family"] = arial.get_name()
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["font.size"] = 14
    plt.gca().spines["right"].set_visible(True)
    plt.gca().spines["top"].set_visible(True)
    plt.gca().spines["bottom"].set_visible(True)
    plt.gca().spines["left"].set_visible(True)
    plt.gca().spines["top"].set_linewidth(2)
    plt.gca().spines["left"].set_linewidth(2)
    plt.gca().spines["bottom"].set_linewidth(2)
    plt.gca().spines["right"].set_linewidth(2)
    plt.xticks(fontproperties=arial.get_name())
    plt.yticks(fontproperties=arial.get_name())
    plt.tick_params(left=True, width=2, length=5)


def selection_path_plot(ax, selection_path, alphas, feature_names=None):
    _init()
    selection_path = np.asarray(selection_path)
    if feature_names is None:
        feature_names = [f"feature {l + 1}" for l in range(selection_path.shape[0])]

    for i, feat in enumerate(selection_path):
        ax.plot(alphas, feat, label=feature_names[i])
    ax.set_xscale("log", base=10)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$\alpha$", fontsize=16)
    ax.set_ylabel("Selection probability", fontsize=16)
    ax.legend()
    return ax


def selection_probs_bar_plot(
    ax,
    selection_probs,
    selection_th=None,
    feature_names=None,
    bar_color="limegreen",
    vline_color="red",
    errorbar=False,
):
    _init()
    selection_probs = np.asarray(selection_probs)
    if feature_names is None:
        feature_names = [f"feature {l + 1}" for l in range(selection_probs.shape[0])]

    # ソート
    sort_indice = np.argsort(selection_probs[:, 0])
    selection_probs_sorted = selection_probs[sort_indice]
    feature_names_sorted = np.array(feature_names)[sort_indice]

    ax.barh(
        feature_names_sorted,
        selection_probs_sorted[:, 0],
        xerr=selection_probs_sorted[:, 1] if errorbar else 0,
        capsize=2,
        height=1.0,
        edgecolor="black",
        color=bar_color,
    )

    if selection_th is not None:
        ax.axvline(
            selection_th,
            linestyle="dashed",
            linewidth=1.5,
            label="Threshold",
            color=vline_color,
        )
        ax.legend()

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Mean of selection probability", fontsize=16)
    ax.set_ylabel("Features", fontsize=16)
    return ax


def weight_diagram(
    ax,
    weights,
    feature_names=None,
    pos_color="red",
    neg_color="dodgerblue",
    mid_color="whitesmoke",
    xscale: Literal["linear", "log2"] = "linear",
):
    _init()
    weights = np.asarray(weights)
    n_features, n_models = weights.shape

    if feature_names is None:
        feature_names = [f"feature {l + 1}" for l in range(n_features)]

    # ranks = np.arange(1, n_models + 1)
    x = np.arange(0.5, n_models + 1.5)
    y = np.arange(n_features + 1)

    im = ax.pcolormesh(
        x,
        y,
        weights,
        cmap=cl.LinearSegmentedColormap.from_list(
            "weight_diagram",
            [neg_color, mid_color, pos_color],
        ),
        norm=cl.TwoSlopeNorm(
            vcenter=0.0,
        ),
    )

    ax.set_xlim(1, n_models + 1)
    if xscale == "log2":
        ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(tick.LogLocator(base=2))
    ax.xaxis.set_major_formatter(tick.LogFormatterMathtext(base=2))
    ax.set_yticklabels(feature_names)
    ax.set_yticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xlabel("Rank", fontsize=16)
    ax.set_ylabel("Features", fontsize=16)
    return im


def coefs_waterfall_plot(
    ax,
    weights,
    rank=0,
    feature_names=None,
    pos_color="red",
    neg_color="dodgerblue",
):
    _init()
    weights = np.asarray(weights)
    coefs = weights[:, rank]
    n_features = coefs.shape[0]

    if feature_names is None:
        feature_names = [f"feature {l + 1}" for l in range(n_features)]

    colors = [pos_color if v >= 0 else neg_color for v in coefs]

    # 回帰係数がゼロとなる特徴量を落とす
    filter_indice = np.where(coefs != 0)[0]
    coefs_filtered = coefs[filter_indice]
    feature_names_filtered = np.array(feature_names)[filter_indice]

    # ソート
    sort_indice = np.argsort(np.abs(coefs_filtered))
    coefs_sorted = coefs_filtered[sort_indice]
    feature_names_sorted = feature_names_filtered[sort_indice]

    ax.barh(
        feature_names_sorted,
        coefs_sorted,
        color=colors,
        height=1.0,
        edgecolor="black",
    )
    ax.axvline(x=0, color="black", linewidth=2)
    ax.set_xlabel("Standardized coefficient", fontsize=16)
    ax.set_ylabel("Features", fontsize=16)
    return ax
