"""
Plotting utilities for BayesianPRModel.

All functions return a matplotlib Figure so callers can save or display as needed.
Requires matplotlib and optionally seaborn for nicer styling.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .model import BayesianPRModel

_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def _apply_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_posteriors(
    model: "BayesianPRModel",
    ci: float = 0.95,
    n_points: int = 500,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """
    Plot the posterior PDFs for precision, recall, and F1 (via KDE on samples).

    Shaded bands show the credible interval; vertical dashed lines mark the posterior mean.

    Parameters
    ----------
    model   : BayesianPRModel
    ci      : credible interval level (default 0.95)
    n_points: resolution for the Beta PDF curves
    figsize : figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    x = np.linspace(0, 1, n_points)

    for ax, (dist, label, color) in zip(
        axes,
        [
            (model.precision_posterior, "Precision", _COLORS[0]),
            (model.recall_posterior, "Recall", _COLORS[1]),
            (None, "F1", _COLORS[2]),
        ],
    ):
        if dist is not None:
            # Closed-form Beta PDF
            y = dist.pdf(x)
            stats = getattr(model, f"{label.lower()}_stats")(ci)
            ax.plot(x, y, color=color, lw=2)
            mask = (x >= stats.ci_low) & (x <= stats.ci_high)
            ax.fill_between(x, y, where=mask, color=color, alpha=0.25,
                            label=f"{int(ci*100)}% CI")
            ax.axvline(stats.mean, color=color, lw=1.5, ls="--", label=f"mean={stats.mean:.3f}")
        else:
            # F1: KDE on Monte Carlo samples
            f1_s = model.f1_samples()
            stats = model.f1_stats(ci)
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(f1_s)
            y = kde(x)
            ax.plot(x, y, color=color, lw=2)
            mask = (x >= stats.ci_low) & (x <= stats.ci_high)
            ax.fill_between(x, y, where=mask, color=color, alpha=0.25,
                            label=f"{int(ci*100)}% CI")
            ax.axvline(stats.mean, color=color, lw=1.5, ls="--", label=f"mean={stats.mean:.3f}")

        _apply_style(ax, label, "Value", "Density")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)

    fig.suptitle(f"Posterior distributions — {model.name}", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_sequential_update(
    prior_alpha: float,
    prior_beta: float,
    observations: list[dict],
    metric: str = "precision",
    ci: float = 0.95,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """
    Show how the posterior evolves as batches of observations arrive sequentially.

    Parameters
    ----------
    prior_alpha, prior_beta : Beta prior hyperparameters
    observations : list of dicts with keys 'tp', 'fp', 'fn'
        Each dict is one batch of new observations.
    metric : 'precision' | 'recall'
    ci     : credible interval level
    figsize: figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    from .model import BayesianPRModel

    if metric not in ("precision", "recall"):
        raise ValueError("metric must be 'precision' or 'recall'")

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    x = np.linspace(0, 1, 500)

    model = BayesianPRModel(prior_alpha=prior_alpha, prior_beta=prior_beta)
    means, ci_lows, ci_highs = [], [], []
    cumulative = 0

    for i, obs in enumerate([{}] + observations):  # index 0 = prior only
        if obs:
            model.update(**obs)
        dist = getattr(model, f"{metric}_posterior")
        y = dist.pdf(x)
        color = plt.cm.viridis(i / max(len(observations), 1))  # type: ignore[attr-defined]
        label = "prior" if i == 0 else f"after batch {i}"
        axes[0].plot(x, y, color=color, lw=1.8, label=label, alpha=0.85)

        s = getattr(model, f"{metric}_stats")(ci)
        means.append(s.mean)
        ci_lows.append(s.ci_low)
        ci_highs.append(s.ci_high)

    axes[0].set_xlim(0, 1)
    _apply_style(axes[0], f"{metric.capitalize()} posterior evolution", "Value", "Density")
    axes[0].legend(fontsize=8, loc="upper left")

    steps = list(range(len(observations) + 1))
    axes[1].plot(steps, means, "o-", color=_COLORS[0], lw=2, label="posterior mean")
    axes[1].fill_between(steps, ci_lows, ci_highs, alpha=0.2, color=_COLORS[0],
                         label=f"{int(ci*100)}% CI")
    axes[1].set_xticks(steps)
    axes[1].set_xticklabels(["prior"] + [f"batch {i+1}" for i in range(len(observations))],
                             rotation=30, ha="right")
    _apply_style(axes[1], f"{metric.capitalize()} mean + CI over time", "Update step", "Value")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    return fig


def plot_comparison(
    models: list["BayesianPRModel"],
    metric: str = "f1",
    ci: float = 0.95,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """
    Overlay posterior distributions for multiple models on the same axis,
    with a second panel showing means and credible intervals as a forest plot.

    Parameters
    ----------
    models  : list of BayesianPRModel instances
    metric  : 'precision' | 'recall' | 'f1'
    ci      : credible interval level
    figsize : figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    if metric not in ("precision", "recall", "f1"):
        raise ValueError("metric must be 'precision', 'recall', or 'f1'")

    fig, (ax_pdf, ax_forest) = plt.subplots(1, 2, figsize=figsize)
    x = np.linspace(0, 1, 500)

    for i, (model, color) in enumerate(zip(models, _COLORS)):
        if metric == "f1":
            from scipy.stats import gaussian_kde
            samples = model.f1_samples()
            kde = gaussian_kde(samples)
            y = kde(x)
            stats = model.f1_stats(ci)
        else:
            dist = getattr(model, f"{metric}_posterior")
            y = dist.pdf(x)
            stats = getattr(model, f"{metric}_stats")(ci)

        ax_pdf.plot(x, y, color=color, lw=2, label=model.name)
        ax_pdf.fill_between(x, y,
                            where=(x >= stats.ci_low) & (x <= stats.ci_high),
                            color=color, alpha=0.15)
        ax_pdf.axvline(stats.mean, color=color, lw=1, ls="--")

        # Forest plot (right panel)
        y_pos = len(models) - 1 - i
        ax_forest.plot(stats.mean, y_pos, "o", color=color, ms=8, zorder=3)
        ax_forest.hlines(y_pos, stats.ci_low, stats.ci_high, color=color, lw=3, alpha=0.7)

    ax_forest.set_yticks(range(len(models)))
    ax_forest.set_yticklabels([m.name for m in reversed(models)])
    ax_forest.set_xlim(0, 1)
    _apply_style(ax_forest, f"{metric.capitalize()} comparison (forest plot)",
                 "Value", "")
    ax_forest.axvline(0.5, color="gray", lw=0.8, ls=":")

    ax_pdf.set_xlim(0, 1)
    _apply_style(ax_pdf, f"{metric.capitalize()} posterior overlay", "Value", "Density")
    ax_pdf.legend(fontsize=9)

    fig.tight_layout()
    return fig


def plot_prior_sensitivity(
    tp: int,
    fp: int,
    fn: int,
    prior_strengths: list[tuple[float, float]] | None = None,
    metric: str = "precision",
    figsize: tuple = (9, 5),
) -> plt.Figure:
    """
    Show how much the prior concentration affects the posterior for fixed observations.

    Parameters
    ----------
    tp, fp, fn        : observed counts
    prior_strengths   : list of (alpha, beta) tuples to compare;
                        defaults to [(1,1), (2,2), (5,5), (10,10), (0.5,2)]
    metric            : 'precision' | 'recall'
    figsize           : figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    from .model import BayesianPRModel

    if prior_strengths is None:
        prior_strengths = [(1, 1), (2, 2), (5, 5), (10, 10), (0.5, 2.0)]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.linspace(0, 1, 500)

    for (a, b), color in zip(prior_strengths, _COLORS):
        model = BayesianPRModel(prior_alpha=a, prior_beta=b)
        model.update(tp=tp, fp=fp, fn=fn)
        dist = getattr(model, f"{metric}_posterior")
        ax.plot(x, dist.pdf(x), color=color, lw=2, label=f"Beta({a},{b}) prior")

    _apply_style(ax, f"Prior sensitivity — {metric} (TP={tp}, FP={fp}, FN={fn})",
                 "Value", "Density")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig
