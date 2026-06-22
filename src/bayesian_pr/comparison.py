"""
Posterior-based comparison and distributional consistency tests for BayesianPRModels.

compare_models
    Estimates P(model_a metric > model_b metric) by drawing from both posteriors.
    Intended for comparing two *different* models evaluated on the *same* data
    distribution. Emits a warning if the model_name fields suggest otherwise.

transfer_test
    Tests whether a model's metric measured on two data distributions (e.g. a
    test set vs. production) is *practically equivalent* or has *meaningfully
    shifted*. Uses a Region of Practical Equivalence (ROPE) decision on the
    posterior of the difference

        delta = p_ref - p_eval

    drawn by Monte Carlo from both Beta posteriors. A symmetric ROPE = [-eps, +eps]
    around zero defines "the same for practical purposes"; the decision compares
    the 95% highest-density interval (HDI) of delta against the ROPE:

      - "equivalent" — HDI lies entirely inside the ROPE (no meaningful shift).
      - "shifted"    — HDI lies entirely outside the ROPE (meaningful shift).
      - "undecided"  — HDI straddles a ROPE boundary (not enough data).

    Symmetric in the two datasets and uses the full uncertainty of both. The
    raw-count form is ``transfer_test_rope``; ``transfer_test`` is the
    BayesianPRModel-based wrapper. Emits a warning if the model_name /
    sample_dist_name fields suggest the inputs are mismatched.
"""

from __future__ import annotations

import warnings
import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .model import BayesianPRModel

from .model import Metric


def compare_models(
    model_a: "BayesianPRModel",
    model_b: "BayesianPRModel",
    metric: Metric | str = Metric.PRECISION,
    n_samples: int = 100_000,
) -> dict:
    """
    Estimate P(model_a metric > model_b metric) via Monte Carlo sampling.

    Draws n_samples from each model's posterior and returns the proportion of
    draws where model_a exceeds model_b, along with the full distribution of
    the difference.

    This function is intended to compare two *different* models evaluated on
    the *same* data distribution. A warning is raised when:
      - Both models share the same model_name (likely comparing the same model).
      - The models have different sample_dist_name values (the observed difference
        may reflect distribution shift rather than a genuine model difference).

    Parameters
    ----------
    model_a, model_b : BayesianPRModel
    metric           : Metric | str  One of Metric.PRECISION, RECALL, F1.
    n_samples        : int

    Returns
    -------
    dict with keys:
        prob_a_better : float  P(metric_a > metric_b)
        mean_diff     : float  E[metric_a − metric_b]
        ci_low        : float  2.5th percentile of the difference distribution
        ci_high       : float  97.5th percentile of the difference distribution
        model_a       : str    model_a.name
        model_b       : str    model_b.name
        metric        : str    metric used
    """
    metric = Metric(metric)

    if model_a.model_name == model_b.model_name:
        warnings.warn(
            f"Both models share model_name='{model_a.model_name}'. "
            "compare_models is intended to compare different models on the same "
            "distribution. Use transfer_test to compare the same model across "
            "different distributions.",
            UserWarning,
            stacklevel=2,
        )

    if (
        model_a.sample_dist_name is not None
        and model_b.sample_dist_name is not None
        and model_a.sample_dist_name != model_b.sample_dist_name
    ):
        warnings.warn(
            f"Models have different sample_dist_name "
            f"('{model_a.sample_dist_name}' vs '{model_b.sample_dist_name}'). "
            "The observed difference may reflect distribution shift rather than "
            "a genuine model difference. Consider using transfer_test instead.",
            UserWarning,
            stacklevel=2,
        )

    if metric not in (Metric.PRECISION, Metric.RECALL, Metric.F1):
        raise ValueError(f"metric must be one of {[m.value for m in Metric]}")

    rng = np.random.default_rng()

    def _samples(model: "BayesianPRModel") -> np.ndarray:
        if metric == Metric.PRECISION:
            return model.precision_posterior.rvs(n_samples, random_state=rng)
        if metric == Metric.RECALL:
            return model.recall_posterior.rvs(n_samples, random_state=rng)
        p = model.precision_posterior.rvs(n_samples, random_state=rng)
        r = model.recall_posterior.rvs(n_samples, random_state=rng)
        denom = p + r
        return np.where(denom > 0, 2 * p * r / denom, 0.0)

    sa, sb = _samples(model_a), _samples(model_b)
    diff = sa - sb

    return {
        "metric":        metric.value,
        "prob_a_better": float((diff > 0).mean()),
        "mean_diff":     float(diff.mean()),
        "ci_low":        float(np.percentile(diff, 2.5)),
        "ci_high":       float(np.percentile(diff, 97.5)),
        "model_a":       model_a.name,
        "model_b":       model_b.name,
    }


@dataclass
class TransferResult:
    """Outcome of a ROPE-based transfer test on the posterior difference delta."""
    rho: float                  # posterior mass inside the ROPE, P(|delta| <= eps)
    hdi_low: float
    hdi_high: float
    status: str                 # "equivalent" | "shifted" | "undecided"
    direction: Optional[str]    # "prod_worse" | "prod_better" | None
    eps: float
    hdi_mass: float


def _hdi(samples: np.ndarray, mass: float) -> tuple[float, float]:
    """Highest-density interval containing `mass` fraction of `samples`.

    Uses the narrowest-window method (NOT equal-tailed quantiles): the posterior
    of a difference of Betas is skewed when either sample is small, and the HDI
    is the correct credible interval in that case.
    """
    s = np.sort(samples)
    n = s.size
    w = int(np.floor(mass * n))
    if w < 1:
        return float(s[0]), float(s[-1])
    widths = s[w:] - s[:n - w]
    i = int(np.argmin(widths))
    return float(s[i]), float(s[i + w])


def _rope_decision(delta: np.ndarray, eps: float, hdi_mass: float):
    """ROPE/HDI decision on a sample of the difference `delta`."""
    rho = float(np.mean((delta >= -eps) & (delta <= eps)))
    lo, hi = _hdi(delta, hdi_mass)
    if lo >= -eps and hi <= eps:
        status, direction = "equivalent", None
    elif lo > eps:
        status, direction = "shifted", "prod_worse"     # ref (test) clearly higher
    elif hi < -eps:
        status, direction = "shifted", "prod_better"     # eval (prod) clearly higher
    else:
        status, direction = "undecided", None
    return rho, lo, hi, status, direction


def transfer_test_rope(
    tp_test: int, fp_test: int,
    tp_prod: int, fp_prod: int,
    eps: float = 0.03,
    lam: float = 1.0,
    hdi_mass: float = 0.95,
    n_samples: int = 100_000,
    seed: Optional[int] = None,
) -> TransferResult:
    """
    ROPE-based test of whether a precision (or recall) metric is practically
    equivalent between a test set and production, from raw counts.

    Posteriors (Beta-Binomial conjugacy with smoothing `lam`):

        p_test ~ Beta(tp_test + lam, fp_test + lam)
        p_prod ~ Beta(tp_prod + lam, fp_prod + lam)
        delta  = p_test - p_prod

    The 95% (``hdi_mass``) HDI of `delta` is compared against ROPE = [-eps, +eps]:
    entirely inside → "equivalent"; entirely outside → "shifted"; straddling a
    boundary → "undecided".

    Pass (TP, FN) in place of (TP, FP) to run the test on recall.
    """
    rng = np.random.default_rng(seed)
    draws_test = rng.beta(tp_test + lam, fp_test + lam, size=n_samples)
    draws_prod = rng.beta(tp_prod + lam, fp_prod + lam, size=n_samples)
    delta = draws_test - draws_prod
    rho, lo, hi, status, direction = _rope_decision(delta, eps, hdi_mass)
    return TransferResult(rho, lo, hi, status, direction, eps, hdi_mass)


def transfer_test_precision(tp_test, fp_test, tp_prod, fp_prod, **kwargs) -> TransferResult:
    """ROPE transfer test on precision (TP / (TP+FP))."""
    return transfer_test_rope(tp_test, fp_test, tp_prod, fp_prod, **kwargs)


def transfer_test_recall(tp_test, fn_test, tp_prod, fn_prod, **kwargs) -> TransferResult:
    """ROPE transfer test on recall (TP / (TP+FN))."""
    return transfer_test_rope(tp_test, fn_test, tp_prod, fn_prod, **kwargs)


def transfer_test(
    model_ref: "BayesianPRModel",
    model_eval: "BayesianPRModel",
    metric: Metric | str = Metric.PRECISION,
    eps: float = 0.03,
    hdi_mass: float = 0.95,
    n_samples: int = 100_000,
    seed: Optional[int] = None,
) -> dict:
    """
    ROPE-based transfer test between two BayesianPRModels of the *same* model on
    two *different* data distributions (e.g. ``model_ref`` = test set,
    ``model_eval`` = production).

    Draws from each model's metric posterior (respecting its prior), forms
    ``delta = metric_ref - metric_eval``, and applies the ROPE/HDI decision.
    Unlike the raw-count :func:`transfer_test_rope`, this samples the posteriors
    directly, so F1 (which has no closed-form posterior) is also supported.

    A warning is raised when:
      - The models have different model_name values (likely a model comparison).
      - Both models share the same sample_dist_name (no distributional contrast).

    Parameters
    ----------
    model_ref, model_eval : BayesianPRModel
    metric    : Metric | str   One of Metric.PRECISION, RECALL, F1.
    eps       : float          ROPE half-width (default 0.03 = 3 points).
    hdi_mass  : float          Credible mass of the HDI (default 0.95).
    n_samples : int            Monte-Carlo draws (default 100_000).
    seed      : int | None     RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        metric, mu_ref, mu_eval, mean_delta, rho, hdi_low, hdi_high,
        eps, hdi_mass, status ("equivalent"|"shifted"|"undecided"),
        direction ("prod_worse"|"prod_better"|None), model_ref, model_eval
    """
    metric = Metric(metric)

    if model_ref.model_name != model_eval.model_name:
        warnings.warn(
            f"Models have different model_name "
            f"('{model_ref.model_name}' vs '{model_eval.model_name}'). "
            "transfer_test is intended to test the same model across different "
            "distributions. Use compare_models to compare different models.",
            UserWarning,
            stacklevel=2,
        )

    if (
        model_ref.sample_dist_name is not None
        and model_eval.sample_dist_name is not None
        and model_ref.sample_dist_name == model_eval.sample_dist_name
    ):
        warnings.warn(
            f"Both models share sample_dist_name='{model_ref.sample_dist_name}'. "
            "transfer_test is intended to compare the same model on two different "
            "distributions.",
            UserWarning,
            stacklevel=2,
        )

    if metric not in (Metric.PRECISION, Metric.RECALL, Metric.F1):
        raise ValueError(f"metric must be one of {[m.value for m in Metric]}")

    rng = np.random.default_rng(seed)

    def _samples(model: "BayesianPRModel") -> np.ndarray:
        if metric == Metric.PRECISION:
            return model.precision_posterior.rvs(n_samples, random_state=rng)
        if metric == Metric.RECALL:
            return model.recall_posterior.rvs(n_samples, random_state=rng)
        p = model.precision_posterior.rvs(n_samples, random_state=rng)
        r = model.recall_posterior.rvs(n_samples, random_state=rng)
        denom = p + r
        return np.where(denom > 0, 2 * p * r / denom, 0.0)

    st, sp = _samples(model_ref), _samples(model_eval)
    delta = st - sp
    rho, lo, hi, status, direction = _rope_decision(delta, eps, hdi_mass)

    return {
        "metric":     metric.value,
        "mu_ref":     float(st.mean()),
        "mu_eval":    float(sp.mean()),
        "mean_delta": float(delta.mean()),
        "rho":        rho,
        "hdi_low":    lo,
        "hdi_high":   hi,
        "eps":        eps,
        "hdi_mass":   hdi_mass,
        "status":     status,
        "direction":  direction,
        "model_ref":  model_ref.name,
        "model_eval": model_eval.name,
    }
