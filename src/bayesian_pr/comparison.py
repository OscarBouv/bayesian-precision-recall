"""
Posterior-based comparison and distributional consistency tests for BayesianPRModels.

compare_models
    Estimates P(model_a metric > model_b metric) by drawing from both posteriors.
    Intended for comparing two *different* models evaluated on the *same* data
    distribution. Emits a warning if the model_name fields suggest otherwise.

transfer_test
    Tests whether a model's posterior over a metric is consistent across two
    data distributions. Conceptually analogous to a two-sample test, but framed
    in terms of the Bayesian posteriors: the reference distribution's posterior
    mean is treated as a fixed point and its tail probability under the second
    distribution's posterior is measured.

        S = min(F_2(μ_1), 1 − F_2(μ_1))

    where μ_1 is the posterior mean from distribution 1 and F_2 is the CDF of
    the posterior fitted on distribution 2. Small S indicates the two posteriors
    are inconsistent (μ_1 sits in the tail of the second distribution).

    Intended for the *same* model evaluated on two *different* distributions.
    Emits a warning if the model_name fields suggest otherwise.
"""

from __future__ import annotations

import warnings
import numpy as np
from typing import TYPE_CHECKING

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


def transfer_test(
    model_ref: "BayesianPRModel",
    model_eval: "BayesianPRModel",
    metric: Metric | str = Metric.PRECISION,
    significance: float = 0.05,
) -> dict:
    """
    Test whether a model's posterior is consistent across two data distributions.

    Treats the posterior mean of model_ref (μ_ref) as a fixed reference point
    and computes its tail probability under model_eval's posterior:

        S = min(F_eval(μ_ref), 1 − F_eval(μ_ref))

    where F_eval is the Beta CDF of model_eval's posterior. S is the probability
    mass in the smaller tail — equivalent to a one-sample location test framed
    in terms of the Bayesian posteriors.

      - Large S: μ_ref sits in the bulk of model_eval's posterior → the two
        distributions yield consistent estimates, or model_eval has too few
        observations to conclude otherwise.
      - Small S (≤ significance): μ_ref sits in the tail → the posteriors are
        statistically inconsistent; the model behaves differently on the two
        distributions.

    This function is intended to test the *same* model across two *different*
    data distributions. A warning is raised when:
      - The models have different model_name values (likely a model comparison,
        not a distribution consistency check).
      - Both models share the same sample_dist_name (no distributional contrast).

    Parameters
    ----------
    model_ref  : BayesianPRModel  Reference distribution (typically larger / more trusted).
    model_eval : BayesianPRModel  Distribution under evaluation.
    metric     : Metric | str     One of Metric.PRECISION or Metric.RECALL.
                                  (F1 is not supported as it has no closed-form CDF.)
    significance : float          Tail-probability threshold below which the
                                  posteriors are declared inconsistent. Default 0.05.

    Returns
    -------
    dict with keys:
        mu_ref          : float  Posterior mean of model_ref (reference point).
        mu_eval         : float  Posterior mean of model_eval.
        delta           : float  mu_ref − mu_eval (signed; positive = ref is higher).
        S               : float  Tail probability; in [0, 0.5].
        inconsistent    : bool   True when S ≤ significance.
        model_ref       : str    model_ref.name
        model_eval      : str    model_eval.name
        metric          : str    metric used
    """
    metric = Metric(metric)
    if metric == Metric.F1:
        raise ValueError(
            "transfer_test requires a closed-form posterior CDF; "
            "F1 is not supported. Use Metric.PRECISION or Metric.RECALL."
        )

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

    ref_dist  = getattr(model_ref,  f"{metric.value}_posterior")
    eval_dist = getattr(model_eval, f"{metric.value}_posterior")

    mu_ref  = float(ref_dist.mean())
    mu_eval = float(eval_dist.mean())
    cdf_val = float(eval_dist.cdf(mu_ref))
    S = min(cdf_val, 1.0 - cdf_val)

    return {
        "metric":       metric.value,
        "mu_ref":       mu_ref,
        "mu_eval":      mu_eval,
        "delta":        mu_ref - mu_eval,
        "S":            S,
        "inconsistent": S <= significance,
        "model_ref":    model_ref.name,
        "model_eval":   model_eval.name,
    }
