"""
Utilities for comparing two BayesianPRModels and detecting production transfer drift.

compare_models
    Estimates P(candidate > baseline) by sampling from both posteriors.
    Thresholds: certain_gain if P > 0.80, certain_drop if P < 0.20.

transfer_test
    Checks whether a model's performance degrades from test to production.
    Treats the test estimate as a fixed reference and asks where it falls in
    the (wider) production posterior.
    S = min(F_prod(p̂_test), 1 − F_prod(p̂_test)); flag when S ≤ 0.05.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import BayesianPRModel


def compare_models(
    candidate: "BayesianPRModel",
    baseline: "BayesianPRModel",
    metric: str = "precision",
    n_samples: int = 100_000,
    verdict_gain: str = "certain_gain",
    verdict_drop: str = "certain_drop",
    verdict_stagnation: str = "stagnation",
) -> dict:
    """
    Estimates P(candidate > baseline) via Monte Carlo sampling from both posteriors.
    Verdicts:
      - "certain_gain"  : P > 0.80
      - "certain_drop"  : P < 0.20
      - "stagnation"    : otherwise

    Parameters
    ----------
    candidate : BayesianPRModel   The new model being evaluated.
    baseline  : BayesianPRModel   The model currently in production.
    metric             : 'precision' | 'recall' | 'f1'
    n_samples          : int
    verdict_gain       : str   Label returned when P > 0.80 (default 'certain_gain').
    verdict_drop       : str   Label returned when P < 0.20 (default 'certain_drop').
    verdict_stagnation : str   Label returned otherwise (default 'stagnation').

    Returns
    -------
    dict with keys:
        prob_candidate_better : float   P(candidate metric > baseline metric)
        mean_diff             : float   E[candidate − baseline]
        ci_low, ci_high       : float   95% CI on the difference
        verdict               : str     'certain_gain' | 'certain_drop' | 'stagnation'
    """
    if metric not in ("precision", "recall", "f1"):
        raise ValueError("metric must be 'precision', 'recall', or 'f1'")

    rng = np.random.default_rng()

    def _samples(model: "BayesianPRModel") -> np.ndarray:
        if metric == "precision":
            return model.precision_posterior.rvs(n_samples, random_state=rng)
        if metric == "recall":
            return model.recall_posterior.rvs(n_samples, random_state=rng)
        p = model.precision_posterior.rvs(n_samples, random_state=rng)
        r = model.recall_posterior.rvs(n_samples, random_state=rng)
        denom = p + r
        return np.where(denom > 0, 2 * p * r / denom, 0.0)

    s_cand = _samples(candidate)
    s_base = _samples(baseline)
    diff = s_cand - s_base

    prob = float((diff > 0).mean())
    mean_diff = float(diff.mean())
    ci_low = float(np.percentile(diff, 2.5))
    ci_high = float(np.percentile(diff, 97.5))

    if prob > 0.80:
        verdict = verdict_gain
    elif prob < 0.20:
        verdict = verdict_drop
    else:
        verdict = verdict_stagnation

    return {
        "metric": metric,
        "prob_candidate_better": prob,
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "verdict": verdict,
        "candidate": candidate.name,
        "baseline": baseline.name,
    }


def transfer_test(
    test_model: "BayesianPRModel",
    prod_model: "BayesianPRModel",
    metric: str = "precision",
    significance: float = 0.05,
) -> dict:
    """
    Treats the test-set posterior mean as a fixed reference p̂_test and asks
    where it falls in the (typically wider) production posterior curve:

        S = min(F_prod(p̂_test), 1 − F_prod(p̂_test))

    A small S means p̂_test sits in the tail of the production curve → the drop
    is real.  A large S means p̂_test sits in the bulk → the two agree, or the
    production sample is still too small to conclude.

    Parameters
    ----------
    test_model   : BayesianPRModel  Fitted on the labelled test set.
    prod_model   : BayesianPRModel  Fitted on sampled production predictions.
    metric       : 'precision' | 'recall'
    significance : float  Flag threshold for S (default 0.05).

    Returns
    -------
    dict with keys:
        p_hat_test      : float  Test-set posterior mean (reference point).
        S               : float  Tail probability in the production posterior.
        significant_drop: bool   True when S ≤ significance (real drift).
        verdict         : str    'transfer_ok' | 'domain_shift_detected'
        apparent_drop   : float  p̂_test − prod posterior mean (signed, + = drop).
    """
    if metric not in ("precision", "recall"):
        raise ValueError("metric must be 'precision' or 'recall'")

    test_dist = getattr(test_model, f"{metric}_posterior")
    prod_dist = getattr(prod_model, f"{metric}_posterior")

    p_hat_test = float(test_dist.mean())
    cdf_val = float(prod_dist.cdf(p_hat_test))
    S = min(cdf_val, 1.0 - cdf_val)

    significant = S <= significance
    verdict = "domain_shift_detected" if significant else "transfer_ok"

    return {
        "metric": metric,
        "p_hat_test": p_hat_test,
        "prod_mean": float(prod_dist.mean()),
        "apparent_drop": p_hat_test - float(prod_dist.mean()),
        "S": S,
        "significant_drop": significant,
        "verdict": verdict,
        "test_model": test_model.name,
        "prod_model": prod_model.name,
    }
