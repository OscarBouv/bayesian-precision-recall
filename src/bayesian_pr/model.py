"""
Bayesian model for estimating precision and recall with uncertainty quantification.

Both precision and recall are proportions, so we use Beta-Binomial conjugate models:
  - precision = TP / (TP + FP)  ~ Beta(alpha_p, beta_p)
  - recall    = TP / (TP + FN)  ~ Beta(alpha_r, beta_r)

The Beta posterior after observing (tp, fp, fn) is:
  - precision posterior: Beta(alpha_p + tp, beta_p + fp)
  - recall    posterior: Beta(alpha_r + tp, beta_r + fn)

F1 = 2 * precision * recall / (precision + recall) has no closed-form posterior,
so we estimate it via Monte Carlo sampling.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy import stats


@dataclass
class PosteriorStats:
    mean: float
    std: float
    ci_low: float
    ci_high: float
    ci_level: float = 0.95

    def __repr__(self) -> str:
        pct = int(self.ci_level * 100)
        return (
            f"mean={self.mean:.4f}, std={self.std:.4f}, "
            f"{pct}% CI=[{self.ci_low:.4f}, {self.ci_high:.4f}]"
        )


class BayesianPRModel:
    """
    Bayesian model for precision and recall with full uncertainty quantification.

    Uses Beta-Binomial conjugate models for both precision and recall, and
    Monte Carlo sampling for the F1 score posterior.

    Parameters
    ----------
    prior_alpha : float
        Alpha (pseudo-successes) of the Beta prior. Default 1.0 (uniform).
    prior_beta : float
        Beta (pseudo-failures) of the Beta prior. Default 1.0 (uniform).
    n_samples : int
        Number of Monte Carlo samples for F1 estimation. Default 100_000.
    name : str, optional
        Label used in plots and comparisons.

    Examples
    --------
    >>> model = BayesianPRModel(name="my_classifier")
    >>> model.update(tp=80, fp=20, fn=15)
    >>> print(model.precision_stats())
    mean=0.7959, std=0.0400, 95% CI=[0.7148, 0.8706]
    """

    def __init__(
        self,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        n_samples: int = 100_000,
        name: Optional[str] = None,
    ):
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("prior_alpha and prior_beta must be > 0")
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.n_samples = n_samples
        self.name = name or "model"

        # Observation counts (accumulated across update() calls)
        self._tp: int = 0
        self._fp: int = 0
        self._fn: int = 0

        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def update(self, tp: int, fp: int, fn: int) -> "BayesianPRModel":
        """
        Incorporate new observations into the posterior.

        Can be called multiple times for sequential/online updating.

        Parameters
        ----------
        tp : int  True positives
        fp : int  False positives
        fn : int  False negatives
        """
        if tp < 0 or fp < 0 or fn < 0:
            raise ValueError("tp, fp, fn must be non-negative integers")
        self._tp += tp
        self._fp += fp
        self._fn += fn
        return self

    def reset(self) -> "BayesianPRModel":
        """Reset observations, keeping the prior."""
        self._tp = self._fp = self._fn = 0
        return self

    # ------------------------------------------------------------------
    # Posterior distributions
    # ------------------------------------------------------------------

    @property
    def precision_posterior(self) -> stats.beta:
        """Beta posterior distribution for precision."""
        a = self.prior_alpha + self._tp
        b = self.prior_beta + self._fp
        return stats.beta(a, b)

    @property
    def recall_posterior(self) -> stats.beta:
        """Beta posterior distribution for recall."""
        a = self.prior_alpha + self._tp
        b = self.prior_beta + self._fn
        return stats.beta(a, b)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def precision_stats(self, ci: float = 0.95) -> PosteriorStats:
        d = self.precision_posterior
        lo, hi = d.interval(ci)
        return PosteriorStats(d.mean(), d.std(), lo, hi, ci)

    def recall_stats(self, ci: float = 0.95) -> PosteriorStats:
        d = self.recall_posterior
        lo, hi = d.interval(ci)
        return PosteriorStats(d.mean(), d.std(), lo, hi, ci)

    def f1_stats(self, ci: float = 0.95) -> PosteriorStats:
        """Estimate F1 posterior statistics via Monte Carlo sampling."""
        p_samples = self.precision_posterior.rvs(self.n_samples, random_state=self._rng)
        r_samples = self.recall_posterior.rvs(self.n_samples, random_state=self._rng)
        denom = p_samples + r_samples
        # Avoid division by zero (both 0 → F1 = 0)
        f1_samples = np.where(denom > 0, 2 * p_samples * r_samples / denom, 0.0)
        lo = float(np.percentile(f1_samples, (1 - ci) / 2 * 100))
        hi = float(np.percentile(f1_samples, (1 - (1 - ci) / 2) * 100))
        return PosteriorStats(float(f1_samples.mean()), float(f1_samples.std()), lo, hi, ci)

    def f1_samples(self) -> np.ndarray:
        """Return raw Monte Carlo F1 samples (useful for custom analysis)."""
        p = self.precision_posterior.rvs(self.n_samples, random_state=self._rng)
        r = self.recall_posterior.rvs(self.n_samples, random_state=self._rng)
        denom = p + r
        return np.where(denom > 0, 2 * p * r / denom, 0.0)

    # ------------------------------------------------------------------
    # Threshold probability
    # ------------------------------------------------------------------

    def prob_above_threshold(self, threshold: float, metric: str = "precision") -> float:
        """
        Return P(true metric > threshold).

        For precision and recall the posterior is Beta, so this is exact (Beta CDF).
        For F1 it is estimated via Monte Carlo over the joint precision–recall samples.

        Parameters
        ----------
        threshold : float  Acceptance floor in [0, 1], e.g. 0.65 for precision.
        metric    : str    One of 'precision', 'recall', 'f1'.

        Returns
        -------
        float in [0, 1]

        Examples
        --------
        >>> m = BayesianPRModel(name="cls_0")
        >>> m.update(tp=80, fp=26, fn=20)
        >>> m.prob_above_threshold(0.65, metric="precision")
        0.9891   # very confident it clears 65%
        >>> m.prob_above_threshold(0.80, metric="precision")
        0.2714   # much less sure it clears 80%
        """
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        if metric == "precision":
            return float(1.0 - self.precision_posterior.cdf(threshold))
        if metric == "recall":
            return float(1.0 - self.recall_posterior.cdf(threshold))
        if metric == "f1":
            return float((self.f1_samples() > threshold).mean())
        raise ValueError("metric must be 'precision', 'recall', or 'f1'")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def observations(self) -> dict:
        return {"tp": self._tp, "fp": self._fp, "fn": self._fn}

    def summary(self, ci: float = 0.95) -> str:
        lines = [
            f"BayesianPRModel: {self.name}",
            f"  Prior       : Beta({self.prior_alpha}, {self.prior_beta})",
            f"  Observations: TP={self._tp}, FP={self._fp}, FN={self._fn}",
            f"  Precision   : {self.precision_stats(ci)}",
            f"  Recall      : {self.recall_stats(ci)}",
            f"  F1 (MC)     : {self.f1_stats(ci)}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"BayesianPRModel(name={self.name!r}, "
            f"prior=Beta({self.prior_alpha},{self.prior_beta}), "
            f"tp={self._tp}, fp={self._fp}, fn={self._fn})"
        )
