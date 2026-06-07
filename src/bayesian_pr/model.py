"""
Bayesian model for estimating precision and recall with uncertainty quantification.

Both precision and recall are proportions, so we use Beta-Binomial conjugate models:
  - precision = TP / (TP + FP)  ~ Beta(alpha_p + TP, beta_p + FP)
  - recall    = TP / (TP + FN)  ~ Beta(alpha_r + TP, beta_r + FN)

fp and fn are independently optional. Providing only fp enables precision
estimation; providing only fn enables recall estimation; providing both enables
all three metrics including F1.

F1 = 2 * precision * recall / (precision + recall) has no closed-form posterior,
so we estimate it via Monte Carlo sampling.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from scipy import stats


class Metric(str, Enum):
    """Supported evaluation metrics."""
    PRECISION = "precision"
    RECALL    = "recall"
    F1        = "f1"


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

    Uses Beta-Binomial conjugate models for precision and recall, and Monte Carlo
    sampling for the F1 posterior.

    ``fp`` and ``fn`` are independently optional in ``update()``:

    - Provide ``fp`` only → precision posterior available.
    - Provide ``fn`` only → recall posterior available.
    - Provide both       → all three metrics (precision, recall, F1) available.

    Attempting to access a metric whose counts have not been provided raises
    ``ValueError`` with an informative message.

    Parameters
    ----------
    model_name : str
        Identifier for the model architecture or training run.
    sample_dist_name : str, optional
        Identifier for the data distribution the observations were drawn from
        (e.g. "test_set_v2", "2024-Q1"). Used to emit warnings when
        ``compare_models`` or ``transfer_test`` are called with mismatched
        identifiers.
    prior_alpha : float
        Alpha hyperparameter of the Beta prior (pseudo-successes). Default 1.0
        gives a uniform prior over [0, 1].
    prior_beta : float
        Beta hyperparameter of the Beta prior (pseudo-failures). Default 1.0.
    n_samples : int
        Number of Monte Carlo samples used for F1 estimation. Default 100_000.

    Examples
    --------
    >>> m = BayesianPRModel(model_name="resnet50", sample_dist_name="test_set_v1")
    >>> m.update(tp=80, fp=26)          # precision only
    >>> print(m.precision_stats())
    mean=0.7500, std=0.0415, 95% CI=[0.6646, 0.8267]

    >>> m.update(tp=80, fp=26, fn=20)   # both metrics
    >>> print(m.recall_stats())
    mean=0.7941, std=0.0398, 95% CI=[0.7109, 0.8664]
    """

    def __init__(
        self,
        model_name: str = "model",
        sample_dist_name: Optional[str] = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        n_samples: int = 100_000,
    ):
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("prior_alpha and prior_beta must be > 0")
        self.model_name       = model_name
        self.sample_dist_name = sample_dist_name
        self.prior_alpha      = prior_alpha
        self.prior_beta       = prior_beta
        self.n_samples        = n_samples

        self._tp: int           = 0
        self._fp: Optional[int] = None  # None until first fp observation
        self._fn: Optional[int] = None  # None until first fn observation

        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable label combining model and distribution identifiers."""
        if self.sample_dist_name:
            return f"{self.model_name} [{self.sample_dist_name}]"
        return self.model_name

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def update(
        self,
        tp: int,
        fp: Optional[int] = None,
        fn: Optional[int] = None,
    ) -> "BayesianPRModel":
        """
        Incorporate new observations into the posterior.

        At least one of ``fp`` or ``fn`` must be provided. Can be called
        multiple times — each call accumulates counts, equivalent to a single
        call with their sum.

        Parameters
        ----------
        tp : int            True positives (required).
        fp : int, optional  False positives. Required for precision estimation.
        fn : int, optional  False negatives. Required for recall estimation.
        """
        if fp is None and fn is None:
            raise ValueError("At least one of fp or fn must be provided.")
        if tp < 0:
            raise ValueError("tp must be a non-negative integer.")
        if fp is not None and fp < 0:
            raise ValueError("fp must be a non-negative integer.")
        if fn is not None and fn < 0:
            raise ValueError("fn must be a non-negative integer.")

        self._tp += tp
        if fp is not None:
            self._fp = (self._fp or 0) + fp
        if fn is not None:
            self._fn = (self._fn or 0) + fn
        return self

    def reset(self) -> "BayesianPRModel":
        """Clear all observations, keeping the prior hyperparameters."""
        self._tp = 0
        self._fp = None
        self._fn = None
        return self

    # ------------------------------------------------------------------
    # Availability checks
    # ------------------------------------------------------------------

    @property
    def has_precision(self) -> bool:
        """True if fp observations have been provided."""
        return self._fp is not None

    @property
    def has_recall(self) -> bool:
        """True if fn observations have been provided."""
        return self._fn is not None

    @property
    def has_f1(self) -> bool:
        """True if both fp and fn observations have been provided."""
        return self._fp is not None and self._fn is not None

    def _require_precision(self) -> None:
        if not self.has_precision:
            raise ValueError(
                "Precision is not available: no fp observations have been provided. "
                "Call update(tp=..., fp=...) to enable precision estimation."
            )

    def _require_recall(self) -> None:
        if not self.has_recall:
            raise ValueError(
                "Recall is not available: no fn observations have been provided. "
                "Call update(tp=..., fn=...) to enable recall estimation."
            )

    def _require_f1(self) -> None:
        missing = []
        if not self.has_precision:
            missing.append("fp")
        if not self.has_recall:
            missing.append("fn")
        if missing:
            raise ValueError(
                f"F1 is not available: {' and '.join(missing)} observations have not been "
                "provided. Call update() with both fp and fn to enable F1 estimation."
            )

    # ------------------------------------------------------------------
    # Posterior distributions
    # ------------------------------------------------------------------

    @property
    def precision_posterior(self) -> stats.beta:
        """Beta posterior distribution for precision. Requires fp observations."""
        self._require_precision()
        return stats.beta(self.prior_alpha + self._tp, self.prior_beta + self._fp)

    @property
    def recall_posterior(self) -> stats.beta:
        """Beta posterior distribution for recall. Requires fn observations."""
        self._require_recall()
        return stats.beta(self.prior_alpha + self._tp, self.prior_beta + self._fn)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def precision_stats(self, ci: float = 0.95) -> PosteriorStats:
        """Posterior mean, std, and equal-tailed credible interval for precision."""
        d = self.precision_posterior
        lo, hi = d.interval(ci)
        return PosteriorStats(d.mean(), d.std(), lo, hi, ci)

    def recall_stats(self, ci: float = 0.95) -> PosteriorStats:
        """Posterior mean, std, and equal-tailed credible interval for recall."""
        d = self.recall_posterior
        lo, hi = d.interval(ci)
        return PosteriorStats(d.mean(), d.std(), lo, hi, ci)

    def f1_stats(self, ci: float = 0.95) -> PosteriorStats:
        """
        Estimate posterior statistics for F1 via Monte Carlo sampling.

        Requires both fp and fn observations.
        """
        self._require_f1()
        p_s = self.precision_posterior.rvs(self.n_samples, random_state=self._rng)
        r_s = self.recall_posterior.rvs(self.n_samples, random_state=self._rng)
        denom = p_s + r_s
        f1_s = np.where(denom > 0, 2 * p_s * r_s / denom, 0.0)
        lo = float(np.percentile(f1_s, (1 - ci) / 2 * 100))
        hi = float(np.percentile(f1_s, (1 - (1 - ci) / 2) * 100))
        return PosteriorStats(float(f1_s.mean()), float(f1_s.std()), lo, hi, ci)

    def f1_samples(self) -> np.ndarray:
        """Return raw Monte Carlo F1 samples. Requires both fp and fn observations."""
        self._require_f1()
        p = self.precision_posterior.rvs(self.n_samples, random_state=self._rng)
        r = self.recall_posterior.rvs(self.n_samples, random_state=self._rng)
        denom = p + r
        return np.where(denom > 0, 2 * p * r / denom, 0.0)

    # ------------------------------------------------------------------
    # Threshold exceedance probability
    # ------------------------------------------------------------------

    def prob_above_threshold(
        self, threshold: float, metric: Metric | str = Metric.PRECISION
    ) -> float:
        """
        Return P(true metric > threshold) under the posterior.

        For precision and recall this is exact via the Beta CDF.
        For F1 it is estimated via Monte Carlo over the joint posterior samples.

        Parameters
        ----------
        threshold : float       Value in [0, 1].
        metric    : Metric | str

        Returns
        -------
        float in [0, 1]
        """
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        metric = Metric(metric)
        if metric == Metric.PRECISION:
            return float(1.0 - self.precision_posterior.cdf(threshold))
        if metric == Metric.RECALL:
            return float(1.0 - self.recall_posterior.cdf(threshold))
        return float((self.f1_samples() > threshold).mean())

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
            f"  Observations: TP={self._tp}"
            + (f", FP={self._fp}" if self._fp is not None else "")
            + (f", FN={self._fn}" if self._fn is not None else ""),
        ]
        if self.has_precision:
            lines.append(f"  Precision   : {self.precision_stats(ci)}")
        if self.has_recall:
            lines.append(f"  Recall      : {self.recall_stats(ci)}")
        if self.has_f1:
            lines.append(f"  F1 (MC)     : {self.f1_stats(ci)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"BayesianPRModel(model_name={self.model_name!r}, "
            f"sample_dist_name={self.sample_dist_name!r}, "
            f"prior=Beta({self.prior_alpha},{self.prior_beta}), "
            f"tp={self._tp}, fp={self._fp}, fn={self._fn})"
        )
