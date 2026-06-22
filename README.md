# Bayesian Precision–Recall

[![PyPI](https://img.shields.io/pypi/v/bayesian-precision-recall)](https://pypi.org/project/bayesian-precision-recall/)

Precision and recall are point estimates on a finite test set — they carry uncertainty.
This library models that uncertainty using **Beta-Binomial conjugate models**, giving you full
posterior distributions for precision, recall, and F1 rather than single numbers, based on
[Goutte & Gaussier (2005)](https://link.springer.com/chapter/10.1007/978-3-540-31865-1_25).

```
precision = TP / (TP + FP)  →  Beta(α + TP,  β + FP)   [exact, closed-form]
recall    = TP / (TP + FN)  →  Beta(α + TP,  β + FN)   [exact, closed-form]
F1                          →  Monte Carlo over the joint posterior
```

---

## Why this matters

| Point-estimate thinking | Bayesian posterior |
|---|---|
| Precision = 0.75 | Precision ∈ [0.66, 0.83] with 95% confidence |
| "Is 75% above our 70% floor?" | "P(true precision > 70%) = 88%" |
| "Model A is better than B" | "P(A precision > B precision) = 86%" |
| Results brittle on small test sets | Uncertainty widens automatically |
| No principled way to use prior knowledge | Beta prior encodes domain expertise |

The same point estimate can reflect very different situations:

- **class_A**: TP=80, FP=26 → precision ≈ 75.5%, 95% CI = [66%, 83%] — a reliable estimate.
- **class_B**: TP=5, FP=6 → precision ≈ 45.5%, 95% CI = [21%, 72%] — too uncertain to act on.

Comparing bare point estimates across different sample sizes is misleading.
The Bayesian model makes that difference explicit and quantifiable.

---

## Installation

```bash
pip install bayesian-precision-recall
```

Or from source:

```bash
git clone https://github.com/your-org/bayesian-precision-recall.git
cd bayesian-precision-recall
pip install -e .
```

**Requirements:** Python ≥ 3.9, numpy, scipy, matplotlib

---

## Quick start

```python
from bayesian_pr import BayesianPRModel

model = BayesianPRModel(model_name="classifier_v2", sample_dist_name="test_set")
model.update(tp=80, fp=26, fn=20)
print(model.summary())
```

```
BayesianPRModel: classifier_v2 [test_set]
  Prior       : Beta(1.0, 1.0)
  Observations: TP=80, FP=26, FN=20
  Precision   : mean=0.7500, std=0.0415, 95% CI=[0.6646, 0.8267]
  Recall      : mean=0.7941, std=0.0398, 95% CI=[0.7109, 0.8664]
  F1 (MC)     : mean=0.7703, std=0.0291, 95% CI=[0.7106, 0.8248]
```

`fp` and `fn` are independently optional — only the metrics whose counts have been provided are available:

```python
# Precision only
model = BayesianPRModel(model_name="cls")
model.update(tp=80, fp=26)
model.precision_stats()   # ✓
model.recall_stats()      # raises ValueError: no fn observations provided

# Recall only
model = BayesianPRModel(model_name="cls")
model.update(tp=80, fn=20)
model.recall_stats()      # ✓
model.precision_stats()   # raises ValueError: no fp observations provided
```

---

## Core API

### `BayesianPRModel`

```python
BayesianPRModel(
    model_name:       str   = "model",
    sample_dist_name: str   = None,      # optional — used for warnings in compare/transfer
    prior_alpha:      float = 1.0,       # Beta prior α  (uniform by default)
    prior_beta:       float = 1.0,       # Beta prior β
    n_samples:        int   = 100_000,
)
```

| Method / Property | Returns | Requires | Description |
|---|---|---|---|
| `.update(tp, fp=None, fn=None)` | `self` | `fp` or `fn` | Accumulate observations (chainable) |
| `.reset()` | `self` | — | Clear observations, keep prior |
| `.has_precision` | `bool` | — | True if fp has been provided |
| `.has_recall` | `bool` | — | True if fn has been provided |
| `.has_f1` | `bool` | — | True if both fp and fn have been provided |
| `.precision_posterior` | `scipy.stats.beta` | fp | Beta posterior for precision |
| `.recall_posterior` | `scipy.stats.beta` | fn | Beta posterior for recall |
| `.precision_stats(ci=0.95)` | `PosteriorStats` | fp | Mean, std, credible interval |
| `.recall_stats(ci=0.95)` | `PosteriorStats` | fn | Same for recall |
| `.f1_stats(ci=0.95)` | `PosteriorStats` | fp + fn | Monte Carlo F1 posterior |
| `.f1_samples()` | `np.ndarray` | fp + fn | Raw MC samples for custom analysis |
| `.prob_above_threshold(t, metric)` | `float` | fp / fn / both | P(true metric > t) |
| `.summary(ci=0.95)` | `str` | — | Prints only available metrics |

### `prob_above_threshold`

```python
# Does this classifier confidently clear the precision floor?
model.prob_above_threshold(threshold=0.7, metric="precision")
# → 0.883   # large sample, point estimate well above floor

# Same question, only 11 predictions behind it:
sparse_model.prob_above_threshold(threshold=0.7, metric="precision")
# → 0.039   # too uncertain — acquire more labels
```

The same 75% point estimate on different sample sizes:

| n predictions | P(precision > 70%) |
|---|---|
| 10  | 69% |
| 50  | 80% |
| 100 | 85% |
| 200 | 94% |
| 500 | 99% |

Confidence comes from volume, not from the rate itself.

### `compare_models`

Intended for comparing two **different models** on the **same data distribution**. Emits a warning if `model_name` fields match or `sample_dist_name` fields differ.

```python
from bayesian_pr import compare_models, Metric

result = compare_models(model_a, model_b, metric=Metric.PRECISION)
# {
#   "prob_a_better": 0.859,
#   "mean_diff":     +0.042,
#   "ci_low":        0.008,
#   "ci_high":       0.076,
#   "model_a":       "classifier_v2 [test_set]",
#   "model_b":       "classifier_v1 [test_set]",
#   "metric":        "precision",
# }
```

### `transfer_test`

Tests whether the **same model**'s metric is *practically equivalent* across **two different data distributions** (e.g. test set vs. production), using a **Region of Practical Equivalence (ROPE)** decision on the posterior difference `Δ = p_ref − p_eval`. The 95% highest-density interval (HDI) of Δ is compared against `ROPE = [−eps, +eps]` (default `eps = 0.03`):

- `"equivalent"` — HDI entirely inside the ROPE (no meaningful shift).
- `"shifted"` — HDI entirely outside the ROPE (meaningful shift; `direction` says which side).
- `"undecided"` — HDI straddles a ROPE boundary (not enough data).

```python
from bayesian_pr import transfer_test, Metric

result = transfer_test(model_ref, model_eval, metric=Metric.PRECISION, eps=0.03, seed=0)
# {
#   "metric":     "precision",
#   "mu_ref":     0.896,
#   "mu_eval":    0.598,
#   "mean_delta": +0.298,
#   "rho":        0.000,          # posterior mass inside the ROPE
#   "hdi_low":    0.205,
#   "hdi_high":   0.388,
#   "status":     "shifted",
#   "direction":  "prod_worse",
#   "model_ref":  "cls [test]",
#   "model_eval": "cls [prod]",
# }
```

For raw counts (no model objects), use `transfer_test_rope(tp_test, fp_test, tp_prod, fp_prod, ...)`, or the metric-named wrappers `transfer_test_precision(...)` / `transfer_test_recall(...)` (pass `(TP, FN)` for recall). Unlike the model wrapper, these return a `TransferResult` dataclass.

The decision is symmetric in the two datasets (swapping them flips `direction`, not `status`) and uses the full uncertainty of both. Because the posterior of Δ is drawn by Monte Carlo, F1 is supported too via the model wrapper.

---

## Prior sensitivity

The Beta prior encodes beliefs before seeing any data. The two hyperparameters
(α, β) represent pseudo-counts of successes and failures respectively. With little
data the choice of prior matters; with large samples the posterior is dominated by
observations regardless of the prior.

![Prior sensitivity](docs/assets/prior_sensitivity.png)

Each curve shows how a different prior (α, β) shapes the posterior after the same
observations (TP=8, FP=4). Weakly informative priors (e.g. `Beta(1,1)` — uniform)
let the data speak; stronger priors require more data to be overcome. The `plot_prior_sensitivity`
visualization lets you audit this effect for your own observation counts.

---

## Visualization

All plot functions return a `matplotlib.Figure`.

```python
from bayesian_pr.visualization import (
    plot_posteriors,          # PDF for precision, recall, F1
    plot_sequential_update,   # Posterior evolution as observations arrive
    plot_comparison,          # Overlay + forest plot across multiple models
    plot_prior_sensitivity,   # Effect of different priors at low vs high N
)

fig = plot_posteriors(model, ci=0.95)
fig.savefig("posteriors.png", dpi=150, bbox_inches="tight")
```

---

## Examples

| File | What it demonstrates |
|---|---|
| [01_basic_usage.py](examples/01_basic_usage.py) | Fit a model, print the posterior summary, plot the distributions |
| [02_prob_above_threshold.py](examples/02_prob_above_threshold.py) | `prob_above_threshold` across classes and sample sizes |
| [03_model_comparison.py](examples/03_model_comparison.py) | Compare candidate vs. baseline via `compare_models` |
| [04_production_transfer.py](examples/04_production_transfer.py) | Stable / inconclusive / distribution-shift scenarios with `transfer_test` |

---

## Mathematical background

Both precision and recall are Bernoulli success rates, so the natural model is Beta-Binomial:

```
likelihood:  TP | n, θ  ~  Binomial(n, θ)
prior:       θ          ~  Beta(α, β)
posterior:   θ | TP, n  ~  Beta(α + TP, β + n − TP)
```

For **precision**, `n = TP + FP` and `successes = TP`.  
For **recall**, `n = TP + FN` and `successes = TP`.  
Beta is the conjugate prior of the Binomial, so the posterior stays in the Beta family — no MCMC needed.

With an uninformed (uniform) prior `Beta(1, 1)`, the posterior **mode** equals the observed metric:

```
mode = (α + TP − 1) / (α + TP + β + FP − 2)
     = TP / (TP + FP)   when α = β = 1
```

This means the Bayesian estimate recovers the classical point estimate as a special case, while the full posterior additionally quantifies uncertainty around it.

**F1** has no closed-form posterior — estimated via Monte Carlo over joint samples:

```python
p_samples  = Beta(α_p, β_p).sample(N)
r_samples  = Beta(α_r, β_r).sample(N)
f1_samples = 2 * p * r / (p + r)
```

**`transfer_test`** works on the posterior of the difference `Δ = p_ref − p_eval`, drawn by
Monte Carlo from both Beta posteriors:

```
draws_ref  ~ Beta(α + TP_ref,  β + FP_ref)
draws_eval ~ Beta(α + TP_eval, β + FP_eval)
Δ = draws_ref − draws_eval
```

A symmetric Region of Practical Equivalence `ROPE = [−eps, +eps]` (default `eps = 0.03`) defines
"the same for practical purposes." The 95% **highest-density interval** (HDI) of Δ — not the
equal-tailed quantiles, since Δ is skewed when either sample is small — is compared to the ROPE:
entirely inside → `equivalent`; entirely outside → `shifted`; straddling → `undecided`.

