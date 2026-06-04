# Bayesian Precision–Recall

Precision and recall are point estimates on a finite test set — they carry uncertainty.
This library models that uncertainty using **Beta-Binomial conjugate models**, giving you full
posterior distributions for precision, recall, and F1 rather than single numbers.

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
| "Is 75% above our 65% floor?" | "P(true precision > 65%) = 98.8%" |
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
git clone https://github.com/your-org/bayesian-precision-recall.git
cd bayesian-precision-recall
pip install -e .
```

**Requirements:** Python ≥ 3.9, numpy, scipy, matplotlib (see [requirements.txt](requirements.txt))

---

## Quick start

```python
from bayesian_pr import BayesianPRModel

model = BayesianPRModel(name="classifier_v2")
model.update(tp=80, fp=26, fn=20)
print(model.summary())
```

```
BayesianPRModel: classifier_v2
  Prior       : Beta(1.0, 1.0)
  Observations: TP=80, FP=26, FN=20
  Precision   : mean=0.7500, std=0.0415, 95% CI=[0.6646, 0.8267]
  Recall      : mean=0.7941, std=0.0398, 95% CI=[0.7109, 0.8664]
  F1 (MC)     : mean=0.7703, std=0.0291, 95% CI=[0.7106, 0.8248]
```

---

## Core API

### `BayesianPRModel`

```python
BayesianPRModel(
    prior_alpha: float = 1.0,    # Beta prior α  (uniform by default)
    prior_beta:  float = 1.0,    # Beta prior β
    n_samples:   int   = 100_000,
    name:        str   = "model",
)
```

| Method | Returns | Description |
|---|---|---|
| `.update(tp, fp, fn)` | `self` | Incorporate new observations (chainable, callable multiple times) |
| `.reset()` | `self` | Clear observations, keep prior |
| `.prob_above_threshold(t, metric)` | `float` | P(true metric > t) |
| `.precision_stats(ci=0.95)` | `PosteriorStats` | Mean, std, credible interval |
| `.recall_stats(ci=0.95)` | `PosteriorStats` | Same for recall |
| `.f1_stats(ci=0.95)` | `PosteriorStats` | Monte Carlo F1 posterior |
| `.precision_posterior` | `scipy.stats.beta` | Full posterior distribution object |
| `.recall_posterior` | `scipy.stats.beta` | Full posterior distribution object |
| `.f1_samples()` | `np.ndarray` | Raw MC samples for custom analysis |
| `.summary(ci=0.95)` | `str` | Human-readable summary |

### `prob_above_threshold`

```python
# Does this classifier confidently clear the precision floor?
model.prob_above_threshold(threshold=0.65, metric="precision")
# → 0.988   # large sample, point estimate well above floor

# Same question, only 11 predictions behind it:
sparse_model.prob_above_threshold(threshold=0.65, metric="precision")
# → 0.085   # too uncertain — acquire more labels
```

The same 70% point estimate on different sample sizes:

| n predictions | P(precision > 65%) |
|---|---|
| 10  | 57% |
| 50  | 75% |
| 100 | 84% |
| 200 | 93% |
| 500 | 99% |

Confidence comes from volume, not from the rate itself.

### `compare_models`

```python
from bayesian_pr import compare_models

result = compare_models(candidate, baseline, metric="precision")
# {
#   "prob_candidate_better": 0.859,
#   "verdict": "certain_gain",
#   "mean_diff": +0.042,
#   "ci_low": 0.008, "ci_high": 0.076,
# }
```

| Verdict | Condition |
|---|---|
| `certain_gain` | P > 0.80 |
| `stagnation` | 0.20 ≤ P ≤ 0.80 |
| `certain_drop` | P < 0.20 |

### `transfer_test`

Checks whether a model's performance holds up on production data.

```python
from bayesian_pr import transfer_test

result = transfer_test(test_model, prod_model, metric="precision")
# {
#   "p_hat_test":       0.794,
#   "prod_mean":        0.381,
#   "apparent_drop":    +0.413,
#   "S":                0.0000,
#   "significant_drop": True,
#   "verdict":          "domain_shift_detected",
# }
```

`S = min(F_prod(p̂_test), 1 − F_prod(p̂_test))` measures where the test-set posterior mean
lands in the production posterior. `S ≤ 0.05` indicates the drop is statistically real.
A high `S` means the two agree, or the production sample is too small to conclude.

---

## Shipping decision example

```python
from bayesian_pr import BayesianPRModel, compare_models, transfer_test

PRECISION_FLOOR = 0.65
RECALL_FLOOR    = 0.35

candidate = BayesianPRModel(name="model_v2")
candidate.update(tp=80, fp=20, fn=25)

baseline  = BayesianPRModel(name="model_v1")
baseline.update(tp=72, fp=26, fn=28)

prod      = BayesianPRModel(name="model_v2 (prod sample)")
prod.update(tp=35, fp=10, fn=12)

# Does it clear the acceptance bar?
p_prob = candidate.prob_above_threshold(PRECISION_FLOOR, "precision")  # 0.999
r_prob = candidate.prob_above_threshold(RECALL_FLOOR,    "recall")     # 1.000

# Is it better than baseline?
g2 = compare_models(candidate, baseline, metric="precision")
# → {"verdict": "certain_gain", "prob_candidate_better": 0.86}

# Does it hold up on production data?
g3 = transfer_test(candidate, prod, metric="precision")
# → {"verdict": "transfer_ok", "S": 0.34}
```

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

**F1** has no closed-form posterior — estimated via Monte Carlo over joint samples:

```python
p_samples  = Beta(α_p, β_p).sample(N)
r_samples  = Beta(α_r, β_r).sample(N)
f1_samples = 2 * p * r / (p + r)
```

**`transfer_test`** treats the test-set posterior mean as a fixed reference and measures its
tail probability in the production posterior:

```
S = min( F_prod(p̂_test),  1 − F_prod(p̂_test) )
```

`F_prod` is the Beta CDF, evaluated analytically. `S ≤ 0.05` → the test estimate sits in a
thin tail of the production curve → the performance drop is real.

Based on: Goutte & Gaussier, *"A Probabilistic Interpretation of Precision, Recall and F-score"*, ECIR 2005.

---

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
