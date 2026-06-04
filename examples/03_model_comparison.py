"""
Example 3 — Model comparison

Instead of comparing two point estimates, compare the full posteriors.
P(candidate > baseline) accounts for the uncertainty in both estimates.
"""

from bayesian_pr import BayesianPRModel, compare_models
from bayesian_pr.visualization import plot_comparison

candidate = BayesianPRModel(name="candidate").update(tp=80, fp=20, fn=25)
baseline  = BayesianPRModel(name="baseline").update(tp=72, fp=26, fn=28)

result = compare_models(candidate, baseline, metric="precision")
print(f"P(candidate > baseline) = {result['prob_candidate_better']:.1%}")
print(f"Expected difference      = {result['mean_diff']:+.4f}")
print(f"95% CI on difference     = [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")
print(f"Verdict                  = {result['verdict']}")

fig = plot_comparison([candidate, baseline], metric="precision")
fig.savefig("03_model_comparison.png", dpi=150, bbox_inches="tight")
