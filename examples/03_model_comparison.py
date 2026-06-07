"""
Example 3 — Model comparison

compare_models estimates P(model_a metric > model_b metric) by sampling from
both posteriors. This accounts for uncertainty in both estimates, unlike a
direct comparison of point estimates.

Both models should be evaluated on the same data distribution. A warning is
raised automatically if the sample_dist_name fields differ.
"""

from bayesian_pr import BayesianPRModel, Metric, compare_models
from bayesian_pr.visualization import plot_comparison

model_a = BayesianPRModel(model_name="classifier_v2", sample_dist_name="test_set")
model_a.update(tp=80, fp=20, fn=25)

model_b = BayesianPRModel(model_name="classifier_v1", sample_dist_name="test_set")
model_b.update(tp=72, fp=26, fn=28)

result = compare_models(model_a, model_b, metric=Metric.PRECISION)
print(f"P(model_a > model_b) = {result['prob_a_better']:.1%}")
print(f"Expected difference  = {result['mean_diff']:+.4f}")
print(f"95% CI on difference = [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")

fig = plot_comparison([model_a, model_b], metric="precision")
fig.savefig("03_model_comparison.png", dpi=150, bbox_inches="tight")
