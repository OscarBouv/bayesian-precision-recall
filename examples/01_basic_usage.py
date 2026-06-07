"""
Example 1 — Basic usage
"""

from bayesian_pr import BayesianPRModel, Metric
from bayesian_pr.visualization import plot_posteriors

model = BayesianPRModel(model_name="classifier_v1", sample_dist_name="test_set")
model.update(tp=80, fp=26, fn=20)

print(model.summary())
print()
print(f"P(precision > 0.65) = {model.prob_above_threshold(0.65, Metric.PRECISION):.1%}")
print(f"P(recall    > 0.35) = {model.prob_above_threshold(0.35, Metric.RECALL):.1%}")

fig = plot_posteriors(model)
fig.savefig("01_basic_usage.png", dpi=150, bbox_inches="tight")
