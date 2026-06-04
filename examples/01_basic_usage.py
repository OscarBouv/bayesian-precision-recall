"""
Example 1 — Basic usage
"""

from bayesian_pr import BayesianPRModel
from bayesian_pr.visualization import plot_posteriors

model = BayesianPRModel(name="classifier_v1")
model.update(tp=80, fp=26, fn=20)

print(model.summary())

fig = plot_posteriors(model)
fig.savefig("01_basic_usage.png", dpi=150, bbox_inches="tight")
