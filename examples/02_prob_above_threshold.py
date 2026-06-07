"""
Example 2 — P(metric > threshold)

The same point estimate carries very different confidence depending on sample
size. This example sweeps three classes at different observation counts and
shows how P(precision > 0.65) changes as a function of n.
"""

from bayesian_pr import BayesianPRModel, Metric
import matplotlib.pyplot as plt
import numpy as np

THRESHOLD = 0.65

models = [
    BayesianPRModel(model_name="class_A").update(tp=80, fp=26, fn=20),
    BayesianPRModel(model_name="class_B").update(tp=18, fp=8,  fn=12),
    BayesianPRModel(model_name="class_C").update(tp=5,  fp=6,  fn=8),
]

print(f"{'Model':<10} {'P est':>6}  {'P(P>0.65)':>10}")
print("-" * 32)
for m in models:
    p_est = m._tp / (m._tp + m._fp)
    print(f"{m.model_name:<10} {p_est:>6.1%}  {m.prob_above_threshold(THRESHOLD, Metric.PRECISION):>10.1%}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

x = np.linspace(0, 1, 500)
for m, color in zip(models, ["#4C72B0", "#DD8452", "#55A868"]):
    probs = [m.prob_above_threshold(t, Metric.PRECISION) for t in x]
    axes[0].plot(x, probs, lw=2, color=color, label=m.model_name)
axes[0].axvline(THRESHOLD, lw=1.5, ls=":", color="crimson", label=f"threshold ({THRESHOLD})")
axes[0].axhline(0.70, lw=1.2, ls="--", color="gray", label="0.70")
axes[0].set_xlabel("Threshold")
axes[0].set_ylabel("P(true precision > threshold)")
axes[0].set_title("Threshold sweep by class")
axes[0].legend(fontsize=9)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# Fixed 70% point estimate, varying n
ns = [10, 20, 50, 100, 200, 500]
probs_n = [
    BayesianPRModel().update(tp=round(n * 0.70), fp=n - round(n * 0.70), fn=0)
    .prob_above_threshold(THRESHOLD, Metric.PRECISION)
    for n in ns
]
axes[1].plot(ns, probs_n, "o-", lw=2, color="#4C72B0")
axes[1].axhline(0.70, lw=1.2, ls="--", color="gray", label="0.70")
axes[1].set_xlabel("n predictions (fixed 70% point estimate)")
axes[1].set_ylabel(f"P(true precision > {THRESHOLD})")
axes[1].set_title("Confidence grows with sample size")
axes[1].legend(fontsize=9)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("02_prob_above_threshold.png", dpi=150, bbox_inches="tight")
