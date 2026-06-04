"""
Example 2 — P(metric > threshold)

Instead of asking "is the point estimate above 65%?", ask
"how much of the posterior curve sits above 65%?"

The same 70% point estimate on 10 vs 200 predictions gives very different
answers — confidence comes from volume, not from the rate itself.
"""

from bayesian_pr import BayesianPRModel
import matplotlib.pyplot as plt
import numpy as np

PRECISION_FLOOR = 0.65
RECALL_FLOOR    = 0.35

# Three classifiers at different sample sizes
models = [
    BayesianPRModel(name="class_A").update(tp=80, fp=26, fn=20),   # large
    BayesianPRModel(name="class_B").update(tp=18, fp=8,  fn=12),   # medium
    BayesianPRModel(name="class_C").update(tp=5,  fp=6,  fn=8),    # sparse
]

print(f"{'Model':<10} {'P est':>6}  {'P(P>65%)':>9}  {'P(R>35%)':>9}")
print("-" * 40)
for m in models:
    p_est = m._tp / (m._tp + m._fp)
    print(
        f"{m.name:<10} {p_est:>6.1%}  "
        f"{m.prob_above_threshold(PRECISION_FLOOR, 'precision'):>9.1%}  "
        f"{m.prob_above_threshold(RECALL_FLOOR, 'recall'):>9.1%}"
    )

# Sample-size effect at a fixed 70% point estimate
print("\nP(precision > 65%) at fixed 70% point estimate:")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

x = np.linspace(0, 1, 500)
for m, color in zip(models, ["#4C72B0", "#DD8452", "#55A868"]):
    probs = [m.prob_above_threshold(t, "precision") for t in x]
    axes[0].plot(x, probs, lw=2, color=color, label=m.name)
axes[0].axvline(PRECISION_FLOOR, lw=1.5, ls=":", color="crimson", label="floor (65%)")
axes[0].axhline(0.70, lw=1.2, ls="--", color="gray", label="confidence bar (70%)")
axes[0].set_xlabel("Threshold")
axes[0].set_ylabel("P(true precision > threshold)")
axes[0].set_title("Threshold sweep by class")
axes[0].legend(fontsize=9)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

ns = [10, 20, 50, 100, 200, 500]
probs_n = []
for n in ns:
    tp = round(n * 0.70)
    m = BayesianPRModel().update(tp=tp, fp=n - tp, fn=0)
    probs_n.append(m.prob_above_threshold(0.65, "precision"))
axes[1].plot(ns, probs_n, "o-", lw=2, color="#4C72B0")
axes[1].axhline(0.70, lw=1.2, ls="--", color="gray", label="70%")
axes[1].set_xlabel("n predictions (at fixed 70% point estimate)")
axes[1].set_ylabel("P(true precision > 65%)")
axes[1].set_title("Confidence grows with sample size")
axes[1].legend(fontsize=9)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("02_prob_above_threshold.png", dpi=150, bbox_inches="tight")
