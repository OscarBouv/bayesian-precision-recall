"""
Example 4 — Distributional consistency test (transfer_test)

transfer_test checks whether the same model's posterior is consistent across
two data distributions. It takes the posterior mean from the reference
distribution (μ_ref) and measures its tail probability under the second
distribution's posterior:

    S = min(F_eval(μ_ref), 1 − F_eval(μ_ref))

Small S (≤ 0.05 by default) indicates the two posteriors are statistically
inconsistent — the model behaves differently on the two distributions.

Both models must share the same model_name. A warning is raised if they don't.
"""

from bayesian_pr import BayesianPRModel, Metric, transfer_test
import matplotlib.pyplot as plt
import numpy as np

# Consistent: both distributions yield similar posteriors
a_ref  = BayesianPRModel(model_name="cls", sample_dist_name="dist_A").update(tp=80, fp=20, fn=25)
a_eval = BayesianPRModel(model_name="cls", sample_dist_name="dist_B").update(tp=38, fp=11, fn=14)

# Inconclusive: eval distribution has too few observations (wide posterior)
b_ref  = BayesianPRModel(model_name="cls", sample_dist_name="dist_A").update(tp=60, fp=14, fn=18)
b_eval = BayesianPRModel(model_name="cls", sample_dist_name="dist_B").update(tp=4,  fp=3,  fn=3)

# Inconsistent: posteriors clearly separated
c_ref  = BayesianPRModel(model_name="cls", sample_dist_name="dist_A").update(tp=90, fp=22, fn=18)
c_eval = BayesianPRModel(model_name="cls", sample_dist_name="dist_B").update(tp=15, fp=25, fn=12)

for label, ref, evl in [
    ("consistent",    a_ref, a_eval),
    ("inconclusive",  b_ref, b_eval),
    ("inconsistent",  c_ref, c_eval),
]:
    r = transfer_test(ref, evl, metric=Metric.PRECISION)
    flag = "✗" if r["inconsistent"] else "✓"
    print(f"{label:<14} μ_ref={r['mu_ref']:.1%}  μ_eval={r['mu_eval']:.1%}"
          f"  Δ={r['delta']:+.1%}  S={r['S']:.4f}  {flag}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
x = np.linspace(0, 1, 500)
scenarios = [
    ("consistent",   a_ref, a_eval, "✓ consistent"),
    ("inconclusive", b_ref, b_eval, "? too few observations"),
    ("inconsistent", c_ref, c_eval, "✗ inconsistent"),
]
for ax, (title, ref, evl, verdict) in zip(axes, scenarios):
    for model, color, label in [(ref, "#4C72B0", "dist_A"), (evl, "#DD8452", "dist_B")]:
        y = model.precision_posterior.pdf(x)
        ax.plot(x, y, lw=2, color=color, label=label)
        ax.fill_between(x, y, alpha=0.12, color=color)
        ax.axvline(model.precision_posterior.mean(), lw=1, ls="--", color=color, alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Precision")
    ax.set_title(f"{title}\n{verdict}", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ax is axes[0]:
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
fig.suptitle("Distributional consistency of precision posteriors", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("04_production_transfer.png", dpi=150, bbox_inches="tight")
