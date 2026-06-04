"""
Example 4 — Production transfer test

Checks whether a model's precision holds up on production data.
S = min(F_prod(p̂_test), 1 − F_prod(p̂_test))
S ≤ 0.05 → the drop is statistically real.
"""

from bayesian_pr import BayesianPRModel, transfer_test
import matplotlib.pyplot as plt
import numpy as np

# Stable: test and prod agree
a_test = BayesianPRModel(name="stable (test)").update(tp=80, fp=20, fn=25)
a_prod = BayesianPRModel(name="stable (prod)").update(tp=38, fp=11, fn=14)

# Inconclusive: prod sample too small
b_test = BayesianPRModel(name="inconclusive (test)").update(tp=60, fp=14, fn=18)
b_prod = BayesianPRModel(name="inconclusive (prod)").update(tp=4,  fp=3,  fn=3)

# Shift: clear drop in production
c_test = BayesianPRModel(name="shift (test)").update(tp=90, fp=22, fn=18)
c_prod = BayesianPRModel(name="shift (prod)").update(tp=15, fp=25, fn=12)

for label, t, p in [("stable", a_test, a_prod),
                     ("inconclusive", b_test, b_prod),
                     ("shift", c_test, c_prod)]:
    r = transfer_test(t, p, metric="precision")
    verdict = "✗ SHIFT" if r["significant_drop"] else "✓ OK"
    print(f"{label:<15} test={r['p_hat_test']:.1%}  prod={r['prod_mean']:.1%}"
          f"  drop={r['apparent_drop']:+.1%}  S={r['S']:.4f}  {verdict}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
x = np.linspace(0, 1, 500)
scenarios = [
    ("stable",        a_test, a_prod, "✓ stable"),
    ("inconclusive",  b_test, b_prod, "? too few prod samples"),
    ("shift",         c_test, c_prod, "✗ distribution shift"),
]
for ax, (title, t_m, p_m, verdict) in zip(axes, scenarios):
    for model, color, label in [(t_m, "#4C72B0", "test"), (p_m, "#DD8452", "prod")]:
        y = model.precision_posterior.pdf(x)
        ax.plot(x, y, lw=2, color=color, label=label)
        ax.fill_between(x, y, alpha=0.12, color=color)
        ax.axvline(model.precision_posterior.mean(), lw=1, ls="--", color=color, alpha=0.7)
    ax.axvline(0.65, lw=1.2, ls=":", color="crimson", alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Precision")
    ax.set_title(f"{title}\n{verdict}", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ax is axes[0]:
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("04_production_transfer.png", dpi=150, bbox_inches="tight")
