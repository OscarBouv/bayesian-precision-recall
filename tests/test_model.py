import pytest
import numpy as np
from bayesian_pr import BayesianPRModel, compare_models, transfer_test
from scipy import stats


def test_posterior_shapes():
    m = BayesianPRModel()
    m.update(tp=50, fp=10, fn=15)
    # scipy froze the beta distribution as beta_frozen (older) or beta_gen_frozen (newer)
    assert hasattr(m.precision_posterior, "pdf")
    assert hasattr(m.recall_posterior, "interval")
    assert 0 < m.precision_posterior.mean() < 1
    assert 0 < m.recall_posterior.mean() < 1


def test_posterior_parameters():
    """Beta posterior parameters should equal prior + counts."""
    m = BayesianPRModel(prior_alpha=2.0, prior_beta=3.0)
    m.update(tp=10, fp=5, fn=8)

    p = m.precision_posterior
    assert p.args[0] == pytest.approx(12.0)   # alpha + tp
    assert p.args[1] == pytest.approx(8.0)    # beta + fp

    r = m.recall_posterior
    assert r.args[0] == pytest.approx(12.0)   # alpha + tp
    assert r.args[1] == pytest.approx(11.0)   # beta + fn


def test_sequential_update_is_additive():
    """Two calls to update() should equal one combined call."""
    m1 = BayesianPRModel()
    m1.update(tp=30, fp=10, fn=5)
    m1.update(tp=20, fp=8, fn=7)

    m2 = BayesianPRModel()
    m2.update(tp=50, fp=18, fn=12)

    assert m1.precision_posterior.mean() == pytest.approx(m2.precision_posterior.mean(), abs=1e-10)
    assert m1.recall_posterior.mean() == pytest.approx(m2.recall_posterior.mean(), abs=1e-10)


def test_reset():
    m = BayesianPRModel(prior_alpha=2, prior_beta=2)
    m.update(tp=50, fp=20, fn=10)
    m.reset()
    assert m.observations == {"tp": 0, "fp": 0, "fn": 0}
    # posterior should equal prior
    assert m.precision_posterior.mean() == pytest.approx(0.5)


def test_f1_samples_shape():
    m = BayesianPRModel(n_samples=10_000)
    m.update(tp=50, fp=10, fn=15)
    samples = m.f1_samples()
    assert samples.shape == (10_000,)
    assert np.all(samples >= 0) and np.all(samples <= 1)


def test_f1_mean_in_expected_range():
    m = BayesianPRModel()
    m.update(tp=100, fp=10, fn=10)
    s = m.f1_stats()
    # With high TP and low FP/FN, F1 should be well above 0.8
    assert s.mean > 0.8
    assert s.ci_low < s.mean < s.ci_high


def test_stats_ci_ordering():
    m = BayesianPRModel()
    m.update(tp=50, fp=15, fn=20)
    for s in [m.precision_stats(), m.recall_stats(), m.f1_stats()]:
        assert s.ci_low < s.mean < s.ci_high


def test_invalid_inputs():
    with pytest.raises(ValueError):
        BayesianPRModel(prior_alpha=-1)
    with pytest.raises(ValueError):
        BayesianPRModel(prior_beta=0)
    m = BayesianPRModel()
    with pytest.raises(ValueError):
        m.update(tp=-1, fp=0, fn=0)


# ── prob_above_threshold ───────────────────────────────────────────────────────

def test_prob_above_threshold_range():
    m = BayesianPRModel()
    m.update(tp=80, fp=20, fn=15)
    for metric in ("precision", "recall", "f1"):
        p = m.prob_above_threshold(0.65, metric=metric)
        assert 0 <= p <= 1

def test_prob_above_threshold_high_precision():
    """With 200 TP and 10 FP, P(precision > 0.65) should be near 1."""
    m = BayesianPRModel()
    m.update(tp=200, fp=10, fn=20)
    assert m.prob_above_threshold(0.65, "precision") > 0.999

def test_prob_above_threshold_low_precision():
    """With 10 TP and 40 FP (20% precision), P(precision > 0.65) should be near 0."""
    m = BayesianPRModel()
    m.update(tp=10, fp=40, fn=5)
    assert m.prob_above_threshold(0.65, "precision") < 0.001

def test_prob_above_threshold_monotone():
    """P(metric > t) must decrease as t increases."""
    m = BayesianPRModel()
    m.update(tp=60, fp=20, fn=20)
    thresholds = [0.3, 0.5, 0.65, 0.80, 0.90]
    probs = [m.prob_above_threshold(t, "precision") for t in thresholds]
    assert all(probs[i] >= probs[i+1] for i in range(len(probs)-1))

def test_prob_above_threshold_invalid():
    m = BayesianPRModel()
    m.update(tp=50, fp=10, fn=10)
    with pytest.raises(ValueError):
        m.prob_above_threshold(1.5, "precision")
    with pytest.raises(ValueError):
        m.prob_above_threshold(0.65, "accuracy")


# ── compare_models (updated API) ──────────────────────────────────────────────

def test_compare_models_structure():
    candidate = BayesianPRModel(name="candidate")
    candidate.update(tp=80, fp=10, fn=10)
    baseline = BayesianPRModel(name="baseline")
    baseline.update(tp=60, fp=15, fn=20)

    result = compare_models(candidate, baseline, metric="precision", n_samples=50_000)
    assert "prob_candidate_better" in result
    assert "verdict" in result
    assert result["verdict"] in ("certain_gain", "certain_drop", "stagnation")
    assert 0 <= result["prob_candidate_better"] <= 1
    assert result["ci_low"] < result["ci_high"]

def test_compare_models_certain_gain():
    candidate = BayesianPRModel(name="strong")
    candidate.update(tp=200, fp=5, fn=5)
    baseline = BayesianPRModel(name="weak")
    baseline.update(tp=80, fp=60, fn=40)

    result = compare_models(candidate, baseline, metric="precision", n_samples=100_000)
    assert result["verdict"] == "certain_gain"
    assert result["prob_candidate_better"] > 0.80

def test_compare_models_certain_drop():
    candidate = BayesianPRModel(name="worse")
    candidate.update(tp=60, fp=60, fn=20)
    baseline = BayesianPRModel(name="better")
    baseline.update(tp=200, fp=5, fn=10)

    result = compare_models(candidate, baseline, metric="precision", n_samples=100_000)
    assert result["verdict"] == "certain_drop"
    assert result["prob_candidate_better"] < 0.20

def test_compare_models_invalid_metric():
    a = BayesianPRModel()
    b = BayesianPRModel()
    with pytest.raises(ValueError):
        compare_models(a, b, metric="accuracy")


# ── transfer_test ─────────────────────────────────────────────────────────────

def test_transfer_test_stable():
    """Same performance on test and prod → no shift."""
    test  = BayesianPRModel(name="test")
    test.update(tp=80, fp=20, fn=20)
    prod  = BayesianPRModel(name="prod")
    prod.update(tp=40, fp=10, fn=10)  # same rate, half the data

    result = transfer_test(test, prod, metric="precision")
    assert not result["significant_drop"]
    assert result["verdict"] == "transfer_ok"

def test_transfer_test_shift_detected():
    """Clear precision drop from test to prod → shift detected."""
    test = BayesianPRModel(name="test")
    test.update(tp=200, fp=20, fn=30)   # ~91% precision, tight curve
    prod = BayesianPRModel(name="prod")
    prod.update(tp=20, fp=60, fn=15)    # ~25% precision in prod

    result = transfer_test(test, prod, metric="precision")
    assert result["significant_drop"]
    assert result["verdict"] == "domain_shift_detected"
    assert result["S"] <= 0.05

def test_transfer_test_s_bounds():
    test = BayesianPRModel()
    test.update(tp=50, fp=10, fn=15)
    prod = BayesianPRModel()
    prod.update(tp=30, fp=8, fn=10)

    result = transfer_test(test, prod, metric="precision")
    assert 0 <= result["S"] <= 0.5   # S is the smaller tail, always ≤ 0.5

def test_transfer_test_invalid_metric():
    a = BayesianPRModel()
    b = BayesianPRModel()
    with pytest.raises(ValueError):
        transfer_test(a, b, metric="f1")
