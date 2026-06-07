import pytest
import warnings
import numpy as np
from bayesian_pr import BayesianPRModel, Metric, compare_models, transfer_test


# ── update() and availability ─────────────────────────────────────────────────

def test_update_fp_only_enables_precision():
    m = BayesianPRModel()
    m.update(tp=50, fp=10)
    assert m.has_precision
    assert not m.has_recall
    assert not m.has_f1


def test_update_fn_only_enables_recall():
    m = BayesianPRModel()
    m.update(tp=50, fn=15)
    assert not m.has_precision
    assert m.has_recall
    assert not m.has_f1


def test_update_both_enables_all():
    m = BayesianPRModel()
    m.update(tp=50, fp=10, fn=15)
    assert m.has_precision
    assert m.has_recall
    assert m.has_f1


def test_update_accumulates_across_calls():
    m = BayesianPRModel()
    m.update(tp=30, fp=10)
    m.update(tp=20, fn=8)
    assert m.has_precision
    assert m.has_recall
    assert m.has_f1
    assert m._tp == 50
    assert m._fp == 10
    assert m._fn == 8


def test_update_requires_at_least_one_of_fp_fn():
    m = BayesianPRModel()
    with pytest.raises(ValueError, match="At least one of fp or fn"):
        m.update(tp=10)


def test_update_rejects_negative_counts():
    m = BayesianPRModel()
    with pytest.raises(ValueError):
        m.update(tp=-1, fp=0)
    with pytest.raises(ValueError):
        m.update(tp=0, fp=-1)
    with pytest.raises(ValueError):
        m.update(tp=0, fn=-1)


def test_precision_unavailable_without_fp():
    m = BayesianPRModel()
    m.update(tp=50, fn=15)
    with pytest.raises(ValueError, match="fp"):
        _ = m.precision_posterior
    with pytest.raises(ValueError, match="fp"):
        m.precision_stats()
    with pytest.raises(ValueError, match="fp"):
        m.prob_above_threshold(0.65, Metric.PRECISION)


def test_recall_unavailable_without_fn():
    m = BayesianPRModel()
    m.update(tp=50, fp=10)
    with pytest.raises(ValueError, match="fn"):
        _ = m.recall_posterior
    with pytest.raises(ValueError, match="fn"):
        m.recall_stats()
    with pytest.raises(ValueError, match="fn"):
        m.prob_above_threshold(0.35, Metric.RECALL)


def test_f1_unavailable_without_both():
    m_fp_only = BayesianPRModel()
    m_fp_only.update(tp=50, fp=10)
    with pytest.raises(ValueError, match="fn"):
        m_fp_only.f1_stats()

    m_fn_only = BayesianPRModel()
    m_fn_only.update(tp=50, fn=15)
    with pytest.raises(ValueError, match="fp"):
        m_fn_only.f1_stats()


def test_reset_clears_fp_fn():
    m = BayesianPRModel(prior_alpha=2, prior_beta=2)
    m.update(tp=50, fp=20, fn=10)
    m.reset()
    assert m.observations == {"tp": 0, "fp": None, "fn": None}
    assert not m.has_precision
    assert not m.has_recall


# ── Posterior parameters ──────────────────────────────────────────────────────

def test_posterior_parameters():
    m = BayesianPRModel(prior_alpha=2.0, prior_beta=3.0)
    m.update(tp=10, fp=5, fn=8)
    assert m.precision_posterior.args[0] == pytest.approx(12.0)
    assert m.precision_posterior.args[1] == pytest.approx(8.0)
    assert m.recall_posterior.args[0]    == pytest.approx(12.0)
    assert m.recall_posterior.args[1]    == pytest.approx(11.0)


def test_name_property():
    m = BayesianPRModel(model_name="resnet50", sample_dist_name="test_set_v1")
    assert m.name == "resnet50 [test_set_v1]"
    assert BayesianPRModel(model_name="resnet50").name == "resnet50"


def test_sequential_update_is_additive():
    m1 = BayesianPRModel()
    m1.update(tp=30, fp=10, fn=5)
    m1.update(tp=20, fp=8,  fn=7)
    m2 = BayesianPRModel()
    m2.update(tp=50, fp=18, fn=12)
    assert m1.precision_posterior.mean() == pytest.approx(m2.precision_posterior.mean(), abs=1e-10)
    assert m1.recall_posterior.mean()    == pytest.approx(m2.recall_posterior.mean(),    abs=1e-10)


def test_f1_samples_shape():
    m = BayesianPRModel(n_samples=10_000)
    m.update(tp=50, fp=10, fn=15)
    s = m.f1_samples()
    assert s.shape == (10_000,)
    assert np.all(s >= 0) and np.all(s <= 1)


def test_f1_mean_in_expected_range():
    m = BayesianPRModel()
    m.update(tp=100, fp=10, fn=10)
    s = m.f1_stats()
    assert s.mean > 0.8
    assert s.ci_low < s.mean < s.ci_high


def test_stats_ci_ordering():
    m = BayesianPRModel()
    m.update(tp=50, fp=15, fn=20)
    for s in [m.precision_stats(), m.recall_stats(), m.f1_stats()]:
        assert s.ci_low < s.mean < s.ci_high


def test_summary_shows_only_available_metrics():
    m_p = BayesianPRModel()
    m_p.update(tp=50, fp=10)
    summary = m_p.summary()
    assert "Precision" in summary
    assert "Recall"    not in summary
    assert "F1"        not in summary

    m_r = BayesianPRModel()
    m_r.update(tp=50, fn=15)
    summary = m_r.summary()
    assert "Precision" not in summary
    assert "Recall"    in summary
    assert "F1"        not in summary

    m_all = BayesianPRModel()
    m_all.update(tp=50, fp=10, fn=15)
    summary = m_all.summary()
    assert "Precision" in summary
    assert "Recall"    in summary
    assert "F1"        in summary


def test_invalid_prior():
    with pytest.raises(ValueError):
        BayesianPRModel(prior_alpha=-1)
    with pytest.raises(ValueError):
        BayesianPRModel(prior_beta=0)


# ── prob_above_threshold ───────────────────────────────────────────────────────

def test_prob_above_threshold_accepts_enum_and_string():
    m = BayesianPRModel()
    m.update(tp=80, fp=20, fn=15)
    assert m.prob_above_threshold(0.65, Metric.PRECISION) == pytest.approx(
        m.prob_above_threshold(0.65, "precision"), abs=1e-6
    )


def test_prob_above_threshold_range():
    m = BayesianPRModel()
    m.update(tp=80, fp=20, fn=15)
    for metric in Metric:
        assert 0 <= m.prob_above_threshold(0.65, metric) <= 1


def test_prob_above_threshold_high():
    m = BayesianPRModel()
    m.update(tp=200, fp=10, fn=20)
    assert m.prob_above_threshold(0.65, Metric.PRECISION) > 0.999


def test_prob_above_threshold_low():
    m = BayesianPRModel()
    m.update(tp=10, fp=40, fn=5)
    assert m.prob_above_threshold(0.65, Metric.PRECISION) < 0.001


def test_prob_above_threshold_monotone():
    m = BayesianPRModel()
    m.update(tp=60, fp=20, fn=20)
    probs = [m.prob_above_threshold(t, Metric.PRECISION) for t in [0.3, 0.5, 0.65, 0.80, 0.90]]
    assert all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1))


def test_prob_above_threshold_invalid_threshold():
    m = BayesianPRModel()
    m.update(tp=50, fp=10, fn=10)
    with pytest.raises(ValueError):
        m.prob_above_threshold(1.5, Metric.PRECISION)


def test_prob_above_threshold_invalid_metric():
    m = BayesianPRModel()
    m.update(tp=50, fp=10, fn=10)
    with pytest.raises(ValueError):
        m.prob_above_threshold(0.65, "accuracy")


# ── compare_models ─────────────────────────────────────────────────────────────

def test_compare_models_structure():
    a = BayesianPRModel(model_name="A", sample_dist_name="test")
    a.update(tp=80, fp=10, fn=10)
    b = BayesianPRModel(model_name="B", sample_dist_name="test")
    b.update(tp=60, fp=15, fn=20)

    result = compare_models(a, b, metric=Metric.PRECISION, n_samples=50_000)
    assert set(result.keys()) >= {"prob_a_better", "mean_diff", "ci_low", "ci_high"}
    assert "verdict" not in result
    assert 0 <= result["prob_a_better"] <= 1
    assert result["ci_low"] < result["ci_high"]


def test_compare_models_clearly_better():
    a = BayesianPRModel(model_name="strong", sample_dist_name="test")
    a.update(tp=200, fp=5, fn=5)
    b = BayesianPRModel(model_name="weak", sample_dist_name="test")
    b.update(tp=80, fp=60, fn=40)
    assert compare_models(a, b, metric=Metric.PRECISION, n_samples=100_000)["prob_a_better"] > 0.99


def test_compare_models_clearly_worse():
    a = BayesianPRModel(model_name="worse", sample_dist_name="test")
    a.update(tp=60, fp=60, fn=20)
    b = BayesianPRModel(model_name="better", sample_dist_name="test")
    b.update(tp=200, fp=5, fn=10)
    assert compare_models(a, b, metric=Metric.PRECISION, n_samples=100_000)["prob_a_better"] < 0.01


def test_compare_models_warns_same_model_name():
    a = BayesianPRModel(model_name="resnet50", sample_dist_name="dist_A")
    a.update(tp=80, fp=20, fn=20)
    b = BayesianPRModel(model_name="resnet50", sample_dist_name="dist_B")
    b.update(tp=40, fp=10, fn=10)
    with pytest.warns(UserWarning, match="share model_name"):
        compare_models(a, b)


def test_compare_models_warns_different_distribution():
    a = BayesianPRModel(model_name="model_A", sample_dist_name="dist_X")
    a.update(tp=80, fp=20, fn=20)
    b = BayesianPRModel(model_name="model_B", sample_dist_name="dist_Y")
    b.update(tp=80, fp=20, fn=20)
    with pytest.warns(UserWarning, match="different sample_dist_name"):
        compare_models(a, b)


def test_compare_models_invalid_metric():
    a = BayesianPRModel()
    a.update(tp=50, fp=10, fn=10)
    b = BayesianPRModel()
    b.update(tp=50, fp=10, fn=10)
    with pytest.raises(ValueError):
        compare_models(a, b, metric="accuracy")


# ── transfer_test ─────────────────────────────────────────────────────────────

def test_transfer_test_structure():
    ref = BayesianPRModel(model_name="m", sample_dist_name="dist_A")
    ref.update(tp=80, fp=20, fn=20)
    evl = BayesianPRModel(model_name="m", sample_dist_name="dist_B")
    evl.update(tp=40, fp=10, fn=10)
    result = transfer_test(ref, evl, metric=Metric.PRECISION)
    assert set(result.keys()) >= {"mu_ref", "mu_eval", "delta", "S", "inconsistent"}
    assert "verdict" not in result
    assert 0 <= result["S"] <= 0.5


def test_transfer_test_consistent():
    ref = BayesianPRModel(model_name="m", sample_dist_name="A")
    ref.update(tp=80, fp=20, fn=20)
    evl = BayesianPRModel(model_name="m", sample_dist_name="B")
    evl.update(tp=40, fp=10, fn=10)
    assert not transfer_test(ref, evl, metric=Metric.PRECISION)["inconsistent"]


def test_transfer_test_inconsistent():
    ref = BayesianPRModel(model_name="m", sample_dist_name="A")
    ref.update(tp=200, fp=20, fn=30)
    evl = BayesianPRModel(model_name="m", sample_dist_name="B")
    evl.update(tp=20, fp=60, fn=15)
    result = transfer_test(ref, evl, metric=Metric.PRECISION)
    assert result["inconsistent"]
    assert result["S"] <= 0.05


def test_transfer_test_warns_different_model_name():
    a = BayesianPRModel(model_name="model_A", sample_dist_name="dist_X")
    a.update(tp=80, fp=20, fn=20)
    b = BayesianPRModel(model_name="model_B", sample_dist_name="dist_Y")
    b.update(tp=40, fp=10, fn=10)
    with pytest.warns(UserWarning, match="different model_name"):
        transfer_test(a, b)


def test_transfer_test_warns_same_distribution():
    a = BayesianPRModel(model_name="m", sample_dist_name="dist_X")
    a.update(tp=80, fp=20, fn=20)
    b = BayesianPRModel(model_name="m", sample_dist_name="dist_X")
    b.update(tp=40, fp=10, fn=10)
    with pytest.warns(UserWarning, match="share sample_dist_name"):
        transfer_test(a, b)


def test_transfer_test_rejects_f1():
    a = BayesianPRModel(model_name="m", sample_dist_name="A")
    a.update(tp=80, fp=20, fn=20)
    b = BayesianPRModel(model_name="m", sample_dist_name="B")
    b.update(tp=40, fp=10, fn=10)
    with pytest.raises(ValueError, match="F1"):
        transfer_test(a, b, metric=Metric.F1)
