import numpy as np
import pytest
from sklearn.metrics import log_loss

from ctr.metrics import (
    binary_entropy,
    calibration_table,
    evaluate,
    expected_calibration_error,
    normalized_entropy,
)


@pytest.fixture
def labels():
    rng = np.random.default_rng(0)
    return rng.binomial(1, 0.2, size=10_000)


def test_binary_entropy_known_values():
    assert binary_entropy(0.5) == pytest.approx(np.log(2))
    assert binary_entropy(0.0) == pytest.approx(0.0, abs=1e-5)


def test_constant_base_rate_predictor_has_ne_of_one(labels):
    rate = labels.mean()
    p = np.full(len(labels), rate)

    assert normalized_entropy(labels, p, rate) == pytest.approx(1.0)


def test_better_predictions_give_ne_below_one():
    rng = np.random.default_rng(1)
    true_p = rng.uniform(0.05, 0.6, size=20_000)
    y = rng.binomial(1, true_p)

    assert normalized_entropy(y, true_p, y.mean()) < 0.95


def test_calibration_table_bins_are_equal_count():
    p = np.linspace(0.01, 0.5, 1_000)
    y = np.zeros(1_000)
    table = calibration_table(y, p, n_bins=10)

    assert len(table) == 10
    assert (table["count"] == 100).all()
    assert table["mean_pred"].is_monotonic_increasing


def test_constant_predictions_share_one_bin(labels):
    table = calibration_table(labels, np.full(len(labels), 0.3), n_bins=10)

    assert len(table) == 1
    assert expected_calibration_error(labels, np.full(len(labels), 0.3)) == pytest.approx(
        abs(0.3 - labels.mean())
    )


def test_ece_is_near_zero_for_calibrated_predictions():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.6, size=200_000)
    y = rng.binomial(1, p)

    assert expected_calibration_error(y, p) < 0.01


def test_evaluate_returns_consistent_metrics(labels):
    p = np.full(len(labels), 0.25)
    result = evaluate(labels, p, base_rate=0.25)

    assert result["rows"] == len(labels)
    assert result["log_loss"] == pytest.approx(log_loss(labels, p))
    assert result["calibration_ratio"] == pytest.approx(0.25 / labels.mean())
    assert result["auc"] == pytest.approx(0.5)


def test_evaluate_handles_single_class():
    result = evaluate(np.zeros(10), np.full(10, 0.1), base_rate=0.1)

    assert np.isnan(result["auc"])
    assert np.isnan(result["calibration_ratio"])
