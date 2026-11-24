import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ctr.calibration import PlattCalibrator
from ctr.metrics import expected_calibration_error


@pytest.fixture
def overconfident():
    """True probabilities, and a model that halves them (a badly calibrated one)."""
    rng = np.random.default_rng(0)
    p_true = rng.uniform(0.05, 0.6, size=50_000)
    y = rng.binomial(1, p_true)
    return p_true * 0.5, y


def test_platt_fixes_miscalibration(overconfident):
    p, y = overconfident
    calibrated = PlattCalibrator().fit(p, y).transform(p)

    assert expected_calibration_error(y, p) > 0.1
    assert expected_calibration_error(y, calibrated) < 0.02


def test_platt_keeps_ranking(overconfident):
    p, y = overconfident
    calibrated = PlattCalibrator().fit(p, y).transform(p)

    assert roc_auc_score(y, calibrated) == pytest.approx(roc_auc_score(y, p))


def test_platt_is_near_identity_when_already_calibrated():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.6, size=100_000)
    y = rng.binomial(1, p)
    calibrator = PlattCalibrator().fit(p, y)

    assert calibrator.a == pytest.approx(1.0, abs=0.05)
    assert calibrator.b == pytest.approx(0.0, abs=0.05)
    assert set(calibrator.params()) == {"platt_a", "platt_b"}
