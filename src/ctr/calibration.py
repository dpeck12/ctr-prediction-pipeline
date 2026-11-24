"""Post-hoc calibration: correcting a model's probabilities on held-out data."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

EPS = 1e-7


class PlattCalibrator:
    """Platt scaling: p' = sigmoid(a * logit(p) + b), with a and b fitted by log loss.

    Two parameters can't overfit and keep the model's ranking, so AUC is unchanged.
    Fit it on the validation day: data the model wasn't trained on, but earlier
    than the test day.
    """

    def fit(self, p: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        x = logit(np.clip(p, EPS, 1 - EPS)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6).fit(x, y)  # effectively unregularised
        self.a = float(lr.coef_[0, 0])
        self.b = float(lr.intercept_[0])
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return expit(self.a * logit(np.clip(p, EPS, 1 - EPS)) + self.b)

    def params(self) -> dict[str, float]:
        return {"platt_a": self.a, "platt_b": self.b}
