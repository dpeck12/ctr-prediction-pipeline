from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ctr.features import HashingFeaturizer
from ctr.schema import LABEL


class LogisticRegressionModel:
    """Hashed one-hot features into logistic regression: the phase 1 baseline.

    C defaults to 0.1, the value phase 1 chose on the validation day.
    """

    name = "logreg"

    def __init__(self, C: float = 0.1, n_features: int = 2**20, max_iter: int = 1000) -> None:
        self.C = C
        self.n_features = n_features
        self.max_iter = max_iter

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> LogisticRegressionModel:
        self.featurizer = HashingFeaturizer(self.n_features)
        self.model = LogisticRegression(C=self.C, max_iter=self.max_iter)
        self.model.fit(self.featurizer.transform(train), train[LABEL].to_numpy())
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.featurizer.transform(df))[:, 1]

    def params(self) -> dict[str, Any]:
        return {"C": self.C, "n_features": self.n_features, "max_iter": self.max_iter}
