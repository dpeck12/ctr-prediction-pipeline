from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from ctr.encoders import GBMEncoder, make_fields, record_fields
from ctr.schema import LABEL


class LightGBMModel:
    """Gradient-boosted trees on count encodings and capped category codes.

    Boosting stops when validation log loss hasn't improved for
    `early_stopping_rounds` rounds, and prediction uses the best round.
    """

    name = "lightgbm"

    def __init__(
        self,
        max_categories: int = 100,
        n_estimators: int = 3000,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        min_child_samples: int = 200,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        seed: int = 0,
    ) -> None:
        self.max_categories = max_categories
        self.early_stopping_rounds = early_stopping_rounds
        self.lgb_params = {
            "objective": "binary",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "subsample_freq": 1,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
            "random_state": seed,
            "verbose": -1,
        }

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> LightGBMModel:
        self.encoder = GBMEncoder(self.max_categories).fit(make_fields(train))
        x_train = self.encoder.transform(make_fields(train))
        x_valid = self.encoder.transform(make_fields(valid))

        self.model = lgb.LGBMClassifier(**self.lgb_params)
        self.model.fit(
            x_train,
            train[LABEL].to_numpy(),
            eval_X=(x_valid,),
            eval_y=(valid[LABEL].to_numpy(),),
            eval_metric="binary_logloss",
            categorical_feature=self.encoder.categorical_columns,
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
        )
        return self

    @property
    def best_iteration(self) -> int:
        return int(self.model.best_iteration_ or self.model.n_estimators)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        x = self.encoder.transform(make_fields(df))
        return self.model.predict_proba(x, num_iteration=self.best_iteration)[:, 1]

    def predict_records(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        x = self.encoder.transform_records([record_fields(r) for r in records])
        # The raw booster skips the sklearn wrapper's pandas validation.
        return np.asarray(self.model.booster_.predict(x, num_iteration=self.best_iteration))

    def params(self) -> dict[str, Any]:
        params = {
            "max_categories": self.max_categories,
            "early_stopping_rounds": self.early_stopping_rounds,
            **self.lgb_params,
        }
        if hasattr(self, "model"):
            params["best_iteration"] = self.best_iteration
        return params
