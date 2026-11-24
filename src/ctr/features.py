"""Turning raw impressions into model inputs with the hashing trick."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher

from ctr.schema import CATEGORICAL_COLUMNS, TIMESTAMP

MISSING = "<NA>"


class HashingFeaturizer:
    """One-hot encode every categorical value by hashing `column=value` into a fixed space.

    Ad data has millions of distinct IDs (sites, apps, devices) and new ones appear
    every day. Hashing needs no fitted vocabulary, so it is stateless, uses fixed
    memory, and handles values it has never seen. Some collisions are the price.

    Each row also gets hour-of-day and day-of-week tokens from its timestamp.
    """

    def __init__(
        self, n_features: int = 2**20, columns: Sequence[str] = CATEGORICAL_COLUMNS
    ) -> None:
        self.n_features = n_features
        self.columns = tuple(columns)
        self._hasher = FeatureHasher(
            n_features=n_features, input_type="string", alternate_sign=False
        )

    @property
    def tokens_per_row(self) -> int:
        return len(self.columns) + 2

    def tokens(self, df: pd.DataFrame) -> list[pd.Series]:
        """The `name=value` strings that get hashed, one Series per feature."""
        cols = [col + "=" + df[col].astype("string").fillna(MISSING) for col in self.columns]
        cols.append("hour_of_day=" + df[TIMESTAMP].dt.hour.astype("string"))
        cols.append("day_of_week=" + df[TIMESTAMP].dt.day_of_week.astype("string"))
        return cols

    def transform(self, df: pd.DataFrame) -> sparse.csr_matrix:
        rows = zip(*(col.tolist() for col in self.tokens(df)), strict=True)
        return self._hasher.transform(rows)
