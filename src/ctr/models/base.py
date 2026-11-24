from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import pandas as pd


class CTRModel(Protocol):
    """A click model trained on raw impressions (as returned by ctr.data.load_avazu)."""

    name: str

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> CTRModel:
        """Train on `train`. `valid` may be used for early stopping, never for gradients."""
        ...

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Click probability for each row."""
        ...

    def predict_records(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Same as predict_proba, for raw impression dicts, without pandas.

        The fast path for serving: a one-row DataFrame costs milliseconds to
        build and encode, while dict lookups take microseconds.
        """
        ...

    def params(self) -> dict[str, Any]:
        """Hyperparameters and fitted facts worth logging (e.g. best iteration)."""
        ...
