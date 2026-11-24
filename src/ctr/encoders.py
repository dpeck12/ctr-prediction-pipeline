"""Categorical encodings learned from the training days.

The hashing featuriser (features.py) is stateless. LightGBM and the deep model
instead need encodings fitted on training data: integer codes per value, and how
often each value was seen. Values that are rare in training, or that first appear
on a later day, share a single "unknown" code.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ctr.schema import CATEGORICAL_COLUMNS, RAW_TIME, TIMESTAMP

TIME_FIELDS = ("hour_of_day", "day_of_week")
FIELDS: tuple[str, ...] = (*CATEGORICAL_COLUMNS, *TIME_FIELDS)
MISSING = "<NA>"


def _is_missing(value: Any) -> bool:
    """None, or NaN (how pandas hands back a missing value in to_dict)."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def record_fields(record: Mapping[str, Any]) -> list[str]:
    """One raw impression (a dict with `hour` as YYMMDDHH) -> its values in FIELDS order.

    The no-pandas twin of make_fields, for scoring single requests quickly. Gives
    exactly the same strings, so both paths hit the same encodings.
    """
    hour = record[RAW_TIME]
    day = date(2000 + int(hour[0:2]), int(hour[2:4]), int(hour[4:6]))
    values = [record.get(col) for col in CATEGORICAL_COLUMNS]
    return [
        *(MISSING if _is_missing(value) else str(value) for value in values),
        str(int(hour[6:8])),
        str(day.weekday()),
    ]


def make_fields(df: pd.DataFrame) -> pd.DataFrame:
    """The categorical columns plus hour-of-day and day-of-week, all as strings."""
    out = df[list(CATEGORICAL_COLUMNS)].astype("string").fillna(MISSING)
    out["hour_of_day"] = df[TIMESTAMP].dt.hour.astype("string")
    out["day_of_week"] = df[TIMESTAMP].dt.day_of_week.astype("string")
    return out


class VocabEncoder:
    """Map each field's values to integers 1..n; rare or unseen values map to 0.

    Used for embedding lookups: one embedding row per known value, plus a shared
    row 0 that the model learns as "some value we know little about".
    """

    def __init__(self, min_count: int = 5) -> None:
        self.min_count = min_count
        self.vocab: dict[str, pd.Index] = {}

    def fit(self, fields: pd.DataFrame) -> VocabEncoder:
        for col in fields.columns:
            counts = fields[col].value_counts()
            self.vocab[col] = pd.Index(counts.index[counts >= self.min_count])
        return self

    @property
    def cardinalities(self) -> list[int]:
        """Embedding rows needed per field, including the unknown row."""
        return [len(values) + 1 for values in self.vocab.values()]

    def transform(self, fields: pd.DataFrame) -> np.ndarray:
        codes = [
            values.get_indexer(fields[col]).astype(np.int64) + 1  # unseen -> -1 -> 0
            for col, values in self.vocab.items()
        ]
        return np.stack(codes, axis=1)

    def transform_records(self, rows: Sequence[Sequence[str]]) -> np.ndarray:
        """Like transform, for rows from record_fields, using plain dict lookups."""
        if not hasattr(self, "_lookup"):
            self._lookup = [
                {value: i + 1 for i, value in enumerate(values)} for values in self.vocab.values()
            ]
        return np.array(
            [
                [lookup.get(v, 0) for lookup, v in zip(self._lookup, row, strict=True)]
                for row in rows
            ],
            dtype=np.int64,
        )


class GBMEncoder:
    """Per field: how often the value occurred in training, and a capped category code.

    Count encoding lets trees use every value, however rare, as "popular vs niche".
    The category code covers each field's `max_categories` most common values for
    LightGBM's native categorical splits; anything else is -1, which LightGBM
    treats as missing.
    """

    def __init__(self, max_categories: int = 100) -> None:
        self.max_categories = max_categories
        self.counts: dict[str, pd.Series] = {}
        self.top: dict[str, pd.Index] = {}

    def fit(self, fields: pd.DataFrame) -> GBMEncoder:
        for col in fields.columns:
            counts = fields[col].value_counts()
            self.counts[col] = counts
            self.top[col] = pd.Index(counts.index[: self.max_categories])
        return self

    @property
    def categorical_columns(self) -> list[str]:
        return [f"{col}_cat" for col in self.counts]

    def transform(self, fields: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col, counts in self.counts.items():
            out[f"{col}_count"] = fields[col].map(counts).fillna(0).astype("float32").to_numpy()
            out[f"{col}_cat"] = self.top[col].get_indexer(fields[col]).astype(np.int32)
        return pd.DataFrame(out, index=fields.index)

    def transform_records(self, rows: Sequence[Sequence[str]]) -> np.ndarray:
        """Like transform (same column order, as floats), for rows from record_fields."""
        if not hasattr(self, "_lookup"):
            self._lookup = [
                (self.counts[col].to_dict(), {v: i for i, v in enumerate(self.top[col])})
                for col in self.counts
            ]
        out = np.empty((len(rows), 2 * len(self._lookup)), dtype=np.float64)
        for r, row in enumerate(rows):
            for f, ((counts, codes), value) in enumerate(zip(self._lookup, row, strict=True)):
                out[r, 2 * f] = counts.get(value, 0)
                out[r, 2 * f + 1] = codes.get(value, -1)
        return out
