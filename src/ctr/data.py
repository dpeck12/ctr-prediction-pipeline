"""Loading the raw data and splitting it by time."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ctr.schema import CATEGORICAL_COLUMNS, ID, LABEL, RAW_TIME, TIMESTAMP


def load_avazu(
    path: str | Path,
    nrows: int | None = None,
    sample_frac: float | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Read the raw Avazu train file (CSV, optionally gzipped).

    Returns one row per impression with the label, a parsed `timestamp` and the
    categorical features as strings. The row `id` is dropped: it is unique per
    row and carries no signal.

    `sample_frac` keeps a random fraction of rows from across the whole file.
    Use it rather than `nrows` to get a smaller dataset that still covers every
    day: the file is sorted by time, so `nrows` alone only returns the first days.
    """
    skiprows = None
    if sample_frac is not None:
        if not 0 < sample_frac <= 1:
            raise ValueError("sample_frac must be in (0, 1]")
        rng = np.random.default_rng(seed)

        def skiprows(i: int) -> bool:
            return i > 0 and rng.random() >= sample_frac  # row 0 is the header

    dtypes = {col: "string" for col in CATEGORICAL_COLUMNS}
    dtypes |= {LABEL: "int8", RAW_TIME: "string"}
    df = pd.read_csv(
        path,
        nrows=nrows,
        skiprows=skiprows,
        dtype=dtypes,
        usecols=lambda col: col != ID,
    )
    df.insert(1, TIMESTAMP, pd.to_datetime(df.pop(RAW_TIME), format="%y%m%d%H"))
    return df


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def split_by_day(df: pd.DataFrame, valid_days: int = 1, test_days: int = 1) -> TimeSplit:
    """Split on whole days: the last `test_days` are test, the days before are validation.

    Splitting by time rather than at random mirrors production, where a model is
    trained on the past and scored on the future, and keeps later impressions
    from leaking into training.
    """
    if valid_days < 1 or test_days < 1:
        raise ValueError("valid_days and test_days must both be at least 1")

    day = df[TIMESTAMP].dt.normalize()
    days = sorted(day.unique())
    if len(days) < valid_days + test_days + 1:
        raise ValueError(
            f"Need at least {valid_days + test_days + 1} days of data, got {len(days)}"
        )

    test_start = days[-test_days]
    valid_start = days[-(test_days + valid_days)]
    return TimeSplit(
        train=df[day < valid_start].reset_index(drop=True),
        valid=df[(day >= valid_start) & (day < test_start)].reset_index(drop=True),
        test=df[day >= test_start].reset_index(drop=True),
    )
