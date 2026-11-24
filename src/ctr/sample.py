"""Synthetic data in the raw Avazu format, for tests and CI.

The real dataset is several GB and sits behind the Kaggle competition rules, so
CI never downloads it. This generator produces a small file with the same columns
and formats, and a click rate that depends on a couple of features so models have
something to learn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ctr.schema import CATEGORICAL_COLUMNS, ID, LABEL, RAW_COLUMNS, RAW_TIME

START = pd.Timestamp("2014-10-21")


def make_synthetic(
    n_rows: int = 5_000, n_days: int = 10, base_ctr: float = 0.17, seed: int = 0
) -> pd.DataFrame:
    """Return a DataFrame shaped like the raw Avazu train file."""
    rng = np.random.default_rng(seed)

    hours = rng.integers(0, n_days * 24, size=n_rows)
    timestamps = START + pd.to_timedelta(np.sort(hours), unit="h")

    df = pd.DataFrame(
        {col: rng.integers(0, 20, size=n_rows).astype(str) for col in CATEGORICAL_COLUMNS}
    )
    df["banner_pos"] = rng.choice(["0", "1"], size=n_rows, p=[0.7, 0.3])
    df["device_type"] = rng.choice(["0", "1", "4"], size=n_rows, p=[0.1, 0.8, 0.1])

    # Click probability rises for banner position 1 and device type 0.
    logit = (
        np.log(base_ctr / (1 - base_ctr))
        + 0.8 * (df["banner_pos"] == "1")
        + 0.5 * (df["device_type"] == "0")
    )
    df[LABEL] = rng.binomial(1, 1 / (1 + np.exp(-logit))).astype("int8")
    df[RAW_TIME] = timestamps.strftime("%y%m%d%H")
    df[ID] = np.arange(n_rows).astype(str)
    return df[list(RAW_COLUMNS)]
