import pandas as pd
import pytest

from ctr.data import load_avazu, split_by_day
from ctr.sample import make_synthetic
from ctr.schema import CATEGORICAL_COLUMNS, ID, LABEL, RAW_TIME, TIMESTAMP


def test_load_parses_time_and_drops_id(raw_csv):
    df = load_avazu(raw_csv)

    assert ID not in df.columns
    assert RAW_TIME not in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df[TIMESTAMP])
    assert df[TIMESTAMP].min() == pd.Timestamp("2014-10-21")
    assert set(CATEGORICAL_COLUMNS) <= set(df.columns)
    assert all(df[col].dtype == "string" for col in CATEGORICAL_COLUMNS)
    assert set(df[LABEL].unique()) <= {0, 1}


def test_load_respects_nrows(raw_csv):
    assert len(load_avazu(raw_csv, nrows=100)) == 100


def test_sample_frac_keeps_a_fraction_from_every_day(raw_csv):
    full = load_avazu(raw_csv)
    sample = load_avazu(raw_csv, sample_frac=0.3, seed=0)

    assert 0.2 * len(full) < len(sample) < 0.4 * len(full)
    assert sample[TIMESTAMP].dt.normalize().nunique() == full[TIMESTAMP].dt.normalize().nunique()


def test_sample_frac_is_reproducible(raw_csv):
    a = load_avazu(raw_csv, sample_frac=0.3, seed=7)
    b = load_avazu(raw_csv, sample_frac=0.3, seed=7)

    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("frac", [0, -0.1, 1.5])
def test_sample_frac_rejects_out_of_range(raw_csv, frac):
    with pytest.raises(ValueError, match="sample_frac"):
        load_avazu(raw_csv, sample_frac=frac)


def test_split_by_day_is_ordered_and_complete(raw_csv):
    df = load_avazu(raw_csv)
    split = split_by_day(df, valid_days=1, test_days=1)

    assert len(split.train) + len(split.valid) + len(split.test) == len(df)
    assert split.train[TIMESTAMP].max() < split.valid[TIMESTAMP].min()
    assert split.valid[TIMESTAMP].max() < split.test[TIMESTAMP].min()
    assert split.test[TIMESTAMP].dt.normalize().nunique() == 1
    assert split.valid[TIMESTAMP].dt.normalize().nunique() == 1


def test_split_by_day_supports_multi_day_windows(raw_csv):
    split = split_by_day(load_avazu(raw_csv), valid_days=2, test_days=3)

    assert split.valid[TIMESTAMP].dt.normalize().nunique() == 2
    assert split.test[TIMESTAMP].dt.normalize().nunique() == 3


def test_split_by_day_rejects_too_few_days(tmp_path):
    path = tmp_path / "short.csv"
    make_synthetic(n_rows=200, n_days=2).to_csv(path, index=False)

    with pytest.raises(ValueError, match="at least 3 days"):
        split_by_day(load_avazu(path))


@pytest.mark.parametrize("valid_days, test_days", [(0, 1), (1, 0)])
def test_split_by_day_rejects_empty_windows(raw_csv, valid_days, test_days):
    with pytest.raises(ValueError, match="at least 1"):
        split_by_day(load_avazu(raw_csv), valid_days=valid_days, test_days=test_days)


def test_synthetic_click_rate_is_plausible():
    df = make_synthetic(n_rows=20_000, seed=2)
    ctr = df[LABEL].mean()

    assert 0.1 < ctr < 0.35
    # Banner position 1 is built to click more often than position 0.
    by_pos = df.groupby("banner_pos")[LABEL].mean()
    assert by_pos["1"] > by_pos["0"]
