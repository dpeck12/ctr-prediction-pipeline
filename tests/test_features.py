import pandas as pd

from ctr.data import load_avazu
from ctr.features import HashingFeaturizer


def test_transform_shape_and_one_token_per_feature(raw_csv):
    df = load_avazu(raw_csv)
    featurizer = HashingFeaturizer(n_features=2**16)
    x = featurizer.transform(df)

    assert x.shape == (len(df), 2**16)
    # Each feature hashes to one cell; a collision within a row adds 2 to one cell.
    assert (x.sum(axis=1) == featurizer.tokens_per_row).all()


def test_transform_is_deterministic(raw_csv):
    df = load_avazu(raw_csv, nrows=50)
    a = HashingFeaturizer(2**16).transform(df)
    b = HashingFeaturizer(2**16).transform(df)

    assert (a != b).nnz == 0


def test_same_value_in_different_columns_hashes_differently():
    df = pd.DataFrame(
        {"a": ["x"], "b": ["x"], "timestamp": [pd.Timestamp("2014-10-21 05:00")]}
    ).astype({"a": "string", "b": "string"})
    tokens = [col.iloc[0] for col in HashingFeaturizer(columns=["a", "b"]).tokens(df)]

    assert tokens == ["a=x", "b=x", "hour_of_day=5", "day_of_week=1"]


def test_missing_values_become_a_token():
    df = pd.DataFrame(
        {"a": pd.array([None], dtype="string"), "timestamp": [pd.Timestamp("2014-10-21")]}
    )
    tokens = HashingFeaturizer(columns=["a"]).tokens(df)

    assert tokens[0].iloc[0] == "a=<NA>"
