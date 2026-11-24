import numpy as np
import pandas as pd

from ctr.encoders import FIELDS, GBMEncoder, VocabEncoder, make_fields
from ctr.schema import TIMESTAMP


def _fields(**columns):
    return pd.DataFrame({k: pd.array(v, dtype="string") for k, v in columns.items()})


def test_make_fields_adds_time_fields_as_strings(raw_csv):
    from ctr.data import load_avazu

    df = load_avazu(raw_csv, nrows=10)
    fields = make_fields(df)

    assert tuple(fields.columns) == FIELDS
    assert (fields["hour_of_day"] == df[TIMESTAMP].dt.hour.astype("string")).all()
    assert all(fields[col].dtype == "string" for col in fields.columns)


def test_vocab_encoder_maps_rare_and_unseen_to_zero():
    train = _fields(a=["x"] * 5 + ["y"] * 2)
    encoder = VocabEncoder(min_count=3).fit(train)
    codes = encoder.transform(_fields(a=["x", "y", "never-seen"]))

    assert encoder.cardinalities == [2]  # "x" plus the unknown row
    assert codes[:, 0].tolist() == [1, 0, 0]
    assert codes.dtype == np.int64


def test_vocab_encoder_codes_fit_each_field_cardinality(raw_csv):
    from ctr.data import load_avazu

    fields = make_fields(load_avazu(raw_csv))
    encoder = VocabEncoder(min_count=1).fit(fields)
    codes = encoder.transform(fields)

    assert codes.shape == (len(fields), len(FIELDS))
    assert (codes >= 1).all()  # min_count=1: every training value is known
    assert (codes.max(axis=0) < np.array(encoder.cardinalities)).all()


def test_gbm_encoder_counts_and_caps_categories():
    train = _fields(a=["x"] * 3 + ["y"] * 2 + ["z"])
    encoder = GBMEncoder(max_categories=2).fit(train)
    out = encoder.transform(_fields(a=["x", "y", "z", "new"]))

    assert out["a_count"].tolist() == [3, 2, 1, 0]
    assert out["a_cat"].tolist() == [0, 1, -1, -1]  # z is outside the top 2; -1 = missing
    assert encoder.categorical_columns == ["a_cat"]
