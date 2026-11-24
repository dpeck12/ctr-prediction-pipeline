import numpy as np
import pytest
from sklearn.metrics import log_loss

from ctr.data import load_avazu, split_by_day
from ctr.models import MODEL_NAMES, build_model
from ctr.sample import make_synthetic
from ctr.schema import LABEL

# Small settings so each model trains in seconds on synthetic data.
FAST = {
    "logreg": {"C": 0.01, "n_features": 2**16},
    "lightgbm": {"n_estimators": 200, "learning_rate": 0.1, "num_leaves": 15},
    "dcn": {"embed_dim": 4, "hidden": (16,), "batch_size": 512, "lr": 3e-3, "max_epochs": 3},
}


@pytest.fixture(scope="module")
def split(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "train.csv"
    make_synthetic(n_rows=20_000, n_days=10, seed=3).to_csv(path, index=False)
    return split_by_day(load_avazu(path))


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_beats_constant_on_test_day(split, name):
    model = build_model(name, **FAST[name]).fit(split.train, split.valid)
    p = model.predict_proba(split.test)
    y = split.test[LABEL].to_numpy()
    constant = np.full(len(y), split.train[LABEL].mean())

    assert p.shape == (len(split.test),)
    assert ((p > 0) & (p < 1)).all()
    assert log_loss(y, p) < log_loss(y, constant)
    assert model.name == name
    assert model.params()


def test_lightgbm_records_best_iteration(split):
    model = build_model("lightgbm", **FAST["lightgbm"]).fit(split.train, split.valid)

    assert 1 <= model.params()["best_iteration"] <= 200


def test_dcn_restores_best_epoch(split):
    model = build_model("dcn", **FAST["dcn"]).fit(split.train, split.valid)
    best = min(model.history, key=lambda h: h["valid_log_loss"])
    p_valid = model.predict_proba(split.valid)

    assert model.params()["best_epoch"] == best["epoch"]
    assert log_loss(split.valid[LABEL], p_valid) == pytest.approx(best["valid_log_loss"])


def test_dcn_is_reproducible(split):
    a = build_model("dcn", **FAST["dcn"], seed=7).fit(split.train, split.valid)
    b = build_model("dcn", **FAST["dcn"], seed=7).fit(split.train, split.valid)

    np.testing.assert_allclose(a.predict_proba(split.test), b.predict_proba(split.test))


def test_unknown_model_name():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("xgboost")
