import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ctr.data import load_avazu
from ctr.export import main as export_main
from ctr.export import train_bundle
from ctr.models import MODEL_NAMES
from ctr.sample import make_synthetic
from ctr.serving.app import create_app
from ctr.serving.benchmark import bench_http, bench_model, percentiles, sample_impressions
from ctr.serving.bundle import ModelBundle
from ctr.serving.schema import MAX_BATCH, PredictRequest, to_frame

FAST = {
    "logreg": {"C": 0.01, "n_features": 2**16},
    "lightgbm": {"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 15},
    "dcn": {"embed_dim": 4, "hidden": (16,), "batch_size": 512, "lr": 3e-3, "max_epochs": 2},
}


@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "train.csv"
    make_synthetic(n_rows=10_000, n_days=10, seed=11).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def bundles(synthetic_csv, tmp_path_factory):
    """One saved bundle per model, keyed by name -> directory."""
    df = load_avazu(synthetic_csv)
    root = tmp_path_factory.mktemp("artifacts")
    return {name: train_bundle(df, name, **FAST[name]).save(root / name) for name in MODEL_NAMES}


@pytest.fixture
def impressions():
    imps = sample_impressions(300, seed=4)
    imps[0]["site_id"] = None  # missing value
    imps[1]["app_id"] = "never-seen-before"  # value not in training data
    return imps


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_bundle_round_trip_keeps_predictions(bundles, impressions, name):
    loaded = ModelBundle.load(bundles[name])
    frame = to_frame(PredictRequest(impressions=impressions).impressions)

    assert loaded.name == name
    assert loaded.metadata["model"] == name
    assert loaded.metadata["data"]["test_days"] == ["2014-10-30", "2014-10-30"]
    assert 0 < loaded.metadata["test_metrics"]["log_loss"] < 1
    assert loaded.model.predict_proba(frame).shape == (len(impressions),)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_fast_path_matches_pandas_path(bundles, impressions, name):
    model = ModelBundle.load(bundles[name]).model
    validated = PredictRequest(impressions=impressions).impressions

    batch = model.predict_proba(to_frame(validated))
    fast = model.predict_records([imp.model_dump() for imp in validated])

    np.testing.assert_allclose(fast, batch, rtol=0, atol=1e-12)


@pytest.fixture(scope="module")
def client(bundles):
    with TestClient(create_app(bundles["lightgbm"])) as c:
        yield c


def test_health_and_model_metadata(client):
    assert client.get("/health").json() == {"status": "ok", "model": "lightgbm"}
    meta = client.get("/model").json()
    assert meta["model"] == "lightgbm"
    assert {"params", "data", "test_metrics", "features", "trained_at"} <= set(meta)


def test_predict_matches_model(client, bundles, impressions):
    response = client.post("/predict", json={"impressions": impressions[:20]})
    model = ModelBundle.load(bundles["lightgbm"]).model
    expected = model.predict_proba(
        to_frame(PredictRequest(impressions=impressions[:20]).impressions)
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "lightgbm"
    np.testing.assert_allclose(body["probabilities"], expected, atol=1e-12)


def test_predict_accepts_numbers_and_missing_fields(client):
    response = client.post("/predict", json={"impressions": [{"hour": "14103014", "C1": 1005}]})

    assert response.status_code == 200
    assert 0 < response.json()["probabilities"][0] < 1


@pytest.mark.parametrize(
    "body",
    [
        {"impressions": []},
        {"impressions": [{"hour": "14103099"}]},  # hour 99
        {"impressions": [{"hour": "2014-10-30"}]},
        {"impressions": [{"C1": "1005"}]},  # hour missing
        {"impressions": [{"hour": "14103014", "unknown_field": "x"}]},
        {"impressions": [{"hour": "14103014"}] * (MAX_BATCH + 1)},
    ],
)
def test_predict_rejects_invalid_requests(client, body):
    assert client.post("/predict", json=body).status_code == 422


def test_percentiles():
    stats = percentiles(list(range(1, 101)))

    assert stats["n"] == 100
    assert stats["p50_ms"] == pytest.approx(50.5)
    assert stats["p99_ms"] == pytest.approx(99.01)


def test_bench_model_times_both_paths(bundles, impressions):
    result = bench_model(ModelBundle.load(bundles["logreg"]), impressions[:10], n=20, warmup=2)

    assert set(result) == {"pandas", "fast"}
    assert result["fast"]["n"] == 20


def test_bench_http_starts_a_real_server(bundles, impressions):
    result = bench_http(bundles["logreg"], impressions[:10], n=10, warmup=2)

    assert result["n"] == 10
    assert result["p50_ms"] > 0


def test_export_cli_saves_bundles(synthetic_csv, tmp_path):
    out = tmp_path / "artifacts"
    exit_code = export_main(
        ["--data", str(synthetic_csv), "--out", str(out), "--sample-frac", "1",
         "--models", "logreg", "--model-kwargs", json.dumps(FAST)]
    )  # fmt: skip

    assert exit_code == 0
    assert {p.name for p in (out / "logreg").iterdir()} == {"model.joblib", "metadata.json"}
    assert json.loads((out / "logreg" / "metadata.json").read_text())["data"]["sample_frac"] == 1
