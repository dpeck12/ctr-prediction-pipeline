import json

import mlflow
import pytest

from ctr.data import load_avazu
from ctr.sample import make_synthetic
from ctr.tracking import flatten_metrics
from ctr.train import main, run_experiment

FAST = {
    "logreg": {"C": 0.01, "n_features": 2**16},
    "lightgbm": {"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 15},
}


@pytest.fixture
def synthetic_csv(tmp_path):
    path = tmp_path / "train.csv"
    make_synthetic(n_rows=10_000, n_days=10, seed=5).to_csv(path, index=False)
    return path


def test_run_experiment_reports_raw_and_calibrated_models(synthetic_csv):
    result = run_experiment(load_avazu(synthetic_csv), ["logreg", "lightgbm"], FAST)
    models = result.summary["models"]

    assert set(models) == {
        "constant", "logreg", "logreg + Platt", "lightgbm", "lightgbm + Platt",
    }  # fmt: skip
    for name in ("logreg", "lightgbm"):
        assert models[name]["fit_seconds"] >= 0
        assert models[name]["batch_predict_us_per_row"] > 0
        # Platt keeps the ranking, so AUC is unchanged.
        assert models[f"{name} + Platt"]["test"]["auc"] == pytest.approx(
            models[name]["test"]["auc"]
        )
    assert set(result.test_predictions) == {"constant", "logreg", "lightgbm"}


def test_run_experiment_without_calibration(synthetic_csv):
    result = run_experiment(load_avazu(synthetic_csv), ["logreg"], FAST, calibrate=False)

    assert set(result.summary["models"]) == {"constant", "logreg"}


def test_cli_writes_report_and_logs_to_mlflow(synthetic_csv, tmp_path):
    out = tmp_path / "report"
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"

    exit_code = main(
        ["--data", str(synthetic_csv), "--out", str(out), "--sample-frac", "1",
         "--models", "logreg", "--model-kwargs", json.dumps(FAST), "--mlflow-uri", uri]
    )  # fmt: skip

    assert exit_code == 0
    assert {p.name for p in out.iterdir()} == {"metrics.json", "results.md", "reliability.png"}
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["ctr-prediction"])
    assert set(runs["tags.mlflow.runName"]) == {"constant", "logreg", "logreg + Platt"}
    logreg = runs[runs["tags.mlflow.runName"] == "logreg"].iloc[0]
    assert logreg["metrics.test_log_loss"] > 0
    assert logreg["params.C"] == "0.01"


def test_cli_can_skip_mlflow(synthetic_csv, tmp_path):
    exit_code = main(
        ["--data", str(synthetic_csv), "--out", str(tmp_path / "r"), "--sample-frac", "1",
         "--models", "logreg", "--model-kwargs", json.dumps(FAST), "--no-mlflow",
         "--no-calibrate"]
    )  # fmt: skip

    assert exit_code == 0
    assert not (tmp_path / "mlflow.db").exists()


def test_flatten_metrics_drops_nan_and_prefixes():
    flat = flatten_metrics("test", {"log_loss": 0.4, "auc": float("nan"), "rows": 10, "x": "a"})

    assert flat == {"test_log_loss": 0.4, "test_rows": 10.0}
