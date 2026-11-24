import json

import pytest

from ctr.data import load_avazu
from ctr.sample import make_synthetic
from ctr.train_baseline import main, train_and_evaluate


@pytest.fixture
def synthetic_df(tmp_path):
    path = tmp_path / "train.csv"
    make_synthetic(n_rows=20_000, n_days=10, seed=3).to_csv(path, index=False)
    return load_avazu(path)


def test_logistic_regression_beats_constant(synthetic_df):
    result = train_and_evaluate(synthetic_df, c_grid=[0.001, 0.01], n_features=2**16)
    models = result.summary["models"]
    logreg, const = models["logistic_regression"]["test"], models["constant"]["test"]

    assert logreg["log_loss"] < const["log_loss"]
    assert logreg["normalized_entropy"] < 1.0
    assert logreg["auc"] > 0.55
    assert models["logistic_regression"]["C"] in (0.001, 0.01)
    assert len(result.summary["c_search"]) == 2


def test_summary_records_the_time_split(synthetic_df):
    summary = train_and_evaluate(synthetic_df, c_grid=[1.0], n_features=2**16).summary

    assert summary["data"]["days"]["test"] == ["2014-10-30", "2014-10-30"]
    assert summary["data"]["days"]["valid"] == ["2014-10-29", "2014-10-29"]
    assert summary["data"]["days"]["train"] == ["2014-10-21", "2014-10-28"]
    assert sum(summary["data"]["rows"].values()) == len(synthetic_df)


def test_cli_writes_report(tmp_path):
    data = tmp_path / "train.gz"
    make_synthetic(n_rows=5_000, n_days=10, seed=4).to_csv(data, index=False, compression="gzip")
    out = tmp_path / "report"

    exit_code = main(
        ["--data", str(data), "--out", str(out), "--sample-frac", "1",
         "--n-features", "65536", "--c-grid", "1.0"]
    )  # fmt: skip

    assert exit_code == 0
    assert {p.name for p in out.iterdir()} == {"metrics.json", "results.md", "reliability.png"}
    summary = json.loads((out / "metrics.json").read_text())
    assert set(summary["models"]) == {"constant", "logistic_regression"}
    assert "| logistic_regression |" in (out / "results.md").read_text()


def test_empty_c_grid_is_rejected(synthetic_df):
    with pytest.raises(AssertionError, match="c_grid"):
        train_and_evaluate(synthetic_df, c_grid=[], n_features=2**16)
