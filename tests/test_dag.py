"""Runs the real Airflow DAG for two days on synthetic data.

Skipped unless Airflow is installed (it isn't part of the dev environment: Airflow
doesn't support Windows). CI runs it in its own job; see .github/workflows/ci.yml.
"""

import json
import os
from pathlib import Path

import pytest

# Not "airflow": the repo's own airflow/ folder would satisfy that as a namespace package.
pytest.importorskip("airflow.sdk")

DAGS = Path(__file__).resolve().parents[1] / "airflow" / "dags"


@pytest.fixture(scope="module")
def dag(tmp_path_factory):
    from ctr.data import load_avazu
    from ctr.pipeline.steps import Paths
    from ctr.pipeline.store import write_days
    from ctr.sample import make_synthetic

    root = tmp_path_factory.mktemp("ctr")
    csv = root / "train.csv"
    make_synthetic(n_rows=20_000, n_days=10, seed=6).to_csv(csv, index=False)
    write_days(load_avazu(csv), Paths.from_root(root).data)
    os.environ["CTR_ROOT"] = str(root)
    os.environ["CTR_MODEL_KWARGS"] = json.dumps({"n_estimators": 50})

    from airflow.dag_processing.dagbag import DagBag

    bag = DagBag(dag_folder=str(DAGS), include_examples=False)
    assert bag.import_errors == {}
    return bag.get_dag("ctr_daily"), root


def test_dag_structure(dag):
    d, _ = dag
    assert set(d.task_dict) == {"check_drift", "train_challenger", "promote", "evaluate"}
    assert d.task_dict["evaluate"].upstream_task_ids == {"promote"}
    assert d.task_dict["train_challenger"].upstream_task_ids == {"check_drift"}
    assert d.catchup is True


def test_two_daily_runs(dag):
    import pendulum

    d, root = dag
    for day in (24, 25):
        run = d.test(logical_date=pendulum.datetime(2014, 10, day, tz="UTC"))
        assert run.state == "success", f"run for 2014-10-{day} ended {run.state}"

    from ctr.pipeline.registry import Registry
    from ctr.pipeline.steps import Paths

    paths = Paths.from_root(root)
    assert set(Registry(paths.registry).versions()) == {"2014-10-24", "2014-10-25"}
    report = paths.reports / "2014-10-25"
    assert {p.name for p in report.iterdir()} == {
        "drift.json", "train.json", "promote.json", "evaluate.json",
    }  # fmt: skip
    evaluation = json.loads((report / "evaluate.json").read_text())
    assert 0 < evaluation["champion_metrics"]["log_loss"] < 1
