import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from ctr.data import load_avazu
from ctr.pipeline.__main__ import main as pipeline_main
from ctr.pipeline.drift import categorical_psi, feature_drift, level, psi, score_psi
from ctr.pipeline.registry import Registry
from ctr.pipeline.steps import DayPlan, Paths, evaluate_day, promote, run_day, train_challenger
from ctr.pipeline.store import available_days, load_days, write_days
from ctr.sample import make_synthetic
from ctr.schema import TIMESTAMP

FAST = {"n_estimators": 50, "learning_rate": 0.1, "num_leaves": 15}


@pytest.fixture
def root(tmp_path):
    csv = tmp_path / "train.csv"
    make_synthetic(n_rows=12_000, n_days=10, seed=13).to_csv(csv, index=False)
    write_days(load_avazu(csv), Paths.from_root(tmp_path).data)
    return tmp_path


@pytest.fixture
def paths(root):
    return Paths.from_root(root)


# --- store ---------------------------------------------------------------


def test_daily_store_round_trip(paths):
    days = available_days(paths.data)
    df = load_days(paths.data, days[:2])

    assert days[0] == date(2014, 10, 21) and len(days) == 10
    assert set(df[TIMESTAMP].dt.date) == set(days[:2])
    assert df["site_id"].dtype == "string"


def test_load_days_reports_missing_days(paths):
    with pytest.raises(FileNotFoundError, match="2014-11-01"):
        load_days(paths.data, [date(2014, 11, 1)])
    with pytest.raises(ValueError):
        load_days(paths.data, [])


# --- drift ---------------------------------------------------------------


def test_psi_is_zero_for_identical_and_grows_with_shift():
    same = psi([50, 30, 20], [50, 30, 20])
    small = psi([50, 30, 20], [45, 33, 22])
    large = psi([50, 30, 20], [10, 30, 60])

    assert same == pytest.approx(0)
    assert 0 < small < 0.1 < large
    assert level(small) == "stable" and level(0.15) == "moderate" and level(large) == "major"


def test_categorical_psi_buckets_unseen_values_as_other():
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    cur = pd.Series(["a"] * 50 + ["new"] * 50)  # half the traffic is a value never seen

    assert categorical_psi(ref, ref) == pytest.approx(0)
    assert categorical_psi(ref, cur, top_k=2) > 0.25


def test_score_psi_detects_a_shifted_score_distribution():
    rng = np.random.default_rng(0)
    ref = rng.beta(2, 8, 10_000)

    assert score_psi(ref, rng.beta(2, 8, 10_000)) < 0.02
    assert score_psi(ref, rng.beta(4, 6, 10_000)) > 0.25


def test_feature_drift_flags_a_changed_field(paths):
    ref = load_days(paths.data, available_days(paths.data)[:3])
    cur = load_days(paths.data, available_days(paths.data)[3:4]).copy()
    calm = feature_drift(ref, cur)
    cur["site_id"] = "brand-new-site"
    shifted = feature_drift(ref, cur)

    assert "day_of_week" not in calm["psi"]
    assert calm["level"] == "stable"
    assert shifted["max_psi_field"] == "site_id"
    assert shifted["level"] == "major"
    assert "site_id" in shifted["moderate_or_worse"]


# --- registry and planning ------------------------------------------------


def test_registry_promote_requires_known_version(tmp_path):
    with pytest.raises(KeyError):
        Registry(tmp_path).promote("nope")
    assert Registry(tmp_path).champion is None


def test_day_plan_uses_expanding_window(paths):
    plan = DayPlan.for_day(date(2014, 10, 26), paths.data)

    assert plan.valid_day == date(2014, 10, 25)
    assert plan.train_days == [date(2014, 10, d) for d in (21, 22, 23, 24)]
    with pytest.raises(ValueError, match="need 2 training days"):
        DayPlan.for_day(date(2014, 10, 23), paths.data)
    with pytest.raises(FileNotFoundError, match="validation day"):
        DayPlan.for_day(date(2014, 11, 5), paths.data)


# --- steps ---------------------------------------------------------------


def test_three_days_of_the_pipeline(paths):
    for day in ("2014-10-24", "2014-10-25", "2014-10-26"):
        result = run_day(paths, day, "lightgbm", **FAST)

    registry = Registry(paths.registry)
    assert set(registry.versions()) == {"2014-10-24", "2014-10-25", "2014-10-26"}
    assert registry.info("2014-10-26")["train_days"][-1] == "2014-10-24"
    assert registry.info("2014-10-26")["path"] == "2014-10-26"  # relative, so it can move

    first = json.loads((paths.reports / "2014-10-24" / "promote.json").read_text())
    assert first == {**first, "promoted": True, "reason": "first model"}
    assert result["promote"]["previous_champion"] is not None
    assert set(result["promote"]["valid_log_loss"]) <= {"2014-10-24", "2014-10-25", "2014-10-26"}

    evaluation = result["evaluate"]
    assert evaluation["champion"] == registry.champion
    assert 0 < evaluation["actual_ctr"] < 1
    if registry.champion != "2014-10-24":
        assert evaluation["frozen_version"] == "2014-10-24"
        assert "frozen_metrics" in evaluation


def test_promote_rejects_a_worse_challenger(paths):
    train_challenger(paths, "2014-10-25", **FAST)
    promote(paths, "2014-10-25")
    # A one-tree model is much worse than the champion on the validation day.
    train_challenger(paths, "2014-10-26", n_estimators=1, learning_rate=0.01)
    result = promote(paths, "2014-10-26")

    assert result["promoted"] is False
    assert result["champion"] == "2014-10-25"
    assert Registry(paths.registry).champion == "2014-10-25"


def test_evaluate_needs_a_champion(paths):
    with pytest.raises(RuntimeError, match="No champion"):
        evaluate_day(paths, "2014-10-25")


# --- CLI -----------------------------------------------------------------


def test_cli_prepare_and_replay(tmp_path):
    csv = tmp_path / "train.csv"
    make_synthetic(n_rows=8_000, n_days=6, seed=21).to_csv(csv, index=False)
    root = ["--root", str(tmp_path)]

    assert pipeline_main([*root, "prepare", "--data", str(csv), "--sample-frac", "1"]) == 0
    assert pipeline_main([*root, "replay", "--model-kwargs", json.dumps(FAST)]) == 0

    reports = tmp_path / "reports" / "pipeline"
    rows = json.loads((reports / "summary.json").read_text())
    assert [r["day"] for r in rows] == ["2014-10-24", "2014-10-25", "2014-10-26"]
    assert (reports / "timeline.png").exists()
    assert "| 2014-10-26 |" in (reports / "summary.md").read_text()
