"""Daily CTR model retraining and monitoring.

One run per day D, from 24 to 30 October 2014 (catchup replays them in order):

    check_drift ──> train_challenger ──> promote ──> evaluate

- check_drift: PSI of every feature, day D vs the training days (no labels needed)
- train_challenger: train on all days before D-1, early stop on D-1, register as version D
- promote: the challenger replaces the champion if it does as well on D-1
- evaluate: once D's clicks are known, score D with the champion and a never-retrained model

The logic lives in ctr.pipeline.steps; this file only wires and schedules it.
Configuration: CTR_ROOT (folder holding data/, artifacts/, reports/), CTR_MODEL
(default lightgbm) and CTR_MODEL_KWARGS (JSON).
"""

from __future__ import annotations

import json
import os
from datetime import timedelta

import pendulum
from airflow.sdk import dag, get_current_context, task

ROOT = os.environ.get("CTR_ROOT", "/opt/ctr")
MODEL = os.environ.get("CTR_MODEL", "lightgbm")
MODEL_KWARGS = json.loads(os.environ.get("CTR_MODEL_KWARGS", "{}"))


def _paths():
    from ctr.pipeline.steps import Paths

    return Paths.from_root(ROOT)


def _day() -> str:
    """The day this run is for: its logical date (set for scheduled and backfill runs)."""
    context = get_current_context()
    logical_date = context.get("logical_date")
    if logical_date is None:  # manual runs in Airflow 3 can have no logical date
        logical_date = context["dag_run"].run_after
    return logical_date.date().isoformat()


@dag(
    dag_id="ctr_daily",
    schedule="@daily",
    start_date=pendulum.datetime(2014, 10, 24, tz="UTC"),
    end_date=pendulum.datetime(2014, 10, 30, tz="UTC"),
    catchup=True,
    max_active_runs=1,  # each day's promotion depends on the previous day's champion
    default_args={"retries": 1, "retry_delay": timedelta(minutes=1)},
    tags=["ctr", "ml"],
    doc_md=__doc__,
)
def ctr_daily():
    @task
    def check_drift() -> dict:
        from ctr.pipeline.steps import check_drift

        result = check_drift(_paths(), _day())
        return {k: result[k] for k in ("max_psi", "max_psi_field", "level", "moderate_or_worse")}

    @task
    def train_challenger() -> dict:
        from ctr.pipeline.steps import train_challenger

        result = train_challenger(_paths(), _day(), MODEL, **MODEL_KWARGS)
        return {"version": result["version"], "valid_log_loss": result["valid"]["log_loss"]}

    @task
    def promote(challenger: dict) -> dict:
        from ctr.pipeline.steps import promote

        result = promote(_paths(), challenger["version"])
        return {"promoted": result["promoted"], "champion": result["champion"]}

    @task
    def evaluate(promotion: dict) -> dict:
        from ctr.pipeline.steps import evaluate_day

        result = evaluate_day(_paths(), _day())
        metrics = result["champion_metrics"]
        return {
            "champion": promotion["champion"],
            "log_loss": metrics["log_loss"],
            "calibration_ratio": metrics["calibration_ratio"],
            "score_psi": result["score_psi_vs_previous_day"],
        }

    drift = check_drift()
    challenger = train_challenger()
    drift >> challenger
    evaluate(promote(challenger))


ctr_daily()
