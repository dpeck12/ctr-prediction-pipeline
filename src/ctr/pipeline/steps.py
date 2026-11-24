"""The four daily steps. Each takes the day it runs for and returns a JSON-able dict.

For a run on day D (the day being served and later scored):

1. check_drift      features of day D vs the training days (no labels needed)
2. train_challenger train on days before D-1, early stop on D-1, register as version D
3. promote          make the challenger champion if it beats the champion on D-1
4. evaluate_day     once D's clicks are known, score D with the champion and with
                    a frozen model that was never retrained

The first runs have no champion yet: the first challenger is promoted
automatically and becomes the frozen reference model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import log_loss

from ctr.export import train_bundle
from ctr.metrics import evaluate
from ctr.pipeline.drift import feature_drift, score_psi
from ctr.pipeline.registry import Registry
from ctr.pipeline.store import available_days, load_days
from ctr.schema import LABEL
from ctr.serving.bundle import ModelBundle

FROZEN_KEY = "frozen"


@dataclass(frozen=True)
class Paths:
    data: Path  # daily Parquet files
    registry: Path  # model versions
    reports: Path  # one JSON per step per day

    @classmethod
    def from_root(cls, root: str | Path) -> Paths:
        root = Path(root)
        return cls(
            root / "data" / "daily", root / "artifacts" / "registry", root / "reports" / "pipeline"
        )


@dataclass(frozen=True)
class DayPlan:
    day: date
    train_days: list[date]
    valid_day: date

    @classmethod
    def for_day(cls, day: date, data_dir: Path, min_train_days: int = 2) -> DayPlan:
        """Expanding window: train on every stored day before D-1, validate on D-1."""
        valid_day = day - timedelta(days=1)
        train_days = [d for d in available_days(data_dir) if d < valid_day]
        if len(train_days) < min_train_days:
            raise ValueError(f"{day}: need {min_train_days} training days, have {len(train_days)}")
        if valid_day not in available_days(data_dir):
            raise FileNotFoundError(f"{day}: no data for validation day {valid_day}")
        return cls(day, train_days, valid_day)


def _save(paths: Paths, day: date, step: str, result: dict[str, Any]) -> dict[str, Any]:
    out = paths.reports / str(day)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{step}.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def _as_date(day: date | str) -> date:
    return day if isinstance(day, date) else date.fromisoformat(day)


def check_drift(paths: Paths, day: date | str) -> dict[str, Any]:
    day = _as_date(day)
    plan = DayPlan.for_day(day, paths.data)
    reference = load_days(paths.data, plan.train_days)
    current = load_days(paths.data, [day])
    result = {"day": day, **feature_drift(reference, current)}
    return _save(paths, day, "drift", result)


def train_challenger(
    paths: Paths, day: date | str, model: str = "lightgbm", **model_kwargs: Any
) -> dict[str, Any]:
    day = _as_date(day)
    plan = DayPlan.for_day(day, paths.data)
    df = load_days(paths.data, [*plan.train_days, plan.valid_day])
    # train_bundle splits by day: train on all but the last day, early stop on the
    # last. With no separate test day here, the validation day doubles as its "test".
    bundle = train_bundle(df, model, test_days=0, **model_kwargs)
    valid = bundle.metadata["test_metrics"]

    version = str(day)
    Registry(paths.registry).register(
        version,
        bundle,
        train_days=[str(d) for d in plan.train_days],
        valid_day=str(plan.valid_day),
        valid_log_loss=valid["log_loss"],
    )
    result = {
        "day": day,
        "version": version,
        "model": model,
        "train_days": [str(plan.train_days[0]), str(plan.train_days[-1])],
        "train_rows": bundle.metadata["data"]["train_rows"],
        "valid_day": plan.valid_day,
        "valid": valid,
    }
    return _save(paths, day, "train", result)


def promote(paths: Paths, day: date | str, tolerance: float = 0.001) -> dict[str, Any]:
    """Champion/challenger on the validation day (D-1), which neither model trained on.

    The challenger used D-1 for early stopping, so it gets a small edge; a
    real setup would hold out a separate slice. `tolerance` is the relative log
    loss increase allowed before a challenger is rejected.
    """
    day = _as_date(day)
    registry = Registry(paths.registry)
    challenger = str(day)
    champion = registry.champion

    if champion is None:
        registry.promote(challenger)
        _set_frozen(paths, challenger)
        return _save(paths, day, "promote", {"day": day, "promoted": True, "champion": challenger,
                                              "reason": "first model"})  # fmt: skip

    plan = DayPlan.for_day(day, paths.data)
    valid = load_days(paths.data, [plan.valid_day])
    y = valid[LABEL].to_numpy()
    scores = {
        version: float(
            log_loss(y, registry.load(version).model.predict_proba(valid), labels=[0, 1])
        )
        for version in {champion, challenger}
    }
    promoted = scores[challenger] <= scores[champion] * (1 + tolerance)
    if promoted:
        registry.promote(challenger)
    result = {
        "day": day,
        "promoted": promoted,
        "champion": challenger if promoted else champion,
        "previous_champion": champion,
        "valid_day": plan.valid_day,
        "valid_log_loss": scores,
        "reason": "challenger within tolerance" if promoted else "challenger worse than champion",
    }
    return _save(paths, day, "promote", result)


def _set_frozen(paths: Paths, version: str) -> None:
    marker = paths.registry / f"{FROZEN_KEY}.txt"
    if not marker.exists():
        marker.write_text(version)


def _frozen_version(paths: Paths) -> str | None:
    marker = paths.registry / f"{FROZEN_KEY}.txt"
    return marker.read_text().strip() if marker.exists() else None


def evaluate_day(paths: Paths, day: date | str) -> dict[str, Any]:
    """Score day D now its clicks are known: the served champion vs a never-retrained model."""
    day = _as_date(day)
    registry = Registry(paths.registry)
    champion = registry.champion
    if champion is None:
        raise RuntimeError("No champion to evaluate")

    current = load_days(paths.data, [day])
    plan = DayPlan.for_day(day, paths.data)
    valid = load_days(paths.data, [plan.valid_day])
    y = current[LABEL].to_numpy()

    def score(version: str) -> tuple[dict[str, Any], np.ndarray]:
        bundle: ModelBundle = registry.load(version)
        base_rate = bundle.metadata["data"]["train_ctr"]
        p = bundle.model.predict_proba(current)
        return evaluate(y, p, base_rate), p

    champion_metrics, p_today = score(champion)
    p_valid = registry.load(champion).model.predict_proba(valid)
    result: dict[str, Any] = {
        "day": day,
        "champion": champion,
        "actual_ctr": float(y.mean()),
        "champion_metrics": champion_metrics,
        # Have the model's scores shifted since the day before? Catches drift the
        # feature checks miss, e.g. many small shifts that add up.
        "score_psi_vs_previous_day": round(score_psi(p_valid, p_today), 4),
    }
    frozen = _frozen_version(paths)
    if frozen and frozen != champion:
        result["frozen_version"] = frozen
        result["frozen_metrics"] = score(frozen)[0]
    return _save(paths, day, "evaluate", result)


def run_day(paths: Paths, day: date | str, model: str = "lightgbm", **kwargs: Any) -> dict:
    """All four steps for one day, in order (what the DAG does for one run)."""
    return {
        "drift": check_drift(paths, day),
        "train": train_challenger(paths, day, model, **kwargs),
        "promote": promote(paths, day),
        "evaluate": evaluate_day(paths, day),
    }
