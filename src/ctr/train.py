"""Phase 2: train several CTR models on the same time split and compare them.

    uv run python -m ctr.train --data data/raw/train.gz

Each model is trained on days 1-8 (early stopping on day 9 where the model
supports it), then Platt-calibrated on day 9 and scored on day 10. Writes
metrics.json, results.md and reliability.png to reports/experiment/ and logs
one MLflow run per model (browse with `uv run mlflow ui --backend-store-uri
sqlite:///mlflow.db`).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ctr.calibration import PlattCalibrator
from ctr.data import load_avazu, split_by_day
from ctr.metrics import evaluate
from ctr.models import MODEL_NAMES, build_model
from ctr.report import ExperimentResult, write_report
from ctr.schema import LABEL, TIMESTAMP
from ctr.tracking import DEFAULT_TRACKING_URI, flatten_metrics, log_run

log = logging.getLogger(__name__)


def _day_range(df: pd.DataFrame) -> list[str]:
    days = df[TIMESTAMP].dt.normalize()
    return [str(days.min().date()), str(days.max().date())]


def run_experiment(
    df: pd.DataFrame,
    model_names: Sequence[str] = MODEL_NAMES,
    model_kwargs: dict[str, dict[str, Any]] | None = None,
    calibrate: bool = True,
) -> ExperimentResult:
    model_kwargs = model_kwargs or {}
    split = split_by_day(df)
    y_valid = split.valid[LABEL].to_numpy()
    y_test = split.test[LABEL].to_numpy()
    base_rate = float(split.train[LABEL].mean())

    preds: dict[str, np.ndarray] = {"constant": np.full(len(y_test), base_rate)}
    models: dict[str, dict] = {
        "constant": {
            "params": {"ctr": base_rate},
            "valid": evaluate(y_valid, np.full(len(y_valid), base_rate), base_rate),
            "test": evaluate(y_test, preds["constant"], base_rate),
        }
    }

    for name in model_names:
        model = build_model(name, **model_kwargs.get(name, {}))
        log.info("Training %s", name)
        t0 = time.perf_counter()
        model.fit(split.train, split.valid)
        fit_seconds = time.perf_counter() - t0

        p_valid = model.predict_proba(split.valid)
        t0 = time.perf_counter()
        p_test = model.predict_proba(split.test)
        predict_us = (time.perf_counter() - t0) / len(y_test) * 1e6

        preds[name] = p_test
        models[name] = {
            "params": model.params(),
            "fit_seconds": round(fit_seconds, 1),
            "batch_predict_us_per_row": round(predict_us, 2),
            "valid": evaluate(y_valid, p_valid, base_rate),
            "test": evaluate(y_test, p_test, base_rate),
        }
        if history := getattr(model, "history", None):
            models[name]["history"] = history
        log.info("%s: test log loss %.5f", name, models[name]["test"]["log_loss"])

        if calibrate:
            calibrator = PlattCalibrator().fit(p_valid, y_valid)
            models[f"{name} + Platt"] = {
                "params": calibrator.params(),
                "test": evaluate(y_test, calibrator.transform(p_test), base_rate),
            }

    summary = {
        "data": {
            "rows": {"train": len(split.train), "valid": len(y_valid), "test": len(y_test)},
            "days": {
                "train": _day_range(split.train),
                "valid": _day_range(split.valid),
                "test": _day_range(split.test),
            },
            "train_ctr": base_rate,
        },
        "models": models,
    }
    return ExperimentResult(summary=summary, y_test=y_test, test_predictions=preds)


def log_to_mlflow(result: ExperimentResult, out_dir: Path, tracking_uri: str) -> None:
    data = result.summary["data"]
    for name, entry in result.summary["models"].items():
        metrics: dict[str, float] = {}
        for split_name in ("valid", "test"):
            if split_name in entry:
                metrics |= flatten_metrics(split_name, entry[split_name])
        for key in ("fit_seconds", "batch_predict_us_per_row"):
            if key in entry:
                metrics[key] = float(entry[key])
        params = {
            **entry.get("params", {}),
            "train_rows": data["rows"]["train"],
            "sample_frac": data.get("sample_frac"),
        }
        log_run(name, params, metrics, artifacts_dir=out_dir, tracking_uri=tracking_uri)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/raw/train.gz"))
    parser.add_argument("--out", type=Path, default=Path("reports/experiment"))
    parser.add_argument("--sample-frac", type=float, default=0.05)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--mlflow-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--model-kwargs",
        type=json.loads,
        default={},
        help='JSON per model, e.g. \'{"dcn": {"max_epochs": 3}}\'',
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    t0 = time.perf_counter()
    sample_frac = None if args.sample_frac >= 1 else args.sample_frac
    df = load_avazu(args.data, sample_frac=sample_frac, seed=args.seed)
    log.info("Loaded %s rows in %.1fs", f"{len(df):,}", time.perf_counter() - t0)

    result = run_experiment(
        df, args.models, model_kwargs=args.model_kwargs, calibrate=not args.no_calibrate
    )
    result.summary["data"]["sample_frac"] = args.sample_frac
    write_report(result, args.out)
    log.info("Wrote report to %s", args.out)

    if not args.no_mlflow:
        log_to_mlflow(result, args.out, args.mlflow_uri)
        log.info("Logged %d runs to MLflow at %s", len(result.summary["models"]), args.mlflow_uri)

    print((args.out / "results.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
