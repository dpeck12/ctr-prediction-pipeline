"""Train models on the standard split and save them as bundles for serving.

    uv run python -m ctr.export --data data/raw/train.gz --models lightgbm dcn

Saves artifacts/<model>/ with model.joblib and metadata.json (training days,
settings and test-day metrics). Point the API at one with MODEL_PATH.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ctr import __version__
from ctr.data import load_avazu, split_by_day
from ctr.encoders import FIELDS
from ctr.metrics import evaluate
from ctr.models import MODEL_NAMES, build_model
from ctr.schema import LABEL, TIMESTAMP
from ctr.serving.bundle import ModelBundle

log = logging.getLogger(__name__)


def _days(df: pd.DataFrame) -> list[str]:
    days = df[TIMESTAMP].dt.normalize()
    return [str(days.min().date()), str(days.max().date())]


def train_bundle(df: pd.DataFrame, name: str, test_days: int = 1, **kwargs: Any) -> ModelBundle:
    """Fit one model on the training days and score it on the held-out days.

    With test_days=1 (the default) the last day is the test day and the day
    before it is for early stopping. With test_days=0 there is no test day: the
    last day is used for early stopping and its metrics are reported instead.
    The daily pipeline uses that, because its test day hasn't happened yet.
    """
    if test_days:
        split = split_by_day(df, test_days=test_days)
        train, valid, test = split.train, split.valid, split.test
    else:
        day = df[TIMESTAMP].dt.normalize()
        if day.nunique() < 2:
            raise ValueError("Need at least 2 days of data")
        train = df[day < day.max()].reset_index(drop=True)
        valid = test = df[day == day.max()].reset_index(drop=True)

    t0 = time.perf_counter()
    model = build_model(name, **kwargs).fit(train, valid)
    fit_seconds = time.perf_counter() - t0

    base_rate = float(train[LABEL].mean())
    test_metrics = evaluate(test[LABEL].to_numpy(), model.predict_proba(test), base_rate)
    metadata = {
        "model": name,
        "package_version": __version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "params": model.params(),
        "fit_seconds": round(fit_seconds, 1),
        "features": list(FIELDS),
        "data": {
            "train_days": _days(train),
            "valid_days": _days(valid),
            "test_days": _days(test),
            "train_rows": len(train),
            "train_ctr": base_rate,
        },
        "test_metrics": test_metrics,
    }
    return ModelBundle(model=model, metadata=metadata)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/raw/train.gz"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--sample-frac", type=float, default=0.05)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=["lightgbm"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-kwargs", type=json.loads, default={})
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    sample_frac = None if args.sample_frac >= 1 else args.sample_frac
    df = load_avazu(args.data, sample_frac=sample_frac, seed=args.seed)
    log.info("Loaded %s rows", f"{len(df):,}")

    for name in args.models:
        bundle = train_bundle(df, name, **args.model_kwargs.get(name, {}))
        bundle.metadata["data"]["sample_frac"] = args.sample_frac
        path = bundle.save(args.out / name)
        log.info(
            "Saved %s to %s (test log loss %.5f)",
            name,
            path,
            bundle.metadata["test_metrics"]["log_loss"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
