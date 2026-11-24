"""Phase 1 baseline: hashed logistic regression, compared against a constant predictor.

Run on the real data (samples 5% of rows, about 2M, from across all 10 days):

    uv run python -m ctr.train_baseline --data data/raw/train.gz

Writes metrics.json, results.md and reliability.png to reports/baseline/.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from ctr.data import load_avazu, split_by_day
from ctr.features import HashingFeaturizer
from ctr.metrics import evaluate
from ctr.report import ExperimentResult, write_report
from ctr.schema import LABEL, TIMESTAMP

log = logging.getLogger(__name__)

DEFAULT_C_GRID = (0.001, 0.01, 0.1, 1.0)

BaselineResult = ExperimentResult  # kept so existing imports keep working


def _day_range(df: pd.DataFrame) -> list[str]:
    days = df[TIMESTAMP].dt.normalize()
    return [str(days.min().date()), str(days.max().date())]


def train_and_evaluate(
    df: pd.DataFrame,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    n_features: int = 2**20,
) -> BaselineResult:
    """Split by day, fit the models on train, pick C on validation, report on test.

    No class weighting or resampling: both would distort the predicted
    probabilities, and calibration matters as much as ranking here.
    """
    split = split_by_day(df)
    featurizer = HashingFeaturizer(n_features)

    t0 = time.perf_counter()
    x_train, x_valid, x_test = (
        featurizer.transform(part) for part in (split.train, split.valid, split.test)
    )
    y_train, y_valid, y_test = (
        part[LABEL].to_numpy() for part in (split.train, split.valid, split.test)
    )
    log.info("Hashed features in %.1fs", time.perf_counter() - t0)

    base_rate = float(y_train.mean())
    preds: dict[str, np.ndarray] = {"constant": np.full(len(y_test), base_rate)}
    models: dict[str, dict] = {
        "constant": {
            "valid": evaluate(y_valid, np.full(len(y_valid), base_rate), base_rate),
            "test": evaluate(y_test, preds["constant"], base_rate),
        }
    }

    search = []
    best: tuple[float, float, LogisticRegression] | None = None
    for c in c_grid:
        t0 = time.perf_counter()
        model = LogisticRegression(C=c, max_iter=1000)
        model.fit(x_train, y_train)
        valid_ll = float(log_loss(y_valid, model.predict_proba(x_valid)[:, 1], labels=[0, 1]))
        log.info("C=%g valid log loss %.5f (%.1fs)", c, valid_ll, time.perf_counter() - t0)
        search.append({"C": c, "valid_log_loss": valid_ll})
        if best is None or valid_ll < best[0]:
            best = (valid_ll, c, model)

    assert best is not None, "c_grid must not be empty"
    _, best_c, model = best
    preds["logistic_regression"] = model.predict_proba(x_test)[:, 1]
    models["logistic_regression"] = {
        "C": best_c,
        "valid": evaluate(y_valid, model.predict_proba(x_valid)[:, 1], base_rate),
        "test": evaluate(y_test, preds["logistic_regression"], base_rate),
    }

    summary = {
        "data": {
            "rows": {"train": len(y_train), "valid": len(y_valid), "test": len(y_test)},
            "days": {
                "train": _day_range(split.train),
                "valid": _day_range(split.valid),
                "test": _day_range(split.test),
            },
            "train_ctr": base_rate,
        },
        "features": {"hashing_space": n_features, "tokens_per_row": featurizer.tokens_per_row},
        "c_search": search,
        "models": models,
    }
    return BaselineResult(summary=summary, y_test=y_test, test_predictions=preds)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/raw/train.gz"))
    parser.add_argument("--out", type=Path, default=Path("reports/baseline"))
    parser.add_argument(
        "--sample-frac", type=float, default=0.05, help="fraction of rows to keep (1 = all)"
    )
    parser.add_argument("--n-features", type=int, default=2**20)
    parser.add_argument("--c-grid", type=float, nargs="+", default=list(DEFAULT_C_GRID))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    t0 = time.perf_counter()
    sample_frac = None if args.sample_frac >= 1 else args.sample_frac
    df = load_avazu(args.data, sample_frac=sample_frac, seed=args.seed)
    log.info("Loaded %s rows in %.1fs", f"{len(df):,}", time.perf_counter() - t0)

    result = train_and_evaluate(df, c_grid=args.c_grid, n_features=args.n_features)
    result.summary["data"]["sample_frac"] = args.sample_frac
    write_report(result, args.out)
    log.info("Wrote report to %s", args.out)
    print(json.dumps(result.summary["models"]["logistic_regression"]["test"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
