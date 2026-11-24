"""Prepare daily data and replay the daily pipeline without Airflow.

    uv run python -m ctr.pipeline prepare --data data/raw/train.gz
    uv run python -m ctr.pipeline replay

`replay` runs the same four steps the Airflow DAG runs, for each day in turn,
then writes reports/pipeline/summary.md and timeline.png.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ctr.data import load_avazu  # noqa: E402
from ctr.pipeline.steps import Paths, run_day  # noqa: E402
from ctr.pipeline.store import available_days, write_days  # noqa: E402

log = logging.getLogger("ctr.pipeline")


def summarise(paths: Paths, days: Sequence[date]) -> list[dict]:
    rows = []
    for day in days:
        folder = paths.reports / str(day)
        drift = json.loads((folder / "drift.json").read_text())
        promo = json.loads((folder / "promote.json").read_text())
        ev = json.loads((folder / "evaluate.json").read_text())
        row = {
            "day": str(day),
            "actual_ctr": ev["actual_ctr"],
            "champion": ev["champion"],
            "promoted": promo["promoted"],
            "champion_log_loss": ev["champion_metrics"]["log_loss"],
            "champion_calibration": ev["champion_metrics"]["calibration_ratio"],
            "frozen_log_loss": ev.get("frozen_metrics", ev["champion_metrics"])["log_loss"],
            "max_feature_psi": drift["max_psi"],
            "max_psi_field": drift["max_psi_field"],
            "score_psi": ev["score_psi_vs_previous_day"],
        }
        rows.append(row)
    return rows


def summary_markdown(rows: list[dict]) -> str:
    lines = [
        "| Day | Click rate | Champion | Promoted | Log loss (retrained) | Log loss (frozen) "
        "| Calibration ratio | Max feature PSI | Score PSI |",
        "| --- | ---: | --- | :---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['day']} | {r['actual_ctr']:.2%} | {r['champion']} | "
            f"{'yes' if r['promoted'] else 'no'} | {r['champion_log_loss']:.5f} | "
            f"{r['frozen_log_loss']:.5f} | {r['champion_calibration']:.3f} | "
            f"{r['max_feature_psi']:.3f} ({r['max_psi_field']}) | {r['score_psi']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def plot_timeline(rows: list[dict], path: Path) -> None:
    x = list(range(len(rows)))
    days = x  # plot against positions, label with dates below
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    axes[0].plot(days, [r["champion_log_loss"] for r in rows], marker="o", label="retrained daily")
    axes[0].plot(
        days, [r["frozen_log_loss"] for r in rows], marker="o", label="frozen (first model)"
    )
    axes[0].set(ylabel="Log loss", title="Next-day performance")
    axes[0].legend()
    axes[1].plot(days, [r["actual_ctr"] for r in rows], marker="o", color="black", label="actual")
    axes[1].plot(
        days,
        [r["actual_ctr"] * r["champion_calibration"] for r in rows],
        marker="o",
        label="mean predicted (champion)",
    )
    axes[1].set(ylabel="Click rate")
    axes[1].legend()
    axes[2].bar(days, [r["max_feature_psi"] for r in rows], color="grey", label="max feature PSI")
    axes[2].axhline(0.1, linestyle="--", color="orange", label="moderate (0.1)")
    axes[2].axhline(0.25, linestyle="--", color="red", label="major (0.25)")
    axes[2].set(ylabel="PSI", xlabel="Day (2014)")
    axes[2].set_xticks(x, [r["day"][5:] for r in rows])
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--root", type=Path, default=Path("."), help="project root for data/reports"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="split the raw file into daily Parquet files")
    prep.add_argument("--data", type=Path, default=Path("data/raw/train.gz"))
    prep.add_argument("--sample-frac", type=float, default=0.05)
    prep.add_argument("--seed", type=int, default=0)

    rep = sub.add_parser("replay", help="run the daily steps for each day in turn")
    rep.add_argument("--model", default="lightgbm")
    rep.add_argument("--model-kwargs", type=json.loads, default={})
    rep.add_argument("--first-day", type=date.fromisoformat, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    paths = Paths.from_root(args.root)

    if args.command == "prepare":
        frac = None if args.sample_frac >= 1 else args.sample_frac
        days = write_days(load_avazu(args.data, sample_frac=frac, seed=args.seed), paths.data)
        log.info("Wrote %d days to %s", len(days), paths.data)
        return 0

    stored = available_days(paths.data)
    # The first runnable day needs two training days and a validation day before it.
    first = args.first_day or stored[0] + timedelta(days=3)
    days = [d for d in stored if d >= first]
    for day in days:
        result = run_day(paths, day, args.model, **args.model_kwargs)
        log.info(
            "%s: drift %s (max PSI %.3f), promoted=%s, log loss %.5f",
            day,
            result["drift"]["level"],
            result["drift"]["max_psi"],
            result["promote"]["promoted"],
            result["evaluate"]["champion_metrics"]["log_loss"],
        )

    rows = summarise(paths, days)
    (paths.reports / "summary.md").write_text(summary_markdown(rows))
    (paths.reports / "summary.json").write_text(json.dumps(rows, indent=2))
    plot_timeline(rows, paths.reports / "timeline.png")
    print(summary_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
