"""Writing evaluation results to disk: JSON, a Markdown table and a reliability plot."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ctr.metrics import calibration_table  # noqa: E402


@dataclass
class ExperimentResult:
    """Everything a run produces: the summary dict and the test-day predictions per model."""

    summary: dict
    y_test: np.ndarray
    test_predictions: dict[str, np.ndarray] = field(default_factory=dict)


METRICS = [
    ("log_loss", "Log loss", "{:.5f}"),
    ("normalized_entropy", "Normalised entropy", "{:.4f}"),
    ("auc", "AUC", "{:.4f}"),
    ("calibration_ratio", "Calibration ratio", "{:.3f}"),
    ("ece", "ECE", "{:.4f}"),
]


def results_markdown(summary: dict) -> str:
    """Test-set metrics, one row per model."""
    data = summary["data"]
    header = "| Model | " + " | ".join(label for _, label, _ in METRICS) + " |"
    rule = "| --- |" + " ---: |" * len(METRICS)
    rows = []
    for name, result in summary["models"].items():
        test = result["test"]
        cells = [fmt.format(test[key]) for key, _, fmt in METRICS]
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    test_days = " to ".join(dict.fromkeys(data["days"]["test"]))
    return "\n".join(
        [
            f"Test set: {data['rows']['test']:,} impressions on {test_days}, "
            f"click rate {summary['models']['constant']['test']['ctr']:.4f}.",
            "",
            header,
            rule,
            *rows,
            "",
        ]
    )


def plot_reliability(result: ExperimentResult, path: Path, n_bins: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    hi = 0.0
    for name, preds in result.test_predictions.items():
        table = calibration_table(result.y_test, preds, n_bins)
        ax.plot(table["mean_pred"], table["actual_ctr"], marker="o", label=name)
        hi = max(hi, table["mean_pred"].max(), table["actual_ctr"].max())
    hi = min(1.0, hi * 1.1)
    ax.plot([0, hi], [0, hi], linestyle="--", color="grey", label="perfect calibration")
    ax.set(
        xlim=(0, hi),
        ylim=(0, hi),
        xlabel="Mean predicted CTR",
        ylabel="Actual CTR",
        title="Reliability on test day",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(result: ExperimentResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result.summary, indent=2))
    (out_dir / "results.md").write_text(results_markdown(result.summary))
    plot_reliability(result, out_dir / "reliability.png")
