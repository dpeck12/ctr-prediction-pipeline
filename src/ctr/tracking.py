"""Logging runs to MLflow, so every experiment's settings and results are kept."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "ctr-prediction"

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, float]:
    """{'log_loss': 0.4} -> {'test_log_loss': 0.4}, dropping NaNs and non-numbers."""
    return {
        f"{prefix}_{key}": float(value)
        for key, value in metrics.items()
        if isinstance(value, int | float) and not math.isnan(value)
    }


def log_run(
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts_dir: Path | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment: str = EXPERIMENT,
) -> str:
    """Log one model's run and return its MLflow run ID."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({key: str(value) for key, value in params.items()})
        mlflow.log_metrics(metrics)
        if artifacts_dir is not None:
            mlflow.log_artifacts(str(artifacts_dir))
        return run.info.run_id
