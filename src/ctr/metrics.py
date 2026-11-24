"""Evaluation metrics for click probabilities.

A CTR model's output is used as a probability (it sets bid prices), so ranking
quality (AUC) is not enough: the probabilities themselves must be accurate. Hence
log loss, normalised entropy and calibration alongside AUC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

EPS = 1e-7


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def binary_entropy(rate: float) -> float:
    """Log loss of always predicting `rate` on data whose click rate is `rate`."""
    r = float(np.clip(rate, EPS, 1 - EPS))
    return -(r * np.log(r) + (1 - r) * np.log(1 - r))


def normalized_entropy(y: np.ndarray, p: np.ndarray, base_rate: float) -> float:
    """Log loss divided by the entropy of the background click rate.

    From He et al. (2014), "Practical Lessons from Predicting Clicks on Ads at
    Facebook". 1.0 means no better than predicting the training CTR for every
    impression; lower is better. Unlike raw log loss it is comparable across
    datasets with different click rates. `base_rate` should be the training CTR.
    """
    return log_loss(y, _clip(p), labels=[0, 1]) / binary_entropy(base_rate)


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs actual click rate in `n_bins` equal-count bins of the predictions.

    Equal-count bins suit CTR models, whose predictions bunch at low values.
    Tied predictions always share a bin, so a constant model gives one bin.
    """
    p = np.asarray(p, dtype=float)
    ranks = pd.Series(p).rank(method="min").to_numpy() - 1
    bins = np.minimum((ranks * n_bins // len(p)).astype(int), n_bins - 1)
    table = (
        pd.DataFrame({"bin": bins, "pred": p, "click": np.asarray(y)})
        .groupby("bin")
        .agg(count=("click", "size"), mean_pred=("pred", "mean"), actual_ctr=("click", "mean"))
        .reset_index(drop=True)
    )
    return table


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Count-weighted mean of |predicted - actual click rate| across calibration bins."""
    t = calibration_table(y, p, n_bins)
    return float((t["count"] * (t["mean_pred"] - t["actual_ctr"]).abs()).sum() / t["count"].sum())


def evaluate(y: np.ndarray, p: np.ndarray, base_rate: float) -> dict[str, float]:
    """All headline metrics for one set of predictions."""
    y = np.asarray(y)
    p = _clip(p)
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    return {
        "rows": int(len(y)),
        "ctr": float(y.mean()),
        "mean_pred": float(p.mean()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "auc": float(auc),
        "normalized_entropy": float(normalized_entropy(y, p, base_rate)),
        "calibration_ratio": float(p.mean() / y.mean()) if y.mean() > 0 else float("nan"),
        "ece": expected_calibration_error(y, p),
    }
