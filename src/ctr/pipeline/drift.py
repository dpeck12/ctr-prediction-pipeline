"""Drift checks: has today's traffic, or the model's output, moved away from training?

Uses the population stability index (PSI), a standard drift score in credit
scoring and ad tech:

    PSI = sum over buckets of (current% - reference%) * ln(current% / reference%)

Common reading: below 0.1 stable, 0.1-0.25 moderate shift, above 0.25 major shift.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ctr.encoders import make_fields

MODERATE = 0.1
MAJOR = 0.25
OTHER = "__other__"
EPS = 1e-4  # stops empty buckets from giving log(0)


def psi(reference: np.ndarray, current: np.ndarray) -> float:
    """PSI between two aligned arrays of bucket counts or proportions."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = np.clip(ref / ref.sum(), EPS, None)
    cur = np.clip(cur / cur.sum(), EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def categorical_psi(reference: pd.Series, current: pd.Series, top_k: int = 20) -> float:
    """PSI over the reference's `top_k` most common values plus one bucket for the rest."""
    top = reference.value_counts().index[:top_k]
    ref = reference.where(reference.isin(top), OTHER).value_counts()
    cur = current.where(current.isin(top), OTHER).value_counts()
    buckets = list(top) + [OTHER]
    return psi(ref.reindex(buckets, fill_value=0), cur.reindex(buckets, fill_value=0))


def score_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI of model scores, in buckets at the reference scores' deciles."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    ref = np.bincount(np.searchsorted(edges, reference, side="right"), minlength=len(edges) + 1)
    cur = np.bincount(np.searchsorted(edges, current, side="right"), minlength=len(edges) + 1)
    return psi(ref, cur)


def level(value: float) -> str:
    if value >= MAJOR:
        return "major"
    if value >= MODERATE:
        return "moderate"
    return "stable"


def feature_drift(
    reference: pd.DataFrame, current: pd.DataFrame, top_k: int = 20
) -> dict[str, Any]:
    """PSI of every model input between the training window and a new day's traffic.

    Needs no labels, so in production it can run as traffic arrives, before
    anyone knows what was clicked.
    """
    ref_fields, cur_fields = make_fields(reference), make_fields(current)
    scores = {
        col: round(categorical_psi(ref_fields[col], cur_fields[col], top_k), 4)
        for col in ref_fields.columns
        if col != "day_of_week"  # each day is a single weekday by construction
    }
    worst = max(scores, key=scores.get)
    return {
        "psi": scores,
        "max_psi": scores[worst],
        "max_psi_field": worst,
        "level": level(scores[worst]),
        "moderate_or_worse": sorted(c for c, v in scores.items() if v >= MODERATE),
    }
