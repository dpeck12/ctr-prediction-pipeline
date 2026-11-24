"""CTR models with a common interface: fit(train, valid), predict_proba(df), params()."""

from __future__ import annotations

from typing import Any

from ctr.models.base import CTRModel

MODEL_NAMES = ("logreg", "lightgbm", "dcn")


def build_model(name: str, **kwargs: Any) -> CTRModel:
    """Create a model by name. Imports lazily so torch only loads when DCN is used."""
    if name == "logreg":
        from ctr.models.logreg import LogisticRegressionModel

        return LogisticRegressionModel(**kwargs)
    if name == "lightgbm":
        from ctr.models.gbm import LightGBMModel

        return LightGBMModel(**kwargs)
    if name == "dcn":
        from ctr.models.dcn import DCNModel

        return DCNModel(**kwargs)
    raise ValueError(f"Unknown model {name!r}; choose from {', '.join(MODEL_NAMES)}")


__all__ = ["MODEL_NAMES", "CTRModel", "build_model"]
