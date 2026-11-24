"""Request and response models for the prediction API."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from ctr.schema import CATEGORICAL_COLUMNS, TIMESTAMP

MAX_BATCH = 1000


class _ImpressionBase(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True, extra="forbid")

    hour: str = Field(description="Impression time as YYMMDDHH, as in the Avazu data")

    @field_validator("hour")
    @classmethod
    def _valid_hour(cls, value: str) -> str:
        try:
            datetime.strptime(value, "%y%m%d%H")
        except ValueError as err:
            raise ValueError("hour must be YYMMDDHH, e.g. 14103014") from err
        return value


# One optional string field per Avazu feature. Missing fields are treated like
# any value the model hasn't seen, so callers can send what they have.
Impression = create_model(
    "Impression",
    __base__=_ImpressionBase,
    **{col: (str | None, None) for col in CATEGORICAL_COLUMNS},
)


class PredictRequest(BaseModel):
    impressions: list[Impression] = Field(min_length=1, max_length=MAX_BATCH)  # type: ignore[valid-type]


class PredictResponse(BaseModel):
    model: str
    probabilities: list[float]


def to_frame(impressions: list[BaseModel]) -> pd.DataFrame:
    """Convert validated impressions into the frame the models were trained on."""
    df = pd.DataFrame(
        [imp.model_dump() for imp in impressions], columns=["hour", *CATEGORICAL_COLUMNS]
    )
    df.insert(0, TIMESTAMP, pd.to_datetime(df.pop("hour"), format="%y%m%d%H"))
    return df.astype({col: "string" for col in CATEGORICAL_COLUMNS})
