"""Impressions stored one Parquet file per day, as a production data lake would hold them."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from pathlib import Path

import pandas as pd

from ctr.schema import TIMESTAMP


def write_days(df: pd.DataFrame, out_dir: str | Path) -> list[date]:
    """Split impressions by calendar day into <out_dir>/<YYYY-MM-DD>.parquet."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    days = []
    for day, part in df.groupby(df[TIMESTAMP].dt.date):
        part.reset_index(drop=True).to_parquet(out_dir / f"{day}.parquet", index=False)
        days.append(day)
    return days


def available_days(data_dir: str | Path) -> list[date]:
    return sorted(date.fromisoformat(p.stem) for p in Path(data_dir).glob("*.parquet"))


def load_days(data_dir: str | Path, days: Iterable[date]) -> pd.DataFrame:
    days = list(days)
    if not days:
        raise ValueError("No days requested")
    missing = [d for d in days if not (Path(data_dir) / f"{d}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"No data for {', '.join(map(str, missing))} in {data_dir}")
    parts = [pd.read_parquet(Path(data_dir) / f"{d}.parquet") for d in days]
    return pd.concat(parts, ignore_index=True)
