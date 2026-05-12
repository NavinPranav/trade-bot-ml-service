"""Map gRPC market messages to pandas frames (bars supplied by the upstream backend).

The Java/backend service streams market data (e.g. from its broker integration). This service
never calls broker APIs directly. By default bars are returned with their original intraday
resolution; pass ``aggregate_daily=True`` only for the classical ML pipeline.
"""
from __future__ import annotations

import pandas as pd
from loguru import logger

from app.config import settings


def _timestamps_to_ist(ts_ms: list[int]) -> pd.DatetimeIndex:
    idx = pd.to_datetime(ts_ms, unit="ms", utc=True)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(settings.timezone)
    return idx


def _daily_ohlcv_from_intraday(df: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar day: open=first, high=max, low=min, close=last, volume=sum."""
    if df.empty:
        return df
    by_date = df.groupby(df.index.date, sort=True)
    out = by_date.agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).tz_localize(settings.timezone)
    out.index.name = "timestamp"
    return out


def _attr_or_key(obj, name):
    """Read a field from either a proto message (attribute) or a plain dict (key)."""
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)


def ohlcv_bars_to_dataframe(bars, *, aggregate_daily: bool = False) -> pd.DataFrame:
    """Convert repeated OhlcvBar to an OHLCV DataFrame.

    Accepts proto OhlcvBar objects or plain dicts with the same keys.

    When ``aggregate_daily=False`` (default) bars are returned at their original intraday
    resolution, which is required for AI/Gemini predictions and for computing technical
    indicators over many intraday bars.  Pass ``aggregate_daily=True`` to collapse intraday
    bars to one row per calendar day (used by the classical ML pipeline).
    """
    if not bars:
        return pd.DataFrame()

    ts = [int(_attr_or_key(b, "timestamp_unix_ms")) for b in bars]
    idx = _timestamps_to_ist(ts)
    df = pd.DataFrame(
        {
            "open": [float(_attr_or_key(b, "open")) for b in bars],
            "high": [float(_attr_or_key(b, "high")) for b in bars],
            "low": [float(_attr_or_key(b, "low")) for b in bars],
            "close": [float(_attr_or_key(b, "close")) for b in bars],
            "volume": [float(int(_attr_or_key(b, "volume"))) for b in bars],
        },
        index=idx,
    )
    df = df.sort_index()
    if df.index.duplicated().any():
        logger.warning("Duplicate OHLCV timestamps in gRPC payload; keeping last row per timestamp")
        df = df[~df.index.duplicated(keep="last")]
    if not aggregate_daily:
        return df
    out = _daily_ohlcv_from_intraday(df)
    if out.index.duplicated().any():
        logger.warning("Duplicate daily OHLCV rows after aggregation; keeping last")
        out = out[~out.index.duplicated(keep="last")]
    return out


def vix_points_to_dataframe(points) -> pd.DataFrame:
    """Accepts proto VixPoint objects or plain dicts with the same keys."""
    if not points:
        return pd.DataFrame()

    ts = [int(_attr_or_key(p, "timestamp_unix_ms")) for p in points]
    idx = _timestamps_to_ist(ts)
    df = pd.DataFrame({"vix": [float(_attr_or_key(p, "vix")) for p in points]}, index=idx)
    df.index.name = "timestamp"
    df = df.sort_index()
    if df.index.duplicated().any():
        logger.warning("Duplicate VIX timestamps in gRPC payload; keeping last per timestamp")
        df = df[~df.index.duplicated(keep="last")]
    daily = df.groupby(df.index.date, sort=True).last()
    daily.index = pd.DatetimeIndex(pd.to_datetime(daily.index)).tz_localize(settings.timezone)
    daily.index.name = "timestamp"
    return daily[["vix"]]
