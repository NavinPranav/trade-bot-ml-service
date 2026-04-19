"""Map gRPC market messages to pandas frames (bars supplied by the upstream backend).

The Java/backend service streams market data (e.g. from its broker integration). This service
never calls broker APIs directly. OHLCV may be daily or multiple intraday bars per session; we
aggregate to one row per calendar day in the configured market timezone for indicator features.
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


def ohlcv_bars_to_dataframe(bars) -> pd.DataFrame:
    """Convert repeated OhlcvBar to a daily OHLCV frame (intraday bars aggregated per day)."""
    if not bars:
        return pd.DataFrame()

    ts = [int(b.timestamp_unix_ms) for b in bars]
    idx = _timestamps_to_ist(ts)
    df = pd.DataFrame(
        {
            "open": [float(b.open) for b in bars],
            "high": [float(b.high) for b in bars],
            "low": [float(b.low) for b in bars],
            "close": [float(b.close) for b in bars],
            "volume": [float(int(b.volume)) for b in bars],
        },
        index=idx,
    )
    df = df.sort_index()
    if df.index.duplicated().any():
        logger.warning("Duplicate OHLCV timestamps in gRPC payload; keeping last row per timestamp")
        df = df[~df.index.duplicated(keep="last")]
    out = _daily_ohlcv_from_intraday(df)
    if out.index.duplicated().any():
        logger.warning("Duplicate daily OHLCV rows after aggregation; keeping last")
        out = out[~out.index.duplicated(keep="last")]
    return out


def vix_points_to_dataframe(points) -> pd.DataFrame:
    if not points:
        return pd.DataFrame()

    ts = [int(p.timestamp_unix_ms) for p in points]
    idx = _timestamps_to_ist(ts)
    df = pd.DataFrame({"vix": [float(p.vix) for p in points]}, index=idx)
    df.index.name = "timestamp"
    df = df.sort_index()
    if df.index.duplicated().any():
        logger.warning("Duplicate VIX timestamps in gRPC payload; keeping last per timestamp")
        df = df[~df.index.duplicated(keep="last")]
    daily = df.groupby(df.index.date, sort=True).last()
    daily.index = pd.DatetimeIndex(pd.to_datetime(daily.index)).tz_localize(settings.timezone)
    daily.index.name = "timestamp"
    return daily[["vix"]]
