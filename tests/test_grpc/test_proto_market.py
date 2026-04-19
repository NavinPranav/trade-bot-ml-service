"""Tests for gRPC market payload → pandas conversion."""

import pandas as pd

from app.grpc_server.generated import prediction_service_pb2 as pb2
from app.grpc_server.proto_market import ohlcv_bars_to_dataframe, vix_points_to_dataframe


def test_ohlcv_bars_to_dataframe_sorted_and_columns():
    # 2024-01-02 and 2024-01-03 UTC midnight ms
    ms2 = 1_704_153_600_000
    ms3 = 1_704_240_000_000
    bars = [
        pb2.OhlcvBar(
            timestamp_unix_ms=ms3,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1_000_000,
        ),
        pb2.OhlcvBar(
            timestamp_unix_ms=ms2,
            open=99.0,
            high=100.0,
            low=98.0,
            close=99.5,
            volume=900_000,
        ),
    ]
    df = ohlcv_bars_to_dataframe(bars)
    assert len(df) == 2
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["close"].iloc[-1] == 100.5


def test_vix_points_to_dataframe():
    ms = 1_704_153_600_000
    pts = [pb2.VixPoint(timestamp_unix_ms=ms, vix=12.5)]
    df = vix_points_to_dataframe(pts)
    assert "vix" in df.columns
    assert float(df["vix"].iloc[0]) == 12.5


def test_empty_bars():
    assert ohlcv_bars_to_dataframe([]).empty
    assert vix_points_to_dataframe([]).empty


def test_intraday_ohlcv_aggregates_to_single_daily_row():
    """Two bars on the same IST session day → one daily OHLCV row (streaming-style payload)."""
    ms1 = int(pd.Timestamp("2024-01-02 04:00:00", tz="UTC").timestamp() * 1000)
    ms2 = int(pd.Timestamp("2024-01-02 10:00:00", tz="UTC").timestamp() * 1000)
    bars = [
        pb2.OhlcvBar(
            timestamp_unix_ms=ms1,
            open=100.0,
            high=102.0,
            low=99.0,
            close=101.0,
            volume=1_000,
        ),
        pb2.OhlcvBar(
            timestamp_unix_ms=ms2,
            open=101.0,
            high=105.0,
            low=100.0,
            close=104.0,
            volume=2_000,
        ),
    ]
    df = ohlcv_bars_to_dataframe(bars)
    assert len(df) == 1
    assert df["open"].iloc[0] == 100.0
    assert df["high"].iloc[0] == 105.0
    assert df["low"].iloc[0] == 99.0
    assert df["close"].iloc[0] == 104.0
    assert df["volume"].iloc[0] == 3_000.0
