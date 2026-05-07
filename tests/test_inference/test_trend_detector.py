"""Tests for the deterministic multi-timeframe trend detector and guardrail.

These do not hit Gemini — they exercise the pure Python helpers added to
``gemini_predictor`` so we can verify regime detection and the BUY-in-downtrend /
SELL-in-uptrend veto without touching the network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.inference.gemini_predictor import (
    _apply_trend_guardrail,
    _detect_trend_context,
    _detect_trend_single_tf,
)


def _make_ohlcv(closes: list[float], *, freq: str = "5min", base_volume: int = 1_000) -> pd.DataFrame:
    """Build a minimal OHLCV frame with a DatetimeIndex around the given closes."""
    n = len(closes)
    idx = pd.date_range("2024-01-02 09:15", periods=n, freq=freq)
    closes_arr = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes_arr[0]], closes_arr[:-1]])
    highs = np.maximum(opens, closes_arr) + 5
    lows = np.minimum(opens, closes_arr) - 5
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes_arr,
            "volume": np.full(n, base_volume, dtype=float),
        },
        index=idx,
    )


def test_detect_trend_uptrend():
    closes = list(np.linspace(50_000, 50_700, 60))   # +1.4% steady ramp
    df = _make_ohlcv(closes)
    out = _detect_trend_single_tf(df)
    assert out["regime"] == "UPTREND"
    assert out["score"] >= 3
    assert out["evidence"]["bars"] == 60


def test_detect_trend_downtrend():
    closes = list(np.linspace(50_700, 50_000, 60))   # −1.4% steady fall
    df = _make_ohlcv(closes)
    out = _detect_trend_single_tf(df)
    assert out["regime"] == "DOWNTREND"
    assert out["score"] <= -3


def test_detect_trend_sideways():
    rng = np.random.default_rng(42)
    base = np.full(60, 50_350.0)
    noise = rng.normal(0, 25.0, size=60)        # tight ±25-pt chop around 50_350
    df = _make_ohlcv(list(base + noise))
    out = _detect_trend_single_tf(df)
    assert out["regime"] == "SIDEWAYS"
    assert -2 <= out["score"] <= 2


def test_detect_trend_handles_too_few_bars():
    df = _make_ohlcv([50_000.0] * 5)
    out = _detect_trend_single_tf(df)
    assert out["regime"] == "UNKNOWN"
    assert out["score"] == 0


def test_multi_timeframe_uptrend_agreement():
    closes = list(np.linspace(50_000, 50_900, 90))
    df = _make_ohlcv(closes)
    ctx = _detect_trend_context(df)
    assert ctx["regime"] == "UPTREND"
    assert ctx["primary_regime"] == "UPTREND"
    # 90 × 5-min resampled to 15-min → ≥ 30 bars; should also be UPTREND.
    assert ctx["higher_regime"] == "UPTREND"
    assert ctx["agreement"] is True


def test_multi_timeframe_disagreement_demotes_to_sideways():
    """Bull-then-bear crash-to-end: 5m says DOWNTREND but the 15m aggregate
    can still read UPTREND. The combined regime must be SIDEWAYS so the
    guardrail does not veto either side."""
    up = list(np.linspace(50_000, 51_000, 60))
    down = list(np.linspace(51_000, 50_400, 30))
    df = _make_ohlcv(up + down)
    ctx = _detect_trend_context(df)
    if (
        ctx["primary_regime"] in ("UPTREND", "DOWNTREND")
        and ctx["higher_regime"] in ("UPTREND", "DOWNTREND")
        and ctx["primary_regime"] != ctx["higher_regime"]
    ):
        assert ctx["regime"] == "SIDEWAYS"


def test_guardrail_vetoes_buy_in_downtrend():
    closes = list(np.linspace(50_700, 50_000, 60))
    df = _make_ohlcv(closes)
    ctx = _detect_trend_context(df)
    final, reason = _apply_trend_guardrail("BUY", ctx)
    assert final == "HOLD"
    assert reason and "BUY vetoed" in reason


def test_guardrail_vetoes_sell_in_uptrend():
    closes = list(np.linspace(50_000, 50_700, 60))
    df = _make_ohlcv(closes)
    ctx = _detect_trend_context(df)
    final, reason = _apply_trend_guardrail("SELL", ctx)
    assert final == "HOLD"
    assert reason and "SELL vetoed" in reason


def test_guardrail_allows_buy_in_uptrend():
    closes = list(np.linspace(50_000, 50_700, 60))
    df = _make_ohlcv(closes)
    ctx = _detect_trend_context(df)
    final, reason = _apply_trend_guardrail("BUY", ctx)
    assert final == "BUY"
    assert reason is None


def test_guardrail_allows_sell_in_downtrend():
    closes = list(np.linspace(50_700, 50_000, 60))
    df = _make_ohlcv(closes)
    ctx = _detect_trend_context(df)
    final, reason = _apply_trend_guardrail("SELL", ctx)
    assert final == "SELL"
    assert reason is None


def test_guardrail_no_veto_in_sideways():
    """SIDEWAYS / UNKNOWN must never veto on its own — the existing
    confidence / R:R / dead-tape gates already handle low-conviction tape."""
    rng = np.random.default_rng(7)
    closes = list(50_350 + rng.normal(0, 25.0, size=60))
    df = _make_ohlcv(closes)
    ctx = _detect_trend_context(df)
    if ctx["regime"] not in ("UPTREND", "DOWNTREND"):
        assert _apply_trend_guardrail("BUY", ctx) == ("BUY", None)
        assert _apply_trend_guardrail("SELL", ctx) == ("SELL", None)


def test_guardrail_passes_through_hold_and_missing_context():
    assert _apply_trend_guardrail("HOLD", None) == ("HOLD", None)
    assert _apply_trend_guardrail("BUY", None) == ("BUY", None)


@pytest.mark.parametrize("direction", ["BUY", "SELL"])
def test_guardrail_handles_unknown_higher_tf(direction: str):
    """When the higher timeframe is UNKNOWN (e.g., resample failed) the
    guardrail still vetoes if the primary timeframe disagrees."""
    if direction == "BUY":
        closes = list(np.linspace(50_700, 50_000, 60))
    else:
        closes = list(np.linspace(50_000, 50_700, 60))
    primary = _detect_trend_single_tf(_make_ohlcv(closes))
    ctx = {
        "regime": primary["regime"],
        "primary_regime": primary["regime"],
        "higher_regime": "UNKNOWN",
        "agreement": False,
        "primary_score": primary["score"],
        "higher_score": 0,
        "primary_reasons": primary["reasons"],
        "higher_reasons": [],
        "evidence": {"primary": primary["evidence"], "higher": {}},
    }
    final, reason = _apply_trend_guardrail(direction, ctx)
    assert final == "HOLD"
    assert reason is not None
