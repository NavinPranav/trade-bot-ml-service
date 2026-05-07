"""Tests for the post-coercion accuracy layers added on top of the trend guardrail:

- Reversal-confirmation override (so legitimate counter-trend reversals pass through).
- Volume confirmation gate (force HOLD on thin tape).
- Regime-aware confidence floor (relax floor for with-trend trades).

All exercises are deterministic and do not hit Gemini.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.inference.gemini_predictor import (
    _PredictionPolicyStore,
    _apply_trend_guardrail,
    _coerce_result,
    _detect_reversal_confirmation,
    _detect_trend_context,
    _effective_confidence_floor,
    _is_with_trend,
)


def _ohlcv_linear(start: float, end: float, n: int = 60) -> pd.DataFrame:
    closes = np.linspace(start, end, n)
    idx = pd.date_range("2024-01-02 09:15", periods=n, freq="5min")
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + 5,
            "low": np.minimum(opens, closes) - 5,
            "close": closes,
            "volume": np.full(n, 1_000.0),
        },
        index=idx,
    )


def _ohlcv_with_bullish_reversal(n: int = 60) -> pd.DataFrame:
    """Downtrend that ends with a bullish-engulfing high-volume bar reclaiming VWAP."""
    df = _ohlcv_linear(50_700, 50_050, n=n - 1).reset_index(drop=True)
    last_open = float(df["close"].iloc[-1])
    last_close = last_open + 80
    last_high = last_close + 3
    last_low = last_open - 5
    last_row = pd.DataFrame(
        {
            "open": [last_open],
            "high": [last_high],
            "low": [last_low],
            "close": [last_close],
            "volume": [3_000.0],
        }
    )
    full = pd.concat([df, last_row], ignore_index=True)
    full.index = pd.date_range("2024-01-02 09:15", periods=len(full), freq="5min")
    return full


def _ohlcv_with_bearish_reversal(n: int = 60) -> pd.DataFrame:
    df = _ohlcv_linear(50_050, 50_700, n=n - 1).reset_index(drop=True)
    last_open = float(df["close"].iloc[-1])
    last_close = last_open - 80
    last_high = last_open + 5
    last_low = last_close - 3
    last_row = pd.DataFrame(
        {
            "open": [last_open],
            "high": [last_high],
            "low": [last_low],
            "close": [last_close],
            "volume": [3_000.0],
        }
    )
    full = pd.concat([df, last_row], ignore_index=True)
    full.index = pd.date_range("2024-01-02 09:15", periods=len(full), freq="5min")
    return full


@pytest.fixture(autouse=True)
def _clear_policy_overrides():
    """Make sure each test runs with a clean policy slate."""
    _PredictionPolicyStore.clear()
    yield
    _PredictionPolicyStore.clear()


# ── Reversal confirmation detector ──────────────────────────────────────────

def test_reversal_detector_finds_bullish_signals():
    df = _ohlcv_with_bullish_reversal()
    indicators = {"rsi_14": 42.0, "volume_ratio": 1.8}
    out = _detect_reversal_confirmation(df, indicators, "BUY")
    assert out["score"] >= 3
    assert out["confirmed"] is True
    assert any("Bullish engulfing" in r or "Hammer" in r for r in out["reasons"])
    assert any("Volume" in r for r in out["reasons"])
    assert any("RSI" in r for r in out["reasons"])


def test_reversal_detector_finds_bearish_signals():
    df = _ohlcv_with_bearish_reversal()
    indicators = {"rsi_14": 58.0, "volume_ratio": 1.8}
    out = _detect_reversal_confirmation(df, indicators, "SELL")
    assert out["score"] >= 3
    assert out["confirmed"] is True


def test_reversal_detector_quiet_tape_no_confirmation():
    df = _ohlcv_linear(50_000, 50_100, n=60)
    indicators = {"rsi_14": 50.0, "volume_ratio": 0.8}
    out = _detect_reversal_confirmation(df, indicators, "BUY")
    assert out["confirmed"] is False
    assert out["score"] <= 2


def test_reversal_detector_skips_for_hold():
    df = _ohlcv_with_bullish_reversal()
    out = _detect_reversal_confirmation(df, {"rsi_14": 42, "volume_ratio": 2.0}, "HOLD")
    assert out["score"] == 0
    assert out["confirmed"] is False


def test_reversal_detector_threshold_tunable_via_policy():
    df = _ohlcv_with_bullish_reversal()
    indicators = {"rsi_14": 42.0, "volume_ratio": 1.8}
    _PredictionPolicyStore.apply_updates({"reversal_confirmation_min_signals": 99})
    out = _detect_reversal_confirmation(df, indicators, "BUY")
    assert out["confirmed"] is False  # threshold so high it can never trip
    assert out["score"] >= 3


# ── Guardrail + reversal interaction ────────────────────────────────────────

def test_guardrail_lets_buy_through_when_reversal_confirmed():
    """Counter-trend BUY in a downtrend is allowed when reversal evidence is strong."""
    df = _ohlcv_with_bullish_reversal()
    ctx = _detect_trend_context(df)
    assert ctx["regime"] == "DOWNTREND"
    reversal = {
        "confirmed": True,
        "score": 4,
        "reasons": ["Bullish engulfing", "Volume 1.8x", "RSI 42 turning", "VWAP reclaim"],
    }
    final, note = _apply_trend_guardrail("BUY", ctx, reversal)
    assert final == "BUY"
    assert note is not None
    assert "BUY allowed" in note
    assert "DOWNTREND" in note


def test_guardrail_still_vetoes_when_reversal_unconfirmed():
    df = _ohlcv_linear(50_700, 50_000, n=60)
    ctx = _detect_trend_context(df)
    reversal = {"confirmed": False, "score": 1, "reasons": []}
    final, note = _apply_trend_guardrail("BUY", ctx, reversal)
    assert final == "HOLD"
    assert note and "vetoed" in note


# ── Regime-aware confidence floor ───────────────────────────────────────────

def test_with_trend_helper_uptrend_buy():
    assert _is_with_trend("BUY", {"regime": "UPTREND"}) is True
    assert _is_with_trend("SELL", {"regime": "UPTREND"}) is False


def test_with_trend_helper_downtrend_sell():
    assert _is_with_trend("SELL", {"regime": "DOWNTREND"}) is True
    assert _is_with_trend("BUY", {"regime": "DOWNTREND"}) is False


def test_with_trend_helper_sideways_or_missing():
    assert _is_with_trend("BUY", {"regime": "SIDEWAYS"}) is False
    assert _is_with_trend("BUY", None) is False
    assert _is_with_trend("HOLD", {"regime": "UPTREND"}) is False


def test_floor_relaxed_when_regime_matches_direction():
    base_floor = float(_PredictionPolicyStore.get("min_confidence"))
    relaxed = float(_PredictionPolicyStore.get("relaxed_confidence_floor_strong_trend"))
    assert relaxed < base_floor

    floor_no_ctx = _effective_confidence_floor("BUY", indicators={}, checklist_signal=None, price=50_000.0)
    floor_with_ctx = _effective_confidence_floor(
        "BUY",
        indicators={},
        checklist_signal=None,
        price=50_000.0,
        trend_context={"regime": "UPTREND"},
    )
    assert floor_no_ctx == base_floor
    assert floor_with_ctx == relaxed


def test_floor_not_relaxed_for_counter_trend():
    base_floor = float(_PredictionPolicyStore.get("min_confidence"))
    floor = _effective_confidence_floor(
        "BUY",
        indicators={},
        checklist_signal=None,
        price=50_000.0,
        trend_context={"regime": "DOWNTREND"},
    )
    assert floor == base_floor


# ── End-to-end _coerce_result behaviour ─────────────────────────────────────

def test_coerce_volume_gate_disabled_by_default():
    """Default policy: volume_confirmation_min_ratio = 0.0 → no gate."""
    raw = {
        "direction": "BUY",
        "confidence": 80.0,
        "magnitude": 0.5,
        "entry_price": 50_000,
        "stop_loss": 49_900,
        "target_price": 50_200,
        "valid_minutes": 15,
        "reason": "Strong setup",
    }
    out = _coerce_result(
        raw,
        50_000.0,
        indicators={"volume_ratio": 0.2},   # very thin
        trend_context={"regime": "UPTREND", "primary_regime": "UPTREND", "higher_regime": "UPTREND"},
    )
    assert out["direction"] == "BUY"
    assert "Volume gate" not in (out["prediction_reason"] or "")


def test_coerce_volume_gate_blocks_when_enabled_and_thin():
    raw = {
        "direction": "BUY",
        "confidence": 80.0,
        "magnitude": 0.5,
        "entry_price": 50_000,
        "stop_loss": 49_900,
        "target_price": 50_200,
        "valid_minutes": 15,
        "reason": "Strong setup",
    }
    _PredictionPolicyStore.apply_updates({"volume_confirmation_min_ratio": 0.7})
    out = _coerce_result(
        raw,
        50_000.0,
        indicators={"volume_ratio": 0.2},
        trend_context={"regime": "UPTREND", "primary_regime": "UPTREND", "higher_regime": "UPTREND"},
    )
    assert out["direction"] == "HOLD"
    assert "Volume gate" in out["prediction_reason"]
    assert "0.20x" in out["prediction_reason"]


def test_coerce_volume_gate_allows_when_volume_meets_ratio():
    raw = {
        "direction": "BUY",
        "confidence": 80.0,
        "magnitude": 0.5,
        "entry_price": 50_000,
        "stop_loss": 49_900,
        "target_price": 50_200,
        "valid_minutes": 15,
        "reason": "Strong setup",
    }
    _PredictionPolicyStore.apply_updates({"volume_confirmation_min_ratio": 0.7})
    out = _coerce_result(
        raw,
        50_000.0,
        indicators={"volume_ratio": 1.4},   # plenty of volume
        trend_context={"regime": "UPTREND", "primary_regime": "UPTREND", "higher_regime": "UPTREND"},
    )
    assert out["direction"] == "BUY"


def test_coerce_reversal_override_lets_buy_through_in_downtrend():
    df = _ohlcv_with_bullish_reversal()
    ctx = _detect_trend_context(df)
    assert ctx["regime"] == "DOWNTREND"
    raw = {
        "direction": "BUY",
        "confidence": 78.0,
        "magnitude": 0.4,
        "entry_price": 50_000,
        "stop_loss": 49_900,
        "target_price": 50_200,
        "valid_minutes": 15,
        "reason": "Reversal candle on heavy volume",
    }
    out = _coerce_result(
        raw,
        50_000.0,
        indicators={"rsi_14": 42.0, "volume_ratio": 1.8},
        trend_context=ctx,
        ohlcv=df,
    )
    assert out["direction"] == "BUY"
    assert "[Trend guardrail]" in out["prediction_reason"]
    assert "BUY allowed despite DOWNTREND" in out["prediction_reason"]


def test_coerce_guardrail_still_vetoes_without_reversal_evidence():
    df = _ohlcv_linear(50_700, 50_000, n=60)   # clean downtrend, no reversal bar
    ctx = _detect_trend_context(df)
    raw = {
        "direction": "BUY",
        "confidence": 78.0,
        "magnitude": 0.4,
        "entry_price": 50_000,
        "stop_loss": 49_900,
        "target_price": 50_200,
        "valid_minutes": 15,
        "reason": "AI thinks bottom is in",
    }
    out = _coerce_result(
        raw,
        50_000.0,
        indicators={"rsi_14": 30.0, "volume_ratio": 0.9},
        trend_context=ctx,
        ohlcv=df,
    )
    assert out["direction"] == "HOLD"
    assert "BUY vetoed" in out["prediction_reason"]


def test_coerce_regime_floor_relaxes_with_trend():
    """A 60% confidence BUY in an UPTREND should pass thanks to the relaxed floor."""
    df = _ohlcv_linear(50_000, 50_700, n=60)
    ctx = _detect_trend_context(df)
    assert ctx["regime"] == "UPTREND"
    raw = {
        "direction": "BUY",
        "confidence": 60.0,        # below default min_confidence (65) but above relaxed (58)
        "magnitude": 0.4,
        "entry_price": 50_700,
        "stop_loss": 50_600,
        "target_price": 50_900,
        "valid_minutes": 15,
        "reason": "With-trend pullback entry",
    }
    out = _coerce_result(
        raw,
        50_700.0,
        indicators={"rsi_14": 55.0, "volume_ratio": 1.2},
        trend_context=ctx,
        ohlcv=df,
    )
    assert out["direction"] == "BUY"


def test_coerce_regime_floor_does_not_relax_against_trend():
    """The same 60% confidence BUY in a SIDEWAYS regime should be coerced to HOLD."""
    rng = np.random.default_rng(11)
    closes = list(50_350 + rng.normal(0, 25.0, size=60))
    df = pd.DataFrame(
        {
            "open": closes,
            "high": [c + 5 for c in closes],
            "low": [c - 5 for c in closes],
            "close": closes,
            "volume": [1_000.0] * 60,
        },
        index=pd.date_range("2024-01-02 09:15", periods=60, freq="5min"),
    )
    ctx = _detect_trend_context(df)
    raw = {
        "direction": "BUY",
        "confidence": 60.0,
        "magnitude": 0.4,
        "entry_price": 50_350,
        "stop_loss": 50_300,
        "target_price": 50_500,
        "valid_minutes": 15,
        "reason": "Choppy tape",
    }
    out = _coerce_result(
        raw,
        50_350.0,
        indicators={"rsi_14": 50.0, "volume_ratio": 1.0},
        trend_context=ctx,
        ohlcv=df,
    )
    assert out["direction"] == "HOLD"   # base floor applies, 60 < 65
