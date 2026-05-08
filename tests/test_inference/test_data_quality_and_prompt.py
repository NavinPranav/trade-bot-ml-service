"""
Unit tests for the priority-fix items added to the ML service:

  * ``_ohlcv_quality_diagnostic`` — surface mixed timeframes, ATR%, stale data.
  * ``_maybe_append_legacy_guidance`` — auto-upgrade legacy custom admin prompts.
  * ``_baseline_min_bars`` and ``LiveTickBuffer.store_baseline`` —
    guard against poisoning the live re-prediction baseline with thin payloads.
  * ``_min_bars_for_horizon`` (main.py) — per-horizon minimum-bars resolution.

These all exercise pure Python helpers — no Gemini, no gRPC, no FastAPI runtime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config import settings
from app.inference.gemini_predictor import (
    _SYSTEM_PROMPT_TEMPLATE,
    _maybe_append_legacy_guidance,
    _ohlcv_quality_diagnostic,
)
from app.grpc_server.live_tick_buffer import LiveTickBuffer, _baseline_min_bars
from app.main import _min_bars_for_horizon


# ── Helpers ────────────────────────────────────────────────────────────────


def _ohlcv(closes: list[float], *, freq: str = "5min", start: str = "2024-01-02 09:15") -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq=freq)
    closes_arr = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes_arr[0]], closes_arr[:-1]])
    highs = np.maximum(opens, closes_arr) + 2.0
    lows = np.minimum(opens, closes_arr) - 2.0
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes_arr, "volume": 1_000.0},
        index=idx,
    )


# ── _ohlcv_quality_diagnostic ─────────────────────────────────────────────


class TestOhlcvQualityDiagnostic:
    def test_clean_5min_frame_has_no_warnings(self):
        # Realistic Sensex-like prices: ~80000 with tiny per-bar drift so ATR% stays sensible.
        rng = np.random.default_rng(seed=42)
        closes = (80_000 + rng.normal(0, 30, size=60).cumsum()).tolist()
        df = _ohlcv(closes)
        out = _ohlcv_quality_diagnostic(df, "15M")
        assert out["bars"] == 60
        assert out["bar_interval_seconds_modal"] == 300
        # Allow a `stale_last_bar_minutes` warning since the test uses a fixed 2024 date,
        # but reject any data-quality warnings (mixed intervals, bad H/L, drift, ATR%).
        unexpected = [
            w for w in out["warnings"]
            if not w.startswith("stale_last_bar_minutes=")
        ]
        assert unexpected == [], unexpected
        assert out["bad_high_low_bars"] == 0
        assert "atr_14" in out

    def test_empty_frame_emits_warning(self):
        out = _ohlcv_quality_diagnostic(pd.DataFrame(), "15M")
        assert out["bars"] == 0
        assert "empty_ohlcv" in out["warnings"]

    def test_mixed_bar_intervals_flagged(self):
        idx = pd.DatetimeIndex(
            [
                "2024-01-02 09:15", "2024-01-02 09:20", "2024-01-02 09:25",
                "2024-01-02 09:30", "2024-01-02 09:31", "2024-01-02 09:32",
                "2024-01-02 09:33", "2024-01-02 09:34", "2024-01-02 09:35",
                "2024-01-02 09:36",
            ]
        )
        n = len(idx)
        df = pd.DataFrame(
            {
                "open": np.linspace(100, 110, n),
                "high": np.linspace(101, 111, n),
                "low": np.linspace(99, 109, n),
                "close": np.linspace(100.5, 110.5, n),
                "volume": np.full(n, 1000.0),
            },
            index=idx,
        )
        out = _ohlcv_quality_diagnostic(df, "15M")
        assert any("mixed_bar_intervals" in w for w in out["warnings"]), out

    def test_high_below_low_flagged(self):
        df = _ohlcv([100 + i for i in range(20)])
        df.loc[df.index[5], "high"] = 50.0
        df.loc[df.index[5], "low"] = 200.0
        out = _ohlcv_quality_diagnostic(df, "15M")
        assert out["bad_high_low_bars"] >= 1
        assert any("high_lt_low_bars" in w for w in out["warnings"])

    def test_window_drift_flagged_on_instrument_swap(self):
        # First 30 bars at index level ~50000, next 30 at ~75 — looks like a
        # (broken) concatenation of Sensex with a stock price.
        head = _ohlcv([50_000 + i for i in range(30)])
        tail = _ohlcv(
            [75 + i * 0.1 for i in range(30)],
            start="2024-01-02 11:45",
        )
        df = pd.concat([head, tail])
        out = _ohlcv_quality_diagnostic(df, "15M")
        assert out["window_drift_pct"] > 25
        assert any("large_window_drift_pct" in w for w in out["warnings"])


# ── _maybe_append_legacy_guidance ─────────────────────────────────────────


class TestPromptUpgrade:
    def test_default_template_unchanged(self):
        out = _maybe_append_legacy_guidance(_SYSTEM_PROMPT_TEMPLATE, is_custom=False)
        assert out == _SYSTEM_PROMPT_TEMPLATE

    def test_default_template_already_has_guidance_when_custom_flag_set(self):
        # Default template DOES include the new blocks, so even when reused
        # under is_custom=True nothing is appended.
        out = _maybe_append_legacy_guidance(_SYSTEM_PROMPT_TEMPLATE, is_custom=True)
        assert out == _SYSTEM_PROMPT_TEMPLATE

    def test_legacy_prompt_gets_trend_and_volume_blocks(self):
        legacy = (
            "Predict the next {target_minutes} minutes for Bank Nifty. "
            "Use the indicators payload to decide direction. min_confidence={min_confidence}. "
            "min_risk_reward={min_risk_reward}."
        )
        out = _maybe_append_legacy_guidance(legacy, is_custom=True)
        assert "TREND CONTEXT" in out
        assert "VOLUME CONFIRMATION" in out
        assert out.startswith(legacy)

    def test_partial_legacy_prompt_with_trend_only_gets_volume_appended_too(self):
        partial = (
            "Predict for {target_minutes}. Use trend_context as a hard prior. "
            "min_confidence={min_confidence}. min_risk_reward={min_risk_reward}."
        )
        out = _maybe_append_legacy_guidance(partial, is_custom=True)
        # Already has 'trend_context' so no append needed when *both* checks pass.
        assert "TREND CONTEXT" in out
        # But it lacks the VOLUME block, so the helper still appends it.
        assert "VOLUME CONFIRMATION" in out


# ── LiveTickBuffer.store_baseline guard ───────────────────────────────────


class TestBaselineGuard:
    def test_store_baseline_accepts_full_payload(self):
        buf = LiveTickBuffer()
        df = _ohlcv([100 + i for i in range(150)])
        buf.store_baseline("15M", df, None, engine="AI", underlying_symbol="BANKNIFTY")
        assert buf.has_baseline()
        assert buf.get_baseline_horizon() == "15M"

    def test_store_baseline_refuses_to_degrade_existing_baseline(self):
        buf = LiveTickBuffer()
        good = _ohlcv([100 + i for i in range(150)])
        buf.store_baseline("15M", good, None, engine="AI", underlying_symbol="BANKNIFTY")

        thin = _ohlcv([110 + i for i in range(5)])
        buf.store_baseline("15M", thin, None, engine="AI", underlying_symbol="BANKNIFTY")

        # The thin payload must be ignored — baseline still has the original 150 bars.
        baseline = buf._baseline_ohlcv  # type: ignore[attr-defined]
        assert baseline is not None and len(baseline) == 150

    def test_store_baseline_accepts_thin_when_no_existing_baseline(self):
        buf = LiveTickBuffer()
        thin = _ohlcv([100 + i for i in range(5)])
        # No prior baseline → we accept (with a warning) so the system isn't completely cold.
        buf.store_baseline("15M", thin, None, engine="AI", underlying_symbol="BANKNIFTY")
        assert buf.has_baseline()


class TestBaselineMinBars:
    def test_per_horizon_floor_resolution(self):
        assert _baseline_min_bars("15M") >= 60
        assert _baseline_min_bars("5M") >= 30

    def test_unknown_horizon_falls_back_to_global(self):
        assert _baseline_min_bars("UNKNOWN") == max(1, settings.min_ohlcv_bars_grpc)


# ── _min_bars_for_horizon (REST endpoint) ─────────────────────────────────


class TestMinBarsForHorizon:
    def test_15m_uses_per_horizon_floor(self):
        assert _min_bars_for_horizon("15M") == settings.min_ohlcv_bars_by_horizon["15M"]

    def test_lower_case_is_normalised(self):
        assert _min_bars_for_horizon("15m") == settings.min_ohlcv_bars_by_horizon["15M"]

    def test_unknown_horizon_falls_back_to_global(self):
        assert _min_bars_for_horizon("XYZ") == max(1, settings.min_ohlcv_bars_grpc)

    def test_none_falls_back_to_global(self):
        assert _min_bars_for_horizon(None) == max(1, settings.min_ohlcv_bars_grpc)
