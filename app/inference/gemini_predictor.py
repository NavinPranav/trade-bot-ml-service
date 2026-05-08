"""Google Gemini intra-day prediction via Generative Language REST API (httpx).

Optimised for Bank Nifty intra-day trading: returns entry price, stop-loss,
target price, risk/reward, and a validity window alongside direction/confidence.
Trading levels are serialised as a JSON suffix in prediction_reason so the Java
backend can parse them without a proto schema change.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import httpx
import numpy as np
import pandas as pd
from loguru import logger


# ── Logging helpers ────────────────────────────────────────────────────────
#
# Every prediction generates a short hex correlation ID (`pid`) which is
# emitted on every per-stage line plus the final structured PRED_RESULT line.
# In prod you can:
#   - grep PRED_RESULT  → one JSON line per prediction (parseable with jq)
#   - grep "pid=abc12345" → all lines for a single prediction
# All structured marker lines are at INFO so the default LOG_LEVEL=INFO surfaces
# them; per-stage detail lines are at INFO too because predictions run only
# every few minutes (≈ 60–360 lines/day total).


def _make_pid() -> str:
    """Short opaque correlation id for one prediction call."""
    return uuid.uuid4().hex[:8]


def _safe_round(v: Any, n: int = 2) -> Any:
    """Round numeric values for log payloads; pass through None/N/A unchanged."""
    if v is None:
        return None
    try:
        return round(float(v), n)
    except (TypeError, ValueError):
        return v


def _log_marker(marker: str, pid: str, payload: Optional[dict[str, Any]] = None) -> None:
    """Emit ``MARKER pid=xxxxxxxx {…json…}`` so prod logs are easy to grep + jq."""
    payload = payload or {}
    body = {"pid": pid, **payload}
    try:
        text = json.dumps(body, default=str, separators=(",", ":"))
    except Exception:
        text = str(body)
    logger.info("{} {}", marker, text)

from app.config import settings
from app.data.ingestion.news_fetcher import get_news_sentiment
from app.data.ingestion.vix_fetcher import derive_vix_from_ohlcv
from app.inference.checklist import run_checklist

# Maps horizon label → minutes ahead we are predicting
_HORIZON_MINUTES: dict[str, int] = {
    "5M": 5,
    "15M": 15,
    "30M": 30,
    "1H": 60,
    "1D": 375,   # full trading session fallback
}

_LEVELS_MARKER = "\n\n[TRADING_LEVELS]"


def _horizon_to_minutes(horizon: str) -> int:
    return _HORIZON_MINUTES.get(horizon.upper(), 15)


def _resolve_realtime(
    ohlcv: pd.DataFrame,
    sensex_quote: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    if sensex_quote and sensex_quote.get("price"):
        return {
            "price": float(sensex_quote["price"]),
            "change": float(sensex_quote.get("change") or 0),
            "change_pct": float(sensex_quote.get("change_pct") or 0),
        }
    if not ohlcv.empty and "close" in ohlcv.columns:
        last = float(ohlcv["close"].iloc[-1])
        return {"price": last, "change": 0.0, "change_pct": 0.0}
    return {"price": 0.0, "change": 0.0, "change_pct": 0.0}


def _ohlcv_quality_diagnostic(
    ohlcv: pd.DataFrame, horizon: str
) -> dict[str, Any]:
    """
    Inspects an OHLCV frame for the most common data-quality issues that produce
    surprising indicator values (especially ATR):

      * mixed timeframes (the broker stitched 1-min and 5-min candles together)
      * absurd high/low ranges (high < low, high == low for many bars, etc.)
      * stale data (last bar timestamp is hours old during market hours)
      * tail vs full-history price gaps (suggests instrument switch / bad concat)

    Returns a JSON-serialisable dict suitable for structured logging plus a list
    of human-readable ``warnings`` strings for grep-friendly alerts.
    """
    out: dict[str, Any] = {
        "horizon": horizon,
        "bars": int(len(ohlcv)) if ohlcv is not None else 0,
        "warnings": [],
    }
    if ohlcv is None or ohlcv.empty:
        out["warnings"].append("empty_ohlcv")
        return out

    df = ohlcv

    # 1) Bar interval consistency — a healthy 5M frame should have most diffs ≈ 300 s.
    try:
        if isinstance(df.index, pd.DatetimeIndex) and len(df) >= 3:
            diffs = (
                df.index.to_series().diff().dropna().dt.total_seconds().astype(int)
            )
            if not diffs.empty:
                most_common = int(diffs.mode().iloc[0])
                unique_diffs = sorted(diffs.unique().tolist())[:8]
                # Filter out market-close-to-next-open jumps (>1 hour) which are normal.
                intra_session = diffs[diffs <= 3600]
                non_modal = (
                    int((intra_session != most_common).sum())
                    if not intra_session.empty
                    else 0
                )
                out["bar_interval_seconds_modal"] = most_common
                out["bar_interval_seconds_unique"] = unique_diffs
                out["bar_interval_non_modal_count"] = non_modal
                if non_modal > max(2, int(0.1 * len(intra_session))):
                    out["warnings"].append(
                        f"mixed_bar_intervals modal={most_common}s "
                        f"non_modal_bars={non_modal}/{len(intra_session)}"
                    )
    except Exception:
        pass

    # 2) High/Low sanity.
    try:
        if {"high", "low"}.issubset(df.columns):
            bad_hl = int(((df["high"] < df["low"])).sum())
            flat_hl = int((df["high"] == df["low"]).sum())
            out["bad_high_low_bars"] = bad_hl
            out["flat_high_low_bars"] = flat_hl
            if bad_hl > 0:
                out["warnings"].append(f"high_lt_low_bars={bad_hl}")
            if flat_hl > max(3, int(0.2 * len(df))):
                out["warnings"].append(
                    f"too_many_flat_bars high==low {flat_hl}/{len(df)}"
                )
    except Exception:
        pass

    # 3) ATR% snapshot (informational — if it’s out of a sensible band, flag it).
    try:
        if {"high", "low", "close"}.issubset(df.columns) and len(df) >= 14:
            n = 14
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            close = df["close"].astype(float)
            prev_close = close.shift(1)
            tr = pd.concat(
                [
                    (high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr = float(tr.rolling(n).mean().iloc[-1])
            last_close = float(close.iloc[-1])
            atr_pct = (atr / last_close * 100.0) if last_close > 0 else None
            out["atr_14"] = round(atr, 2)
            out["last_close"] = round(last_close, 2)
            out["atr_pct_of_price"] = (
                round(atr_pct, 3) if atr_pct is not None else None
            )
            # Index intra-day ATR% is typically 0.05% – 0.5%. Anything > 2% on a 5M frame
            # is almost always bad data.
            if atr_pct is not None and atr_pct > 2.0:
                out["warnings"].append(f"atr_pct_high={atr_pct:.2f}%")
            if atr_pct is not None and atr_pct < 0.005:
                out["warnings"].append(f"atr_pct_near_zero={atr_pct:.4f}%")
    except Exception:
        pass

    # 4) Stale-data sniff — last bar should be within ~30 minutes during market hours.
    try:
        if isinstance(df.index, pd.DatetimeIndex) and len(df) >= 1:
            last_ts = df.index[-1]
            try:
                last_naive = pd.Timestamp(last_ts).tz_localize(None)
            except Exception:
                last_naive = pd.Timestamp(last_ts)
            now_naive = pd.Timestamp.utcnow().tz_localize(None)
            age_min = max(
                0.0, (now_naive - last_naive).total_seconds() / 60.0
            )
            out["last_bar_age_minutes"] = round(age_min, 1)
            if age_min > 90:
                out["warnings"].append(
                    f"stale_last_bar_minutes={age_min:.0f}"
                )
    except Exception:
        pass

    # 5) Recent vs early window price level — if the index switched to a different
    # instrument the close price will jump by orders of magnitude.
    try:
        if "close" in df.columns and len(df) >= 30:
            head_med = float(df["close"].head(10).median())
            tail_med = float(df["close"].tail(10).median())
            if head_med > 0:
                drift_pct = abs(tail_med - head_med) / head_med * 100.0
                out["window_drift_pct"] = round(drift_pct, 2)
                if drift_pct > 25.0:
                    out["warnings"].append(
                        f"large_window_drift_pct={drift_pct:.1f}%"
                    )
    except Exception:
        pass

    return out


def _ohlcv_tail_records(ohlcv: pd.DataFrame, max_rows: int = 120) -> list[dict[str, Any]]:
    tail = ohlcv.tail(max_rows)
    rows: list[dict[str, Any]] = []
    for idx, row in tail.iterrows():
        rec: dict[str, Any] = {"ts": str(idx)}
        for col in ("open", "high", "low", "close", "volume"):
            if col in tail.columns:
                try:
                    rec[col] = round(float(row[col]), 2)
                except (TypeError, ValueError):
                    rec[col] = None
        rows.append(rec)
    return rows


def _vix_tail_summary(vix: pd.DataFrame, max_rows: int = 5) -> list[dict[str, Any]]:
    if vix.empty:
        return []
    cols = [c for c in ("close", "open", "high", "low") if c in vix.columns]
    if not cols:
        return []
    use = vix[cols].tail(max_rows)
    out: list[dict[str, Any]] = []
    for idx, row in use.iterrows():
        item: dict[str, Any] = {"ts": str(idx)}
        for c in cols:
            try:
                item[c] = round(float(row[c]), 2)
            except (TypeError, ValueError):
                item[c] = None
        out.append(item)
    return out


def _compute_indicator_snapshot(ohlcv: pd.DataFrame) -> dict[str, Any]:
    """Compute a technical indicator snapshot from the last bar of OHLCV."""
    snap: dict[str, Any] = {}
    try:
        import ta  # noqa: PLC0415 — optional at call-time only
        close = ohlcv["close"].astype(float)
        high = ohlcv["high"].astype(float)
        low = ohlcv["low"].astype(float)
        n = len(close)

        def _last(s: pd.Series) -> float | None:
            if s.empty:
                return None
            v = s.iloc[-1]
            return round(float(v), 3) if pd.notna(v) else None

        # RSI
        if n >= 15:
            snap["rsi_14"] = _last(ta.momentum.rsi(close, window=14))

        # EMAs
        if n >= 9:
            ema9_s = ta.trend.ema_indicator(close, window=9)
            snap["ema_9"] = _last(ema9_s)
            if snap.get("ema_9"):
                snap["price_vs_ema9_pct"] = round(
                    (float(close.iloc[-1]) - snap["ema_9"]) / snap["ema_9"] * 100, 3
                )
        if n >= 20:
            snap["ema_20"] = _last(ta.trend.ema_indicator(close, window=20))
        if n >= 21:
            snap["ema_21"] = _last(ta.trend.ema_indicator(close, window=21))
            if snap.get("ema_9") and snap.get("ema_21"):
                snap["ema9_above_ema21"] = snap["ema_9"] > snap["ema_21"]
        if n >= 50:
            snap["ema_50"] = _last(ta.trend.ema_indicator(close, window=50))

        # MACD (needs 26 bars)
        if n >= 27:
            macd_obj = ta.trend.MACD(close)
            snap["macd"] = _last(macd_obj.macd())
            snap["macd_signal"] = _last(macd_obj.macd_signal())
            snap["macd_hist"] = _last(macd_obj.macd_diff())

        # Bollinger Bands
        if n >= 20:
            bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
            snap["bb_upper"] = _last(bb.bollinger_hband())
            snap["bb_lower"] = _last(bb.bollinger_lband())
            snap["bb_pct"] = _last(bb.bollinger_pband())
            snap["bb_width"] = _last(bb.bollinger_wband())

        # ATR
        if n >= 14:
            snap["atr_14"] = _last(
                ta.volatility.average_true_range(high, low, close, window=14)
            )

        # Volume ratio
        try:
            vol = ohlcv["volume"].astype(float)
            if n >= 20 and vol.sum() > 0:
                vol_sma = vol.rolling(20).mean()
                last_sma = float(vol_sma.iloc[-1])
                if last_sma > 0:
                    snap["volume_ratio"] = round(float(vol.iloc[-1]) / last_sma, 2)
        except Exception:
            pass

    except ImportError:
        logger.warning("ta library not installed — skipping indicator snapshot")
    except Exception as e:
        logger.warning(f"Indicator snapshot failed: {e}")

    return snap


# ── Deterministic multi-timeframe trend detector ────────────────────────────
#
# The Gemini prompt sometimes emits BUY in a clear downtrend (or SELL in a clear
# uptrend) because it overweights short-term reversal patterns / mean-reversion
# cues. This module computes a model-free regime label from raw OHLCV using a
# scoring system and exposes a guardrail that vetoes counter-trend directional
# calls when both the primary (5m) and the higher (15m) timeframe agree.

_TREND_SCORE_THRESHOLD = 3
_TREND_MIN_BARS = 30
_TREND_INTRADAY_VWAP_BARS = 75   # approx one IST trading day on 5-minute bars
_TREND_REGRESSION_BARS = 30
_TREND_SWING_SEGMENT = 5         # last/prior 5-bar segments for higher-high / lower-low
_TREND_VWAP_DEVIATION_BPS = 0.05  # +/- 0.05 % away from VWAP counts as above/below


def _last_float(series: pd.Series) -> float | None:
    if series is None or series.empty:
        return None
    v = series.iloc[-1]
    if pd.isna(v):
        return None
    return float(v)


def _ema_local(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _normalised_slope(series: pd.Series, lookback: int = 5) -> float:
    """Last value vs `lookback` bars ago, normalised by the older value (returns a fraction)."""
    if series is None or len(series) < lookback + 1:
        return 0.0
    earlier = float(series.iloc[-lookback - 1])
    latest = float(series.iloc[-1])
    base = abs(earlier) or 1.0
    return (latest - earlier) / base


def _intraday_vwap_value(ohlcv: pd.DataFrame) -> float | None:
    """Volume-weighted average price using the last ~75 bars (≈ one trading day on 5m)."""
    try:
        if ohlcv is None or ohlcv.empty or "volume" not in ohlcv.columns:
            return None
        tail = ohlcv.tail(_TREND_INTRADAY_VWAP_BARS)
        vol = tail["volume"].astype(float)
        if vol.sum() <= 0:
            return None
        typical = (
            tail["high"].astype(float)
            + tail["low"].astype(float)
            + tail["close"].astype(float)
        ) / 3.0
        return float((typical * vol).sum() / vol.sum())
    except Exception:
        return None


def _swing_structure_score(ohlcv: pd.DataFrame, segment: int = _TREND_SWING_SEGMENT) -> int:
    """+1 for higher-highs+higher-lows, -1 for lower-highs+lower-lows, 0 otherwise."""
    if ohlcv is None or len(ohlcv) < 2 * segment:
        return 0
    recent = ohlcv.tail(segment)
    prior = ohlcv.iloc[-(2 * segment):-segment]
    if recent.empty or prior.empty:
        return 0
    rec_high = float(recent["high"].max())
    rec_low = float(recent["low"].min())
    pri_high = float(prior["high"].max())
    pri_low = float(prior["low"].min())
    if rec_high > pri_high and rec_low > pri_low:
        return 1
    if rec_high < pri_high and rec_low < pri_low:
        return -1
    return 0


def _detect_trend_single_tf(ohlcv: pd.DataFrame) -> dict[str, Any]:
    """Score-based regime detector for a single timeframe.

    Returns ``{regime, score, reasons, evidence}`` where ``regime`` is one of
    ``UPTREND`` / ``DOWNTREND`` / ``SIDEWAYS`` / ``UNKNOWN``. Pure pandas/numpy
    so it stays fast and dependency-free.
    """
    if ohlcv is None or ohlcv.empty or len(ohlcv) < _TREND_MIN_BARS:
        return {"regime": "UNKNOWN", "score": 0, "reasons": [], "evidence": {"bars": 0 if ohlcv is None else len(ohlcv)}}

    close = ohlcv["close"].astype(float)
    last_close = float(close.iloc[-1])

    score = 0
    reasons: list[str] = []
    evidence: dict[str, Any] = {"bars": int(len(ohlcv)), "last_close": round(last_close, 2)}

    # 1) EMA stack + EMA50 slope (strongest single signal).
    ema9 = _ema_local(close, 9)
    ema21 = _ema_local(close, 21)
    ema50 = _ema_local(close, 50) if len(close) >= 50 else None
    e9 = _last_float(ema9)
    e21 = _last_float(ema21)
    e50 = _last_float(ema50) if ema50 is not None else None
    if e9 is not None and e21 is not None:
        evidence["ema_9"] = round(e9, 2)
        evidence["ema_21"] = round(e21, 2)
    if e50 is not None:
        evidence["ema_50"] = round(e50, 2)
        slope_50 = _normalised_slope(ema50, lookback=5)
        evidence["ema_50_slope_pct"] = round(slope_50 * 100, 4)
        if e9 is not None and e21 is not None:
            if e9 > e21 > e50 and slope_50 >= 0:
                score += 2
                reasons.append("EMA stack bullish (EMA9>EMA21>EMA50) with EMA50 sloping up")
            elif e9 < e21 < e50 and slope_50 <= 0:
                score -= 2
                reasons.append("EMA stack bearish (EMA9<EMA21<EMA50) with EMA50 sloping down")
    elif e9 is not None and e21 is not None:
        # Fallback when not enough bars for EMA50.
        if e9 > e21:
            score += 1
            reasons.append("EMA9 above EMA21 (EMA50 unavailable)")
        elif e9 < e21:
            score -= 1
            reasons.append("EMA9 below EMA21 (EMA50 unavailable)")

    # 2) Price vs intraday VWAP (institutional bias proxy).
    vwap = _intraday_vwap_value(ohlcv)
    if vwap and vwap > 0:
        evidence["vwap"] = round(vwap, 2)
        deviation_pct = (last_close - vwap) / vwap * 100
        evidence["price_vs_vwap_pct"] = round(deviation_pct, 3)
        if deviation_pct > _TREND_VWAP_DEVIATION_BPS:
            score += 1
            reasons.append("Price above intraday VWAP")
        elif deviation_pct < -_TREND_VWAP_DEVIATION_BPS:
            score -= 1
            reasons.append("Price below intraday VWAP")

    # 3) Swing structure: HH/HL vs LH/LL.
    swing = _swing_structure_score(ohlcv)
    evidence["swing_structure"] = swing
    if swing > 0:
        score += 1
        reasons.append("Higher-highs and higher-lows over last 5 bars")
    elif swing < 0:
        score -= 1
        reasons.append("Lower-highs and lower-lows over last 5 bars")

    # 4) Linear regression slope of last 30 closes (catches gradual drifts EMAs can lag).
    try:
        seg = close.tail(_TREND_REGRESSION_BARS).reset_index(drop=True)
        if len(seg) >= 10:
            xs = np.arange(len(seg), dtype=float)
            slope, _intercept = np.polyfit(xs, seg.to_numpy(dtype=float), 1)
            base = abs(float(seg.iloc[0])) or 1.0
            slope_pct_per_bar = (slope / base) * 100
            evidence["regression_slope_pct_per_bar"] = round(slope_pct_per_bar, 4)
            if slope_pct_per_bar > 0.01:
                score += 1
                reasons.append("Positive regression slope on last 30 closes")
            elif slope_pct_per_bar < -0.01:
                score -= 1
                reasons.append("Negative regression slope on last 30 closes")
    except Exception:
        pass

    if score >= _TREND_SCORE_THRESHOLD:
        regime = "UPTREND"
    elif score <= -_TREND_SCORE_THRESHOLD:
        regime = "DOWNTREND"
    else:
        regime = "SIDEWAYS"

    return {"regime": regime, "score": int(score), "reasons": reasons, "evidence": evidence}


def _resample_to_higher_tf(ohlcv: pd.DataFrame, rule: str = "15min") -> pd.DataFrame:
    """Resample 5-minute OHLCV to a higher timeframe; returns empty frame on failure."""
    try:
        if ohlcv is None or ohlcv.empty:
            return pd.DataFrame()
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            return pd.DataFrame()
        agg: dict[str, str] = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in ohlcv.columns:
            agg["volume"] = "sum"
        return ohlcv.resample(rule).agg(agg).dropna(subset=["close"])
    except Exception:
        return pd.DataFrame()


def _detect_trend_context(ohlcv: pd.DataFrame) -> dict[str, Any]:
    """Multi-timeframe trend assessment: primary (raw) + higher (15m resample).

    The combined ``regime`` is the primary timeframe regime, demoted to
    ``SIDEWAYS`` whenever the two timeframes disagree directionally — this is
    what feeds the post-AI guardrail.
    """
    primary = _detect_trend_single_tf(ohlcv)
    higher_df = _resample_to_higher_tf(ohlcv, rule="15min")
    higher = _detect_trend_single_tf(higher_df) if not higher_df.empty else {
        "regime": "UNKNOWN", "score": 0, "reasons": [], "evidence": {"bars": 0},
    }

    primary_regime = primary.get("regime") or "UNKNOWN"
    higher_regime = higher.get("regime") or "UNKNOWN"

    agreement = (
        primary_regime == higher_regime
        and primary_regime in ("UPTREND", "DOWNTREND")
    )
    combined = primary_regime
    if (
        primary_regime in ("UPTREND", "DOWNTREND")
        and higher_regime in ("UPTREND", "DOWNTREND")
        and primary_regime != higher_regime
    ):
        combined = "SIDEWAYS"

    return {
        "regime": combined,
        "primary_regime": primary_regime,
        "higher_regime": higher_regime,
        "agreement": agreement,
        "primary_score": int(primary.get("score") or 0),
        "higher_score": int(higher.get("score") or 0),
        "primary_reasons": primary.get("reasons") or [],
        "higher_reasons": higher.get("reasons") or [],
        "evidence": {
            "primary": primary.get("evidence") or {},
            "higher": higher.get("evidence") or {},
        },
    }


def _detect_reversal_confirmation(
    ohlcv: pd.DataFrame,
    indicators: Optional[dict[str, Any]],
    direction: str,
) -> dict[str, Any]:
    """Score deterministic counter-trend reversal evidence for a directional call.

    Returns ``{"confirmed": bool, "score": int, "reasons": [...]}``. ``confirmed``
    becomes ``True`` once the score crosses the ``reversal_confirmation_min_signals``
    policy threshold. Used to override the trend guardrail when a genuine reversal
    is unfolding (e.g. bullish engulfing on a volume spike at the end of a downtrend).

    All checks are pure pandas/numpy. No ``ta`` import here — we read the
    pre-computed RSI / volume_ratio out of the indicator snapshot.
    """
    out: dict[str, Any] = {"confirmed": False, "score": 0, "reasons": [], "evidence": {}}
    if direction not in ("BUY", "SELL"):
        return out
    if ohlcv is None or len(ohlcv) < 2:
        return out

    last = ohlcv.iloc[-1]
    prev = ohlcv.iloc[-2]
    try:
        o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
        po, _ph, _pl, pc = float(prev["open"]), float(prev["high"]), float(prev["low"]), float(prev["close"])
    except (KeyError, TypeError, ValueError):
        return out
    body = abs(c - o)
    safe_body = max(body, 1e-6)

    score = 0
    reasons: list[str] = []
    evidence: dict[str, Any] = {"last_close": round(c, 2), "prev_close": round(pc, 2)}

    rsi_val = (indicators or {}).get("rsi_14")
    vol_ratio = (indicators or {}).get("volume_ratio")
    vwap = _intraday_vwap_value(ohlcv)
    if vwap is not None:
        evidence["vwap"] = round(vwap, 2)
    if rsi_val is not None:
        evidence["rsi_14"] = rsi_val
    if vol_ratio is not None:
        evidence["volume_ratio"] = vol_ratio

    if direction == "BUY":
        # 1) Bullish engulfing — last bar is green, prev red, body engulfs prev body.
        if c > o and pc < po and c >= po and o <= pc:
            score += 1
            reasons.append("Bullish engulfing pattern")
        # 2) Hammer / pin bar — long lower wick, small upper wick, bull-ish body.
        lower_wick = (o - l) if c >= o else (c - l)
        upper_wick = (h - c) if c >= o else (h - o)
        if lower_wick > 2.0 * safe_body and upper_wick < safe_body and c >= o:
            score += 1
            reasons.append("Hammer-style lower wick")
        # 3) Volume spike — buyers stepping in.
        try:
            if vol_ratio is not None and float(vol_ratio) >= 1.3:
                score += 1
                reasons.append(f"Volume {float(vol_ratio):.2f}x average")
        except (TypeError, ValueError):
            pass
        # 4) RSI rising from oversold zone — momentum turning up.
        try:
            if rsi_val is not None and 35 <= float(rsi_val) <= 55:
                score += 1
                reasons.append(f"RSI {float(rsi_val):.1f} rising from oversold")
        except (TypeError, ValueError):
            pass
        # 5) VWAP reclaim — last close back above VWAP, prev close was below.
        if vwap is not None and pc < vwap and c >= vwap:
            score += 1
            reasons.append("Reclaimed VWAP from below")
    else:  # SELL
        if c < o and pc > po and c <= po and o >= pc:
            score += 1
            reasons.append("Bearish engulfing pattern")
        upper_wick = (h - o) if c <= o else (h - c)
        lower_wick = (o - l) if c <= o else (c - l)
        if upper_wick > 2.0 * safe_body and lower_wick < safe_body and c <= o:
            score += 1
            reasons.append("Shooting-star upper wick")
        try:
            if vol_ratio is not None and float(vol_ratio) >= 1.3:
                score += 1
                reasons.append(f"Volume {float(vol_ratio):.2f}x average")
        except (TypeError, ValueError):
            pass
        try:
            if rsi_val is not None and 45 <= float(rsi_val) <= 65:
                score += 1
                reasons.append(f"RSI {float(rsi_val):.1f} falling from overbought")
        except (TypeError, ValueError):
            pass
        if vwap is not None and pc > vwap and c <= vwap:
            score += 1
            reasons.append("Lost VWAP from above")

    try:
        required = max(1, int(_policy("reversal_confirmation_min_signals")))
    except (KeyError, TypeError, ValueError):
        required = 3

    out.update({
        "confirmed": score >= required,
        "score": int(score),
        "reasons": reasons,
        "evidence": evidence,
        "required_signals": int(required),
    })
    return out


def _apply_trend_guardrail(
    direction: str,
    trend_context: Optional[dict[str, Any]],
    reversal: Optional[dict[str, Any]] = None,
) -> tuple[str, Optional[str]]:
    """Veto BUY in a downtrend / SELL in an uptrend, unless reversal evidence overrides.

    Returns ``(final_direction, downgrade_reason_or_None)``. The veto is
    deliberately strict: if any of {primary, higher, combined} timeframes
    explicitly disagrees with the directional call we coerce to HOLD. SIDEWAYS
    or UNKNOWN never triggers the veto on its own — the existing confidence /
    R:R / dead-tape gates already cover those.

    When ``reversal["confirmed"]`` is ``True`` the veto is skipped and the
    directional call is allowed through. The caller is responsible for
    annotating ``prediction_reason`` so the user can see both the regime
    mismatch and the reversal evidence that justified passing through.
    """
    if direction not in ("BUY", "SELL"):
        return direction, None
    if not trend_context:
        return direction, None

    combined = trend_context.get("regime") or "UNKNOWN"
    primary = trend_context.get("primary_regime") or "UNKNOWN"
    higher = trend_context.get("higher_regime") or "UNKNOWN"

    opposing_label = "DOWNTREND" if direction == "BUY" else "UPTREND"
    opposing = {combined, primary, higher} & {opposing_label}
    if not opposing:
        return direction, None

    if reversal and reversal.get("confirmed"):
        # Pass-through allowed. We still annotate so the operator sees both sides.
        score = reversal.get("score", 0)
        reasons = ", ".join(reversal.get("reasons") or []) or "pattern + indicator confluence"
        note = (
            f"{direction} allowed despite {opposing_label} regime "
            f"(combined={combined}, primary={primary}, higher={higher}) — "
            f"reversal confirmation score={score}: {reasons}"
        )
        return direction, note

    veto = (
        f"{direction} vetoed by trend guardrail — combined={combined}, "
        f"primary(5m)={primary}, higher(15m)={higher}"
    )
    return "HOLD", veto


_SYSTEM_PROMPT_TEMPLATE = """\
You are an elite intra-day Bank Nifty trader and options analyst with 15 years of experience trading Indian derivatives. You specialize in reading price action, order flow, and volatility to predict short-term moves.

CURRENT MARKET SNAPSHOT:
- Spot: {spot_price}
- Today Open: {open} | High: {high} | Low: {low}
- Previous Close: {prev_close} | Day Change: {day_change_pct}%
- EMA 20: {ema_20} | EMA 50: {ema_50}
- RSI(14): {rsi_14}
- VWAP: {vwap}
- ATR(14): {atr_14}
- MACD: {macd} | Signal: {macd_signal} | Histogram: {macd_histogram}
- India VIX: {india_vix} ({vix_change}% change)
- PCR (OI): {pcr_oi} | PCR (Volume): {pcr_volume}
- Max Pain: {max_pain}
- Volume vs 20-day avg: {volume_ratio}x
- Support: S1={support_1}, S2={support_2}
- Resistance: R1={resistance_1}, R2={resistance_2}
- Pivot: {pivot}
- Market Phase: {market_phase}
- Time (IST): {time_ist}

LAST 5 COMPLETED CANDLES (5-min, newest first):
{last_5_candles}

YOUR TASK:
Predict what Bank Nifty will do in the NEXT {target_minutes} MINUTES.
You are also given recent_ohlcv_bars in the user payload: use the full raw window (normally the latest 120 bars)
to judge trend persistence, pullbacks, breakouts, and lower-high/lower-low or higher-high/higher-low structure.

TREND CONTEXT (deterministic, multi-timeframe):
The user payload contains a trend_context object computed directly from OHLCV with keys:
regime (combined), primary_regime (5-min), higher_regime (15-min), agreement (bool),
primary_score, higher_score (each in roughly −5..+5), and reasons describing why each
timeframe is labelled UPTREND / DOWNTREND / SIDEWAYS / UNKNOWN. Treat this as a hard
prior: if higher_regime is DOWNTREND, do not output BUY unless you have very strong
reversal confirmation (e.g., bullish engulfing on heavy volume + RSI cross up from
oversold + reclaim of VWAP). The same applies in reverse for SELL in an UPTREND.
A post-AI guardrail in the service will downgrade BUY-in-DOWNTREND or SELL-in-UPTREND
to HOLD unless deterministic reversal evidence (engulfing/hammer + volume spike + RSI
zone turn + VWAP reclaim) is present, so do not emit speculative counter-trend calls.

VOLUME CONFIRMATION:
A separate post-AI gate may force HOLD when last-bar volume vs the 20-bar average is
below the configured ratio (default disabled). When you see volume_ratio < 0.7 in the
indicators payload, treat the move as low-conviction and lean toward HOLD or, at minimum,
reduce confidence by 10 points so the existing floor catches it.

THINK THROUGH THIS STEP BY STEP BEFORE ANSWERING:
1. TREND: Is EMA 20 above or below EMA 50? What is the slope — widening or narrowing?
2. MOMENTUM: Is RSI overbought (>70), oversold (<30), or in the buy zone (55-70) / sell zone (30-45) / neutral (45-55)?
3. PRICE vs VWAP: Is price above VWAP (institutional buying) or below (selling)? How far from VWAP?
4. CANDLE PATTERN: Are the last 5 candles making higher highs (bullish) or lower lows (bearish)? Any reversal patterns (doji, hammer, engulfing)?
5. OPTIONS DATA: PCR > 1.2 is bullish (put writers confident), PCR < 0.8 is bearish. If PCR/max pain are N/A, ignore this factor instead of treating it as bearish, neutral, or a HOLD reason.
6. VOLATILITY: Is VIX rising (expect bigger move) or falling (expect range)? Use ATR to size stop loss realistically.
7. SUPPORT/RESISTANCE: Is price near any key level? A bounce off support = BUY, rejection at resistance = SELL, breakout = strong move.
8. TIME OF DAY: 9:15-9:30 = volatile/unreliable, 12:00-13:00 = low volume/choppy, 14:00-15:15 = trend resumes. Adjust confidence based on time.
9. VOLUME: Volume > 1.5x average = conviction in the move. Volume < 0.7x = low conviction, prefer HOLD.
10. CONFLUENCE: Count only available, non-N/A signals. For a clear directional market, BUY/SELL is valid with 3+ aligned signals when trend + price-vs-VWAP + momentum/candles agree. Use HOLD for conflict, chop, poor R:R, or true low confidence — not merely because one optional signal is unavailable.

STRICT RULES:
1. If confidence < {min_confidence} → direction MUST be "HOLD". No exceptions.
2. Risk-reward must be >= {min_risk_reward} for BUY or SELL. If ATR doesn't support a 1:{min_risk_reward} R:R within {target_minutes} minutes, use HOLD.
3. stop_loss must be realistic — use 0.3x to 0.5x ATR from entry. Never equal to entry or target.
4. target_price must be realistic — use 0.5x to 1.0x ATR from entry in the predicted direction.
5. If RSI is between 45-55 AND price is within 0.1% of VWAP AND candles are not trending → this is a NO TRADE zone → HOLD.
6. If only 15 minutes remain before market close (after 15:15 IST) → HOLD.
7. If VIX > 20 → widen stop loss by 1.5x and reduce confidence by 10.
8. magnitude = expected % move (positive for BUY, negative for SELL, 0 for HOLD).
9. predicted_volatility = your estimate of annualized volatility based on current VIX and ATR.
10. valid_minutes = how long this prediction stays valid (never more than {target_minutes}).

CONFIDENCE CALIBRATION:
- 90-100%: All 10 signals aligned + strong candle pattern + high volume breakout (extremely rare)
- 75-89%: 7-8 signals aligned + clear trend + good volume
- 65-74%: 3-6 available signals aligned + moderate directional conviction
- Below {min_confidence}%: Not enough confluence → MUST output HOLD

Respond with ONE JSON object only. No markdown, no backticks, no explanation outside the JSON.
Keys EXACTLY: direction, entry_price, stop_loss, target_price, risk_reward, confidence, magnitude, predicted_volatility, valid_minutes, reason

The "reason" field must be a single sentence citing the 2-3 strongest signals that drove your decision (e.g., "EMA bullish crossover + RSI 62 in buy zone + price bouncing off S1 support with 1.8x volume").\
"""


class _PromptStore:
    """Module-level singleton that holds the currently active admin-set prompt template.

    When empty the default _SYSTEM_PROMPT_TEMPLATE is used. The {target_minutes}
    placeholder is substituted at call time, so every stored template must contain it.
    """
    _custom: str = ""

    @classmethod
    def set(cls, template: str) -> None:
        cls._custom = template.strip()

    @classmethod
    def get(cls) -> str:
        return cls._custom

    @classmethod
    def clear(cls) -> None:
        cls._custom = ""


class _ModelStore:
    """Module-level singleton for the admin-selected Gemini model ID.

    Falls back to settings.gemini_model (GEMINI_MODEL env var) when not set,
    so existing deployments keep working without any admin action.
    """
    _model_id: str = ""

    @classmethod
    def set(cls, model_id: str) -> None:
        cls._model_id = model_id.strip()

    @classmethod
    def get(cls) -> str:
        return cls._model_id or settings.gemini_model.strip()

    @classmethod
    def clear(cls) -> None:
        cls._model_id = ""


class _ChecklistWeightStore:
    """Module-level singleton for the admin-configured checklist signal weight (0–100 %).

    Default is 40 — meaning 40% of the AI conviction should come from the
    rule-based checklist result, with the remaining 60% from its own OHLCV analysis.
    """
    _weight: int = 40

    @classmethod
    def set(cls, weight: int) -> None:
        cls._weight = max(0, min(100, int(weight)))

    @classmethod
    def get(cls) -> int:
        return cls._weight


class _PredictionPolicyStore:
    """Admin-tunable prediction policy. Unset keys fall back to process env (Settings)."""

    _overrides: dict[str, Any] = {}

    _SETTINGS_ATTR: dict[str, str] = {
        "min_confidence": "gemini_min_confidence",
        "min_risk_reward": "gemini_min_risk_reward",
        "strong_trend_min_ema_gap_pct": "gemini_strong_trend_min_ema_gap_pct",
        "relaxed_confidence_floor_strong_trend": "gemini_relaxed_confidence_floor_strong_trend",
        "sell_near_support_min_confidence": "gemini_sell_near_support_min_confidence",
        "min_atr_pct_of_price": "gemini_min_atr_pct_of_price",
        "rate_limit_max_retries": "gemini_429_max_retries",
        "rate_limit_retry_base_delay_sec": "gemini_429_retry_base_delay_sec",
        "trend_guardrail_enabled": "gemini_trend_guardrail_enabled",
        "reversal_confirmation_min_signals": "gemini_reversal_confirmation_min_signals",
        "volume_confirmation_min_ratio": "gemini_volume_confirmation_min_ratio",
    }

    _CAMEL: dict[str, str] = {
        "min_confidence": "minConfidence",
        "min_risk_reward": "minRiskReward",
        "strong_trend_min_ema_gap_pct": "strongTrendMinEmaGapPct",
        "relaxed_confidence_floor_strong_trend": "relaxedConfidenceFloorStrongTrend",
        "sell_near_support_min_confidence": "sellNearSupportMinConfidence",
        "min_atr_pct_of_price": "minAtrPctOfPrice",
        "rate_limit_max_retries": "rateLimitMaxRetries",
        "rate_limit_retry_base_delay_sec": "rateLimitRetryBaseDelaySec",
        "trend_guardrail_enabled": "trendGuardrailEnabled",
        "reversal_confirmation_min_signals": "reversalConfirmationMinSignals",
        "volume_confirmation_min_ratio": "volumeConfirmationMinRatio",
    }

    @classmethod
    def get(cls, key: str) -> Any:
        if key in cls._overrides:
            return cls._overrides[key]
        attr = cls._SETTINGS_ATTR.get(key)
        if not attr:
            raise KeyError(key)
        return getattr(settings, attr)

    @classmethod
    def apply_updates(cls, updates: dict[str, Any]) -> None:
        for k, v in updates.items():
            if k not in cls._SETTINGS_ATTR or v is None:
                continue
            if k in ("rate_limit_max_retries", "reversal_confirmation_min_signals"):
                v = int(v)
            elif k == "trend_guardrail_enabled":
                if isinstance(v, str):
                    v = v.strip().lower() in ("1", "true", "yes", "on")
                else:
                    v = bool(v)
            elif k in ("min_confidence", "min_risk_reward", "strong_trend_min_ema_gap_pct",
                       "relaxed_confidence_floor_strong_trend", "sell_near_support_min_confidence",
                       "min_atr_pct_of_price", "rate_limit_retry_base_delay_sec",
                       "volume_confirmation_min_ratio"):
                v = float(v)
            cls._overrides[k] = v

    @classmethod
    def clear(cls) -> None:
        cls._overrides.clear()

    @classmethod
    def has_active_overrides(cls) -> bool:
        return bool(cls._overrides)

    @classmethod
    def to_public_dict(cls) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k in cls._SETTINGS_ATTR:
            out[cls._CAMEL[k]] = cls.get(k)
        out["hasActiveOverrides"] = cls.has_active_overrides()
        return out


def _policy(key: str) -> Any:
    return _PredictionPolicyStore.get(key)


class _SafeDict(dict):
    """dict subclass that leaves unknown {key} placeholders intact instead of raising KeyError."""
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _fmt(v: Any, decimals: int = 2) -> str:
    """Format a numeric value for display, returning 'N/A' for None/missing."""
    if v is None or v == "N/A":
        return "N/A"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _build_snapshot_context(
    ohlcv: pd.DataFrame,
    vix: pd.DataFrame,
    indicators: dict[str, Any],
    checklist_signal: dict[str, Any],
    realtime: dict[str, Any],
    target_minutes: int,
) -> dict[str, Any]:
    """Compute all named substitution values for the system prompt template."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    IST = ZoneInfo("Asia/Kolkata")
    now_ist = datetime.now(IST)

    spot = float(realtime.get("price") or 0)

    # Today's OHLC and previous close — group by IST date
    today_df = pd.DataFrame()
    prev_close = spot
    try:
        idx = ohlcv.index
        if getattr(idx, "tz", None) is None:
            idx_ist = idx.tz_localize("UTC").tz_convert(IST)
        else:
            idx_ist = idx.tz_convert(IST)
        today_str = now_ist.strftime("%Y-%m-%d")
        today_mask = [d.strftime("%Y-%m-%d") == today_str for d in idx_ist]
        today_df = ohlcv[today_mask]
        not_today = ohlcv[[not m for m in today_mask]]
        if not not_today.empty:
            prev_close = float(not_today["close"].iloc[-1])
    except Exception:
        pass

    if not today_df.empty:
        today_open = float(today_df["open"].iloc[0])
        today_high = float(today_df["high"].max())
        today_low = float(today_df["low"].min())
    else:
        today_open = today_high = today_low = spot

    day_change_pct = round((spot - prev_close) / prev_close * 100, 2) if prev_close else 0.0

    # Levels from checklist
    step2 = checklist_signal.get("step2_vwap") or {}
    step5 = checklist_signal.get("step5_levels") or {}

    # VIX value and % change vs previous bar
    india_vix_val: Any = "N/A"
    vix_change_val: Any = "N/A"
    if not vix.empty and "close" in vix.columns:
        vc = vix["close"].astype(float).dropna()
        if len(vc) >= 1:
            india_vix_val = round(float(vc.iloc[-1]), 2)
        if len(vc) >= 2 and float(vc.iloc[-2]) != 0:
            vix_change_val = round((float(vc.iloc[-1]) - float(vc.iloc[-2])) / float(vc.iloc[-2]) * 100, 2)

    # Market phase
    hm = now_ist.hour * 100 + now_ist.minute
    if hm < 930:
        market_phase = "Opening — high volatility (9:15–9:30)"
    elif hm < 1130:
        market_phase = "Morning trend — institutional activity (9:30–11:30)"
    elif hm < 1300:
        market_phase = "Midday consolidation — low volume / choppy (11:30–13:00)"
    elif hm < 1500:
        market_phase = "Afternoon trend — often directional (13:00–15:00)"
    else:
        market_phase = "Pre-close — thin market, avoid new entries (15:00–15:30)"

    # Last 5 completed candles (newest first)
    valid = ohlcv[ohlcv["high"].astype(float) > 0]
    tail = valid.iloc[-6:-1] if len(valid) >= 6 else valid.iloc[:-1]
    lines: list[str] = []
    for ts, row in tail.iloc[::-1].iterrows():
        o = float(row["open"]); h = float(row["high"])
        l = float(row["low"]); c = float(row["close"])
        v = float(row.get("volume", 0))
        arrow = "▲" if c >= o else "▼"
        lines.append(
            f"  {str(ts)[:16]}  O={o:.0f}  H={h:.0f}  L={l:.0f}  C={c:.0f}  "
            f"{arrow} body={abs(c - o):.0f}  vol={v:.0f}"
        )
    last_5_candles = "\n".join(lines) if lines else "  (insufficient data)"

    return {
        # Price
        "spot_price":    _fmt(spot),
        "open":          _fmt(today_open),
        "high":          _fmt(today_high),
        "low":           _fmt(today_low),
        "prev_close":    _fmt(prev_close),
        "day_change_pct": _fmt(day_change_pct),
        # EMAs
        "ema_20":        _fmt(indicators.get("ema_20")),
        "ema_50":        _fmt(indicators.get("ema_50")),
        # Momentum
        "rsi_14":        _fmt(indicators.get("rsi_14"), 1),
        "vwap":          _fmt(step2.get("vwap")),
        "atr_14":        _fmt(indicators.get("atr_14")),
        "macd":          _fmt(indicators.get("macd")),
        "macd_signal":   _fmt(indicators.get("macd_signal")),
        "macd_histogram": _fmt(indicators.get("macd_hist")),
        # Volatility
        "india_vix":     _fmt(india_vix_val),
        "vix_change":    _fmt(vix_change_val),
        # Options data (no source — shown as N/A)
        "pcr_oi":        "N/A",
        "pcr_volume":    "N/A",
        "max_pain":      "N/A",
        # Volume
        "volume_ratio":  _fmt(indicators.get("volume_ratio")),
        # Levels
        "support_1":     _fmt(step5.get("s1")),
        "support_2":     _fmt(step5.get("s2")),
        "resistance_1":  _fmt(step5.get("r1")),
        "resistance_2":  _fmt(step5.get("r2")),
        "pivot":         _fmt(step5.get("pivot")),
        # Context
        "market_phase":  market_phase,
        "time_ist":      now_ist.strftime("%H:%M IST"),
        "last_5_candles": last_5_candles,
        # Policy
        "target_minutes": target_minutes,
        "min_confidence": _policy("min_confidence"),
        "min_risk_reward": _policy("min_risk_reward"),
    }


_TREND_CONTEXT_GUIDANCE = (
    "TREND CONTEXT (deterministic, multi-timeframe):\n"
    "The user payload contains a trend_context object computed directly from OHLCV with keys "
    "regime (combined), primary_regime (5-min), higher_regime (15-min), agreement (bool), "
    "primary_score, higher_score (each in roughly −5..+5), and reasons describing why each "
    "timeframe is labelled UPTREND / DOWNTREND / SIDEWAYS / UNKNOWN. Treat this as a hard "
    "prior: if higher_regime is DOWNTREND, do not output BUY unless you have very strong "
    "reversal confirmation (bullish engulfing on heavy volume + RSI cross up from oversold + "
    "reclaim of VWAP). The same applies in reverse for SELL in an UPTREND. A post-AI guardrail "
    "in the service will downgrade BUY-in-DOWNTREND or SELL-in-UPTREND to HOLD unless "
    "deterministic reversal evidence (engulfing/hammer + volume spike + RSI zone turn + VWAP "
    "reclaim) is present, so do not emit speculative counter-trend calls.\n\n"
    "VOLUME CONFIRMATION:\n"
    "A separate post-AI gate may force HOLD when last-bar volume vs the 20-bar average is "
    "below the configured ratio (default disabled). When you see volume_ratio < 0.7 in the "
    "indicators payload, treat the move as low-conviction and lean toward HOLD or, at minimum, "
    "reduce confidence by 10 points so the existing floor catches it."
)


def _maybe_append_legacy_guidance(template: str, is_custom: bool) -> str:
    """
    Detects legacy custom admin prompts that pre-date the multi-timeframe trend guardrail and
    volume-confirmation gate, and appends the missing instructions so the AI is always aware of
    them. The default ``_SYSTEM_PROMPT_TEMPLATE`` already contains these blocks; we only patch
    user-saved custom templates.
    """
    if not is_custom:
        return template
    has_trend = "TREND CONTEXT" in template or "trend_context" in template
    has_volume = "VOLUME CONFIRMATION" in template or "volume_ratio" in template
    if has_trend and has_volume:
        return template
    return f"{template}\n\n{_TREND_CONTEXT_GUIDANCE}"


def _build_system_prompt(
    target_minutes: int,
    checklist_weight: int = 40,
    snapshot_ctx: dict[str, Any] | None = None,
) -> str:
    remaining = 100 - checklist_weight
    custom = _PromptStore.get()
    template = custom or _SYSTEM_PROMPT_TEMPLATE
    template = _maybe_append_legacy_guidance(template, is_custom=bool(custom))

    fmt = _SafeDict(snapshot_ctx or {})
    # Ensure the three base keys are always present even for custom prompts that lack a snapshot
    fmt.setdefault("target_minutes", target_minutes)
    fmt.setdefault("min_confidence", _policy("min_confidence"))
    fmt.setdefault("min_risk_reward", _policy("min_risk_reward"))

    base = template.format_map(fmt)

    suffix = (
        f"\n\nCHECKLIST SIGNAL (weight: {checklist_weight}%):\n"
        "The checklist_signal key in the market data payload contains the 8-step rule-based "
        "checklist result (EMA trend, VWAP, RSI, pivot levels, entry candle, strike selection, "
        "risk management, no-trade conditions). Weight its overall verdict at "
        f"{checklist_weight}% of your final conviction; derive the remaining "
        f"{remaining}% from your own OHLCV and indicator analysis.\n\n"
        "NEWS SENTIMENT:\n"
        "The news_sentiment key in the payload contains the latest Indian financial market news "
        "sentiment scored by VADER (overall: BULLISH/BEARISH/NEUTRAL, score −1 to +1, top headlines). "
        "Use it as a macro backdrop — BEARISH news raises the bar for a BUY signal and vice versa. "
        "If overall is UNAVAILABLE, ignore it."
    )
    return base + suffix


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group(0))
        raise


def _normalize_direction(raw: dict[str, Any]) -> str:
    v = raw.get("direction") if raw.get("direction") is not None else raw.get("signal")
    if v is None:
        return "HOLD"
    s = str(v).strip().upper()
    if s in ("BUY", "HOLD", "SELL"):
        return s
    if s in ("BULLISH", "LONG", "CALL", "STRADDLE_BUY", "LONG_VOL"):
        return "BUY"
    if s in ("BEARISH", "SHORT", "PUT", "SHORT_VOL"):
        return "SELL"
    return "HOLD"


_GEMINI_QUOTA_USER_NOTICE = (
    "Google Gemini rate limit reached — HOLD placeholder shown. "
    "Check billing/quota limits and retry shortly."
)
_GEMINI_OVERLOAD_USER_NOTICE = (
    "Google Gemini capacity error (HTTP 503) — HOLD placeholder shown. Retry shortly."
)

_GEMINI_RETRYABLE_HTTP = frozenset({502, 503})
_GEMINI_MAX_LOAD_RETRIES = 4


def _strong_trend_from_indicators(indicators: dict[str, Any], price: float) -> bool:
    """EMA9 vs EMA21 separation vs spot — used to relax minimum confidence (fewer HOLDs in trends)."""
    if price <= 0:
        return False
    e9 = indicators.get("ema_9")
    e21 = indicators.get("ema_21")
    if e9 is None or e21 is None:
        return False
    try:
        gap_pct = abs(float(e9) - float(e21)) / price * 100
    except (TypeError, ValueError):
        return False
    return gap_pct >= float(_policy("strong_trend_min_ema_gap_pct"))


def _is_with_trend(model_direction: str, trend_context: Optional[dict[str, Any]]) -> bool:
    """True when a directional call agrees with the combined multi-timeframe regime."""
    if not trend_context or model_direction not in ("BUY", "SELL"):
        return False
    regime = trend_context.get("regime")
    if model_direction == "BUY" and regime == "UPTREND":
        return True
    if model_direction == "SELL" and regime == "DOWNTREND":
        return True
    return False


def _effective_confidence_floor(
    model_direction: str,
    indicators: Optional[dict[str, Any]],
    checklist_signal: Optional[dict[str, Any]],
    price: float,
    trend_context: Optional[dict[str, Any]] = None,
) -> float:
    """Minimum confidence required before coercing non-HOLD to HOLD.

    Two relaxation paths into ``relaxed_confidence_floor_strong_trend``:
    - **EMA gap proxy** (legacy): EMA9 vs EMA21 separation is wide.
    - **Regime agreement** (new): the deterministic multi-timeframe regime
      matches the directional call. This is the more robust path because it
      requires confirmation across two timeframes plus VWAP and swing
      structure, not just one EMA pair.
    """
    floor = float(_policy("min_confidence"))
    is_strong_via_ema = bool(
        indicators and price > 0 and _strong_trend_from_indicators(indicators, price)
    )
    is_with_trend_via_regime = _is_with_trend(model_direction, trend_context)
    if is_strong_via_ema or is_with_trend_via_regime:
        floor = float(_policy("relaxed_confidence_floor_strong_trend"))
    step5 = (checklist_signal or {}).get("step5_levels") or {}
    if (
        checklist_signal is not None
        and model_direction == "SELL"
        and step5.get("signal") == "NEAR_SUPPORT"
    ):
        floor = max(floor, float(_policy("sell_near_support_min_confidence")))
    return floor


def _dead_market_from_indicators(indicators: Optional[dict[str, Any]], price: float) -> bool:
    """Optional chop filter: ATR(14) as %% of price below threshold → treat as non-tradeable."""
    thr = float(_policy("min_atr_pct_of_price"))
    if thr <= 0 or not indicators or price <= 0:
        return False
    atr = indicators.get("atr_14")
    if atr is None:
        return False
    try:
        atr_pct = float(atr) / price * 100
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    return atr_pct < thr


def _coerce_result(
    raw: dict[str, Any],
    realtime_price: float,
    *,
    indicators: Optional[dict[str, Any]] = None,
    checklist_signal: Optional[dict[str, Any]] = None,
    trend_context: Optional[dict[str, Any]] = None,
    ohlcv: Optional[pd.DataFrame] = None,
    pid: Optional[str] = None,
) -> Dict[str, Any]:
    # Model output before policy (confidence / R:R). Used so entry/SL/target in history
    # still reflect what Gemini predicted even when we downgrade the tradable signal to HOLD.
    model_direction = _normalize_direction(raw)
    direction = model_direction
    trend_guardrail_reason: Optional[str] = None
    volume_gate_reason: Optional[str] = None
    pid = pid or _make_pid()
    log_pfx = f"[pid={pid}]"
    # Track which gates fired so we can emit a single-line PRED_GATES summary.
    gates: dict[str, Any] = {
        "dead_market": False,
        "low_confidence": False,
        "low_rr": False,
        "volume_gate": False,
        "trend_guardrail": False,
        "reversal_override": False,
    }

    def _f(key: str, default: float = 0.0) -> float:
        try:
            return float(raw.get(key) or default)
        except (TypeError, ValueError):
            return default

    magnitude = _f("magnitude")
    confidence = max(0.0, min(100.0, _f("confidence")))
    predicted_volatility = max(0.0, min(100.0, _f("predicted_volatility")))
    valid_minutes = max(1, int(_f("valid_minutes", 15)))

    min_rr = float(_policy("min_risk_reward"))

    if _dead_market_from_indicators(indicators, realtime_price) and model_direction in ("BUY", "SELL"):
        logger.info(
            "{} ATR%% vs price below min_atr_pct_of_price — overriding {} → HOLD (low volatility gate)",
            log_pfx,
            model_direction,
        )
        gates["dead_market"] = True
        direction = "HOLD"

    eff_floor = _effective_confidence_floor(
        model_direction, indicators, checklist_signal, realtime_price, trend_context
    )
    if confidence < eff_floor and model_direction != "HOLD":
        logger.info(
            "{} Confidence {:.1f}% < {:.1f} — overriding {} → HOLD (no-trade zone)",
            log_pfx,
            confidence,
            eff_floor,
            model_direction,
        )
        gates["low_confidence"] = True
        direction = "HOLD"

    # Trading levels (always derive fallbacks from model_direction so below-threshold rows keep levels)
    entry_price = _f("entry_price", realtime_price) or realtime_price
    stop_loss = _f("stop_loss")
    target_price = _f("target_price")

    # Fallback stop/target from magnitude if Gemini didn't provide them
    if not stop_loss and model_direction not in ("HOLD",):
        atr_pct = 0.003  # 0.3% fallback stop
        if model_direction == "BUY":
            stop_loss = round(entry_price * (1 - atr_pct), 2)
        else:
            stop_loss = round(entry_price * (1 + atr_pct), 2)

    if not target_price and magnitude:
        target_price = round(realtime_price * (1 + magnitude / 100), 2)
    elif not target_price:
        target_price = realtime_price

    # Risk/reward
    risk_reward = _f("risk_reward")
    if not risk_reward and stop_loss and entry_price and target_price:
        risk = abs(entry_price - stop_loss)
        reward = abs(target_price - entry_price)
        risk_reward = round(reward / risk, 2) if risk > 0 else 0.0

    # Enforce minimum R:R for non-HOLD signals
    if direction != "HOLD" and risk_reward < min_rr:
        logger.info(
            "{} R:R {:.2f} < {:.2f} — overriding {} → HOLD",
            log_pfx,
            risk_reward,
            min_rr,
            direction,
        )
        gates["low_rr"] = True
        direction = "HOLD"

    # Volume confirmation gate: reject BUY/SELL when last-bar volume vs 20-bar avg is
    # below the configured ratio. Default threshold is 0.0 (disabled) so this is
    # opt-in. Indicators contain the pre-computed `volume_ratio` so we don't need OHLCV.
    if direction in ("BUY", "SELL") and indicators is not None:
        try:
            min_vol_ratio = float(_policy("volume_confirmation_min_ratio"))
        except (KeyError, TypeError, ValueError):
            min_vol_ratio = 0.0
        vol_ratio = indicators.get("volume_ratio")
        if min_vol_ratio > 0 and vol_ratio is not None:
            try:
                vr = float(vol_ratio)
            except (TypeError, ValueError):
                vr = None
            if vr is not None and vr < min_vol_ratio:
                logger.info(
                    "{} Volume confirmation failed — {:.2f}x < {:.2f}x — overriding {} → HOLD",
                    log_pfx,
                    vr,
                    min_vol_ratio,
                    direction,
                )
                gates["volume_gate"] = True
                volume_gate_reason = (
                    f"{direction} blocked by volume confirmation gate — "
                    f"{vr:.2f}x average < {min_vol_ratio:.2f}x required"
                )
                direction = "HOLD"

    # Reversal-confirmation evidence (used as override for the trend guardrail).
    # Computed only for directional calls and only when we have OHLCV — otherwise
    # we leave the guardrail as a hard veto.
    reversal_info: Optional[dict[str, Any]] = None
    if direction in ("BUY", "SELL") and ohlcv is not None:
        try:
            reversal_info = _detect_reversal_confirmation(ohlcv, indicators, direction)
        except Exception as e:
            logger.warning("Reversal confirmation check failed: {}", e)
            reversal_info = None

    # Trend guardrail: veto BUY in a downtrend / SELL in an uptrend (multi-timeframe).
    # Runs after the other gates so the log line stays meaningful (we only veto signals
    # that would otherwise pass confidence + R:R + dead-tape checks). Reversal
    # confirmation can override the veto with an annotated reason.
    try:
        guardrail_on = bool(_policy("trend_guardrail_enabled"))
    except KeyError:
        guardrail_on = True
    if guardrail_on and direction in ("BUY", "SELL"):
        new_direction, reason = _apply_trend_guardrail(direction, trend_context, reversal_info)
        if new_direction != direction:
            logger.info(
                "{} Trend guardrail tripped — overriding {} → {} ({})",
                log_pfx,
                direction,
                new_direction,
                reason,
            )
            gates["trend_guardrail"] = True
            direction = new_direction
            trend_guardrail_reason = reason
        elif reason and reversal_info and reversal_info.get("confirmed"):
            # Same direction but the guardrail had an opposing regime; reversal override
            # let it through. Surface this in prediction_reason for transparency.
            logger.info(
                "{} Trend guardrail bypassed by reversal confirmation: {}", log_pfx, reason
            )
            gates["reversal_override"] = True
            trend_guardrail_reason = reason

    # Testing / QA: always expose numeric SL and TP for HOLD (incl. low-confidence / no-trade),
    # so UI and history rows always show a bracket. Uses symmetric ±0.3% around entry when
    # levels are missing or target collapsed to entry (spot).
    if direction == "HOLD":
        _atr = 0.003
        ep = float(entry_price)
        if not stop_loss or stop_loss <= 0:
            stop_loss = round(ep * (1 - _atr), 2)
        tp_cand = target_price if target_price else 0.0
        try:
            tp_f = float(tp_cand)
        except (TypeError, ValueError):
            tp_f = 0.0
        if not tp_f or tp_f <= 0 or abs(tp_f - ep) < 1e-6:
            target_price = round(ep * (1 + _atr), 2)
        else:
            target_price = round(tp_f, 2)
        risk = abs(ep - float(stop_loss))
        reward = abs(float(target_price) - ep)
        risk_reward = round(reward / risk, 2) if risk > 0 else 0.0

    current_sensex = realtime_price
    target_sensex = target_price if target_price else round(
        current_sensex * (1 + magnitude / 100), 2
    ) if magnitude else current_sensex

    reason_raw = raw.get("reason") or raw.get("rationale") or raw.get("explanation") or ""
    reason = str(reason_raw).strip()[:4000]
    prefixes: list[str] = []
    if volume_gate_reason:
        prefixes.append(f"[Volume gate] {volume_gate_reason}")
    if trend_guardrail_reason:
        prefixes.append(f"[Trend guardrail] {trend_guardrail_reason}")
    if prefixes:
        head = " ".join(prefixes)
        reason = f"{head} | Original model rationale: {reason}" if reason else head
        reason = reason[:4000]

    out: Dict[str, Any] = {
        "direction": direction,
        "magnitude": round(magnitude, 4),
        "confidence": round(confidence, 2),
        "predicted_volatility": round(predicted_volatility, 4),
        "current_sensex": current_sensex,
        "target_sensex": round(target_sensex, 2),
        "entry_price": round(entry_price, 2),
        "stop_loss": round(stop_loss, 2) if stop_loss else None,
        "target_price": round(target_price, 2),
        "risk_reward": round(risk_reward, 2),
        "valid_minutes": valid_minutes,
        "prediction_reason": reason,
    }
    notice = raw.get("ai_quota_notice")
    if notice:
        out["ai_quota_notice"] = str(notice).strip()[:4000]

    # Per-gate decision summary for monitoring. Stash diagnostics on a private
    # key so the predict() caller can include them in the final PRED_RESULT
    # JSON line. The key starts with `_` so the gRPC response builder ignores it.
    gates_summary = {
        "model_direction": model_direction,
        "final_direction": direction,
        "raw_confidence": round(confidence, 2),
        "effective_confidence_floor": round(eff_floor, 2),
        "raw_risk_reward": round(risk_reward, 2) if risk_reward else None,
        "min_risk_reward": round(min_rr, 2),
        "gates": gates,
        "reversal": {
            "score": (reversal_info or {}).get("score"),
            "required": (reversal_info or {}).get("required_signals"),
            "confirmed": (reversal_info or {}).get("confirmed"),
            "reasons": (reversal_info or {}).get("reasons") or [],
        } if reversal_info is not None else None,
        "guardrail_reason": trend_guardrail_reason,
        "volume_gate_reason": volume_gate_reason,
    }
    out["_diagnostics"] = gates_summary
    _log_marker("PRED_GATES", pid, gates_summary)
    return out


def _gemini_text_from_response(data: dict[str, Any]) -> str:
    cands = data.get("candidates") or []
    if not cands:
        raise RuntimeError(f"Gemini returned no candidates: {data!r}"[:4096])
    parts = (cands[0].get("content") or {}).get("parts") or []
    if not parts:
        raise RuntimeError(f"Gemini candidate has no parts: {data!r}"[:4096])
    t = parts[0].get("text")
    if not t:
        raise RuntimeError(f"Gemini part has no text: {data!r}"[:4096])
    return str(t)


class GeminiPredictor:
    def predict(
        self,
        horizon: str = "15M",
        ohlcv: Optional[pd.DataFrame] = None,
        vix: Optional[pd.DataFrame] = None,
        sensex_quote: Optional[Dict[str, Any]] = None,
        underlying_symbol: str = "",
    ) -> Dict[str, Any]:
        pid = _make_pid()
        log_pfx = f"[pid={pid}]"
        sym = (underlying_symbol or "BANKNIFTY").strip() or "BANKNIFTY"
        bars_count = 0 if ohlcv is None else int(len(ohlcv))
        vix_count = 0 if vix is None else int(len(vix))

        _log_marker("PRED_START", pid, {
            "horizon": horizon,
            "symbol": sym,
            "ohlcv_bars": bars_count,
            "vix_bars": vix_count,
            "has_quote": bool(sensex_quote and sensex_quote.get("price")),
            "quote_price": _safe_round((sensex_quote or {}).get("price")),
        })

        key = (settings.gemini_api_key or "").strip()
        if not key:
            logger.error("{} GEMINI_API_KEY is not set — predict aborted", log_pfx)
            raise RuntimeError("GEMINI_API_KEY is not set")

        if ohlcv is None or ohlcv.empty:
            logger.error("{} empty OHLCV — returning HOLD placeholder", log_pfx)
            placeholder = {
                "error": "No market data available",
                "direction": "HOLD",
                "magnitude": 0, "confidence": 0,
                "predicted_volatility": 0, "current_sensex": 0, "target_sensex": 0,
                "entry_price": 0, "stop_loss": 0, "target_price": 0,
                "risk_reward": 0, "valid_minutes": 15,
                "prediction_reason": "No OHLCV bars available — cannot assess trend.",
            }
            _log_marker("PRED_RESULT", pid, {
                "horizon": horizon,
                "symbol": sym,
                "final_direction": "HOLD",
                "abort_reason": "empty_ohlcv",
            })
            return placeholder

        if vix is None or vix.empty:
            logger.info("{} VIX missing — deriving proxy from OHLCV", log_pfx)
            vix = derive_vix_from_ohlcv(ohlcv)

        # Data-quality diagnostic: inspect bar interval consistency, high/low sanity, ATR%.
        # Surfaces the root cause when ATR seems anomalous (mixed timeframes, bad highs/lows,
        # stale data, broker glitches, etc.). Logged once per prediction and grep-able as
        # PRED_DATA_QUALITY pid=...
        try:
            dq = _ohlcv_quality_diagnostic(ohlcv, horizon)
            _log_marker("PRED_DATA_QUALITY", pid, dq)
            if dq.get("warnings"):
                logger.warning(
                    "{} OHLCV data quality warnings: {}",
                    log_pfx, "; ".join(dq["warnings"]),
                )
        except Exception as dq_err:
            logger.debug("{} data-quality diagnostic failed: {}", log_pfx, dq_err)

        target_minutes = _horizon_to_minutes(horizon)
        realtime = _resolve_realtime(ohlcv, sensex_quote)

        # Compute technical indicators for the current bar
        indicators = _compute_indicator_snapshot(ohlcv)
        _log_marker("PRED_INDICATORS", pid, {
            "rsi_14": indicators.get("rsi_14"),
            "ema_9": indicators.get("ema_9"),
            "ema_21": indicators.get("ema_21"),
            "ema_50": indicators.get("ema_50"),
            "ema9_above_ema21": indicators.get("ema9_above_ema21"),
            "atr_14": indicators.get("atr_14"),
            "macd": indicators.get("macd"),
            "macd_signal": indicators.get("macd_signal"),
            "macd_hist": indicators.get("macd_hist"),
            "bb_pct": indicators.get("bb_pct"),
            "bb_width": indicators.get("bb_width"),
            "volume_ratio": indicators.get("volume_ratio"),
            "price": _safe_round(realtime.get("price")),
            "change_pct_today": _safe_round(realtime.get("change_pct"), 3),
        })

        # Deterministic multi-timeframe trend regime (used by both the prompt
        # and the post-AI guardrail to suppress counter-trend BUY/SELL).
        try:
            trend_context = _detect_trend_context(ohlcv)
        except Exception as e:
            logger.warning("{} Trend context computation failed: {}", log_pfx, e)
            trend_context = {
                "regime": "UNKNOWN",
                "primary_regime": "UNKNOWN",
                "higher_regime": "UNKNOWN",
                "agreement": False,
                "primary_score": 0,
                "higher_score": 0,
                "primary_reasons": [],
                "higher_reasons": [],
                "evidence": {"primary": {}, "higher": {}, "error": str(e)},
            }
        _log_marker("PRED_TREND", pid, {
            "regime": trend_context.get("regime"),
            "primary_regime": trend_context.get("primary_regime"),
            "higher_regime": trend_context.get("higher_regime"),
            "agreement": trend_context.get("agreement"),
            "primary_score": trend_context.get("primary_score"),
            "higher_score": trend_context.get("higher_score"),
            "primary_reasons": trend_context.get("primary_reasons"),
            "higher_reasons": trend_context.get("higher_reasons"),
        })

        checklist_weight = _ChecklistWeightStore.get()
        try:
            checklist_signal = run_checklist(ohlcv)
        except Exception as e:
            logger.warning("{} Checklist computation failed: {}", log_pfx, e)
            checklist_signal = {"overall": "NO_DATA", "error": str(e)}
        _log_marker("PRED_CHECKLIST", pid, {
            "overall": checklist_signal.get("overall"),
            "weight_pct": checklist_weight,
            "step5_signal": (checklist_signal.get("step5_levels") or {}).get("signal"),
        })

        try:
            news_sentiment = get_news_sentiment()
        except Exception as e:
            logger.warning("{} News sentiment fetch failed: {}", log_pfx, e)
            news_sentiment = {"overall": "UNAVAILABLE", "error": str(e)}
        _log_marker("PRED_NEWS", pid, {
            "overall": news_sentiment.get("overall"),
            "score": news_sentiment.get("score"),
            "bullish_count": news_sentiment.get("bullish_count"),
            "bearish_count": news_sentiment.get("bearish_count"),
            "neutral_count": news_sentiment.get("neutral_count"),
            "fetched_at_ist": news_sentiment.get("fetched_at_ist"),
        })

        snapshot_ctx = _build_snapshot_context(
            ohlcv, vix, indicators, checklist_signal, realtime, target_minutes
        )

        user_payload = {
            "horizon": horizon,
            "target_minutes": target_minutes,
            "underlying_symbol": underlying_symbol or "BANKNIFTY",
            "current_price": realtime["price"],
            "change_pct_today": round(realtime["change_pct"], 3),
            "recent_ohlcv_bars": _ohlcv_tail_records(ohlcv),
            "technical_indicators": indicators,
            "india_vix": _vix_tail_summary(vix),
            "checklist_signal": checklist_signal,
            "news_sentiment": news_sentiment,
            "trend_context": trend_context,
        }
        user_text = json.dumps(user_payload, default=str)

        system_prompt = _build_system_prompt(target_minutes, checklist_weight, snapshot_ctx)
        base = settings.gemini_base_url.rstrip("/")
        model = _ModelStore.get()
        path = f"{base}/models/{model}:generateContent"
        url = f"{path}?{urlencode({'key': key})}"

        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": {
                "temperature": 0.15,
                "responseMimeType": "application/json",
            },
        }
        headers = {"Content-Type": "application/json"}

        _log_marker("PRED_GEMINI_REQUEST", pid, {
            "model": model,
            "user_payload_chars": len(user_text),
            "system_prompt_chars": len(system_prompt),
            "target_minutes": target_minutes,
        })

        def _post_once(client: httpx.Client) -> httpx.Response:
            r = client.post(url, headers=headers, json=body)
            if r.status_code == 400:
                body_plain = {
                    "systemInstruction": body["systemInstruction"],
                    "contents": body["contents"],
                    "generationConfig": {"temperature": 0.15},
                }
                r = client.post(url, headers=headers, json=body_plain)
            return r

        gemini_t0 = time.monotonic()
        gemini_attempts = 0
        with httpx.Client(timeout=120.0) as client:
            resp: Optional[httpx.Response] = None
            for attempt in range(_GEMINI_MAX_LOAD_RETRIES):
                gemini_attempts += 1
                resp = _post_once(client)
                if resp.status_code in _GEMINI_RETRYABLE_HTTP and attempt < _GEMINI_MAX_LOAD_RETRIES - 1:
                    wait_s = min(8.0, 0.75 * (2 ** attempt))
                    logger.warning(
                        "{} Gemini {} (transient) attempt {}/{} retry in {:.1f}s",
                        log_pfx,
                        resp.status_code, attempt + 1, _GEMINI_MAX_LOAD_RETRIES, wait_s,
                    )
                    time.sleep(wait_s)
                    continue
                break

            assert resp is not None

            # Rate limits: backoff retries before HOLD placeholder (same spirit as /admin/analyse).
            if resp.status_code == 429:
                max_r = max(1, int(_policy("rate_limit_max_retries")))
                base_d = float(_policy("rate_limit_retry_base_delay_sec"))
                for attempt in range(max_r - 1):
                    wait_s = min(90.0, base_d * (2 ** attempt))
                    logger.warning(
                        "{} Gemini 429 rate limit — retry {}/{} in {:.1f}s",
                        log_pfx,
                        attempt + 1,
                        max_r - 1,
                        wait_s,
                    )
                    time.sleep(wait_s)
                    gemini_attempts += 1
                    resp = _post_once(client)
                    if resp.status_code != 429:
                        break

            gemini_latency_ms = int((time.monotonic() - gemini_t0) * 1000)

            if resp.status_code == 429:
                logger.warning(
                    "{} Gemini 429 exhausted retries after {} attempts ({}ms) — HOLD fallback",
                    log_pfx, gemini_attempts, gemini_latency_ms,
                )
                _log_marker("PRED_GEMINI_RESPONSE", pid, {
                    "status": 429, "attempts": gemini_attempts,
                    "latency_ms": gemini_latency_ms, "outcome": "rate_limit_fallback",
                })
                fb = _coerce_result(
                    {
                        "direction": "HOLD", "magnitude": 0.0, "confidence": 0.0,
                        "predicted_volatility": 0.0, "valid_minutes": target_minutes,
                        "ai_quota_notice": _GEMINI_QUOTA_USER_NOTICE,
                        "reason": "Gemini rate limit hit (HTTP 429) — HOLD placeholder, not a model forecast.",
                    },
                    float(realtime["price"]),
                    indicators=indicators,
                    trend_context=trend_context,
                    ohlcv=ohlcv,
                    pid=pid,
                )
                self._log_pred_result(pid, horizon, sym, target_minutes, fb,
                                      gemini_latency_ms, gemini_attempts, status=429,
                                      fallback="rate_limit")
                return fb

            if resp.status_code in _GEMINI_RETRYABLE_HTTP:
                logger.warning(
                    "{} Gemini still {} after {} attempts ({}ms) — HOLD fallback",
                    log_pfx, resp.status_code, gemini_attempts, gemini_latency_ms,
                )
                _log_marker("PRED_GEMINI_RESPONSE", pid, {
                    "status": resp.status_code, "attempts": gemini_attempts,
                    "latency_ms": gemini_latency_ms, "outcome": "capacity_fallback",
                })
                fb = _coerce_result(
                    {
                        "direction": "HOLD", "magnitude": 0.0, "confidence": 0.0,
                        "predicted_volatility": 0.0, "valid_minutes": target_minutes,
                        "ai_quota_notice": _GEMINI_OVERLOAD_USER_NOTICE,
                        "reason": "Gemini capacity error (HTTP 502/503) — HOLD placeholder.",
                    },
                    float(realtime["price"]),
                    indicators=indicators,
                    trend_context=trend_context,
                    ohlcv=ohlcv,
                    pid=pid,
                )
                self._log_pred_result(pid, horizon, sym, target_minutes, fb,
                                      gemini_latency_ms, gemini_attempts, status=resp.status_code,
                                      fallback="capacity")
                return fb

            if resp.status_code >= 400:
                detail = (resp.text or "")[:2048]
                logger.error(
                    "{} Gemini HTTP {} model={!r} latency={}ms attempts={}: {}",
                    log_pfx, resp.status_code, model, gemini_latency_ms, gemini_attempts, detail,
                )
                _log_marker("PRED_GEMINI_RESPONSE", pid, {
                    "status": resp.status_code, "attempts": gemini_attempts,
                    "latency_ms": gemini_latency_ms, "outcome": "error",
                    "error_detail": detail[:512],
                })
                raise RuntimeError(f"Gemini HTTP {resp.status_code}: {detail}")

            data = resp.json()

        text = _gemini_text_from_response(data)
        raw = _parse_json_object(text)

        # Inject valid_minutes from horizon if Gemini didn't set it
        if not raw.get("valid_minutes"):
            raw["valid_minutes"] = target_minutes

        _log_marker("PRED_GEMINI_RESPONSE", pid, {
            "status": resp.status_code,
            "attempts": gemini_attempts,
            "latency_ms": gemini_latency_ms,
            "outcome": "ok",
            "raw_direction": _normalize_direction(raw),
            "raw_confidence": _safe_round(raw.get("confidence")),
            "raw_magnitude": _safe_round(raw.get("magnitude"), 4),
            "raw_risk_reward": _safe_round(raw.get("risk_reward")),
            "raw_valid_minutes": raw.get("valid_minutes"),
            "raw_entry_price": _safe_round(raw.get("entry_price")),
            "raw_stop_loss": _safe_round(raw.get("stop_loss")),
            "raw_target_price": _safe_round(raw.get("target_price")),
        })

        result = _coerce_result(
            raw,
            float(realtime["price"]),
            indicators=indicators,
            checklist_signal=checklist_signal,
            trend_context=trend_context,
            ohlcv=ohlcv,
            pid=pid,
        )

        self._log_pred_result(pid, horizon, sym, target_minutes, result,
                              gemini_latency_ms, gemini_attempts, status=resp.status_code,
                              fallback=None, trend_context=trend_context,
                              news_sentiment=news_sentiment, checklist_signal=checklist_signal,
                              indicators=indicators)
        return result

    @staticmethod
    def _log_pred_result(
        pid: str,
        horizon: str,
        symbol: str,
        target_minutes: int,
        result: Dict[str, Any],
        gemini_latency_ms: int,
        gemini_attempts: int,
        *,
        status: Optional[int] = None,
        fallback: Optional[str] = None,
        trend_context: Optional[dict[str, Any]] = None,
        news_sentiment: Optional[dict[str, Any]] = None,
        checklist_signal: Optional[dict[str, Any]] = None,
        indicators: Optional[dict[str, Any]] = None,
    ) -> None:
        """Emit the canonical end-of-prediction structured log line.

        ``PRED_RESULT pid=… {json}`` is the single line per-prediction blackbox.
        Every field needed to debug or aggregate prediction quality is here.
        Filter live with ``grep PRED_RESULT | jq .``.
        """
        diag = result.get("_diagnostics") or {}
        payload = {
            "horizon": horizon,
            "symbol": symbol,
            "target_minutes": target_minutes,
            "model_direction": diag.get("model_direction"),
            "final_direction": result.get("direction"),
            "confidence": result.get("confidence"),
            "effective_floor": diag.get("effective_confidence_floor"),
            "magnitude": result.get("magnitude"),
            "risk_reward": result.get("risk_reward"),
            "min_risk_reward": diag.get("min_risk_reward"),
            "valid_minutes": result.get("valid_minutes"),
            "entry_price": result.get("entry_price"),
            "stop_loss": result.get("stop_loss"),
            "target_price": result.get("target_price"),
            "current_sensex": result.get("current_sensex"),
            "target_sensex": result.get("target_sensex"),
            "predicted_volatility": result.get("predicted_volatility"),
            "gates": diag.get("gates"),
            "reversal": diag.get("reversal"),
            "guardrail_reason": diag.get("guardrail_reason"),
            "volume_gate_reason": diag.get("volume_gate_reason"),
            "trend": (
                {
                    "regime": (trend_context or {}).get("regime"),
                    "primary_regime": (trend_context or {}).get("primary_regime"),
                    "higher_regime": (trend_context or {}).get("higher_regime"),
                    "agreement": (trend_context or {}).get("agreement"),
                    "primary_score": (trend_context or {}).get("primary_score"),
                    "higher_score": (trend_context or {}).get("higher_score"),
                }
                if trend_context else None
            ),
            "checklist_overall": (checklist_signal or {}).get("overall") if checklist_signal else None,
            "news_overall": (news_sentiment or {}).get("overall") if news_sentiment else None,
            "news_score": (news_sentiment or {}).get("score") if news_sentiment else None,
            "indicators_summary": (
                {
                    "rsi_14": indicators.get("rsi_14"),
                    "atr_14": indicators.get("atr_14"),
                    "volume_ratio": indicators.get("volume_ratio"),
                    "ema9_above_ema21": indicators.get("ema9_above_ema21"),
                }
                if indicators else None
            ),
            "gemini": {
                "status": status,
                "latency_ms": gemini_latency_ms,
                "attempts": gemini_attempts,
                "fallback": fallback,
            },
            "ai_quota_notice": result.get("ai_quota_notice"),
        }
        _log_marker("PRED_RESULT", pid, payload)
        # Keep the human-friendly compact summary too — easy to read in tail logs.
        logger.info(
            "[pid={}] DONE [{}min] model={}→final={} | conf={:.0f}% (floor={}) | "
            "entry={} SL={} TP={} RR={} | gemini={}ms/{}attempts | regime={}",
            pid,
            target_minutes,
            diag.get("model_direction") or "?",
            result.get("direction") or "?",
            float(result.get("confidence") or 0.0),
            diag.get("effective_confidence_floor"),
            result.get("entry_price"),
            result.get("stop_loss"),
            result.get("target_price"),
            result.get("risk_reward"),
            gemini_latency_ms,
            gemini_attempts,
            (trend_context or {}).get("regime") if trend_context else "n/a",
        )
