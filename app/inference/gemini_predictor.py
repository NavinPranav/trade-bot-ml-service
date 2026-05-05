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
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import httpx
import pandas as pd
from loguru import logger

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


def _ohlcv_tail_records(ohlcv: pd.DataFrame, max_rows: int = 60) -> list[dict[str, Any]]:
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

THINK THROUGH THIS STEP BY STEP BEFORE ANSWERING:
1. TREND: Is EMA 20 above or below EMA 50? What is the slope — widening or narrowing?
2. MOMENTUM: Is RSI overbought (>70), oversold (<30), or in the buy zone (55-70) / sell zone (30-45) / neutral (45-55)?
3. PRICE vs VWAP: Is price above VWAP (institutional buying) or below (selling)? How far from VWAP?
4. CANDLE PATTERN: Are the last 5 candles making higher highs (bullish) or lower lows (bearish)? Any reversal patterns (doji, hammer, engulfing)?
5. OPTIONS DATA: PCR > 1.2 is bullish (put writers confident), PCR < 0.8 is bearish. Is max pain above or below spot?
6. VOLATILITY: Is VIX rising (expect bigger move) or falling (expect range)? Use ATR to size stop loss realistically.
7. SUPPORT/RESISTANCE: Is price near any key level? A bounce off support = BUY, rejection at resistance = SELL, breakout = strong move.
8. TIME OF DAY: 9:15-9:30 = volatile/unreliable, 12:00-13:00 = low volume/choppy, 14:00-15:15 = trend resumes. Adjust confidence based on time.
9. VOLUME: Volume > 1.5x average = conviction in the move. Volume < 0.7x = low conviction, prefer HOLD.
10. CONFLUENCE: Count how many of the above agree. BUY/SELL only with 4+ signals aligned. Otherwise HOLD.

STRICT RULES:
1. If confidence < {min_confidence} → direction MUST be "HOLD". No exceptions.
2. Risk-reward must be >= {min_risk_reward} for BUY or SELL. If ATR doesn't support a 1:{min_risk_reward} R:R within {target_minutes} minutes, use HOLD.
3. stop_loss must be realistic — use 0.3x to 0.5x ATR from entry. Never equal to entry or target.
4. target_price must be realistic — use 0.5x to 1.0x ATR from entry in the predicted direction.
5. If RSI is between 45-55 AND price is within 0.1% of VWAP → this is a NO TRADE zone → HOLD.
6. If only 15 minutes remain before market close (after 15:15 IST) → HOLD.
7. If VIX > 20 → widen stop loss by 1.5x and reduce confidence by 10.
8. magnitude = expected % move (positive for BUY, negative for SELL, 0 for HOLD).
9. predicted_volatility = your estimate of annualized volatility based on current VIX and ATR.
10. valid_minutes = how long this prediction stays valid (never more than {target_minutes}).

CONFIDENCE CALIBRATION:
- 90-100%: All 10 signals aligned + strong candle pattern + high volume breakout (extremely rare)
- 75-89%: 7-8 signals aligned + clear trend + good volume
- 65-74%: 5-6 signals aligned + moderate conviction
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
            if k == "rate_limit_max_retries":
                v = int(v)
            elif k in ("min_confidence", "min_risk_reward", "strong_trend_min_ema_gap_pct",
                       "relaxed_confidence_floor_strong_trend", "sell_near_support_min_confidence",
                       "min_atr_pct_of_price", "rate_limit_retry_base_delay_sec"):
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


def _build_system_prompt(
    target_minutes: int,
    checklist_weight: int = 40,
    snapshot_ctx: dict[str, Any] | None = None,
) -> str:
    remaining = 100 - checklist_weight
    template = _PromptStore.get() or _SYSTEM_PROMPT_TEMPLATE

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


def _effective_confidence_floor(
    model_direction: str,
    indicators: Optional[dict[str, Any]],
    checklist_signal: Optional[dict[str, Any]],
    price: float,
) -> float:
    """Minimum confidence required before coercing non-HOLD to HOLD."""
    floor = float(_policy("min_confidence"))
    if indicators and price > 0 and _strong_trend_from_indicators(indicators, price):
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
) -> Dict[str, Any]:
    # Model output before policy (confidence / R:R). Used so entry/SL/target in history
    # still reflect what Gemini predicted even when we downgrade the tradable signal to HOLD.
    model_direction = _normalize_direction(raw)
    direction = model_direction

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
            "ATR%% vs price below min_atr_pct_of_price — overriding {} → HOLD (low volatility gate)",
            model_direction,
        )
        direction = "HOLD"

    eff_floor = _effective_confidence_floor(
        model_direction, indicators, checklist_signal, realtime_price
    )
    if confidence < eff_floor and model_direction != "HOLD":
        logger.info(
            "Confidence {:.1f}% < {:.1f} — overriding {} → HOLD (no-trade zone)",
            confidence,
            eff_floor,
            model_direction,
        )
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
            "R:R {:.2f} < {:.2f} — overriding {} → HOLD",
            risk_reward,
            min_rr,
            direction,
        )
        direction = "HOLD"

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
        key = (settings.gemini_api_key or "").strip()
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        if ohlcv is None or ohlcv.empty:
            logger.error("GeminiPredictor: empty OHLCV")
            return {
                "error": "No market data available",
                "direction": "HOLD",
                "magnitude": 0, "confidence": 0,
                "predicted_volatility": 0, "current_sensex": 0, "target_sensex": 0,
                "entry_price": 0, "stop_loss": 0, "target_price": 0,
                "risk_reward": 0, "valid_minutes": 15,
                "prediction_reason": "No OHLCV bars available — cannot assess trend.",
            }

        if vix is None or vix.empty:
            vix = derive_vix_from_ohlcv(ohlcv)

        target_minutes = _horizon_to_minutes(horizon)
        realtime = _resolve_realtime(ohlcv, sensex_quote)

        # Compute technical indicators for the current bar
        indicators = _compute_indicator_snapshot(ohlcv)

        checklist_weight = _ChecklistWeightStore.get()
        try:
            checklist_signal = run_checklist(ohlcv)
        except Exception as e:
            logger.warning("Checklist computation failed: {}", e)
            checklist_signal = {"overall": "NO_DATA", "error": str(e)}

        try:
            news_sentiment = get_news_sentiment()
        except Exception as e:
            logger.warning("News sentiment fetch failed: {}", e)
            news_sentiment = {"overall": "UNAVAILABLE", "error": str(e)}

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

        with httpx.Client(timeout=120.0) as client:
            resp: Optional[httpx.Response] = None
            for attempt in range(_GEMINI_MAX_LOAD_RETRIES):
                resp = _post_once(client)
                if resp.status_code in _GEMINI_RETRYABLE_HTTP and attempt < _GEMINI_MAX_LOAD_RETRIES - 1:
                    wait_s = min(8.0, 0.75 * (2 ** attempt))
                    logger.warning(
                        "Gemini {} (transient) attempt {}/{} retry in {:.1f}s",
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
                        "Gemini 429 rate limit — retry {}/{} in {:.1f}s",
                        attempt + 1,
                        max_r - 1,
                        wait_s,
                    )
                    time.sleep(wait_s)
                    resp = _post_once(client)
                    if resp.status_code != 429:
                        break

            if resp.status_code == 429:
                logger.warning("Gemini 429 exhausted retries — HOLD fallback")
                return _coerce_result(
                    {
                        "direction": "HOLD", "magnitude": 0.0, "confidence": 0.0,
                        "predicted_volatility": 0.0, "valid_minutes": target_minutes,
                        "ai_quota_notice": _GEMINI_QUOTA_USER_NOTICE,
                        "reason": "Gemini rate limit hit (HTTP 429) — HOLD placeholder, not a model forecast.",
                    },
                    float(realtime["price"]),
                )

            if resp.status_code in _GEMINI_RETRYABLE_HTTP:
                logger.warning("Gemini still {} after retries — HOLD fallback", resp.status_code)
                return _coerce_result(
                    {
                        "direction": "HOLD", "magnitude": 0.0, "confidence": 0.0,
                        "predicted_volatility": 0.0, "valid_minutes": target_minutes,
                        "ai_quota_notice": _GEMINI_OVERLOAD_USER_NOTICE,
                        "reason": "Gemini capacity error (HTTP 502/503) — HOLD placeholder.",
                    },
                    float(realtime["price"]),
                )

            if resp.status_code >= 400:
                detail = (resp.text or "")[:2048]
                logger.error("Gemini HTTP {} model={!r}: {}", resp.status_code, model, detail)
                raise RuntimeError(f"Gemini HTTP {resp.status_code}: {detail}")

            data = resp.json()

        text = _gemini_text_from_response(data)
        raw = _parse_json_object(text)

        # Inject valid_minutes from horizon if Gemini didn't set it
        if not raw.get("valid_minutes"):
            raw["valid_minutes"] = target_minutes

        result = _coerce_result(
            raw,
            float(realtime["price"]),
            indicators=indicators,
            checklist_signal=checklist_signal,
        )

        logger.info(
            "Gemini intraday [{}min] {} | conf={:.0f}% | entry={} SL={} TP={} RR={}",
            target_minutes,
            result["direction"],
            result["confidence"],
            result.get("entry_price"),
            result.get("stop_loss"),
            result.get("target_price"),
            result.get("risk_reward"),
        )
        return result
