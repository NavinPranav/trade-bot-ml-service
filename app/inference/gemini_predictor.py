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
from app.data.ingestion.vix_fetcher import derive_vix_from_ohlcv

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

        # EMA 9 and 21
        if n >= 9:
            ema9_s = ta.trend.ema_indicator(close, window=9)
            snap["ema_9"] = _last(ema9_s)
            if snap.get("ema_9"):
                snap["price_vs_ema9_pct"] = round(
                    (float(close.iloc[-1]) - snap["ema_9"]) / snap["ema_9"] * 100, 3
                )
        if n >= 21:
            ema21_s = ta.trend.ema_indicator(close, window=21)
            snap["ema_21"] = _last(ema21_s)
            if snap.get("ema_9") and snap.get("ema_21"):
                snap["ema9_above_ema21"] = snap["ema_9"] > snap["ema_21"]

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


_SYSTEM_PROMPT_TEMPLATE = (
    "You are an expert intra-day Bank Nifty trader and options analyst. "
    "You are given recent intra-day OHLCV candles and pre-computed technical indicators for the current bar. "
    "Predict what Bank Nifty will do in the NEXT {target_minutes} MINUTES (intra-day call — not end of day). "
    "India VIX context is also supplied when available. "
    "\n\n"
    "Respond with ONE JSON object only (no markdown fences), with EXACTLY these keys:\n"
    "  direction        — string: BUY | SELL | HOLD\n"
    "  entry_price      — number: ideal entry price (use last close / LTP as anchor)\n"
    "  stop_loss        — number: price to exit if wrong (mandatory, never 0)\n"
    "  target_price     — number: price you expect in {target_minutes} min if direction plays out\n"
    "  risk_reward      — number: (|target - entry|) / (|entry - stop|), always > 0\n"
    "  confidence       — number 0-100: confidence in this call\n"
    "  magnitude        — number: expected signed % move (e.g. 0.35 or -0.2)\n"
    "  predicted_volatility — number 0-100: how volatile this window looks\n"
    "  valid_minutes    — number: how many minutes this signal remains valid (= {target_minutes})\n"
    "  reason           — string: 2-4 sentences covering which indicators fired, "
    "price structure, and why BUY/SELL/HOLD\n"
    "\n"
    "STRICT RULES:\n"
    "  1. If confidence < 65 → set direction = HOLD (weak signals must not trade).\n"
    "  2. Risk-reward must be >= 1.5 for BUY or SELL; if not achievable, use HOLD.\n"
    "  3. stop_loss is mandatory and must never equal entry_price or target_price.\n"
    "  4. For Bank Nifty intra-day, moves > 0.6% in 5 min and > 1.2% in 15 min are unusual; be realistic.\n"
    "  5. Output ONLY valid JSON — no extra text, no markdown."
)


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


def _build_system_prompt(target_minutes: int) -> str:
    template = _PromptStore.get() or _SYSTEM_PROMPT_TEMPLATE
    return template.format(target_minutes=target_minutes)


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


def _coerce_result(raw: dict[str, Any], realtime_price: float) -> Dict[str, Any]:
    direction = _normalize_direction(raw)

    def _f(key: str, default: float = 0.0) -> float:
        try:
            return float(raw.get(key) or default)
        except (TypeError, ValueError):
            return default

    magnitude = _f("magnitude")
    confidence = max(0.0, min(100.0, _f("confidence")))
    predicted_volatility = max(0.0, min(100.0, _f("predicted_volatility")))
    valid_minutes = max(1, int(_f("valid_minutes", 15)))

    # Enforce no-trade zone: confidence < 65 → HOLD
    if confidence < 65.0 and direction != "HOLD":
        logger.info(
            f"Confidence {confidence:.1f}% < 65 — overriding {direction} → HOLD (no-trade zone)"
        )
        direction = "HOLD"

    # Trading levels
    entry_price = _f("entry_price", realtime_price) or realtime_price
    stop_loss = _f("stop_loss")
    target_price = _f("target_price")

    # Fallback stop/target from magnitude if Gemini didn't provide them
    if not stop_loss and direction != "HOLD":
        atr_pct = 0.003  # 0.3% fallback stop
        if direction == "BUY":
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

    # Enforce R:R >= 1.5 for non-HOLD signals
    if direction != "HOLD" and risk_reward < 1.5:
        logger.info(
            f"R:R {risk_reward:.2f} < 1.5 — overriding {direction} → HOLD"
        )
        direction = "HOLD"

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

        user_payload = {
            "horizon": horizon,
            "target_minutes": target_minutes,
            "underlying_symbol": underlying_symbol or "BANKNIFTY",
            "current_price": realtime["price"],
            "change_pct_today": round(realtime["change_pct"], 3),
            "recent_ohlcv_bars": _ohlcv_tail_records(ohlcv),
            "technical_indicators": indicators,
            "india_vix": _vix_tail_summary(vix),
        }
        user_text = json.dumps(user_payload, default=str)

        system_prompt = _build_system_prompt(target_minutes)
        base = settings.gemini_base_url.rstrip("/")
        model = settings.gemini_model.strip()
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

            if resp.status_code == 429:
                logger.warning("Gemini 429 quota/rate limit — HOLD fallback")
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

        result = _coerce_result(raw, float(realtime["price"]))

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
