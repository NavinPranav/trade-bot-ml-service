"""Google Gemini prediction via Generative Language REST API (httpx).

No Redis or gRPC short-circuit caching: every predict() call hits the Gemini API.
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


def _ohlcv_tail_records(ohlcv: pd.DataFrame, max_rows: int = 40) -> list[dict[str, Any]]:
    tail = ohlcv.tail(max_rows)
    rows: list[dict[str, Any]] = []
    for idx, row in tail.iterrows():
        rec: dict[str, Any] = {"date": str(idx)}
        for col in ("open", "high", "low", "close", "volume"):
            if col in tail.columns:
                try:
                    rec[col] = float(row[col])
                except (TypeError, ValueError):
                    rec[col] = None
        rows.append(rec)
    return rows


def _vix_tail_summary(vix: pd.DataFrame, max_rows: int = 10) -> list[dict[str, Any]]:
    if vix.empty:
        return []
    cols = [c for c in ("close", "open", "high", "low") if c in vix.columns]
    if not cols:
        return []
    use = vix[cols].tail(max_rows)
    out: list[dict[str, Any]] = []
    for idx, row in use.iterrows():
        item: dict[str, Any] = {"date": str(idx)}
        for c in cols:
            try:
                item[c] = float(row[c])
            except (TypeError, ValueError):
                item[c] = None
        out.append(item)
    return out


_SYSTEM_PROMPT = (
    "You are an options-focused market analyst for Indian indices/equities. "
    "Use the supplied horizon, underlying_symbol, recent OHLCV, and India VIX context. "
    "Recommend a single actionable stance for options trading (directional or premium-selling "
    "view as appropriate). "
    "Respond with one JSON object only (no markdown), keys exactly: "
    "direction (string, MUST be exactly one of BUY, HOLD, SELL — "
    "BUY = favor long/bullish option structures or buying vol when justified; "
    "SELL = favor bearish structures, reducing longs, or selling premium when justified; "
    "HOLD = no clear edge, wait, or keep existing exposure unchanged), "
    "magnitude (number, signed percent move you expect for the underlying over the horizon, e.g. 0.8 or -1.2), "
    "confidence (number 0-100 in the BUY/SELL/HOLD call), "
    "predicted_volatility (number 0-100, higher = more uncertain or volatile regime for sizing), "
    "reason (string, 2–5 short sentences in plain English: what you saw in recent OHLCV and VIX, "
    "why that supports BUY vs SELL vs HOLD, and how that ties to magnitude and confidence). "
    "Output is for analytics and education only, not personalized investment advice."
)


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group(0))
        raise


def _normalize_option_signal(raw: dict[str, Any]) -> str:
    """Map model output to BUY | HOLD | SELL for PredictionResponse.direction (Gemini path)."""
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
    if s in ("NEUTRAL", "NO_TRADE", "WAIT", "FLAT", "NONE"):
        return "HOLD"
    return "HOLD"


_GEMINI_QUOTA_USER_NOTICE = (
    "Google Gemini rate limit or daily quota was reached, so the AI could not run. "
    "The signal below is a neutral placeholder (HOLD), not a model forecast. "
    "Try again later or check billing/plan limits: https://ai.google.dev/gemini-api/docs/rate-limits"
)

_GEMINI_OVERLOAD_USER_NOTICE = (
    "Google Gemini returned a temporary capacity error (HTTP 503). "
    "The signal below is a neutral placeholder (HOLD). Retry shortly."
)

# Transient upstream errors — retry before failing the request.
_GEMINI_RETRYABLE_HTTP = frozenset({502, 503})
_GEMINI_MAX_LOAD_RETRIES = 4


def _coerce_result(raw: dict[str, Any], realtime_price: float) -> Dict[str, Any]:
    direction = _normalize_option_signal(raw)
    try:
        magnitude = float(raw.get("magnitude", 0))
    except (TypeError, ValueError):
        magnitude = 0.0
    try:
        confidence = float(raw.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(100.0, confidence))
    try:
        predicted_volatility = float(raw.get("predicted_volatility", 0))
    except (TypeError, ValueError):
        predicted_volatility = 0.0
    predicted_volatility = max(0.0, min(100.0, predicted_volatility))

    current_sensex = realtime_price
    target_sensex = 0.0
    if current_sensex and magnitude:
        target_sensex = round(current_sensex * (1 + magnitude / 100), 2)

    reason_raw = raw.get("reason")
    if reason_raw is None:
        reason_raw = raw.get("rationale") if raw.get("rationale") is not None else raw.get("explanation")
    reason = str(reason_raw or "").strip()
    if len(reason) > 4000:
        reason = reason[:4000] + "…"

    out: Dict[str, Any] = {
        "direction": direction,
        "magnitude": round(magnitude, 4),
        "confidence": round(confidence, 2),
        "predicted_volatility": round(predicted_volatility, 4),
        "current_sensex": current_sensex,
        "target_sensex": target_sensex,
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
        horizon: str = "1D",
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
                "error": "No market data available (supply OHLCV from the backend)",
                "direction": "NEUTRAL",
                "magnitude": 0,
                "confidence": 0,
                "predicted_volatility": 0,
                "current_sensex": 0,
                "target_sensex": 0,
                "prediction_reason": "No OHLCV bars were available, so the model could not assess trend or volatility.",
            }

        if vix is None or vix.empty:
            vix = derive_vix_from_ohlcv(ohlcv)

        realtime = _resolve_realtime(ohlcv, sensex_quote)
        user_payload = {
            "horizon": horizon,
            "underlying_symbol": underlying_symbol or None,
            "recent_ohlcv": _ohlcv_tail_records(ohlcv),
            "india_vix_recent": _vix_tail_summary(vix),
        }
        user_text = json.dumps(user_payload, default=str)

        base = settings.gemini_base_url.rstrip("/")
        model = settings.gemini_model.strip()
        path = f"{base}/models/{model}:generateContent"
        url = f"{path}?{urlencode({'key': key})}"

        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_text}],
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }

        headers = {"Content-Type": "application/json"}

        def _post_once(client: httpx.Client) -> httpx.Response:
            r = client.post(url, headers=headers, json=body)
            if r.status_code == 400:
                detail = (r.text or "")[:2048]
                logger.warning(
                    "Gemini generateContent 400 with JSON mode model={!r}: {}",
                    model,
                    detail,
                )
                body_plain = {
                    "systemInstruction": body["systemInstruction"],
                    "contents": body["contents"],
                    "generationConfig": {"temperature": 0.2},
                }
                r = client.post(url, headers=headers, json=body_plain)
            return r

        with httpx.Client(timeout=120.0) as client:
            resp: Optional[httpx.Response] = None
            for attempt in range(_GEMINI_MAX_LOAD_RETRIES):
                resp = _post_once(client)
                if resp.status_code in _GEMINI_RETRYABLE_HTTP and attempt < _GEMINI_MAX_LOAD_RETRIES - 1:
                    wait_s = min(8.0, 0.75 * (2**attempt))
                    logger.warning(
                        "Gemini generateContent {} (transient) model={!r} attempt {}/{} — retry in {:.1f}s",
                        resp.status_code,
                        model,
                        attempt + 1,
                        _GEMINI_MAX_LOAD_RETRIES,
                        wait_s,
                    )
                    time.sleep(wait_s)
                    continue
                break

            assert resp is not None
            # Free-tier / billing quota and per-minute rate limits (RESOURCE_EXHAUSTED → HTTP 429)
            if resp.status_code == 429:
                detail = (resp.text or "")[:1024]
                logger.warning(
                    "Gemini quota or rate limit (HTTP 429) model={!r} — returning HOLD fallback so callers stay up. {}",
                    model,
                    detail,
                )
                return _coerce_result(
                    {
                        "direction": "HOLD",
                        "magnitude": 0.0,
                        "confidence": 0.0,
                        "predicted_volatility": 0.0,
                        "ai_quota_notice": _GEMINI_QUOTA_USER_NOTICE,
                        "reason": (
                            "Gemini returned HTTP 429 (quota or rate limit), so no fresh model pass ran. "
                            "The HOLD placeholder is not based on current bars."
                        ),
                    },
                    float(realtime["price"]),
                )
            # After retries, still overloaded — soft HOLD like 429 so the API stays 200.
            if resp.status_code in _GEMINI_RETRYABLE_HTTP:
                detail = (resp.text or "")[:1024]
                logger.warning(
                    "Gemini still {} after {} attempts model={!r} — HOLD fallback. {}",
                    resp.status_code,
                    _GEMINI_MAX_LOAD_RETRIES,
                    model,
                    detail,
                )
                return _coerce_result(
                    {
                        "direction": "HOLD",
                        "magnitude": 0.0,
                        "confidence": 0.0,
                        "predicted_volatility": 0.0,
                        "ai_quota_notice": _GEMINI_OVERLOAD_USER_NOTICE,
                        "reason": (
                            "Gemini returned repeated capacity errors (HTTP 502/503), so no fresh model pass ran. "
                            "The HOLD placeholder is not based on current bars."
                        ),
                    },
                    float(realtime["price"]),
                )
            if resp.status_code >= 400:
                detail = (resp.text or "")[:2048]
                logger.error(
                    "Gemini generateContent failed status={} model={!r}: {}",
                    resp.status_code,
                    model,
                    detail,
                )
                raise RuntimeError(
                    f"Gemini generateContent HTTP {resp.status_code} (model={model!r}): {detail}"
                )
            data = resp.json()

        text = _gemini_text_from_response(data)
        raw = _parse_json_object(text)
        result = _coerce_result(raw, float(realtime["price"]))

        logger.info(
            "Gemini prediction: {} | conf={}% | mag={}%",
            result["direction"],
            result["confidence"],
            result["magnitude"],
        )
        return result
