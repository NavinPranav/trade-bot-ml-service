"""OpenAI / ChatGPT intra-day prediction via Chat Completions REST API (httpx).

Shares all market-context building and post-coercion logic with GeminiPredictor
(imported from gemini_predictor module-level functions). Only the API call layer
differs: OpenAI Chat Completions instead of Gemini generateContent.

Active tool and model are read from _ActiveModelStore at call time — no model ID
is hardcoded here.  The admin sets the model via PUT /admin/model and the store
is updated; this predictor picks up the change on the next call.

Supported model families:
  GPT-4o series  : gpt-4o, gpt-4o-mini
  GPT-4.1 series : gpt-4.1, gpt-4.1-mini, gpt-4.1-nano
  GPT-4 Turbo    : gpt-4-turbo
  o-series (reasoning): o1, o1-mini, o3-mini, o4-mini
    — temperature is NOT sent for reasoning models (API rejects it)
    — response_format json_object IS supported for o1 / o3-mini / o4-mini
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
from loguru import logger

from app.config import settings
from app.inference.active_model_store import _ActiveModelStore

# ── Re-use all market-context and coercion functions from GeminiPredictor ──
# These are module-level functions (not methods) so they're cleanly importable.
from app.inference.gemini_predictor import (
    _ChecklistWeightStore,
    _ParameterTogglesStore,
    _PromptStore,
    _build_snapshot_context,
    _build_system_prompt,
    _coerce_result,
    _compute_indicator_snapshot,
    _compute_options_summary,
    _detect_trend_context,
    _horizon_to_minutes,
    _log_marker,
    _make_pid,
    _normalize_direction,
    _ohlcv_quality_diagnostic,
    _ohlcv_tail_records,
    _parse_json_object,
    _resolve_realtime,
    _safe_round,
    _vix_tail_summary,
)

# ── Constants ──────────────────────────────────────────────────────────────

_OPENAI_RETRYABLE_HTTP = {502, 503}
_OPENAI_MAX_LOAD_RETRIES = 3

_OPENAI_QUOTA_NOTICE = (
    "OpenAI rate limit hit — HOLD placeholder returned.  "
    "Upgrade your tier or reduce request frequency."
)
_OPENAI_OVERLOAD_NOTICE = (
    "OpenAI capacity error (5xx) — HOLD placeholder returned.  "
    "Retry in a few seconds."
)

# Reasoning models don't accept a temperature parameter.
# Detect by prefix: o1, o3-mini, o4-mini, o1-mini, etc.
_O_SERIES_PREFIXES = ("o1", "o3", "o4")


def _is_reasoning_model(model_id: str) -> bool:
    m = model_id.lower()
    return any(m == p or m.startswith(p + "-") for p in _O_SERIES_PREFIXES)


def _openai_text_from_response(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenAI returned no choices: {str(data)[:2048]}")
    msg = (choices[0].get("message") or {}).get("content")
    if not msg:
        raise RuntimeError(f"OpenAI choice has no content: {str(data)[:2048]}")
    return str(msg)


class OpenAIPredictor:
    def predict(
        self,
        horizon: str = "15M",
        ohlcv: Optional[pd.DataFrame] = None,
        vix: Optional[pd.DataFrame] = None,
        sensex_quote: Optional[Dict[str, Any]] = None,
        underlying_symbol: str = "",
        options_chain: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        pid = _make_pid()
        log_pfx = f"[pid={pid}]"
        sym = (underlying_symbol or "BANKNIFTY").strip() or "BANKNIFTY"
        model_id = _ActiveModelStore.get_model_id()
        bars_count = 0 if ohlcv is None else int(len(ohlcv))
        vix_count = 0 if vix is None else int(len(vix))

        _log_marker("PRED_START", pid, {
            "horizon": horizon,
            "symbol": sym,
            "tool": "OPENAI",
            "model": model_id,
            "ohlcv_bars": bars_count,
            "vix_bars": vix_count,
            "has_quote": bool(sensex_quote and sensex_quote.get("price")),
            "quote_price": _safe_round((sensex_quote or {}).get("price")),
        })

        # ── API key guard ──────────────────────────────────────────────────
        api_key = (settings.openai_api_key or "").strip()
        if not api_key:
            logger.error("{} OPENAI_API_KEY is not set — predict aborted", log_pfx)
            raise RuntimeError("OPENAI_API_KEY is not set")

        # ── Empty-OHLCV guard ──────────────────────────────────────────────
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
            _log_marker("PRED_RESULT", pid, {"horizon": horizon, "symbol": sym,
                                              "final_direction": "HOLD", "abort_reason": "empty_ohlcv"})
            return placeholder

        # ── VIX fallback ───────────────────────────────────────────────────
        if vix is None or vix.empty:
            logger.info("{} VIX missing — deriving proxy from OHLCV", log_pfx)
            from app.data.ingestion.vix_fetcher import derive_vix_from_ohlcv
            vix = derive_vix_from_ohlcv(ohlcv)

        # ── Data quality diagnostic ────────────────────────────────────────
        try:
            dq = _ohlcv_quality_diagnostic(ohlcv, horizon)
            _log_marker("PRED_DATA_QUALITY", pid, dq)
            if dq.get("warnings"):
                logger.warning("{} OHLCV data quality warnings: {}",
                               log_pfx, "; ".join(dq["warnings"]))
        except Exception as dq_err:
            logger.debug("{} data-quality diagnostic failed: {}", log_pfx, dq_err)

        target_minutes = _horizon_to_minutes(horizon)
        realtime = _resolve_realtime(ohlcv, sensex_quote)

        # ── Technical indicators ───────────────────────────────────────────
        indicators = _compute_indicator_snapshot(ohlcv)
        _log_marker("PRED_INDICATORS", pid, {
            "rsi_14":          indicators.get("rsi_14"),
            "ema_9":           indicators.get("ema_9"),
            "ema_21":          indicators.get("ema_21"),
            "ema_50":          indicators.get("ema_50"),
            "ema9_above_ema21":indicators.get("ema9_above_ema21"),
            "atr_14":          indicators.get("atr_14"),
            "macd":            indicators.get("macd"),
            "macd_signal":     indicators.get("macd_signal"),
            "macd_hist":       indicators.get("macd_hist"),
            "bb_pct":          indicators.get("bb_pct"),
            "bb_width":        indicators.get("bb_width"),
            "volume_ratio":    indicators.get("volume_ratio"),
            "price":           _safe_round(realtime.get("price")),
            "change_pct_today":_safe_round(realtime.get("change_pct"), 3),
        })

        # ── Multi-timeframe trend regime ───────────────────────────────────
        try:
            trend_context = _detect_trend_context(ohlcv)
        except Exception as e:
            logger.warning("{} Trend context failed: {}", log_pfx, e)
            trend_context = {
                "regime": "UNKNOWN", "primary_regime": "UNKNOWN",
                "higher_regime": "UNKNOWN", "agreement": False,
                "primary_score": 0, "higher_score": 0,
                "primary_reasons": [], "higher_reasons": [],
                "evidence": {"primary": {}, "higher": {}, "error": str(e)},
            }
        _log_marker("PRED_TREND", pid, {
            "regime":          trend_context.get("regime"),
            "primary_regime":  trend_context.get("primary_regime"),
            "higher_regime":   trend_context.get("higher_regime"),
            "agreement":       trend_context.get("agreement"),
            "primary_score":   trend_context.get("primary_score"),
            "higher_score":    trend_context.get("higher_score"),
            "primary_reasons": trend_context.get("primary_reasons"),
            "higher_reasons":  trend_context.get("higher_reasons"),
        })

        # ── Checklist & news ───────────────────────────────────────────────
        checklist_weight = _ChecklistWeightStore.get()
        if _ParameterTogglesStore.is_enabled("checklist_signal"):
            try:
                from app.inference.checklist import run_checklist
                checklist_signal = run_checklist(ohlcv)
            except Exception as e:
                logger.warning("{} Checklist failed: {}", log_pfx, e)
                checklist_signal = {"overall": "NO_DATA", "error": str(e)}
        else:
            checklist_signal = {"overall": "DISABLED", "note": "Checklist signal disabled by admin"}
        _log_marker("PRED_CHECKLIST", pid, {
            "overall": checklist_signal.get("overall"),
            "weight_pct": checklist_weight,
            "enabled": _ParameterTogglesStore.is_enabled("checklist_signal"),
        })

        if _ParameterTogglesStore.is_enabled("news_sentiment"):
            try:
                from app.data.ingestion.news_fetcher import get_news_sentiment
                news_sentiment = get_news_sentiment()
            except Exception as e:
                logger.warning("{} News sentiment failed: {}", log_pfx, e)
                news_sentiment = {"overall": "UNAVAILABLE", "error": str(e)}
        else:
            news_sentiment = {"overall": "DISABLED", "note": "News sentiment disabled by admin"}
        _log_marker("PRED_NEWS", pid, {
            "overall":       news_sentiment.get("overall"),
            "score":         news_sentiment.get("score"),
            "enabled":       _ParameterTogglesStore.is_enabled("news_sentiment"),
        })

        # ── Options flow ───────────────────────────────────────────────────
        spot_price = float(realtime.get("price") or 0)
        options_summary: dict[str, Any] = {}
        if options_chain is not None and not options_chain.empty:
            try:
                options_summary = _compute_options_summary(options_chain, spot_price)
            except Exception as oe:
                logger.warning("{} Options summary failed (non-fatal): {}", log_pfx, oe)

        # ── Build snapshot context (shared with GeminiPredictor) ──────────
        snapshot_ctx = _build_snapshot_context(
            ohlcv, vix, indicators, checklist_signal, realtime, target_minutes,
            trend_context=trend_context,
            options_summary=options_summary,
        )
        if not _ParameterTogglesStore.is_enabled("previous_day_levels"):
            for k in ("prev_high", "prev_low", "prev_close", "prev_range",
                      "prev_close_position", "range_vs_atr",
                      "cpr_high", "cpr_low", "cpr_width"):
                snapshot_ctx[k] = "N/A"
        if not _ParameterTogglesStore.is_enabled("support_resistance_levels"):
            for k in ("pivot", "support_1", "support_2", "resistance_1", "resistance_2",
                      "resistance_nearest", "resistance_dist",
                      "support_nearest", "support_dist"):
                snapshot_ctx[k] = "N/A"
        if not _ParameterTogglesStore.is_enabled("india_vix"):
            snapshot_ctx["india_vix"] = "N/A"
            snapshot_ctx["vix_change"] = "N/A"

        # ── User payload ───────────────────────────────────────────────────
        user_payload: dict[str, Any] = {
            "horizon": horizon,
            "target_minutes": target_minutes,
            "underlying_symbol": underlying_symbol or "BANKNIFTY",
            "current_price": realtime["price"],
            "change_pct_today": round(realtime["change_pct"], 3),
            "recent_ohlcv_bars": _ohlcv_tail_records(ohlcv),
            "technical_indicators": indicators,
            "trend_context": trend_context,
        }
        if _ParameterTogglesStore.is_enabled("india_vix"):
            user_payload["india_vix"] = _vix_tail_summary(vix)
        if _ParameterTogglesStore.is_enabled("checklist_signal"):
            user_payload["checklist_signal"] = checklist_signal
        if _ParameterTogglesStore.is_enabled("news_sentiment"):
            user_payload["news_sentiment"] = news_sentiment
        if options_summary:
            user_payload["options_flow"] = options_summary

        _log_marker("PRED_PARAMS", pid, _ParameterTogglesStore.to_dict())
        user_text = json.dumps(user_payload, default=str)
        system_prompt = _build_system_prompt(target_minutes, checklist_weight, snapshot_ctx)

        # ── Build OpenAI Chat Completions request ──────────────────────────
        is_reasoning = _is_reasoning_model(model_id)
        body: dict[str, Any] = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_text},
            ],
            "response_format": {"type": "json_object"},
        }
        if not is_reasoning:
            body["temperature"] = 0.15

        url = f"{settings.openai_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        _log_marker("PRED_LLM_REQUEST", pid, {
            "tool": "OPENAI",
            "model": model_id,
            "is_reasoning_model": is_reasoning,
            "user_payload_chars": len(user_text),
            "system_prompt_chars": len(system_prompt),
            "target_minutes": target_minutes,
        })

        # ── HTTP call with retry logic ─────────────────────────────────────
        llm_t0 = time.monotonic()
        llm_attempts = 0

        with httpx.Client(timeout=120.0) as client:
            resp: Optional[httpx.Response] = None

            # Transient 502/503 retry
            for attempt in range(_OPENAI_MAX_LOAD_RETRIES):
                llm_attempts += 1
                resp = client.post(url, headers=headers, json=body)
                if resp.status_code in _OPENAI_RETRYABLE_HTTP and attempt < _OPENAI_MAX_LOAD_RETRIES - 1:
                    wait_s = min(8.0, 0.75 * (2 ** attempt))
                    logger.warning(
                        "{} OpenAI {} (transient) attempt {}/{} retry in {:.1f}s",
                        log_pfx, resp.status_code, attempt + 1, _OPENAI_MAX_LOAD_RETRIES, wait_s,
                    )
                    time.sleep(wait_s)
                    continue
                break

            assert resp is not None

            # 429 rate-limit retry
            if resp.status_code == 429:
                max_r = 4
                base_d = 8.0
                for attempt in range(max_r - 1):
                    wait_s = min(90.0, base_d * (2 ** attempt))
                    logger.warning(
                        "{} OpenAI 429 rate limit — retry {}/{} in {:.1f}s",
                        log_pfx, attempt + 1, max_r - 1, wait_s,
                    )
                    time.sleep(wait_s)
                    llm_attempts += 1
                    resp = client.post(url, headers=headers, json=body)
                    if resp.status_code != 429:
                        break

            llm_latency_ms = int((time.monotonic() - llm_t0) * 1000)

            if resp.status_code == 429:
                logger.warning(
                    "{} OpenAI 429 exhausted retries after {} attempts ({}ms) — HOLD fallback",
                    log_pfx, llm_attempts, llm_latency_ms,
                )
                _log_marker("PRED_LLM_RESPONSE", pid, {
                    "tool": "OPENAI", "model": model_id,
                    "status": 429, "attempts": llm_attempts,
                    "latency_ms": llm_latency_ms, "outcome": "rate_limit_fallback",
                })
                fb = _coerce_result(
                    {
                        "direction": "HOLD", "magnitude": 0.0, "confidence": 0.0,
                        "predicted_volatility": 0.0, "valid_minutes": target_minutes,
                        "ai_quota_notice": _OPENAI_QUOTA_NOTICE,
                        "reason": "OpenAI rate limit (HTTP 429) — HOLD placeholder.",
                    },
                    float(realtime["price"]),
                    indicators=indicators, trend_context=trend_context, ohlcv=ohlcv,
                    pid=pid, horizon=horizon,
                )
                self._log_pred_result(pid, horizon, sym, target_minutes, model_id, fb,
                                      llm_latency_ms, llm_attempts, status=429,
                                      fallback="rate_limit")
                return fb

            if resp.status_code in _OPENAI_RETRYABLE_HTTP:
                logger.warning(
                    "{} OpenAI still {} after {} attempts ({}ms) — HOLD fallback",
                    log_pfx, resp.status_code, llm_attempts, llm_latency_ms,
                )
                _log_marker("PRED_LLM_RESPONSE", pid, {
                    "tool": "OPENAI", "model": model_id,
                    "status": resp.status_code, "attempts": llm_attempts,
                    "latency_ms": llm_latency_ms, "outcome": "capacity_fallback",
                })
                fb = _coerce_result(
                    {
                        "direction": "HOLD", "magnitude": 0.0, "confidence": 0.0,
                        "predicted_volatility": 0.0, "valid_minutes": target_minutes,
                        "ai_quota_notice": _OPENAI_OVERLOAD_NOTICE,
                        "reason": f"OpenAI capacity error (HTTP {resp.status_code}) — HOLD placeholder.",
                    },
                    float(realtime["price"]),
                    indicators=indicators, trend_context=trend_context, ohlcv=ohlcv,
                    pid=pid, horizon=horizon,
                )
                self._log_pred_result(pid, horizon, sym, target_minutes, model_id, fb,
                                      llm_latency_ms, llm_attempts, status=resp.status_code,
                                      fallback="capacity")
                return fb

            if resp.status_code >= 400:
                detail = (resp.text or "")[:2048]
                logger.error(
                    "{} OpenAI HTTP {} model={!r} latency={}ms attempts={}: {}",
                    log_pfx, resp.status_code, model_id, llm_latency_ms, llm_attempts, detail,
                )
                _log_marker("PRED_LLM_RESPONSE", pid, {
                    "tool": "OPENAI", "model": model_id,
                    "status": resp.status_code, "attempts": llm_attempts,
                    "latency_ms": llm_latency_ms, "outcome": "error",
                    "error_detail": detail[:512],
                })
                raise RuntimeError(f"OpenAI HTTP {resp.status_code}: {detail}")

            data = resp.json()

        text = _openai_text_from_response(data)
        raw = _parse_json_object(text)

        if not raw.get("valid_minutes"):
            raw["valid_minutes"] = target_minutes

        _log_marker("PRED_LLM_RESPONSE", pid, {
            "tool": "OPENAI", "model": model_id,
            "status": resp.status_code,
            "attempts": llm_attempts,
            "latency_ms": llm_latency_ms,
            "outcome": "ok",
            "raw_direction":   _normalize_direction(raw),
            "raw_confidence":  _safe_round(raw.get("confidence")),
            "raw_magnitude":   _safe_round(raw.get("magnitude"), 4),
            "raw_risk_reward": _safe_round(raw.get("risk_reward")),
            "raw_valid_minutes": raw.get("valid_minutes"),
            "raw_entry_price": _safe_round(raw.get("entry_price")),
            "raw_stop_loss":   _safe_round(raw.get("stop_loss")),
            "raw_target_price":_safe_round(raw.get("target_price")),
            "usage": data.get("usage"),
        })

        result = _coerce_result(
            raw,
            float(realtime["price"]),
            indicators=indicators,
            checklist_signal=checklist_signal,
            trend_context=trend_context,
            ohlcv=ohlcv,
            pid=pid,
            horizon=horizon,
        )

        self._log_pred_result(pid, horizon, sym, target_minutes, model_id, result,
                              llm_latency_ms, llm_attempts, status=resp.status_code,
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
        model_id: str,
        result: Dict[str, Any],
        llm_latency_ms: int,
        llm_attempts: int,
        *,
        status: Optional[int] = None,
        fallback: Optional[str] = None,
        trend_context: Optional[dict[str, Any]] = None,
        news_sentiment: Optional[dict[str, Any]] = None,
        checklist_signal: Optional[dict[str, Any]] = None,
        indicators: Optional[dict[str, Any]] = None,
    ) -> None:
        diag = result.get("_diagnostics") or {}
        payload = {
            "horizon": horizon,
            "symbol": symbol,
            "target_minutes": target_minutes,
            "tool": "OPENAI",
            "model": model_id,
            "model_direction": diag.get("model_direction"),
            "final_direction": result.get("direction"),
            "confidence": result.get("confidence"),
            "raw_confidence": diag.get("raw_confidence"),
            "calibrated_confidence": diag.get("calibrated_confidence"),
            "calibration_applied": diag.get("calibration_applied"),
            "effective_floor": diag.get("effective_confidence_floor"),
            "magnitude": result.get("magnitude"),
            "risk_reward": result.get("risk_reward"),
            "valid_minutes": result.get("valid_minutes"),
            "entry_price": result.get("entry_price"),
            "stop_loss": result.get("stop_loss"),
            "target_price": result.get("target_price"),
            "current_sensex": result.get("current_sensex"),
            "target_sensex": result.get("target_sensex"),
            "predicted_volatility": result.get("predicted_volatility"),
            "gates": diag.get("gates"),
            "guardrail_reason": diag.get("guardrail_reason"),
            "trend": (
                {
                    "regime":         (trend_context or {}).get("regime"),
                    "primary_regime": (trend_context or {}).get("primary_regime"),
                    "higher_regime":  (trend_context or {}).get("higher_regime"),
                    "agreement":      (trend_context or {}).get("agreement"),
                }
                if trend_context else None
            ),
            "checklist_overall": (checklist_signal or {}).get("overall") if checklist_signal else None,
            "news_overall":      (news_sentiment   or {}).get("overall") if news_sentiment   else None,
            "indicators_summary": (
                {
                    "rsi_14":          indicators.get("rsi_14"),
                    "atr_14":          indicators.get("atr_14"),
                    "volume_ratio":    indicators.get("volume_ratio"),
                    "ema9_above_ema21":indicators.get("ema9_above_ema21"),
                }
                if indicators else None
            ),
            "llm": {
                "status": status,
                "latency_ms": llm_latency_ms,
                "attempts": llm_attempts,
                "fallback": fallback,
            },
            "ai_quota_notice": result.get("ai_quota_notice"),
        }
        _log_marker("PRED_RESULT", pid, payload)
        diag_applied = bool(diag.get("calibration_applied"))
        cal_suffix = f" (raw={diag.get('raw_confidence')}%)" if diag_applied else ""
        logger.info(
            "[pid={}] DONE [{}min] model={}→final={} | conf={:.0f}%{} (floor={}) | "
            "entry={} SL={} TP={} RR={} | openai={}ms/{}attempts | regime={}",
            pid,
            target_minutes,
            diag.get("model_direction") or "?",
            result.get("direction") or "?",
            float(result.get("confidence") or 0.0),
            cal_suffix,
            diag.get("effective_confidence_floor"),
            result.get("entry_price"),
            result.get("stop_loss"),
            result.get("target_price"),
            result.get("risk_reward"),
            llm_latency_ms,
            llm_attempts,
            (trend_context or {}).get("regime") if trend_context else "n/a",
        )
