"""
Main entry point — starts both FastAPI (health/admin/predict) and gRPC server.

On Render (or any single-port host), FastAPI is the primary service on the
externally-routed port (PORT).  gRPC runs on a secondary internal-only port
(GRPC_PORT, default 50051).  The Java backend calls /predict over HTTPS when
gRPC is not reachable.
"""
import asyncio
import json
import os
import re
import threading
from datetime import date
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from app.config import settings
from app.logging_setup import configure_logging
from app.grpc_server.server import serve_grpc
from app.data.storage.db import engine, Base

configure_logging(settings.log_level)

RENDER_PORT = int(os.environ.get("PORT", "0"))
# On Render the single external port must be FastAPI (HTTP) so the /predict REST
# endpoint is reachable. gRPC moves to a secondary internal-only port.
HTTP_PORT = RENDER_PORT if RENDER_PORT else int(os.environ.get("HTTP_PORT", "8000"))
GRPC_PORT = int(os.environ.get("GRPC_PORT", "50051" if RENDER_PORT else str(settings.grpc_port)))

app = FastAPI(title="Sensex ML Service", version="0.1.0")


# ── Pydantic models for REST predict endpoint ──────────────────────────

class OhlcvBarRest(BaseModel):
    timestamp_unix_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: int = 0

class VixPointRest(BaseModel):
    timestamp_unix_ms: int
    vix: float

class SensexQuoteRest(BaseModel):
    price: float = 0.0
    change: float = 0.0
    change_pct: float = 0.0

class PredictRequest(BaseModel):
    horizon: str = "1D"
    sensex_ohlcv: List[OhlcvBarRest]
    india_vix: List[VixPointRest] = Field(default_factory=list)
    sensex_quote: Optional[SensexQuoteRest] = None
    underlying_symbol: str = ""
    instrument_token: str = ""
    engine: str = Field(default="AI", description="AI (Gemini) or ML (classical ensemble)")

class PredictResponse(BaseModel):
    prediction_date: str
    horizon: str
    direction: str
    magnitude: float
    confidence: float
    predicted_volatility: float
    current_sensex: float = 0.0
    target_sensex: float = 0.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    risk_reward: Optional[float] = None
    valid_minutes: Optional[int] = None
    ai_quota_notice: str = ""
    prediction_reason: str = ""


# ── Admin management models ────────────────────────────────────────────

class PromptUpdateRequest(BaseModel):
    prompt_text: str

class ModelUpdateRequest(BaseModel):
    tool: str
    model_id: str

class ChecklistWeightRequest(BaseModel):
    weight: int = Field(ge=0, le=100, description="Checklist signal weight as a percentage (0–100)")


class PredictionRecord(BaseModel):
    id: Optional[int] = None
    predictionDate: Optional[str] = None
    horizon: str = ""
    direction: str = ""
    confidence: Optional[float] = None
    entryPrice: Optional[float] = None
    stopLoss: Optional[float] = None
    targetSensex: Optional[float] = None
    actualClosePrice: Optional[float] = None
    outcomeStatus: Optional[str] = None
    actualPnlPct: Optional[float] = None
    predictionReason: Optional[str] = None
    aiTool: Optional[str] = None
    aiModel: Optional[str] = None

class AnalyseRequest(BaseModel):
    predictions: List[PredictionRecord]


# ── Health / debug routes ──────────────────────────────────────────────

@app.get("/")
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "grpc_listen_port": GRPC_PORT,
        "http_listen_port": HTTP_PORT,
    }


@app.get("/debug/ml")
async def debug_ml():
    from app.grpc_diagnostics import snapshot
    return snapshot()


@app.get("/models/status")
async def model_status():
    try:
        from app.inference.predictor import ML_PIPELINE_ACTIVE, Predictor
        if not ML_PIPELINE_ACTIVE:
            return {
                "status": "ML_DISABLED",
                "message": "ML ensemble is turned off (ML_PIPELINE_ACTIVE=False in predictor.py)",
            }
        predictor = Predictor()
        return predictor.get_model_health()
    except ImportError:
        return {"status": "ML_UNAVAILABLE", "message": "ML packages not installed"}


# ── Admin prompt management endpoints ─────────────────────────────────

@app.get("/admin/prompt")
async def get_active_prompt():
    from app.inference.gemini_predictor import _PromptStore, _SYSTEM_PROMPT_TEMPLATE
    custom = _PromptStore.get()
    return {
        "active": bool(custom),
        "prompt_text": custom if custom else _SYSTEM_PROMPT_TEMPLATE,
        "is_custom": bool(custom),
    }


@app.put("/admin/prompt")
async def set_active_prompt(req: PromptUpdateRequest):
    from app.inference.gemini_predictor import _PromptStore
    text = req.prompt_text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="prompt_text must not be empty")
    if "{target_minutes}" not in text:
        raise HTTPException(
            status_code=400,
            detail="prompt_text must contain the {target_minutes} dynamic variable"
        )
    _PromptStore.set(text)
    logger.info("Admin prompt updated ({} chars)", len(text))
    return {"status": "ok", "prompt_length": len(text)}


@app.delete("/admin/prompt")
async def reset_prompt():
    from app.inference.gemini_predictor import _PromptStore
    _PromptStore.clear()
    logger.info("Admin prompt reset to default")
    return {"status": "ok", "message": "Reverted to default system prompt"}


# ── Admin model management endpoints ──────────────────────────────────

@app.get("/admin/model")
async def get_active_model():
    from app.inference.gemini_predictor import _ModelStore
    from app.config import settings
    active = _ModelStore.get()
    return {
        "model_id": active,
        "is_override": bool(_ModelStore._model_id),
        "env_default": settings.gemini_model,
    }


@app.put("/admin/model")
async def set_active_model(req: ModelUpdateRequest):
    from app.inference.gemini_predictor import _ModelStore
    model_id = req.model_id.strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id must not be empty")
    _ModelStore.set(model_id)
    logger.info("Admin model updated to '{}' (tool={})", model_id, req.tool)
    return {"status": "ok", "model_id": model_id, "tool": req.tool}


@app.delete("/admin/model")
async def reset_model():
    from app.inference.gemini_predictor import _ModelStore
    from app.config import settings
    _ModelStore.clear()
    logger.info("Admin model reset to env default '{}'", settings.gemini_model)
    return {"status": "ok", "model_id": settings.gemini_model, "message": "Reverted to env default"}


# ── Admin checklist weight endpoints ─────────────────────────────────────────

@app.get("/admin/checklist-weight")
async def get_checklist_weight():
    from app.inference.gemini_predictor import _ChecklistWeightStore
    weight = _ChecklistWeightStore.get()
    return {"weight": weight, "remaining": 100 - weight}


@app.put("/admin/checklist-weight")
async def set_checklist_weight(req: ChecklistWeightRequest):
    from app.inference.gemini_predictor import _ChecklistWeightStore
    _ChecklistWeightStore.set(req.weight)
    logger.info("Admin checklist weight updated to {}%", req.weight)
    return {"status": "ok", "weight": req.weight, "remaining": 100 - req.weight}


# ── Prediction analysis endpoint ──────────────────────────────────────────────

@app.post("/admin/analyse")
async def analyse_predictions(req: AnalyseRequest):
    if not req.predictions:
        raise HTTPException(status_code=400, detail="predictions list must not be empty")

    from app.inference.gemini_predictor import _ModelStore

    key = (settings.gemini_api_key or "").strip()
    if not key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not set")

    pred_lines: list[str] = []
    for p in req.predictions:
        line = (
            f"[ID={p.id}] Date={p.predictionDate} Horizon={p.horizon} "
            f"Direction={p.direction} Confidence={p.confidence}% "
            f"Outcome={p.outcomeStatus or 'PENDING'} PnL={p.actualPnlPct}%"
        )
        if p.entryPrice is not None:
            line += f" Entry={p.entryPrice} SL={p.stopLoss} Target={p.targetSensex}"
        if p.predictionReason:
            reason_snippet = p.predictionReason[:600].replace("\n", " ")
            line += f"\n  Reason: {reason_snippet}"
        pred_lines.append(line)

    predictions_text = "\n".join(pred_lines)

    system_prompt = (
        "You are an expert intra-day Bank Nifty trading analyst reviewing a batch of AI-generated predictions. "
        "For each prediction you are given the signal direction, confidence, stated reason, and actual outcome. "
        "Your job is to identify what went wrong, what can be improved, and which reasons were not apt or were misleading. "
        "Respond with ONE JSON object only (no markdown fences), with EXACTLY these keys:\n"
        '  "overall_assessment": string — 2-4 sentences summarising the overall quality of this batch\n'
        '  "what_went_wrong": array of strings — specific problems found (e.g. overconfidence, wrong momentum reading)\n'
        '  "what_can_improve": array of strings — concrete, actionable improvement suggestions for the AI model\n'
        '  "reason_quality": array of objects with keys id (int), quality_score (0-10), feedback (string) — '
        "one entry per prediction that has a reason; critique whether the reason matched the outcome\n"
        '  "patterns": array of strings — recurring patterns observed across predictions (good or bad)\n'
        '  "recommendations": array of strings — prioritised next steps to improve prediction accuracy\n'
        "Output ONLY valid JSON — no extra text, no markdown fences."
    )

    user_text = (
        f"Analyse these {len(req.predictions)} Bank Nifty predictions and their outcomes:\n\n"
        f"{predictions_text}"
    )

    model = _ModelStore.get()
    base = settings.gemini_base_url.rstrip("/")
    url = f"{base}/models/{model}:generateContent?{urlencode({'key': key})}"

    body: dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }

    _max_retries = 3
    _retry_delays = [10, 20, 40]

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = None
            for attempt in range(_max_retries):
                resp = client.post(url, headers={"Content-Type": "application/json"}, json=body)
                if resp.status_code == 429 and attempt < _max_retries - 1:
                    wait_s = _retry_delays[attempt]
                    logger.warning(
                        "Gemini 429 (analyse) attempt {}/{} — retrying in {}s",
                        attempt + 1, _max_retries, wait_s,
                    )
                    import time as _time
                    _time.sleep(wait_s)
                    continue
                break

        if resp.status_code == 429:
            logger.warning("Gemini 429 exhausted retries for analyse")
            return {
                "error": "Gemini rate limit reached — wait a minute and try again",
                "overall_assessment": None,
                "what_went_wrong": [],
                "what_can_improve": [],
                "reason_quality": [],
                "patterns": [],
                "recommendations": [],
            }

        if resp.status_code >= 400:
            detail = (resp.text or "")[:1024]
            logger.error("Gemini analyse HTTP {}: {}", resp.status_code, detail)
            raise HTTPException(status_code=502, detail=f"Gemini HTTP {resp.status_code}: {detail}")

        data = resp.json()
        cands = data.get("candidates") or []
        parts = (cands[0].get("content") or {}).get("parts") if cands else []
        text = (parts[0].get("text") or "") if parts else ""
        text = text.strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                result = json.loads(m.group(0))
            else:
                logger.error("Could not parse Gemini analysis JSON: {!r}", text[:500])
                raise HTTPException(status_code=502, detail="Gemini returned unparseable analysis")

        logger.info("Analysis complete for {} predictions", len(req.predictions))
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analysis endpoint failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── REST prediction endpoint (mirrors gRPC GetPrediction / GetGeminiPrediction) ──

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    from app.grpc_server.proto_market import ohlcv_bars_to_dataframe, vix_points_to_dataframe
    from app.grpc_server.live_tick_buffer import get_live_tick_buffer
    from app.inference.gemini_predictor import GeminiPredictor

    min_bars = settings.min_ohlcv_bars_grpc
    if len(req.sensex_ohlcv) < min_bars:
        raise HTTPException(
            status_code=400,
            detail=(
                f"sensex_ohlcv must contain at least {min_bars} trading days of bars "
                "(after aggregating intraday data; configure min_ohlcv_bars_grpc if needed)"
            ),
        )

    bars_as_dicts = [b.model_dump() for b in req.sensex_ohlcv]
    ohlcv = ohlcv_bars_to_dataframe(bars_as_dicts)
    if ohlcv.empty:
        raise HTTPException(
            status_code=400,
            detail="sensex_ohlcv did not parse to a non-empty dataframe (check timestamps/types)",
        )

    if req.india_vix:
        vix_dicts = [v.model_dump() for v in req.india_vix]
        vix = vix_points_to_dataframe(vix_dicts)
    else:
        logger.warning("india_vix empty; deriving VIX proxy from sensex_ohlcv")
        from app.data.ingestion.vix_fetcher import derive_vix_from_ohlcv
        vix = derive_vix_from_ohlcv(ohlcv)

    quote: Optional[Dict[str, Any]] = None
    if req.sensex_quote and req.sensex_quote.price > 0:
        quote = req.sensex_quote.model_dump()

    use_ai = req.engine.upper() == "AI"
    sym = req.underlying_symbol or ""

    try:
        if use_ai:
            logger.info(f"REST /predict (AI): horizon={req.horizon} bars={len(ohlcv)} vix={len(vix)} underlying={sym!r}")
            predictor = GeminiPredictor()
            result = predictor.predict(
                horizon=req.horizon,
                ohlcv=ohlcv,
                vix=vix,
                sensex_quote=quote,
                underlying_symbol=sym,
            )
        else:
            logger.info(f"REST /predict (ML): horizon={req.horizon} bars={len(ohlcv)} vix={len(vix)}")
            try:
                from app.inference.predictor import ML_PIPELINE_ACTIVE, Predictor
                if not ML_PIPELINE_ACTIVE:
                    raise HTTPException(
                        status_code=501,
                        detail="ML prediction pipeline is disabled; use engine=AI or set ML_PIPELINE_ACTIVE=True",
                    )
                predictor = Predictor()
                result = predictor.predict(
                    horizon=req.horizon,
                    ohlcv=ohlcv,
                    vix=vix,
                    sensex_quote=quote,
                )
            except ImportError:
                raise HTTPException(status_code=501, detail="ML prediction pipeline not available; use engine=AI")

        buf = get_live_tick_buffer()
        buf.store_baseline(
            req.horizon,
            ohlcv,
            vix,
            engine=req.engine.upper(),
            underlying_symbol=req.underlying_symbol or "",
            instrument_token=req.instrument_token or "",
        )

        return PredictResponse(
            prediction_date=str(date.today()),
            horizon=req.horizon,
            direction=result.get("direction", "HOLD"),
            magnitude=float(result.get("magnitude", 0)),
            confidence=float(result.get("confidence", 0)),
            predicted_volatility=float(result.get("predicted_volatility", 0)),
            current_sensex=float(result.get("current_sensex", 0)),
            target_sensex=float(result.get("target_sensex", 0)),
            entry_price=float(result.get("entry_price")) if result.get("entry_price") is not None else None,
            stop_loss=float(result.get("stop_loss")) if result.get("stop_loss") is not None else None,
            target_price=float(result.get("target_price")) if result.get("target_price") is not None else None,
            risk_reward=float(result.get("risk_reward")) if result.get("risk_reward") is not None else None,
            valid_minutes=int(result.get("valid_minutes")) if result.get("valid_minutes") is not None else None,
            ai_quota_notice=str(result.get("ai_quota_notice", "") or ""),
            prediction_reason=str(result.get("prediction_reason", "") or ""),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"REST /predict failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── gRPC + Uvicorn startup ─────────────────────────────────────────────

def start_grpc():
    """Run gRPC server in a background thread."""
    logger.info(f"Starting gRPC server on port {GRPC_PORT}")
    asyncio.run(serve_grpc(GRPC_PORT))


if __name__ == "__main__":
    logger.info("Sensex ML Service starting...")
    logger.info(f"gRPC port: {GRPC_PORT}, HTTP port: {HTTP_PORT}")

    grpc_thread = threading.Thread(target=start_grpc, daemon=True)
    grpc_thread.start()

    uvicorn.run(app, host="0.0.0.0", port=HTTP_PORT, log_level="info")