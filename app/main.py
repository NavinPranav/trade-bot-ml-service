"""
Main entry point — starts both FastAPI (health/admin/predict) and gRPC server.

On Render (or any single-port host), FastAPI is the primary service on the
externally-routed port (PORT).  gRPC runs on a secondary internal-only port
(GRPC_PORT, default 50051).  The Java backend calls /predict over HTTPS when
gRPC is not reachable.
"""
import asyncio
import os
import threading
from datetime import date
from typing import Any, Dict, List, Optional

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
    ai_quota_notice: str = ""


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
            ai_quota_notice=str(result.get("ai_quota_notice", "") or ""),
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