"""
Main prediction orchestrator (ML ensemble).

When ML_PIPELINE_ACTIVE is False, this module exposes a stub Predictor so the
service starts without arch / xgboost / torch / etc. Use GetGeminiPrediction
or REST engine=AI. Re-enable ML by setting ML_PIPELINE_ACTIVE True and restoring
the implementation from version control (see git history for this file).
"""
from datetime import date
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

from app.config import settings
from app.data.ingestion.vix_fetcher import derive_vix_from_ohlcv
from app.data.storage.redis_cache import cache_prediction, get_cached_prediction
from app.inference.ohlcv_fingerprint import ohlcv_cache_fingerprint

# --- Toggle ML ensemble (GARCH / ARIMA / XGBoost / LightGBM / LSTM / stacking) ---
ML_PIPELINE_ACTIVE = False


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


class Predictor:
    """Stub when ML_PIPELINE_ACTIVE is False; full ensemble lives in git history."""

    def __init__(self):
        logger.warning("ML Predictor: ensemble pipeline is disabled (ML_PIPELINE_ACTIVE=False)")

    def predict(
        self,
        horizon: str = "1D",
        ohlcv: Optional[pd.DataFrame] = None,
        vix: Optional[pd.DataFrame] = None,
        sensex_quote: Optional[Dict[str, Any]] = None,
        *,
        bypass_cache: bool = False,
        cache_engine: str = "ML",
    ) -> Dict[str, Any]:
        if ohlcv is None or ohlcv.empty:
            logger.error("predict() aborted: empty OHLCV (backend must send bars in the request)")
            return {
                "error": "No market data available (supply OHLCV from the backend)",
                "direction": "NEUTRAL",
                "magnitude": 0,
                "confidence": 0,
                "predicted_volatility": 0,
                "prediction_reason": "",
            }

        if vix is None or vix.empty:
            logger.warning("VIX not supplied; deriving proxy from OHLCV (send VixPoint rows for better accuracy)")
            vix = derive_vix_from_ohlcv(ohlcv)

        cache_fp = ohlcv_cache_fingerprint(ohlcv)
        if not bypass_cache:
            cached = get_cached_prediction(horizon, cache_fp, engine=cache_engine)
        else:
            cached = None
        if cached:
            return cached

        realtime = _resolve_realtime(ohlcv, sensex_quote)
        result: Dict[str, Any] = {
            "error": "ML ensemble disabled (set ML_PIPELINE_ACTIVE and install ML deps)",
            "direction": "NEUTRAL",
            "magnitude": 0.0,
            "confidence": 0.0,
            "predicted_volatility": 0.0,
            "current_sensex": realtime.get("price", 0.0),
            "target_sensex": realtime.get("price", 0.0),
            "prediction_reason": "",
        }
        if not bypass_cache:
            cache_prediction(horizon, result, ttl=300, data_fingerprint=cache_fp, engine=cache_engine)
        return result

    def predict_volatility(
        self,
        days_ahead: int = 5,
        ohlcv: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        if ohlcv is None or ohlcv.empty:
            return {
                "predicted_rv": 0.0,
                "current_iv": 0.0,
                "iv_percentile": 50,
                "signal": "IV_FAIR",
            }
        return {
            "predicted_rv": 0.0,
            "current_iv": 0.0,
            "iv_percentile": 50,
            "signal": "IV_FAIR",
            "error": "ML volatility pipeline disabled",
        }

    def get_feature_importance(self) -> Dict[str, float]:
        return {}

    def get_model_health(self) -> Dict[str, Any]:
        return {
            "model_version": "0.0.0-disabled",
            "last_trained": str(date.today()),
            "recent_accuracy": 0.0,
            "predictions_today": 0,
            "status": "DISABLED",
        }
