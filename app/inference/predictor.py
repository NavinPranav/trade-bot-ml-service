"""
Main prediction orchestrator.
Loads all models, runs feature pipeline, combines via ensemble.
"""
from datetime import date
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

from app.config import settings
from app.data.ingestion.vix_fetcher import derive_vix_from_ohlcv
from app.data.storage.redis_cache import cache_prediction, get_cached_prediction
from app.inference.ohlcv_fingerprint import ohlcv_cache_fingerprint
from app.features.feature_pipeline import FeaturePipeline
from app.models.statistical.garch_model import GARCHModel
from app.models.statistical.arima_model import ARIMAModel
from app.models.classical_ml.xgboost_model import XGBoostDirectionModel
from app.models.classical_ml.lightgbm_model import LightGBMMagnitudeModel
from app.models.deep_learning.lstm_model import LSTMModel
from app.models.ensemble.stacking_ensemble import StackingEnsemble


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

    def __init__(self):
        model_dir = settings.model_dir
        self.feature_pipeline = FeaturePipeline()
        feature_cols = self.feature_pipeline.get_feature_columns()

        # Initialize all models
        self.garch = GARCHModel(model_dir)
        self.arima = ARIMAModel(model_dir)
        self.xgboost = XGBoostDirectionModel(model_dir, feature_cols)
        self.lightgbm = LightGBMMagnitudeModel(model_dir, feature_cols)
        self.lstm = LSTMModel(model_dir)
        self.ensemble = StackingEnsemble(model_dir)

        # Try loading saved models
        for m in [self.garch, self.arima, self.xgboost, self.lightgbm, self.lstm]:
            m.load()

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
            }

        if vix is None or vix.empty:
            logger.warning("VIX not supplied; deriving proxy from OHLCV (send VixPoint rows for better accuracy)")
            vix = derive_vix_from_ohlcv(ohlcv)
        vix_for_features = vix if not vix.empty else None

        cache_fp = ohlcv_cache_fingerprint(ohlcv)
        if not bypass_cache:
            cached = get_cached_prediction(horizon, cache_fp, engine=cache_engine)
        else:
            cached = None
        if cached:
            logger.info(
                "Prediction (cache hit): direction={} conf={}% mag={}% cur={} tgt={}",
                cached.get("direction"),
                cached.get("confidence"),
                cached.get("magnitude"),
                cached.get("current_sensex"),
                cached.get("target_sensex"),
            )
            return cached

        logger.info(f"Running prediction pipeline for horizon={horizon}")

        # Build features
        df = self.feature_pipeline.build(ohlcv, vix=vix_for_features)
        if df.empty:
            logger.error("Feature pipeline returned empty (check OHLCV length vs indicator warmup)")
            return {"error": "Feature pipeline returned empty", "direction": "NEUTRAL",
                    "magnitude": 0, "confidence": 0, "predicted_volatility": 0}

        # Run all models
        predictions = []

        # Statistical
        try:
            garch_pred = self.garch.predict(df)
            predictions.append(garch_pred)
        except Exception as e:
            logger.error(f"GARCH prediction failed: {e}")

        try:
            arima_pred = self.arima.predict(df)
            predictions.append(arima_pred)
        except Exception as e:
            logger.error(f"ARIMA prediction failed: {e}")

        # Classical ML
        try:
            xgb_pred = self.xgboost.predict(df)
            predictions.append(xgb_pred)
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")

        try:
            lgb_pred = self.lightgbm.predict(df)
            predictions.append(lgb_pred)
        except Exception as e:
            logger.error(f"LightGBM prediction failed: {e}")

        # Deep Learning
        try:
            lstm_pred = self.lstm.predict(df)
            predictions.append(lstm_pred)
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")

        # Ensemble
        result = self.ensemble.combine_predictions(predictions)

        realtime = _resolve_realtime(ohlcv, sensex_quote)
        result["current_sensex"] = realtime.get("price", 0)
        if result["current_sensex"] and result["magnitude"]:
            result["target_sensex"] = round(
                result["current_sensex"] * (1 + result["magnitude"] / 100), 2
            )

        if not bypass_cache:
            cache_prediction(
                horizon, result, ttl=300, data_fingerprint=cache_fp, engine=cache_engine
            )

        logger.info(f"Prediction: {result['direction']} | conf={result['confidence']}% | mag={result['magnitude']}%")
        return result

    def predict_volatility(
        self,
        days_ahead: int = 5,
        ohlcv: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        if ohlcv is None or ohlcv.empty:
            logger.error("predict_volatility: no OHLCV supplied (backend must send bars)")
            return {
                "predicted_rv": 0.0,
                "current_iv": 0.0,
                "iv_percentile": 50,
                "signal": "IV_FAIR",
            }

        df = self.feature_pipeline.build(ohlcv)

        garch_result = self.garch.predict(df)
        current_iv = df.get("hvol_20", [0]).iloc[-1] if "hvol_20" in df.columns else 0

        predicted_rv = garch_result.get("predicted_volatility", 0)
        iv_percentile = 50  # TODO: compute from historical IV data

        if predicted_rv > current_iv * 1.1:
            signal = "IV_LOW"
        elif predicted_rv < current_iv * 0.9:
            signal = "IV_HIGH"
        else:
            signal = "IV_FAIR"

        return {
            "predicted_rv": predicted_rv,
            "current_iv": round(float(current_iv), 4),
            "iv_percentile": iv_percentile,
            "signal": signal,
        }

    def get_feature_importance(self) -> Dict[str, float]:
        return self.xgboost.get_feature_importance()

    def get_model_health(self) -> Dict[str, Any]:
        return {
            "model_version": "0.1.0",
            "last_trained": str(date.today()),
            "recent_accuracy": 0.0,  # TODO: compute from stored predictions
            "predictions_today": 0,
            "status": "HEALTHY" if self.xgboost.is_trained else "STALE",
        }