import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import grpc
from loguru import logger

from app.config import settings
from app.grpc_server.proto_market import ohlcv_bars_to_dataframe, vix_points_to_dataframe
from app.grpc_server.live_tick_buffer import get_live_tick_buffer, live_tick_routing_key
from app.inference.gemini_predictor import GeminiPredictor

_predict_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-predict")

_ML_UNAVAILABLE_MSG = (
    "ML prediction pipeline is not available (ML packages not installed). "
    "Use GetGeminiPrediction for AI-based predictions."
)


def _try_import_predictor():
    try:
        from app.inference.predictor import ML_PIPELINE_ACTIVE, Predictor
        if not ML_PIPELINE_ACTIVE:
            logger.info("ML Predictor disabled (ML_PIPELINE_ACTIVE=False)")
            return None
        return Predictor()
    except ImportError:
        logger.warning("ML Predictor unavailable (missing packages)")
        return None


def _try_import_backtest():
    try:
        from app.backtesting.backtest_engine import BacktestEngine
        return BacktestEngine()
    except ImportError:
        logger.warning("BacktestEngine unavailable (missing packages)")
        return None


class PredictionServicer:
    """Implements the PredictionService gRPC interface."""

    def __init__(self):
        self.predictor = _try_import_predictor()
        self.gemini_predictor = GeminiPredictor()
        self.backtest_engine = _try_import_backtest()

    def _build_response(self, pb2, horizon: str, result: dict):
        """Build a PredictionResponse proto from a result dict.

        Trading levels (entry_price, stop_loss, target_price, risk_reward,
        valid_minutes) are serialised as a JSON suffix in prediction_reason
        so the Java backend can extract them without a proto schema change.
        """
        reason_text = str(result.get("prediction_reason", "") or "").strip()

        levels: dict = {}
        for k in ("entry_price", "stop_loss", "target_price", "risk_reward", "valid_minutes"):
            v = result.get(k)
            if v is not None:
                levels[k] = v

        if levels:
            reason_with_levels = f"{reason_text}\n\n[TRADING_LEVELS]{json.dumps(levels)}"
        else:
            reason_with_levels = reason_text

        return pb2.PredictionResponse(
            prediction_date=str(date.today()),
            horizon=horizon,
            direction=result["direction"],
            magnitude=result["magnitude"],
            confidence=result["confidence"],
            predicted_volatility=result["predicted_volatility"],
            current_sensex=result.get("current_sensex", 0),
            target_sensex=result.get("target_sensex", 0),
            ai_quota_notice=str(result.get("ai_quota_notice", "") or ""),
            prediction_reason=reason_with_levels,
        )

    async def GetPrediction(self, request, context):
        from app.grpc_server.generated import prediction_service_pb2 as pb2
        if self.predictor is None:
            await context.abort(grpc.StatusCode.UNIMPLEMENTED, _ML_UNAVAILABLE_MSG)
        return await self._do_prediction(request, context, pb2, engine="ML")

    async def GetGeminiPrediction(self, request, context):
        from app.grpc_server.generated import prediction_service_pb2 as pb2
        return await self._do_prediction(request, context, pb2, engine="AI")

    async def _do_prediction(self, request, context, pb2, engine: str):
        """Shared handler for both GetPrediction (ML) and GetGeminiPrediction (AI).

        For ML only: if a live re-prediction result exists in the buffer (from
        _repredict_from_live), return it to make periodic refresh cheap.
        Gemini always runs a fresh API call (no live-buffer or Redis cache for AI).
        """
        buf = get_live_tick_buffer()
        if engine == "ML":
            cached_live = buf.get_cached_live_prediction()
            if cached_live and buf.has_baseline() and cached_live.get("direction"):
                logger.debug(
                    f"GetPrediction: returning cached live result ({cached_live.get('direction')})"
                )
                return self._build_response(pb2, buf.get_baseline_horizon(), cached_live)

        min_bars = settings.min_ohlcv_bars_grpc
        if len(request.sensex_ohlcv) < min_bars:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"sensex_ohlcv must contain at least {min_bars} trading days of bars "
                "(after aggregating intraday data; configure min_ohlcv_bars_grpc if needed)",
            )
        ohlcv = ohlcv_bars_to_dataframe(request.sensex_ohlcv)
        if ohlcv.empty:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "sensex_ohlcv did not parse to a non-empty dataframe (check timestamps/types)",
            )

        if len(request.india_vix) >= 1:
            vix = vix_points_to_dataframe(request.india_vix)
        else:
            logger.warning(
                "india_vix empty; deriving VIX proxy from sensex_ohlcv "
                "(send VixPoint rows from the backend for better accuracy)."
            )
            from app.data.ingestion.vix_fetcher import derive_vix_from_ohlcv
            vix = derive_vix_from_ohlcv(ohlcv)

        quote = None
        sq = request.sensex_quote
        if sq.ByteSize() > 0:
            quote = {
                "price": sq.price,
                "change": sq.change,
                "change_pct": sq.change_pct,
            }

        try:
            sym = request.underlying_symbol or ""
            rpc_name = "GetGeminiPrediction" if engine == "AI" else "GetPrediction"
            logger.info(
                f"{rpc_name}: horizon={request.horizon} bars={len(ohlcv)} "
                f"vix={len(vix)} underlying={sym!r}"
            )

            if engine == "AI":
                result = self.gemini_predictor.predict(
                    horizon=request.horizon,
                    ohlcv=ohlcv,
                    vix=vix,
                    sensex_quote=quote,
                    underlying_symbol=sym,
                )
            else:
                result = self.predictor.predict(
                    horizon=request.horizon,
                    ohlcv=ohlcv,
                    vix=vix,
                    sensex_quote=quote,
                )

            buf.store_baseline(
                request.horizon,
                ohlcv,
                vix,
                engine=engine,
                underlying_symbol=request.underlying_symbol or "",
                instrument_token=request.instrument_token or "",
            )

            return self._build_response(pb2, request.horizon, result)
        except Exception as e:
            logger.exception(f"{engine} prediction failed: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def GetVolatilityForecast(self, request, context):
        from app.grpc_server.generated import prediction_service_pb2 as pb2

        if self.predictor is None:
            await context.abort(grpc.StatusCode.UNIMPLEMENTED, _ML_UNAVAILABLE_MSG)

        min_bars = settings.min_ohlcv_bars_grpc
        if len(request.sensex_ohlcv) < min_bars:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"sensex_ohlcv must contain at least {min_bars} trading days of bars",
            )

        ohlcv = ohlcv_bars_to_dataframe(request.sensex_ohlcv)
        if ohlcv.empty:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Failed to parse sensex_ohlcv",
            )

        try:
            logger.info(
                f"GetVolatilityForecast: days_ahead={request.days_ahead} bars={len(ohlcv)} "
                f"underlying={request.underlying_symbol!r}"
            )
            result = self.predictor.predict_volatility(
                days_ahead=request.days_ahead,
                ohlcv=ohlcv,
            )
            return pb2.VolatilityResponse(
                predicted_rv=result["predicted_rv"],
                current_iv=result["current_iv"],
                iv_percentile=result["iv_percentile"],
                signal=result["signal"],
            )
        except Exception as e:
            logger.exception(f"Volatility forecast failed: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def RunBacktest(self, request, context):
        if self.backtest_engine is None:
            await context.abort(grpc.StatusCode.UNIMPLEMENTED, _ML_UNAVAILABLE_MSG)
        try:
            from app.grpc_server.generated import prediction_service_pb2 as pb2

            logger.info(f"RunBacktest: {request.strategy_type} from {request.start_date} to {request.end_date}")
            params = json.loads(request.parameters_json) if request.parameters_json else {}

            async for progress in self.backtest_engine.run_async(
                strategy_type=request.strategy_type,
                start_date=request.start_date,
                end_date=request.end_date,
                params=params,
            ):
                yield pb2.BacktestProgress(
                    progress_percent=progress["percent"],
                    status=progress["status"],
                    result_json=json.dumps(progress.get("result", {})),
                )
        except Exception as e:
            logger.exception(f"Backtest failed: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def GetFeatureImportance(self, request, context):
        if self.predictor is None:
            await context.abort(grpc.StatusCode.UNIMPLEMENTED, _ML_UNAVAILABLE_MSG)
        try:
            from app.grpc_server.generated import prediction_service_pb2 as pb2

            importance = self.predictor.get_feature_importance()
            return pb2.FeatureImportanceResponse(
                features=[
                    pb2.FeatureScore(feature_name=name, importance=score)
                    for name, score in importance.items()
                ]
            )
        except Exception as e:
            logger.exception(f"Feature importance failed: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def GetModelHealth(self, request, context):
        if self.predictor is None:
            await context.abort(grpc.StatusCode.UNIMPLEMENTED, _ML_UNAVAILABLE_MSG)
        try:
            from app.grpc_server.generated import prediction_service_pb2 as pb2

            health = self.predictor.get_model_health()
            return pb2.ModelHealthResponse(**health)
        except Exception as e:
            logger.exception(f"Model health failed: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def StreamLiveTicks(self, request_iterator, context):
        """Ingest real-time ticks streamed from the Java backend via Angel One SmartAPI.

        Each tick updates the buffer. When the debounce window elapses after new ticks,
        re-prediction is dispatched to a thread pool so it doesn't block tick ingestion.
        """
        from app.grpc_server.generated import prediction_service_pb2 as pb2

        buf = get_live_tick_buffer()
        loop = asyncio.get_running_loop()
        n = 0
        repredictions = 0
        try:
            async for tick in request_iterator:
                n += 1
                buf.update_tick(tick)

                if n % 100 == 0 or n == 1:
                    ltp = float(getattr(tick, "last_traded_price", 0.0) or 0.0)
                    close_p = float(getattr(tick, "close", 0.0) or 0.0)
                    chg = float(getattr(tick, "change", 0.0) or 0.0)
                    chg_pct = float(getattr(tick, "change_pct", 0.0) or 0.0)
                    label = live_tick_routing_key(tick) or "?"
                    logger.info(
                        f"[LIVE] {label}: ₹{ltp:.2f} | "
                        f"Δ: {chg:+.2f} ({chg_pct:+.2f}%) | "
                        f"H: {tick.high:.2f} L: {tick.low:.2f} | "
                        f"O: {tick.open:.2f} C: {close_p:.2f} | "
                        f"Vol: {tick.volume} | ticks={n}"
                    )

                route_key = live_tick_routing_key(tick)
                if (
                    route_key
                    and buf.tick_matches_baseline(route_key)
                    and buf.should_repredict()
                    and buf.start_repredict()
                ):
                    ok = await loop.run_in_executor(
                        _predict_executor,
                        self._repredict_from_live,
                        buf,
                    )
                    if ok:
                        repredictions += 1

        except Exception as e:
            logger.exception(f"StreamLiveTicks failed: {e}")
            return pb2.StreamAck(accepted=False, message=str(e)[:512])

        msg = f"stream closed: {n} tick(s), {repredictions} re-prediction(s)"
        logger.info(msg)
        return pb2.StreamAck(accepted=True, message=msg)

    def _repredict_from_live(self, buf) -> bool:
        """Merge live tick into baseline OHLCV and re-run prediction (runs in thread pool).

        Routes to the Gemini or ML predictor based on the engine stored with the baseline.
        Uses only ticks for the baseline underlying (not INDIA VIX or other stream symbols).
        """
        merged = buf.get_merged_ohlcv()
        if merged is None or merged.empty:
            buf.mark_predicted({})
            return False

        horizon = buf.get_baseline_horizon()
        engine = buf.get_baseline_engine()
        vix = buf.get_baseline_vix()
        baseline_sym = buf.get_baseline_underlying()

        tick = buf.get_baseline_tick()
        quote = None
        if tick and tick.ltp > 0:
            quote = {"price": tick.ltp, "change": tick.change, "change_pct": tick.change_pct}

        try:
            use_gemini = engine == "AI"
            label = "GEMINI" if use_gemini else "ML"
            logger.info(
                f"[LIVE RE-PREDICT {label}] horizon={horizon} merged_bars={len(merged)} "
                f"underlying={baseline_sym!r} ltp={tick.ltp if tick else 'N/A'}"
            )
            if use_gemini:
                result = self.gemini_predictor.predict(
                    horizon=horizon,
                    ohlcv=merged,
                    vix=vix,
                    sensex_quote=quote,
                    underlying_symbol=baseline_sym,
                )
            else:
                result = self.predictor.predict(
                    horizon=horizon,
                    ohlcv=merged,
                    vix=vix,
                    sensex_quote=quote,
                )
            buf.mark_predicted(result)
            logger.info(
                f"[LIVE RE-PREDICT {label}] result: {result.get('direction')} "
                f"conf={result.get('confidence')}% mag={result.get('magnitude')}%"
            )
            return True
        except Exception as e:
            logger.error(f"Live re-prediction failed: {e}")
            buf.mark_predicted({})
            return False
