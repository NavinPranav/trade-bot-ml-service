"""
Model training orchestrator — trains all models in sequence.
"""
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from loguru import logger

from app.config import settings
from app.features.feature_pipeline import FeaturePipeline
from app.training.prepared_data import DEFAULT_DATA_DIR, load_training_bundle
from app.models.statistical.garch_model import GARCHModel
from app.models.statistical.arima_model import ARIMAModel
from app.models.classical_ml.xgboost_model import XGBoostDirectionModel
from app.models.classical_ml.lightgbm_model import LightGBMMagnitudeModel


class ModelTrainer:

    def __init__(self):
        self.model_dir = settings.model_dir
        self.feature_pipeline = FeaturePipeline()
        self.feature_cols = self.feature_pipeline.get_feature_columns()

    def train_all(self, ohlcv: pd.DataFrame, vix: pd.DataFrame) -> Dict[str, Dict]:
        logger.info("Starting full model training pipeline...")

        df = self.feature_pipeline.build(ohlcv, vix=vix)

        if df.empty:
            logger.error("No data available for training")
            return {"error": "no data"}

        results = {}

        # 1. GARCH (Statistical)
        logger.info("Training GARCH...")
        garch = GARCHModel(self.model_dir)
        results["garch"] = garch.train(df)
        garch.save()

        # 2. ARIMA (Statistical)
        logger.info("Training ARIMA...")
        arima = ARIMAModel(self.model_dir)
        results["arima"] = arima.train(df)
        arima.save()

        # 3. XGBoost (Classical ML)
        logger.info("Training XGBoost...")
        xgb = XGBoostDirectionModel(self.model_dir, self.feature_cols)
        results["xgboost"] = xgb.train(df)
        xgb.save()

        # 4. LightGBM (Classical ML)
        logger.info("Training LightGBM...")
        lgb = LightGBMMagnitudeModel(self.model_dir, self.feature_cols)
        results["lightgbm"] = lgb.train(df)
        lgb.save()

        # 5. LSTM (Deep Learning) — stub
        logger.info("LSTM training: stub (implement full pipeline)")
        results["lstm"] = {"status": "stub"}

        logger.info(f"Training complete. Results: {results}")
        return results

    def train_from_prepared(self, data_dir: Optional[Path] = None) -> Dict[str, Dict]:
        """Load `data/training/*.csv` from prepare_training_data.py and run train_all."""
        path = data_dir or DEFAULT_DATA_DIR
        ohlcv, vix = load_training_bundle(path)
        return self.train_all(ohlcv, vix)