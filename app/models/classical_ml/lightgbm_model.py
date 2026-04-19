"""
LightGBM — Magnitude Regression

Machine learning model. Predicts the % magnitude of the next move.

Library: lightgbm
Input: Feature matrix
Output: Predicted % price change
"""
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from loguru import logger

from app.models.base import BaseModel


class LightGBMMagnitudeModel(BaseModel):

    def __init__(self, model_dir: Path, feature_columns: list = None):
        super().__init__("lightgbm_magnitude", model_dir)
        self.feature_columns = feature_columns or []
        self.params = {
            "objective": "regression",
            "metric": "mae",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbose": -1,
            "random_state": 42,
            "n_jobs": -1,
        }

    def train(self, df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        feature_cols = [c for c in self.feature_columns if c in df.columns]
        X = df[feature_cols].values
        y = df["target_return"].values * 100  # Percentage

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

        y_pred = self.model.predict(X_test)
        self.is_trained = True

        metrics = {
            "mae": round(mean_absolute_error(y_test, y_pred), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            "direction_accuracy": round(
                np.mean(np.sign(y_pred) == np.sign(y_test)), 4
            ),
        }
        logger.info(f"LightGBM trained: MAE={metrics['mae']}, dir_acc={metrics['direction_accuracy']}")
        return metrics

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.is_trained and not self.load():
            return {"magnitude": 0, "error": "model not loaded"}

        feature_cols = [c for c in self.feature_columns if c in df.columns]
        X = df[feature_cols].iloc[-1:].values
        magnitude = float(self.model.predict(X)[0])

        return {
            "magnitude": round(magnitude, 4),
            "model": "LightGBM",
        }

    def save(self) -> Path:
        path = self.model_dir / "lightgbm_model.pkl"
        if self.model:
            with open(path, "wb") as f:
                pickle.dump({"model": self.model, "features": self.feature_columns}, f)
        return path

    def load(self) -> bool:
        path = self.model_dir / "lightgbm_model.pkl"
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.feature_columns = data["features"]
            self.is_trained = True
            return True
        return False