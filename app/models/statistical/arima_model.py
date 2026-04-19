"""
ARIMA — Trend and Momentum Baseline

NOT a machine learning model. Uses autoregressive integrated moving average
for time-series forecasting. Good baseline for price level prediction.

Library: statsmodels
Input: Sensex close price series
Output: Predicted price level for next N days
"""
import pickle
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from loguru import logger

from app.models.base import BaseModel


class ARIMAModel(BaseModel):

    def __init__(self, model_dir: Path):
        super().__init__("arima", model_dir)
        self.order = (2, 1, 2)  # (p, d, q)

    def train(self, df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        from statsmodels.tsa.arima.model import ARIMA

        close = df["close"].dropna()
        if len(close) < 100:
            return {"status": "failed", "reason": "insufficient data"}

        self.model = ARIMA(close, order=self.order).fit()
        self.is_trained = True

        logger.info(f"ARIMA{self.order} trained: AIC={self.model.aic:.2f}")
        return {"aic": self.model.aic, "bic": self.model.bic}

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.is_trained and not self.load():
            return {"predicted_price": 0, "error": "model not loaded"}

        forecast = self.model.forecast(steps=5)
        current = df["close"].iloc[-1]
        predicted = forecast.iloc[0]
        direction = "BULLISH" if predicted > current else "BEARISH"

        return {
            "predicted_price": round(float(predicted), 2),
            "current_price": round(float(current), 2),
            "direction": direction,
            "pct_change": round(float((predicted - current) / current * 100), 4),
            "model": f"ARIMA{self.order}",
        }

    def save(self) -> Path:
        path = self.model_dir / "arima_model.pkl"
        if self.model:
            with open(path, "wb") as f:
                pickle.dump(self.model, f)
        return path

    def load(self) -> bool:
        path = self.model_dir / "arima_model.pkl"
        if path.exists():
            with open(path, "rb") as f:
                self.model = pickle.load(f)
            self.is_trained = True
            return True
        return False