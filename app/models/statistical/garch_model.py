"""
GARCH(1,1) — Volatility Forecasting

NOT a machine learning model. This is a statistical model that uses
maximum likelihood estimation to fit parameters for conditional variance.

Purpose: Predict realized volatility → compare with implied volatility
         → determine if options are cheap or expensive.

Library: arch
Input: Sensex returns series
Output: Predicted annualized volatility for next N days
"""
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from loguru import logger

from app.models.base import BaseModel


class GARCHModel(BaseModel):

    def __init__(self, model_dir: Path):
        super().__init__("garch", model_dir)
        self.forecast_horizon = 5

    def train(self, df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        from arch import arch_model

        returns = df["returns"].dropna() * 100  # Scale to percentage

        if len(returns) < 100:
            logger.warning("GARCH: insufficient data for training")
            return {"status": "failed", "reason": "insufficient data"}

        # Fit GARCH(1,1)
        model = arch_model(returns, vol="Garch", p=1, q=1, mean="AR", lags=1)
        self.model = model.fit(disp="off", show_warning=False)

        self.is_trained = True
        logger.info(f"GARCH trained: AIC={self.model.aic:.2f}, BIC={self.model.bic:.2f}")

        return {
            "aic": self.model.aic,
            "bic": self.model.bic,
            "log_likelihood": self.model.loglikelihood,
            "alpha": float(self.model.params.get("alpha[1]", 0)),
            "beta": float(self.model.params.get("beta[1]", 0)),
        }

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Forecast volatility by fitting GARCH on the supplied returns window.

        A saved ``garch_model.pkl`` is optional: inference always re-fits on ``df`` (see below).
        """
        if not self.is_trained:
            self.load()

        from arch import arch_model

        if "returns" not in df.columns or df["returns"].dropna().empty:
            logger.warning("GARCH: no returns column or empty series")
            return {"predicted_volatility": 0, "error": "no returns", "model": "GARCH(1,1)"}

        returns = df["returns"].dropna() * 100
        if len(returns) < 30:
            logger.warning(f"GARCH: need at least ~30 return observations; got {len(returns)}")
            return {"predicted_volatility": 0, "error": "insufficient returns", "model": "GARCH(1,1)"}

        # Re-fit on latest data for forecast (does not use pickled self.model)
        model = arch_model(returns, vol="Garch", p=1, q=1, mean="AR", lags=1)
        fitted = model.fit(disp="off", show_warning=False, last_obs=len(returns))

        # Forecast next N days
        forecast = fitted.forecast(horizon=self.forecast_horizon)
        variance_forecast = forecast.variance.iloc[-1].values

        # Annualize: daily variance → annualized volatility
        daily_vol = np.sqrt(variance_forecast.mean()) / 100
        annualized_vol = daily_vol * np.sqrt(252)

        return {
            "predicted_volatility": round(float(annualized_vol), 4),
            "daily_vol": round(float(daily_vol), 6),
            "forecast_days": self.forecast_horizon,
            "model": "GARCH(1,1)",
        }

    def save(self) -> Path:
        path = self.model_dir / "garch_model.pkl"
        if self.model is not None:
            with open(path, "wb") as f:
                pickle.dump(self.model, f)
            logger.info(f"GARCH model saved to {path}")
        return path

    def load(self) -> bool:
        path = self.model_dir / "garch_model.pkl"
        if path.exists():
            with open(path, "rb") as f:
                self.model = pickle.load(f)
            self.is_trained = True
            logger.info("GARCH model loaded from disk")
            return True
        return False