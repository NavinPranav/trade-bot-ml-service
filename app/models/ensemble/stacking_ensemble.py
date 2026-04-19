"""
Ensemble / Meta-Learner — Combines All Models

Takes predictions from statistical, ML, and deep learning models
and learns which model to trust in which market regime.

Library: scikit-learn (LogisticRegression as meta-learner)
Input: Predictions from all base models
Output: Final direction, magnitude, volatility, confidence
"""
import pickle
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from loguru import logger

from app.models.base import BaseModel


class StackingEnsemble(BaseModel):

    def __init__(self, model_dir: Path):
        super().__init__("ensemble", model_dir)
        self.meta_model = LogisticRegression(random_state=42)

    def combine_predictions(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simple weighted combination of base model predictions.
        Used before the meta-learner is trained.
        """
        directions = []
        confidences = []
        magnitudes = []
        volatility = 0.0

        weights = {
            "XGBoost": 0.35,
            "LightGBM": 0.20,
            "GARCH(1,1)": 0.20,
            "LSTM": 0.10,
            "LSTM (stub)": 0.05,
            "default": 0.10,
        }

        for pred in predictions:
            model_name = pred.get("model", "default")
            weight = weights.get(model_name, weights["default"])

            if "direction" in pred:
                score = 1 if pred["direction"] == "BULLISH" else -1 if pred["direction"] == "BEARISH" else 0
                directions.append(score * weight)

            if "confidence" in pred:
                confidences.append(pred["confidence"] * weight)

            if "magnitude" in pred:
                magnitudes.append(pred["magnitude"] * weight)

            if "predicted_volatility" in pred:
                volatility = pred["predicted_volatility"]

        # Aggregate
        direction_score = sum(directions)
        if direction_score > 0.1:
            final_direction = "BULLISH"
        elif direction_score < -0.1:
            final_direction = "BEARISH"
        else:
            final_direction = "NEUTRAL"

        final_confidence = min(sum(confidences) / max(sum(weights.values()), 1) * 1.5, 99)
        final_magnitude = sum(magnitudes)

        return {
            "direction": final_direction,
            "magnitude": round(final_magnitude, 4),
            "confidence": round(final_confidence, 2),
            "predicted_volatility": round(volatility, 4),
            "direction_score": round(direction_score, 4),
            "models_used": len(predictions),
        }

    def train(self, df: pd.DataFrame = None, **kwargs) -> Dict[str, float]:
        # TODO: Train meta-learner on historical base model predictions
        logger.warning("Ensemble meta-learner training: not yet implemented. Using weighted average.")
        return {"status": "using_weighted_average"}

    def predict(self, df=None) -> Dict[str, Any]:
        return {"error": "Use combine_predictions() instead"}

    def save(self) -> Path:
        path = self.model_dir / "ensemble_meta.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.meta_model, f)
        return path

    def load(self) -> bool:
        path = self.model_dir / "ensemble_meta.pkl"
        if path.exists():
            with open(path, "rb") as f:
                self.meta_model = pickle.load(f)
            self.is_trained = True
            return True
        return False