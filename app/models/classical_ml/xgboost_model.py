"""
XGBoost — Direction Classification

Machine learning model. Learns decision rules from hand-engineered features.
The workhorse of quant finance — fast, accurate, interpretable via feature importance.

Library: xgboost
Input: Feature matrix (50+ indicators)
Output: Direction (BULLISH/BEARISH) + probability
"""
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from loguru import logger

from app.models.base import BaseModel


class XGBoostDirectionModel(BaseModel):

    def __init__(self, model_dir: Path, feature_columns: list = None):
        super().__init__("xgboost_direction", model_dir)
        self.feature_columns = feature_columns or []
        self.params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }

    def train(self, df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        feature_cols = [c for c in self.feature_columns if c in df.columns]
        if not feature_cols:
            return {"status": "failed", "reason": "no feature columns found"}

        X = df[feature_cols].values
        y = df["target_direction"].values

        # Walk-forward split: train on first 80%, test on last 20%
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = self.model.predict(X_test)
        self.is_trained = True

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
        logger.info(f"XGBoost trained: accuracy={metrics['accuracy']}, f1={metrics['f1']}")
        return metrics

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.is_trained and not self.load():
            return {"direction": "NEUTRAL", "confidence": 0, "error": "model not loaded"}

        feature_cols = [c for c in self.feature_columns if c in df.columns]
        X = df[feature_cols].iloc[-1:].values

        prob = self.model.predict_proba(X)[0]
        direction = "BULLISH" if prob[1] > 0.5 else "BEARISH"
        confidence = float(max(prob)) * 100

        return {
            "direction": direction,
            "confidence": round(confidence, 2),
            "prob_bullish": round(float(prob[1]), 4),
            "prob_bearish": round(float(prob[0]), 4),
            "model": "XGBoost",
        }

    def get_feature_importance(self) -> Dict[str, float]:
        if self.model is None:
            return {}
        importance = self.model.feature_importances_
        return {
            col: round(float(imp), 4)
            for col, imp in sorted(
                zip(self.feature_columns, importance),
                key=lambda x: x[1], reverse=True
            )[:20]
        }

    def save(self) -> Path:
        path = self.model_dir / "xgboost_model.pkl"
        if self.model:
            with open(path, "wb") as f:
                pickle.dump({"model": self.model, "features": self.feature_columns}, f)
        return path

    def load(self) -> bool:
        path = self.model_dir / "xgboost_model.pkl"
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.feature_columns = data["features"]
            self.is_trained = True
            logger.info("XGBoost model loaded from disk")
            return True
        return False