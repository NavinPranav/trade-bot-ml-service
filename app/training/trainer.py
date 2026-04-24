"""
Model training orchestrator — trains all models in sequence.

ML training is disabled (ML_TRAINING_ACTIVE=False). Re-enable from git history
or set the flag True after installing statistical + ML packages.
"""
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from loguru import logger

from app.training.prepared_data import DEFAULT_DATA_DIR, load_training_bundle

ML_TRAINING_ACTIVE = False


class ModelTrainer:

    def __init__(self):
        logger.warning("ModelTrainer: ML training pipeline is disabled (ML_TRAINING_ACTIVE=False)")

    def train_all(self, ohlcv: pd.DataFrame, vix: pd.DataFrame) -> Dict[str, Dict]:
        # Previously: FeaturePipeline.build → GARCHModel / ARIMAModel / XGBoostDirectionModel /
        # LightGBMMagnitudeModel train+save, LSTM stub. Restore from git when ML_TRAINING_ACTIVE.
        logger.warning("train_all skipped: ML training disabled")
        return {
            "disabled": True,
            "message": "ML training is turned off (see ML_TRAINING_ACTIVE in app/training/trainer.py)",
        }

    def train_from_prepared(self, data_dir: Optional[Path] = None) -> Dict[str, Dict]:
        """Load `data/training/*.csv` from prepare_training_data.py and run train_all."""
        path = data_dir or DEFAULT_DATA_DIR
        ohlcv, vix = load_training_bundle(path)
        return self.train_all(ohlcv, vix)
