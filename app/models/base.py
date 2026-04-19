"""Abstract base model interface — all models implement this."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import pandas as pd


class BaseModel(ABC):
    """
    Every model (statistical, ML, or deep learning) implements this interface.
    This ensures the ensemble can treat all models uniformly.
    """

    def __init__(self, name: str, model_dir: Path):
        self.name = name
        self.model_dir = model_dir / name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.is_trained = False

    @abstractmethod
    def train(self, df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Train the model. Returns metrics dict."""
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run inference. Returns prediction dict."""
        pass

    @abstractmethod
    def save(self) -> Path:
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self) -> bool:
        """Load model from disk. Returns True if successful."""
        pass

    def get_name(self) -> str:
        return self.name