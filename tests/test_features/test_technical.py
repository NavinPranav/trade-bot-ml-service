import pandas as pd
import numpy as np
from app.features.technical import add_technical_features


def test_add_technical_features():
    np.random.seed(42)
    n = 250
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "open": np.random.uniform(59000, 61000, n),
        "high": np.random.uniform(60000, 62000, n),
        "low": np.random.uniform(58000, 60000, n),
        "close": np.cumsum(np.random.randn(n) * 100) + 60000,
        "volume": np.random.randint(1000000, 5000000, n),
        "returns": np.random.randn(n) * 0.01,
    }, index=dates)

    result = add_technical_features(df)
    assert "rsi_14" in result.columns
    assert "macd" in result.columns
    assert "bb_upper" in result.columns
    assert "atr_14" in result.columns
    assert "sma_20" in result.columns
    assert len(result) == n