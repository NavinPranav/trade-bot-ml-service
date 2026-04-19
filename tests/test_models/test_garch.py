import pandas as pd
import numpy as np
from pathlib import Path
from app.models.statistical.garch_model import GARCHModel


def test_garch_train_predict(tmp_path):
    np.random.seed(42)
    n = 500
    returns = np.random.randn(n) * 0.01
    df = pd.DataFrame({"returns": returns}, index=pd.date_range("2022-01-01", periods=n, freq="B"))

    model = GARCHModel(tmp_path)
    metrics = model.train(df)
    assert "aic" in metrics

    pred = model.predict(df)
    assert "predicted_volatility" in pred
    assert pred["predicted_volatility"] > 0