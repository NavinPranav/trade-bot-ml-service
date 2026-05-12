"""India VIX utilities.

Production path: VIX is supplied by the Java backend as VixPoint rows in the gRPC request.
Fallback path: derive a historical-volatility proxy from the OHLCV data already in the request.
"""
import numpy as np
import pandas as pd
from loguru import logger


def derive_vix_from_ohlcv(ohlcv: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute annualised historical volatility from OHLCV close prices as a VIX proxy.

    Returns a DataFrame with a single 'vix' column aligned to the OHLCV index,
    suitable for passing directly to the feature pipeline.
    """
    if ohlcv.empty or "close" not in ohlcv.columns or len(ohlcv) < 2:
        logger.warning("derive_vix_from_ohlcv: insufficient data (need at least 2 bars)")
        return pd.DataFrame()

    effective_window = min(window, max(2, len(ohlcv) - 1))
    log_returns = np.log(ohlcv["close"] / ohlcv["close"].shift(1))
    hvol = log_returns.rolling(effective_window).std() * np.sqrt(252) * 100
    df = hvol.rename("vix").to_frame().dropna()
    logger.info(f"Derived VIX proxy from OHLCV ({len(df)} rows, {effective_window}-day HVol)")
    return df
