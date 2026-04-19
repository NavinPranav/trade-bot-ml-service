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
    if ohlcv.empty or "close" not in ohlcv.columns:
        logger.warning("derive_vix_from_ohlcv: empty or missing 'close' column")
        return pd.DataFrame()

    log_returns = np.log(ohlcv["close"] / ohlcv["close"].shift(1))
    hvol = log_returns.rolling(window).std() * np.sqrt(252) * 100  # annualised %
    df = hvol.rename("vix").to_frame().dropna()
    logger.info(f"Derived VIX proxy from OHLCV ({len(df)} rows, {window}-day HVol)")
    return df
