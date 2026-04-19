"""Data cleaning and normalization."""
import pandas as pd
import numpy as np
from loguru import logger


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    n_in = len(df)
    # Only treat 0 as missing for prices — volume may legitimately be 0; all-zero volume must not
    # poison rows (replace(0, NaN) on the whole frame + dropna() was wiping every row).
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    df = df.dropna(subset=["close"])
    df = df.ffill()

    if "volume" in df.columns:
        vol = df["volume"].replace(0, np.nan)
        df["volume"] = vol.ffill().bfill().fillna(0.0)

    # Add returns column
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # Warmup: first row has no prior close — drop only those, not rows with e.g. NaN in unused cols
    df = df.dropna(subset=["returns", "log_returns"])
    if df.empty and n_in > 0:
        logger.error(
            "clean_ohlcv: no rows left (need ≥2 valid closes for returns). "
            "If gRPC sent many intraday bars for one session, daily aggregation may collapse to 1 row."
        )
    else:
        logger.info(f"Cleaned OHLCV data: {len(df)} rows")
    return df


def normalize_features(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[f"{col}_norm"] = (df[col] - mean) / std

    return df