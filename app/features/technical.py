"""Technical indicator features using the 'ta' library."""
import pandas as pd
import ta
from loguru import logger


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "close" not in df.columns:
        logger.warning("Cannot compute technical features: empty or missing 'close'")
        return df

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df.get("volume", pd.Series(dtype=float))

    # ── Trend ──
    df["sma_20"] = ta.trend.sma_indicator(close, window=20)
    df["sma_50"] = ta.trend.sma_indicator(close, window=50)
    if len(close) >= 200:
        df["sma_200"] = ta.trend.sma_indicator(close, window=200)
    else:
        # Short history: all-NaN SMA-200 would wipe rows on dropna(); proxy with SMA-50.
        df["sma_200"] = df["sma_50"]
    df["ema_9"] = ta.trend.ema_indicator(close, window=9)
    df["ema_21"] = ta.trend.ema_indicator(close, window=21)

    macd = ta.trend.MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_histogram"] = macd.macd_diff()

    # ── Momentum ──
    df["rsi_14"] = ta.momentum.rsi(close, window=14)
    df["stoch_k"] = ta.momentum.stoch(high, low, close, window=14)
    df["stoch_d"] = ta.momentum.stoch_signal(high, low, close, window=14)
    df["williams_r"] = ta.momentum.williams_r(high, low, close, lbp=14)
    df["roc"] = ta.momentum.roc(close, window=10)

    # ── Volatility ──
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    df["bb_pct"] = bb.bollinger_pband()

    df["atr_14"] = ta.volatility.average_true_range(high, low, close, window=14)

    # Historical volatility (annualized)
    df["hvol_10"] = df["returns"].rolling(10).std() * (252 ** 0.5) if "returns" in df.columns else 0
    df["hvol_20"] = df["returns"].rolling(20).std() * (252 ** 0.5) if "returns" in df.columns else 0

    # ── Volume ──
    if not volume.empty and volume.sum() > 0:
        df["obv"] = ta.volume.on_balance_volume(close, volume)
        df["volume_sma_20"] = volume.rolling(20).mean()
        df["volume_ratio"] = volume / df["volume_sma_20"]

    # ── Derived ──
    df["price_vs_sma20"] = (close - df["sma_20"]) / df["sma_20"]
    df["price_vs_sma50"] = (close - df["sma_50"]) / df["sma_50"]
    df["sma_20_50_cross"] = (df["sma_20"] > df["sma_50"]).astype(int)

    logger.info(f"Added {len([c for c in df.columns if c not in ['open','high','low','close','volume']])} technical features")
    return df