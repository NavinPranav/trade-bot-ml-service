"""Orchestrates all feature builders into a single feature matrix."""
import pandas as pd
from loguru import logger

from app.features.technical import add_technical_features
from app.features.macro_features import add_macro_features
from app.data.preprocessor import clean_ohlcv


class FeaturePipeline:

    def build(self, ohlcv: pd.DataFrame, vix: pd.DataFrame = None,
              fii_dii: pd.DataFrame = None, sentiment_score: float = 0.0) -> pd.DataFrame:

        # Step 1: Clean raw data
        df = clean_ohlcv(ohlcv)
        if df.empty:
            logger.error("Feature pipeline: no data after cleaning")
            return df

        # Step 2: Technical indicators
        df = add_technical_features(df)

        # Step 3: Macro features
        df = add_macro_features(df, vix_data=vix, fii_dii=fii_dii)

        # Step 4: Sentiment (scalar added as column)
        df["sentiment_score"] = sentiment_score

        # Step 5: Calendar features
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        df["is_expiry_week"] = 0  # TODO: compute from expiry calendar

        # Step 6: Target variable (next-day direction)
        df["target_direction"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df["target_return"] = df["close"].pct_change().shift(-1)

        # Drop NaN rows from indicator warmup
        initial_len = len(df)
        df = df.dropna()
        logger.info(f"Feature pipeline: {len(df)} rows ({initial_len - len(df)} dropped from warmup)")

        return df

    def get_feature_columns(self) -> list:
        return [
            # Technical
            "sma_20", "sma_50", "ema_9", "ema_21",
            "macd", "macd_signal", "macd_histogram",
            "rsi_14", "stoch_k", "stoch_d", "williams_r", "roc",
            "bb_width", "bb_pct", "atr_14", "hvol_10", "hvol_20",
            "price_vs_sma20", "price_vs_sma50", "sma_20_50_cross",
            # Macro
            "vix", "vix_change", "vix_above_20",
            "fii_net_buy", "dii_net_buy", "fii_dii_net", "fii_5d_sum",
            # Sentiment
            "sentiment_score",
            # Calendar
            "day_of_week", "month",
        ]