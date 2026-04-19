"""Macro/market-wide features: FII/DII, VIX, USD/INR."""
import pandas as pd
from loguru import logger


def add_macro_features(df: pd.DataFrame, vix_data: pd.DataFrame = None,
                       fii_dii: pd.DataFrame = None) -> pd.DataFrame:

    # VIX features
    if vix_data is not None and not vix_data.empty and "vix" in vix_data.columns:
        df = df.join(vix_data[["vix"]], how="left")
        df["vix"] = df["vix"].ffill()
        df["vix_change"] = df["vix"].pct_change()
        df["vix_sma_5"] = df["vix"].rolling(5).mean()
        df["vix_above_20"] = (df["vix"] > 20).astype(int)
        logger.info("Added VIX features")

    # FII/DII features
    if fii_dii is not None and not fii_dii.empty:
        if "date" in fii_dii.columns:
            fii_dii = fii_dii.set_index("date")
        df = df.join(fii_dii[["fii_net_buy", "dii_net_buy"]], how="left")
        df["fii_net_buy"] = df.get("fii_net_buy", pd.Series(dtype=float)).ffill().fillna(0)
        df["dii_net_buy"] = df.get("dii_net_buy", pd.Series(dtype=float)).ffill().fillna(0)
        df["fii_dii_net"] = df["fii_net_buy"] + df["dii_net_buy"]
        df["fii_5d_sum"] = df["fii_net_buy"].rolling(5).sum()
        logger.info("Added FII/DII features")

    return df