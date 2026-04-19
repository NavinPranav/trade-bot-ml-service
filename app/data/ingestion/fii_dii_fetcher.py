"""Fetch FII/DII activity data from NSE."""
import pandas as pd
from loguru import logger


def fetch_fii_dii_activity() -> pd.DataFrame:
    """
    Fetch FII/DII buy/sell data.
    In production, scrape from NSE or use a data provider API.
    """
    logger.warning("FII/DII fetcher: using placeholder. Integrate NSE scraper for production.")
    return pd.DataFrame(columns=["date", "fii_net_buy", "dii_net_buy"])