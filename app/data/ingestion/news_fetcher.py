"""Fetch financial news headlines for sentiment analysis."""
from typing import List, Dict
from loguru import logger


def fetch_news_headlines(query: str = "Sensex BSE market India") -> List[Dict]:
    """
    Fetch recent financial news headlines.
    In production, use MoneyControl RSS, Google News API, or NewsAPI.
    """
    logger.warning("News fetcher: placeholder. Integrate NewsAPI/RSS for production.")
    return []