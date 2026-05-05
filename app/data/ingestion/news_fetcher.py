"""Financial news fetcher with VADER sentiment scoring.

Fetches Bank Nifty / Indian market headlines from NewsAPI.org, scores each
headline+description with VADER, and returns an aggregated sentiment dict.
Results are cached in-memory for `news_fetch_interval_sec` seconds so that
every live-tick prediction reuses the same batch without burning API quota.

NewsAPI free tier: 100 requests/day.  At a 15-min TTL during 6.25-hr trading
session that is ~25 requests/day — well within the limit.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import httpx
from loguru import logger

from app.config import settings

IST = ZoneInfo("Asia/Kolkata")

_NEWS_API_URL = "https://newsapi.org/v2/everything"

_QUERY = (
    "bank nifty OR nifty 50 OR sensex OR RBI OR BSE OR NSE india OR "
    "FII india OR DII india OR SEBI OR indian stock market"
)

_FINANCIAL_KEYWORDS = {
    "nifty", "sensex", "bse", "nse", "rbi", "fii", "dii", "sebi",
    "stock", "market", "equity", "index", "rally", "crash", "bulls",
    "bears", "trade", "rupee", "inflation", "rate", "gdp", "budget",
    "banknifty", "bank nifty", "midcap", "smallcap",
}


def _is_financial(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _FINANCIAL_KEYWORDS)


def _vader_score(text: str) -> float:
    """Return VADER compound score (-1.0 to +1.0) for a text string."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # noqa: PLC0415
        analyzer = getattr(_vader_score, "_analyzer", None)
        if analyzer is None:
            analyzer = SentimentIntensityAnalyzer()
            _vader_score._analyzer = analyzer  # type: ignore[attr-defined]
        return float(analyzer.polarity_scores(text)["compound"])
    except ImportError:
        logger.warning("vaderSentiment not installed — sentiment scoring unavailable")
        return 0.0


def _label(score: float) -> str:
    if score >= 0.05:
        return "BULLISH"
    if score <= -0.05:
        return "BEARISH"
    return "NEUTRAL"


def _now_ist() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")


def _error_result(msg: str) -> Dict[str, Any]:
    return {
        "overall": "UNAVAILABLE", "score": 0.0, "article_count": 0,
        "bullish_count": 0, "bearish_count": 0, "neutral_count": 0,
        "fetched_at_ist": _now_ist(), "top_headlines": [], "error": msg,
    }


def fetch_news_sentiment() -> Dict[str, Any]:
    """Fetch latest financial headlines from NewsAPI and return a sentiment dict.

    Returns a dict with keys:
        overall         — BULLISH | BEARISH | NEUTRAL | UNAVAILABLE
        score           — float −1.0 to +1.0 (average compound)
        article_count   — int
        bullish_count   — int
        bearish_count   — int
        neutral_count   — int
        fetched_at_ist  — IST timestamp string
        top_headlines   — list of { title, source, score, label }
        error           — str (only present on failure)
    """
    key = (settings.news_api_key or "").strip()
    if not key:
        return _error_result("NEWS_API_KEY not configured")

    params = {
        "apiKey": key,
        "q": _QUERY,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 30,
    }

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(_NEWS_API_URL, params=params)

        if resp.status_code == 401:
            logger.warning("NewsAPI 401 — invalid API key")
            return _error_result("Invalid NewsAPI key (HTTP 401)")
        if resp.status_code == 429:
            logger.warning("NewsAPI 429 — daily quota exhausted")
            return _error_result("NewsAPI rate limit hit (HTTP 429)")
        if resp.status_code >= 400:
            logger.warning("NewsAPI HTTP {}", resp.status_code)
            return _error_result(f"NewsAPI HTTP {resp.status_code}")

        data = resp.json()
        if data.get("status") != "ok":
            msg = data.get("message", "unknown error")
            logger.warning("NewsAPI error: {}", msg)
            return _error_result(f"NewsAPI error: {msg}")

        articles: List[Dict[str, Any]] = data.get("articles") or []

    except Exception as e:
        logger.warning("NewsAPI fetch failed: {}", e)
        return _error_result(str(e))

    scored: List[Dict[str, Any]] = []
    for art in articles:
        title = (art.get("title") or "").strip()
        desc = (art.get("description") or "").strip()
        source = ((art.get("source") or {}).get("name") or "").strip()

        if not title or title.lower() == "[removed]":
            continue
        combined = f"{title}. {desc}" if desc else title
        if not _is_financial(combined):
            continue

        compound = _vader_score(combined)
        scored.append({
            "title": title[:200],
            "source": source,
            "score": round(compound, 4),
            "label": _label(compound),
        })

    if not scored:
        return {
            "overall": "NEUTRAL", "score": 0.0, "article_count": 0,
            "bullish_count": 0, "bearish_count": 0, "neutral_count": 0,
            "fetched_at_ist": _now_ist(), "top_headlines": [],
        }

    avg_score = sum(a["score"] for a in scored) / len(scored)
    bullish = sum(1 for a in scored if a["label"] == "BULLISH")
    bearish = sum(1 for a in scored if a["label"] == "BEARISH")
    neutral = len(scored) - bullish - bearish

    top = sorted(scored, key=lambda a: abs(a["score"]), reverse=True)[:5]

    logger.info(
        "NewsAPI sentiment: {} articles → overall={} score={:.3f} bull={} bear={} neutral={}",
        len(scored), _label(avg_score), avg_score, bullish, bearish, neutral,
    )

    return {
        "overall": _label(avg_score),
        "score": round(avg_score, 4),
        "article_count": len(scored),
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": neutral,
        "fetched_at_ist": _now_ist(),
        "top_headlines": top,
    }


# ── In-process TTL cache ──────────────────────────────────────────────────────

class _SentimentCache:
    """Thread-safe TTL cache so multiple live-tick calls share one fetch."""
    _lock = threading.Lock()
    _data: Optional[Dict[str, Any]] = None
    _fetched_at: float = 0.0

    @classmethod
    def get(cls) -> Dict[str, Any]:
        ttl = float(settings.news_fetch_interval_sec)
        now = time.monotonic()
        with cls._lock:
            if cls._data is not None and (now - cls._fetched_at) < ttl:
                return cls._data

        fresh = fetch_news_sentiment()
        with cls._lock:
            cls._data = fresh
            cls._fetched_at = time.monotonic()
        return fresh

    @classmethod
    def invalidate(cls) -> None:
        with cls._lock:
            cls._data = None
            cls._fetched_at = 0.0

    @classmethod
    def last(cls) -> Optional[Dict[str, Any]]:
        with cls._lock:
            return cls._data


def get_news_sentiment() -> Dict[str, Any]:
    """Public entry point: returns cached sentiment, refreshing if TTL has expired."""
    return _SentimentCache.get()
