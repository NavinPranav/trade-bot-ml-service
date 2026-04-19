import json
from typing import Optional

import redis
from loguru import logger

from app.config import settings

_redis: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


def _prediction_key(
    horizon: str, data_fingerprint: Optional[str], engine: str = "ML"
) -> str:
    eng = (engine or "ML").upper()
    if data_fingerprint:
        return f"prediction:{eng}:{horizon}:{data_fingerprint}"
    return f"prediction:{eng}:{horizon}"


def cache_prediction(
    horizon: str,
    data: dict,
    ttl: int = 300,
    data_fingerprint: Optional[str] = None,
    *,
    engine: str = "ML",
):
    key = _prediction_key(horizon, data_fingerprint, engine=engine)
    try:
        r = get_redis()
        r.setex(key, ttl, json.dumps(data, default=str))
        logger.debug(f"Cached prediction for {key} (TTL={ttl}s)")
    except redis.RedisError as e:
        logger.warning(f"Redis cache write skipped: {e}")


def get_cached_prediction(
    horizon: str, data_fingerprint: Optional[str] = None, *, engine: str = "ML"
) -> Optional[dict]:
    key = _prediction_key(horizon, data_fingerprint, engine=engine)
    try:
        r = get_redis()
        raw = r.get(key)
        if raw:
            logger.debug(f"Cache hit for prediction:{horizon}")
            return json.loads(raw)
    except redis.RedisError as e:
        logger.warning(f"Redis cache read skipped: {e}")
    return None