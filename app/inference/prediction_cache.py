"""Redis-based prediction caching. Re-exports from storage."""
from app.data.storage.redis_cache import cache_prediction, get_cached_prediction

__all__ = ["cache_prediction", "get_cached_prediction"]