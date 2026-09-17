"""
Redis store with in-memory dict fallback.
"""
from functools import lru_cache
from typing import Optional

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class InMemoryStore:
    """Fallback store when Redis is unavailable."""

    backend = "in_memory"

    def __init__(self):
        self._data: dict = {}

    def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    def set(self, key: str, value: str, ex: int = None):
        self._data[key] = value

    def delete(self, key: str):
        self._data.pop(key, None)

    def exists(self, key: str) -> bool:
        return key in self._data

    def expire(self, key: str, ttl: int) -> bool:
        # The in-memory fallback has no TTL machinery; nothing to do.
        return key in self._data

    # Set operations (used for the per-token session index)
    def sadd(self, key: str, value: str):
        self._data.setdefault(key, set()).add(value)

    def srem(self, key: str, value: str):
        bucket = self._data.get(key)
        if isinstance(bucket, set):
            bucket.discard(value)

    def smembers(self, key: str) -> list:
        return list(self._data.get(key, set()))


class RedisStore:
    def __init__(self):
        self._client = None
        self._fallback = InMemoryStore()
        self._init()

    @property
    def backend(self) -> str:
        """Which store actually serves reads and writes.

        Reported rather than inferred: the in-memory fallback answers a
        set/get roundtrip exactly like a real server does, so a health check
        cannot tell them apart by roundtripping alone.
        """
        return "redis" if self._client is not None else "in_memory_fallback"

    def _init(self):
        try:
            import redis
            client = redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            client.ping()
            self._client = client
            logger.info("redis_connected", url=settings.REDIS_URL)
        except Exception as e:
            logger.warning("redis_unavailable_using_memory", error=str(e))
            self._client = None

    def get(self, key: str) -> Optional[str]:
        if self._client:
            try:
                return self._client.get(key)
            except Exception as e:
                logger.warning("redis_get_failed", key=key, error=str(e))
        return self._fallback.get(key)

    def set(self, key: str, value: str, ex: int = None):
        ttl = ex or settings.REDIS_TTL
        if self._client:
            try:
                self._client.set(key, value, ex=ttl)
                return
            except Exception as e:
                logger.warning("redis_set_failed", key=key, error=str(e))
        self._fallback.set(key, value)

    def delete(self, key: str):
        if self._client:
            try:
                self._client.delete(key)
                return
            except Exception:
                pass
        self._fallback.delete(key)

    def exists(self, key: str) -> bool:
        if self._client:
            try:
                return bool(self._client.exists(key))
            except Exception:
                pass
        return self._fallback.exists(key)

    def expire(self, key: str, ttl: int) -> bool:
        if self._client:
            try:
                return bool(self._client.expire(key, ttl))
            except Exception as e:
                logger.warning("redis_expire_failed", key=key, error=str(e))
        return self._fallback.expire(key, ttl)

    # Set operations (used for the per-token session index)
    def sadd(self, key: str, value: str):
        if self._client:
            try:
                self._client.sadd(key, value)
                # Keep the index on the same lifespan as the data it points
                # to: an index that outlives its sessions makes the sessions
                # list show empty husks ("新对话 / 1970/1/1").
                self._client.expire(key, settings.REDIS_TTL)
                return
            except Exception as e:
                logger.warning("redis_sadd_failed", key=key, error=str(e))
        self._fallback.sadd(key, value)

    def srem(self, key: str, value: str):
        if self._client:
            try:
                self._client.srem(key, value)
                return
            except Exception:
                pass
        self._fallback.srem(key, value)

    def smembers(self, key: str) -> list:
        if self._client:
            try:
                return list(self._client.smembers(key) or [])
            except Exception as e:
                logger.warning("redis_smembers_failed", key=key, error=str(e))
        return self._fallback.smembers(key)


@lru_cache(maxsize=1)
def get_redis_store() -> RedisStore:
    return RedisStore()
