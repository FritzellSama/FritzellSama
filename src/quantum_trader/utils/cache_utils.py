"""Cache Utility Functions.

Production-ready caching utilities for Redis, in-memory caching,
and distributed cache management with TTL, invalidation, and monitoring.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Any, Callable, TypeVar, Union
from datetime import datetime, timezone, timedelta
import hashlib
import json
import pickle
from structlog import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class CacheKey:
    """Cache key generator with namespacing.

    Attributes:
        namespace: Key namespace
        separator: Separator for key parts
    """

    def __init__(self, namespace: str, separator: str = ":") -> None:
        """Initialize cache key generator.

        Args:
            namespace: Namespace for keys
            separator: Separator character
        """
        self.namespace = namespace
        self.separator = separator

    def generate(self, *parts: Any, **kwargs: Any) -> str:
        """Generate cache key from parts.

        Args:
            *parts: Key components
            **kwargs: Additional key-value pairs

        Returns:
            Generated cache key

        Raises:
            ValueError: If parts invalid
        """
        try:
            key_parts = [self.namespace]

            # Add positional parts
            for part in parts:
                if isinstance(part, (str, int, float)):
                    key_parts.append(str(part))
                elif isinstance(part, Decimal):
                    key_parts.append(str(part))
                elif isinstance(part, datetime):
                    key_parts.append(part.isoformat())
                else:
                    # Hash complex objects
                    key_parts.append(self._hash_object(part))

            # Add keyword parts
            if kwargs:
                sorted_kwargs = sorted(kwargs.items())
                for k, v in sorted_kwargs:
                    key_parts.append(f"{k}={v}")

            cache_key = self.separator.join(key_parts)

            logger.debug("cache_key_generated", key=cache_key, parts=len(key_parts))

            return cache_key

        except Exception as e:
            logger.error("cache_key_generation_failed", error=str(e))
            raise ValueError(f"Failed to generate cache key: {e}")

    def _hash_object(self, obj: Any) -> str:
        """Generate hash for complex object.

        Args:
            obj: Object to hash

        Returns:
            Hash string
        """
        try:
            obj_str = json.dumps(obj, sort_keys=True, default=str)
        except (TypeError, ValueError):
            obj_str = str(obj)

        return hashlib.md5(obj_str.encode()).hexdigest()[:12]


class InMemoryCache:
    """Thread-safe in-memory cache with TTL.

    Attributes:
        max_size: Maximum cache entries
        default_ttl: Default TTL in seconds
    """

    def __init__(self, max_size: int, default_ttl: float) -> None:
        """Initialize in-memory cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl

        self._cache: Dict[str, tuple[Any, float]] = {}
        self._lock = asyncio.Lock()

        logger.info(
            "inmemory_cache_initialized",
            max_size=max_size,
            default_ttl=default_ttl
        )

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            if key not in self._cache:
                logger.debug("cache_miss", key=key)
                return None

            value, expiry = self._cache[key]

            # Check if expired
            import time
            if time.time() > expiry:
                del self._cache[key]
                logger.debug("cache_expired", key=key)
                return None

            logger.debug("cache_hit", key=key)
            return value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (uses default if None)
        """
        import time

        async with self._lock:
            # Evict if cache full
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict_lru()

            ttl_seconds = ttl if ttl is not None else self.default_ttl
            expiry = time.time() + ttl_seconds

            self._cache[key] = (value, expiry)

            logger.debug("cache_set", key=key, ttl=ttl_seconds)

    async def delete(self, key: str) -> bool:
        """Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug("cache_deleted", key=key)
                return True
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info("cache_cleared", entries_removed=count)

    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        # Find entry with earliest expiry (simple LRU approximation)
        lru_key = min(self._cache.items(), key=lambda x: x[1][1])[0]
        del self._cache[lru_key]
        logger.debug("cache_evicted", key=lru_key)

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        async with self._lock:
            import time
            current_time = time.time()

            expired_count = sum(
                1 for _, expiry in self._cache.values()
                if current_time > expiry
            )

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "expired_entries": expired_count,
                "utilization_pct": (len(self._cache) / self.max_size * 100)
                                   if self.max_size > 0 else 0
            }


class RedisCache:
    """Async Redis cache wrapper.

    Requires redis-py with async support.

    Attributes:
        redis_client: Redis async client
        key_prefix: Prefix for all keys
        default_ttl: Default TTL in seconds
    """

    def __init__(
        self,
        redis_client: Any,
        key_prefix: str,
        default_ttl: float
    ) -> None:
        """Initialize Redis cache.

        Args:
            redis_client: Redis async client instance
            key_prefix: Prefix for keys
            default_ttl: Default TTL in seconds
        """
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl

        logger.info(
            "redis_cache_initialized",
            prefix=key_prefix,
            default_ttl=default_ttl
        )

    def _make_key(self, key: str) -> str:
        """Create prefixed key.

        Args:
            key: Original key

        Returns:
            Prefixed key
        """
        return f"{self.key_prefix}:{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        try:
            redis_key = self._make_key(key)
            value = await self.redis.get(redis_key)

            if value is None:
                logger.debug("redis_cache_miss", key=key)
                return None

            # Deserialize
            deserialized = pickle.loads(value)
            logger.debug("redis_cache_hit", key=key)
            return deserialized

        except Exception as e:
            logger.error("redis_get_failed", key=key, error=str(e))
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> bool:
        """Set value in Redis.

        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds

        Returns:
            True if successful
        """
        try:
            redis_key = self._make_key(key)
            ttl_seconds = ttl if ttl is not None else self.default_ttl

            # Serialize
            serialized = pickle.dumps(value)

            # Set with TTL
            await self.redis.setex(
                redis_key,
                int(ttl_seconds),
                serialized
            )

            logger.debug("redis_cache_set", key=key, ttl=ttl_seconds)
            return True

        except Exception as e:
            logger.error("redis_set_failed", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from Redis.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        try:
            redis_key = self._make_key(key)
            result = await self.redis.delete(redis_key)
            logger.debug("redis_cache_deleted", key=key, deleted=bool(result))
            return bool(result)

        except Exception as e:
            logger.error("redis_delete_failed", key=key, error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: Cache key

        Returns:
            True if exists
        """
        try:
            redis_key = self._make_key(key)
            result = await self.redis.exists(redis_key)
            return bool(result)

        except Exception as e:
            logger.error("redis_exists_failed", key=key, error=str(e))
            return False

    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter.

        Args:
            key: Cache key
            amount: Increment amount

        Returns:
            New value or None on error
        """
        try:
            redis_key = self._make_key(key)
            result = await self.redis.incrby(redis_key, amount)
            return int(result)

        except Exception as e:
            logger.error("redis_increment_failed", key=key, error=str(e))
            return None

    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple keys.

        Args:
            keys: List of cache keys

        Returns:
            Dictionary of key-value pairs
        """
        try:
            redis_keys = [self._make_key(k) for k in keys]
            values = await self.redis.mget(redis_keys)

            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    try:
                        result[key] = pickle.loads(value)
                    except Exception as e:
                        logger.error("deserialize_failed", key=key, error=str(e))

            logger.debug("redis_get_many", requested=len(keys), found=len(result))
            return result

        except Exception as e:
            logger.error("redis_get_many_failed", error=str(e))
            return {}

    async def set_many(
        self,
        items: Dict[str, Any],
        ttl: Optional[float] = None
    ) -> bool:
        """Set multiple keys.

        Args:
            items: Dictionary of key-value pairs
            ttl: TTL in seconds

        Returns:
            True if all successful
        """
        try:
            pipeline = self.redis.pipeline()
            ttl_seconds = ttl if ttl is not None else self.default_ttl

            for key, value in items.items():
                redis_key = self._make_key(key)
                serialized = pickle.dumps(value)
                pipeline.setex(redis_key, int(ttl_seconds), serialized)

            await pipeline.execute()

            logger.debug("redis_set_many", count=len(items), ttl=ttl_seconds)
            return True

        except Exception as e:
            logger.error("redis_set_many_failed", error=str(e))
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern.

        Args:
            pattern: Key pattern (e.g., "user:*")

        Returns:
            Number of keys deleted
        """
        try:
            full_pattern = self._make_key(pattern)
            cursor = 0
            deleted = 0

            while True:
                cursor, keys = await self.redis.scan(
                    cursor,
                    match=full_pattern,
                    count=100
                )

                if keys:
                    deleted += await self.redis.delete(*keys)

                if cursor == 0:
                    break

            logger.info("redis_pattern_cleared", pattern=pattern, deleted=deleted)
            return deleted

        except Exception as e:
            logger.error("redis_clear_pattern_failed", pattern=pattern, error=str(e))
            return 0


def cache_result(
    cache: Union[InMemoryCache, RedisCache],
    key_func: Callable[..., str],
    ttl: Optional[float] = None
) -> Callable:
    """Decorator to cache function results.

    Args:
        cache: Cache instance
        key_func: Function to generate cache key from args
        ttl: TTL in seconds

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key
            cache_key = key_func(*args, **kwargs)

            # Try to get from cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            await cache.set(cache_key, result, ttl)

            return result

        return wrapper

    return decorator


async def warm_cache(
    cache: Union[InMemoryCache, RedisCache],
    data_loader: Callable[[], Dict[str, Any]],
    ttl: Optional[float] = None
) -> int:
    """Warm cache with initial data.

    Args:
        cache: Cache instance
        data_loader: Function that returns dict of key-value pairs
        ttl: TTL in seconds

    Returns:
        Number of items cached
    """
    try:
        data = data_loader()

        if isinstance(cache, RedisCache):
            await cache.set_many(data, ttl)
        else:
            for key, value in data.items():
                await cache.set(key, value, ttl)

        logger.info("cache_warmed", items=len(data))
        return len(data)

    except Exception as e:
        logger.error("cache_warm_failed", error=str(e))
        return 0
