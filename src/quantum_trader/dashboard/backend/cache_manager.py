"""Redis-based cache manager with async operations and TTL management.

Production-ready cache manager for dashboard data with:
- Async Redis operations
- Thread-safe access
- Automatic expiration (TTL)
- Key namespacing
- Graceful connection handling
- Health monitoring
"""

import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

import redis.asyncio as aioredis
from structlog import get_logger

logger = get_logger(__name__)


class CacheError(Exception):
    """Cache operation error."""
    pass


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder for Decimal types."""

    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class CacheManager:
    """Redis cache manager for dashboard data.

    Features:
    - Async Redis operations
    - Thread-safe with connection pooling
    - Automatic key expiration (TTL)
    - Namespace support for key isolation
    - Graceful connection handling
    - Health check monitoring

    Attributes:
        config: Cache configuration
        redis: Redis client instance
        namespace: Key namespace prefix
        default_ttl: Default TTL in seconds
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize cache manager.

        Args:
            config: Configuration dict with:
                - redis_host: Redis host
                - redis_port: Redis port
                - redis_db: Redis database number
                - redis_password: Redis password (optional)
                - namespace: Key namespace prefix
                - default_ttl: Default TTL in seconds
                - max_connections: Max connection pool size

        Raises:
            ValueError: If required config missing
        """
        self.config = config
        self._validate_config()

        self.namespace = config.get('namespace', 'quantum_trader')
        self.default_ttl = config.get('default_ttl', 300)  # 5 minutes default

        # Redis client initialized on connect
        self.redis: Optional[aioredis.Redis] = None
        self._lock = asyncio.Lock()

        # Metrics
        self.metrics = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }

        logger.info(
            "cache_manager_initialized",
            namespace=self.namespace,
            default_ttl=self.default_ttl
        )

    def _validate_config(self) -> None:
        """Validate required configuration.

        Raises:
            ValueError: If required config missing
        """
        required = ['redis_host', 'redis_port', 'redis_db']

        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required config: {key}")

    async def connect(self) -> None:
        """Initialize Redis connection with pooling."""
        if self.redis is not None:
            return

        try:
            pool = aioredis.ConnectionPool(
                host=self.config['redis_host'],
                port=self.config['redis_port'],
                db=self.config['redis_db'],
                password=self.config.get('redis_password'),
                max_connections=self.config.get('max_connections', 50),
                decode_responses=True,
                socket_timeout=self.config.get('socket_timeout', 5),
                socket_connect_timeout=self.config.get('connect_timeout', 5)
            )

            self.redis = aioredis.Redis(connection_pool=pool)

            # Test connection
            await self.redis.ping()

            logger.info(
                "cache_connected",
                host=self.config['redis_host'],
                port=self.config['redis_port'],
                db=self.config['redis_db']
            )

        except Exception as e:
            logger.error("cache_connection_failed", error=str(e))
            raise CacheError(f"Failed to connect to Redis: {e}")

    async def disconnect(self) -> None:
        """Close Redis connection gracefully."""
        if self.redis is not None:
            await self.redis.close()
            self.redis = None
            logger.info("cache_disconnected")

    def _make_key(self, key: str) -> str:
        """Create namespaced key.

        Args:
            key: Raw key

        Returns:
            Namespaced key
        """
        return f"{self.namespace}:{key}"

    def _serialize(self, value: Any) -> str:
        """Serialize value to JSON string.

        Args:
            value: Value to serialize

        Returns:
            JSON string
        """
        return json.dumps(value, cls=DecimalEncoder)

    def _deserialize(self, value: str) -> Any:
        """Deserialize JSON string to value.

        Args:
            value: JSON string

        Returns:
            Deserialized value
        """
        return json.loads(value)

    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default

        Raises:
            CacheError: If operation fails
        """
        if self.redis is None:
            await self.connect()

        try:
            namespaced_key = self._make_key(key)
            value = await self.redis.get(namespaced_key)

            if value is None:
                self.metrics['misses'] += 1
                logger.debug("cache_miss", key=key)
                return default

            self.metrics['hits'] += 1
            logger.debug("cache_hit", key=key)
            return self._deserialize(value)

        except Exception as e:
            self.metrics['errors'] += 1
            logger.error("cache_get_error", key=key, error=str(e))
            raise CacheError(f"Failed to get key {key}: {e}")

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = default_ttl)

        Returns:
            True if successful

        Raises:
            CacheError: If operation fails
        """
        if self.redis is None:
            await self.connect()

        try:
            namespaced_key = self._make_key(key)
            serialized_value = self._serialize(value)
            ttl = ttl if ttl is not None else self.default_ttl

            await self.redis.setex(
                namespaced_key,
                ttl,
                serialized_value
            )

            self.metrics['sets'] += 1
            logger.debug("cache_set", key=key, ttl=ttl)
            return True

        except Exception as e:
            self.metrics['errors'] += 1
            logger.error("cache_set_error", key=key, error=str(e))
            raise CacheError(f"Failed to set key {key}: {e}")

    async def delete(self, key: str) -> bool:
        """Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found

        Raises:
            CacheError: If operation fails
        """
        if self.redis is None:
            await self.connect()

        try:
            namespaced_key = self._make_key(key)
            result = await self.redis.delete(namespaced_key)

            self.metrics['deletes'] += 1
            logger.debug("cache_delete", key=key, deleted=bool(result))
            return bool(result)

        except Exception as e:
            self.metrics['errors'] += 1
            logger.error("cache_delete_error", key=key, error=str(e))
            raise CacheError(f"Failed to delete key {key}: {e}")

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if exists

        Raises:
            CacheError: If operation fails
        """
        if self.redis is None:
            await self.connect()

        try:
            namespaced_key = self._make_key(key)
            result = await self.redis.exists(namespaced_key)
            return bool(result)

        except Exception as e:
            self.metrics['errors'] += 1
            logger.error("cache_exists_error", key=key, error=str(e))
            raise CacheError(f"Failed to check key {key}: {e}")

    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache.

        Args:
            keys: List of cache keys

        Returns:
            Dict of key-value pairs (missing keys excluded)

        Raises:
            CacheError: If operation fails
        """
        if self.redis is None:
            await self.connect()

        try:
            namespaced_keys = [self._make_key(k) for k in keys]
            values = await self.redis.mget(namespaced_keys)

            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = self._deserialize(value)
                    self.metrics['hits'] += 1
                else:
                    self.metrics['misses'] += 1

            logger.debug("cache_get_many", keys=len(keys), found=len(result))
            return result

        except Exception as e:
            self.metrics['errors'] += 1
            logger.error("cache_get_many_error", error=str(e))
            raise CacheError(f"Failed to get multiple keys: {e}")

    async def set_many(
        self,
        data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Set multiple values in cache.

        Args:
            data: Dict of key-value pairs
            ttl: Time-to-live in seconds

        Returns:
            True if successful

        Raises:
            CacheError: If operation fails
        """
        if self.redis is None:
            await self.connect()

        try:
            ttl = ttl if ttl is not None else self.default_ttl

            async with self.redis.pipeline() as pipe:
                for key, value in data.items():
                    namespaced_key = self._make_key(key)
                    serialized_value = self._serialize(value)
                    pipe.setex(namespaced_key, ttl, serialized_value)

                await pipe.execute()

            self.metrics['sets'] += len(data)
            logger.debug("cache_set_many", keys=len(data), ttl=ttl)
            return True

        except Exception as e:
            self.metrics['errors'] += 1
            logger.error("cache_set_many_error", error=str(e))
            raise CacheError(f"Failed to set multiple keys: {e}")

    async def clear_namespace(self) -> int:
        """Clear all keys in namespace.

        Returns:
            Number of keys deleted

        Raises:
            CacheError: If operation fails
        """
        if self.redis is None:
            await self.connect()

        try:
            pattern = f"{self.namespace}:*"
            keys = []

            async for key in self.redis.scan_iter(match=pattern, count=100):
                keys.append(key)

            if keys:
                deleted = await self.redis.delete(*keys)
            else:
                deleted = 0

            logger.info("cache_namespace_cleared", namespace=self.namespace, deleted=deleted)
            return deleted

        except Exception as e:
            self.metrics['errors'] += 1
            logger.error("cache_clear_error", error=str(e))
            raise CacheError(f"Failed to clear namespace: {e}")

    async def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for key.

        Args:
            key: Cache key

        Returns:
            TTL in seconds, None if no expiration or not found

        Raises:
            CacheError: If operation fails
        """
        if self.redis is None:
            await self.connect()

        try:
            namespaced_key = self._make_key(key)
            ttl = await self.redis.ttl(namespaced_key)

            if ttl == -2:  # Key doesn't exist
                return None
            if ttl == -1:  # Key exists but no expiration
                return None

            return ttl

        except Exception as e:
            self.metrics['errors'] += 1
            logger.error("cache_get_ttl_error", key=key, error=str(e))
            raise CacheError(f"Failed to get TTL for key {key}: {e}")

    async def health_check(self) -> bool:
        """Check cache connectivity and health.

        Returns:
            True if healthy, False otherwise
        """
        try:
            if self.redis is None:
                await self.connect()

            await self.redis.ping()
            return True

        except Exception as e:
            logger.error("cache_health_check_failed", error=str(e))
            return False

    def get_metrics(self) -> Dict[str, int]:
        """Get cache metrics.

        Returns:
            Metrics dictionary
        """
        metrics = self.metrics.copy()

        # Calculate hit rate
        total = metrics['hits'] + metrics['misses']
        if total > 0:
            metrics['hit_rate'] = round(metrics['hits'] / total * 100, 2)
        else:
            metrics['hit_rate'] = 0.0

        return metrics

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
