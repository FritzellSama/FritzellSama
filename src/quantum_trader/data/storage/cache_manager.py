"""
Cache manager for high-performance data caching.

This module provides a Redis-based caching layer for market data with
automatic expiration, compression, and serialization.
"""

import asyncio
import json
import zlib
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone, timedelta
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class CacheError(Exception):
    """Base exception for cache errors."""

    pass


class CacheManager:
    """
    High-performance cache manager using Redis.

    Provides caching functionality with automatic expiration, compression,
    and efficient serialization for trading data.

    Attributes:
        config: Cache configuration
        redis_url: Redis connection URL
        default_ttl: Default time-to-live in seconds

    Example:
        ```python
        config = {
            'redis_url': 'redis://localhost:6379',
            'default_ttl_seconds': 300,
            'enable_compression': True,
            'compression_threshold_bytes': 1024,
            'key_prefix': 'quantum_trader',
            'max_retries': 3,
            'retry_delay_ms': 100,
            'pool_size': 10,
            'pool_timeout_seconds': 5
        }

        cache = CacheManager(config)
        await cache.connect()

        # Store data
        await cache.set('market_data:BTC/USDT', data, ttl=60)

        # Retrieve data
        data = await cache.get('market_data:BTC/USDT')

        # Store DataFrame
        await cache.set_dataframe('ohlcv:BTC/USDT', df)

        # Retrieve DataFrame
        df = await cache.get_dataframe('ohlcv:BTC/USDT')
        ```
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize cache manager.

        Args:
            config: Cache configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        self._validate_config(config)

        self.config = config
        self.redis_url = config['redis_url']
        self.default_ttl = config.get('default_ttl_seconds', 300)
        self.enable_compression = config.get('enable_compression', True)
        self.compression_threshold = config.get('compression_threshold_bytes', 1024)
        self.key_prefix = config.get('key_prefix', 'quantum_trader')
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay_ms = config.get('retry_delay_ms', 100)
        self.pool_size = config.get('pool_size', 10)
        self.pool_timeout = config.get('pool_timeout_seconds', 5)

        self._redis_client: Optional[Any] = None
        self._connected: bool = False
        self._stats: Dict[str, int] = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0,
            'compressions': 0
        }

        logger.info(
            "Cache manager initialized",
            redis_url=self.redis_url,
            default_ttl=self.default_ttl,
            enable_compression=self.enable_compression
        )

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate cache configuration.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        if 'redis_url' not in config:
            raise ValueError("Missing required config field: redis_url")

        if not config['redis_url'] or not isinstance(config['redis_url'], str):
            raise ValueError("redis_url must be a non-empty string")

    async def connect(self) -> None:
        """
        Connect to Redis.

        Raises:
            CacheError: If connection fails
        """
        if self._connected:
            logger.warning("Cache manager already connected")
            return

        try:
            # Import redis here to avoid import errors if not installed
            import redis.asyncio as redis

            self._redis_client = await redis.from_url(
                self.redis_url,
                encoding='utf-8',
                decode_responses=False,
                max_connections=self.pool_size,
                socket_timeout=self.pool_timeout,
                socket_connect_timeout=self.pool_timeout
            )

            # Test connection
            await self._redis_client.ping()

            self._connected = True
            logger.info("Connected to Redis", url=self.redis_url)

        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            raise CacheError(f"Failed to connect to Redis: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from Redis gracefully."""
        if not self._connected or not self._redis_client:
            return

        try:
            await self._redis_client.close()
            await self._redis_client.connection_pool.disconnect()

            self._connected = False
            logger.info("Disconnected from Redis", stats=self._stats)

        except Exception as e:
            logger.error("Error disconnecting from Redis", error=str(e))

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found

        Raises:
            CacheError: If operation fails
        """
        if not self._connected:
            raise CacheError("Cache manager not connected")

        try:
            full_key = self._make_key(key)
            data = await self._redis_client.get(full_key)

            if data is None:
                self._stats['misses'] += 1
                return None

            value = self._deserialize(data)
            self._stats['hits'] += 1
            return value

        except Exception as e:
            self._stats['errors'] += 1
            logger.error("Error getting from cache", key=key, error=str(e))
            raise CacheError(f"Failed to get from cache: {e}") from e

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)

        Returns:
            True if successful

        Raises:
            CacheError: If operation fails
        """
        if not self._connected:
            raise CacheError("Cache manager not connected")

        try:
            full_key = self._make_key(key)
            data = self._serialize(value)
            ttl_seconds = ttl if ttl is not None else self.default_ttl

            await self._redis_client.setex(full_key, ttl_seconds, data)

            self._stats['sets'] += 1
            return True

        except Exception as e:
            self._stats['errors'] += 1
            logger.error("Error setting cache", key=key, error=str(e))
            raise CacheError(f"Failed to set cache: {e}") from e

    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if key existed and was deleted

        Raises:
            CacheError: If operation fails
        """
        if not self._connected:
            raise CacheError("Cache manager not connected")

        try:
            full_key = self._make_key(key)
            result = await self._redis_client.delete(full_key)

            self._stats['deletes'] += 1
            return result > 0

        except Exception as e:
            self._stats['errors'] += 1
            logger.error("Error deleting from cache", key=key, error=str(e))
            raise CacheError(f"Failed to delete from cache: {e}") from e

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists

        Raises:
            CacheError: If operation fails
        """
        if not self._connected:
            raise CacheError("Cache manager not connected")

        try:
            full_key = self._make_key(key)
            result = await self._redis_client.exists(full_key)
            return result > 0

        except Exception as e:
            self._stats['errors'] += 1
            logger.error("Error checking cache existence", key=key, error=str(e))
            raise CacheError(f"Failed to check cache existence: {e}") from e

    async def set_dataframe(
        self,
        key: str,
        df: pl.DataFrame,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store Polars DataFrame in cache.

        Args:
            key: Cache key
            df: DataFrame to cache
            ttl: Time-to-live in seconds

        Returns:
            True if successful

        Raises:
            CacheError: If operation fails
        """
        try:
            # Convert DataFrame to dict format
            data = df.to_dict(as_series=False)
            return await self.set(key, data, ttl)

        except Exception as e:
            logger.error("Error caching DataFrame", key=key, error=str(e))
            raise CacheError(f"Failed to cache DataFrame: {e}") from e

    async def get_dataframe(self, key: str) -> Optional[pl.DataFrame]:
        """
        Retrieve Polars DataFrame from cache.

        Args:
            key: Cache key

        Returns:
            DataFrame or None if not found

        Raises:
            CacheError: If operation fails
        """
        try:
            data = await self.get(key)

            if data is None:
                return None

            df = pl.DataFrame(data)
            return df

        except Exception as e:
            logger.error("Error retrieving DataFrame", key=key, error=str(e))
            raise CacheError(f"Failed to retrieve DataFrame: {e}") from e

    async def clear_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.

        Args:
            pattern: Key pattern (e.g., 'market_data:*')

        Returns:
            Number of keys deleted

        Raises:
            CacheError: If operation fails
        """
        if not self._connected:
            raise CacheError("Cache manager not connected")

        try:
            full_pattern = self._make_key(pattern)
            cursor = 0
            deleted = 0

            while True:
                cursor, keys = await self._redis_client.scan(
                    cursor=cursor,
                    match=full_pattern,
                    count=100
                )

                if keys:
                    deleted += await self._redis_client.delete(*keys)

                if cursor == 0:
                    break

            logger.info("Cleared cache pattern", pattern=pattern, deleted=deleted)
            return deleted

        except Exception as e:
            self._stats['errors'] += 1
            logger.error("Error clearing cache pattern", pattern=pattern, error=str(e))
            raise CacheError(f"Failed to clear cache pattern: {e}") from e

    def _make_key(self, key: str) -> str:
        """
        Create full cache key with prefix.

        Args:
            key: Base key

        Returns:
            Full key with prefix
        """
        return f"{self.key_prefix}:{key}"

    def _serialize(self, value: Any) -> bytes:
        """
        Serialize value for storage.

        Args:
            value: Value to serialize

        Returns:
            Serialized bytes
        """
        try:
            # Convert Decimal to string for JSON serialization
            json_str = json.dumps(value, default=self._json_serializer)
            data = json_str.encode('utf-8')

            # Compress if enabled and data is large enough
            if self.enable_compression and len(data) >= self.compression_threshold:
                data = zlib.compress(data, level=6)
                self._stats['compressions'] += 1
                # Add compression marker
                data = b'\x01' + data
            else:
                # Add uncompressed marker
                data = b'\x00' + data

            return data

        except Exception as e:
            logger.error("Serialization error", error=str(e))
            raise

    def _deserialize(self, data: bytes) -> Any:
        """
        Deserialize value from storage.

        Args:
            data: Serialized bytes

        Returns:
            Deserialized value
        """
        try:
            # Check compression marker
            if data[0:1] == b'\x01':
                # Decompress
                data = zlib.decompress(data[1:])
            else:
                # Remove uncompressed marker
                data = data[1:]

            json_str = data.decode('utf-8')
            value = json.loads(json_str, parse_float=Decimal)
            return value

        except Exception as e:
            logger.error("Deserialization error", error=str(e))
            raise

    def _json_serializer(self, obj: Any) -> Any:
        """
        Custom JSON serializer for special types.

        Args:
            obj: Object to serialize

        Returns:
            Serializable representation
        """
        if isinstance(obj, Decimal):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, bytes):
            return obj.decode('utf-8')
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary of statistics
        """
        stats = self._stats.copy()

        if stats['hits'] + stats['misses'] > 0:
            stats['hit_rate'] = stats['hits'] / (stats['hits'] + stats['misses'])
        else:
            stats['hit_rate'] = 0

        return stats

    def is_connected(self) -> bool:
        """
        Check if connected to Redis.

        Returns:
            True if connected
        """
        return self._connected

    async def health_check(self) -> bool:
        """
        Perform health check.

        Returns:
            True if cache is healthy

        Raises:
            CacheError: If health check fails
        """
        if not self._connected:
            return False

        try:
            await self._redis_client.ping()
            return True

        except Exception as e:
            logger.error("Health check failed", error=str(e))
            raise CacheError(f"Health check failed: {e}") from e
