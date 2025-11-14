"""
Data reader for efficient retrieval of market data from TimescaleDB.

This module provides optimized data reading with query caching, connection
pooling, and parallel query execution.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class ReaderError(Exception):
    """Base exception for reader errors."""

    pass


class DataReader:
    """
    High-performance data reader for TimescaleDB.

    Provides efficient querying of historical market data with connection
    pooling, query optimization, and result caching.

    Attributes:
        config: Reader configuration
        connection_string: Database connection string

    Example:
        ```python
        config = {
            'connection_string': 'postgresql://user:pass@localhost:5432/quantum_trader',
            'pool_min_size': 5,
            'pool_max_size': 20,
            'pool_timeout_seconds': 30,
            'query_timeout_seconds': 60,
            'enable_query_cache': True,
            'cache_ttl_seconds': 300,
            'max_rows_per_query': 100000,
            'enable_parallel_queries': True,
            'chunk_size': 10000
        }

        reader = DataReader(config)
        await reader.connect()

        # Read OHLCV data
        df = await reader.get_ohlcv(
            symbol='BTC/USDT',
            timeframe='1m',
            start_time=datetime.now(timezone.utc) - timedelta(hours=1),
            end_time=datetime.now(timezone.utc)
        )

        # Read trades
        trades_df = await reader.get_trades(
            symbol='ETH/USDT',
            start_time=datetime.now(timezone.utc) - timedelta(minutes=5),
            limit=1000
        )
        ```
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize data reader.

        Args:
            config: Reader configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        self._validate_config(config)

        self.config = config
        self.connection_string = config['connection_string']
        self.pool_min_size = config.get('pool_min_size', 5)
        self.pool_max_size = config.get('pool_max_size', 20)
        self.pool_timeout = config.get('pool_timeout_seconds', 30)
        self.query_timeout = config.get('query_timeout_seconds', 60)
        self.enable_cache = config.get('enable_query_cache', True)
        self.cache_ttl = config.get('cache_ttl_seconds', 300)
        self.max_rows = config.get('max_rows_per_query', 100000)
        self.enable_parallel = config.get('enable_parallel_queries', True)
        self.chunk_size = config.get('chunk_size', 10000)

        self._pool: Optional[Any] = None
        self._connected: bool = False
        self._query_cache: Dict[str, Tuple[pl.DataFrame, datetime]] = {}
        self._stats: Dict[str, int] = {
            'queries_executed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'rows_fetched': 0,
            'query_errors': 0
        }

        logger.info(
            "Data reader initialized",
            pool_max_size=self.pool_max_size,
            enable_cache=self.enable_cache,
            enable_parallel=self.enable_parallel
        )

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate reader configuration.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        if 'connection_string' not in config:
            raise ValueError("Missing required config field: connection_string")

        if not config['connection_string'] or not isinstance(config['connection_string'], str):
            raise ValueError("connection_string must be a non-empty string")

    async def connect(self) -> None:
        """
        Connect to database.

        Raises:
            ReaderError: If connection fails
        """
        if self._connected:
            logger.warning("Data reader already connected")
            return

        try:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=self.pool_min_size,
                max_size=self.pool_max_size,
                timeout=self.pool_timeout,
                command_timeout=self.query_timeout
            )

            # Test connection
            async with self._pool.acquire() as conn:
                await conn.fetchval('SELECT 1')

            self._connected = True
            logger.info("Connected to database")

        except Exception as e:
            logger.error("Failed to connect to database", error=str(e))
            raise ReaderError(f"Failed to connect to database: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from database gracefully."""
        if not self._connected or not self._pool:
            return

        try:
            await self._pool.close()
            self._connected = False
            self._query_cache.clear()

            logger.info("Disconnected from database", stats=self._stats)

        except Exception as e:
            logger.error("Error disconnecting from database", error=str(e))

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pl.DataFrame:
        """
        Get OHLCV candle data.

        Args:
            symbol: Trading symbol
            timeframe: Candle timeframe
            start_time: Start timestamp
            end_time: End timestamp (defaults to now)
            limit: Maximum number of rows

        Returns:
            DataFrame with OHLCV data

        Raises:
            ReaderError: If query fails
        """
        try:
            if end_time is None:
                end_time = datetime.now(timezone.utc)

            query = """
                SELECT
                    timestamp,
                    symbol,
                    timeframe,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    trades
                FROM ohlcv
                WHERE symbol = $1
                    AND timeframe = $2
                    AND timestamp >= $3
                    AND timestamp <= $4
                ORDER BY timestamp DESC
            """

            params = [symbol, timeframe, start_time, end_time]

            if limit is not None:
                query += " LIMIT $5"
                params.append(limit)

            df = await self._execute_query(query, params)
            return df

        except Exception as e:
            logger.error(
                "Error getting OHLCV data",
                symbol=symbol,
                timeframe=timeframe,
                error=str(e)
            )
            raise ReaderError(f"Failed to get OHLCV data: {e}") from e

    async def get_trades(
        self,
        symbol: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pl.DataFrame:
        """
        Get trade data.

        Args:
            symbol: Trading symbol
            start_time: Start timestamp
            end_time: End timestamp (defaults to now)
            limit: Maximum number of rows

        Returns:
            DataFrame with trade data

        Raises:
            ReaderError: If query fails
        """
        try:
            if end_time is None:
                end_time = datetime.now(timezone.utc)

            query = """
                SELECT
                    timestamp,
                    symbol,
                    exchange,
                    price,
                    volume,
                    side,
                    trade_id
                FROM trades
                WHERE symbol = $1
                    AND timestamp >= $2
                    AND timestamp <= $3
                ORDER BY timestamp DESC
            """

            params = [symbol, start_time, end_time]

            if limit is not None:
                query += " LIMIT $4"
                params.append(limit)

            df = await self._execute_query(query, params)
            return df

        except Exception as e:
            logger.error(
                "Error getting trade data",
                symbol=symbol,
                error=str(e)
            )
            raise ReaderError(f"Failed to get trade data: {e}") from e

    async def get_orderbook_snapshot(
        self,
        symbol: str,
        timestamp: Optional[datetime] = None,
        depth: int = 20
    ) -> Optional[Dict[str, Any]]:
        """
        Get orderbook snapshot.

        Args:
            symbol: Trading symbol
            timestamp: Snapshot timestamp (defaults to latest)
            depth: Number of price levels per side

        Returns:
            Orderbook dictionary or None

        Raises:
            ReaderError: If query fails
        """
        try:
            if timestamp is None:
                timestamp = datetime.now(timezone.utc)

            query = """
                SELECT
                    timestamp,
                    symbol,
                    exchange,
                    bids,
                    asks
                FROM orderbook_snapshots
                WHERE symbol = $1
                    AND timestamp <= $2
                ORDER BY timestamp DESC
                LIMIT 1
            """

            params = [symbol, timestamp]

            df = await self._execute_query(query, params)

            if len(df) == 0:
                return None

            row = df.row(0, named=True)

            orderbook = {
                'timestamp': row['timestamp'],
                'symbol': row['symbol'],
                'exchange': row['exchange'],
                'bids': row['bids'][:depth] if row['bids'] else [],
                'asks': row['asks'][:depth] if row['asks'] else []
            }

            return orderbook

        except Exception as e:
            logger.error(
                "Error getting orderbook snapshot",
                symbol=symbol,
                error=str(e)
            )
            raise ReaderError(f"Failed to get orderbook snapshot: {e}") from e

    async def get_latest_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get latest price for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Latest price or None

        Raises:
            ReaderError: If query fails
        """
        try:
            query = """
                SELECT close
                FROM ohlcv
                WHERE symbol = $1
                ORDER BY timestamp DESC
                LIMIT 1
            """

            df = await self._execute_query(query, [symbol])

            if len(df) == 0:
                return None

            price_str = df['close'][0]
            return Decimal(price_str)

        except Exception as e:
            logger.error("Error getting latest price", symbol=symbol, error=str(e))
            raise ReaderError(f"Failed to get latest price: {e}") from e

    async def get_symbols(self) -> List[str]:
        """
        Get list of available symbols.

        Returns:
            List of symbol strings

        Raises:
            ReaderError: If query fails
        """
        try:
            query = """
                SELECT DISTINCT symbol
                FROM ohlcv
                ORDER BY symbol
            """

            df = await self._execute_query(query, [])
            symbols = df['symbol'].to_list()
            return symbols

        except Exception as e:
            logger.error("Error getting symbols", error=str(e))
            raise ReaderError(f"Failed to get symbols: {e}") from e

    async def _execute_query(
        self,
        query: str,
        params: List[Any]
    ) -> pl.DataFrame:
        """
        Execute query and return DataFrame.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Query result as DataFrame

        Raises:
            ReaderError: If query fails
        """
        if not self._connected:
            raise ReaderError("Data reader not connected")

        try:
            # Check cache
            if self.enable_cache:
                cache_key = self._make_cache_key(query, params)
                cached = self._get_from_cache(cache_key)
                if cached is not None:
                    self._stats['cache_hits'] += 1
                    return cached

            self._stats['cache_misses'] += 1

            # Execute query
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

            # Convert to DataFrame
            if not rows:
                return pl.DataFrame()

            data = [dict(row) for row in rows]
            df = pl.DataFrame(data)

            self._stats['queries_executed'] += 1
            self._stats['rows_fetched'] += len(df)

            # Cache result
            if self.enable_cache:
                self._add_to_cache(cache_key, df)

            return df

        except Exception as e:
            self._stats['query_errors'] += 1
            logger.error("Query execution error", error=str(e), query=query[:100])
            raise ReaderError(f"Query execution failed: {e}") from e

    def _make_cache_key(self, query: str, params: List[Any]) -> str:
        """
        Create cache key from query and parameters.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Cache key string
        """
        param_str = ','.join(str(p) for p in params)
        return f"{query}:{param_str}"

    def _get_from_cache(self, key: str) -> Optional[pl.DataFrame]:
        """
        Get result from cache.

        Args:
            key: Cache key

        Returns:
            Cached DataFrame or None
        """
        if key not in self._query_cache:
            return None

        df, cached_at = self._query_cache[key]

        # Check if cache is expired
        age = (datetime.now(timezone.utc) - cached_at).total_seconds()
        if age > self.cache_ttl:
            del self._query_cache[key]
            return None

        return df

    def _add_to_cache(self, key: str, df: pl.DataFrame) -> None:
        """
        Add result to cache.

        Args:
            key: Cache key
            df: DataFrame to cache
        """
        self._query_cache[key] = (df, datetime.now(timezone.utc))

        # Limit cache size
        max_cache_size = 100
        if len(self._query_cache) > max_cache_size:
            # Remove oldest entry
            oldest_key = min(
                self._query_cache.keys(),
                key=lambda k: self._query_cache[k][1]
            )
            del self._query_cache[oldest_key]

    def clear_cache(self) -> None:
        """Clear query cache."""
        self._query_cache.clear()
        logger.info("Query cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get reader statistics.

        Returns:
            Dictionary of statistics
        """
        stats = self._stats.copy()

        if stats['cache_hits'] + stats['cache_misses'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / (
                stats['cache_hits'] + stats['cache_misses']
            )
        else:
            stats['cache_hit_rate'] = 0

        stats['cache_size'] = len(self._query_cache)
        return stats

    def is_connected(self) -> bool:
        """
        Check if connected to database.

        Returns:
            True if connected
        """
        return self._connected

    async def health_check(self) -> bool:
        """
        Perform health check.

        Returns:
            True if database is accessible

        Raises:
            ReaderError: If health check fails
        """
        if not self._connected:
            return False

        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
            return True

        except Exception as e:
            logger.error("Health check failed", error=str(e))
            raise ReaderError(f"Health check failed: {e}") from e
