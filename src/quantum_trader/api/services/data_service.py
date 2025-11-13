"""
Data service for Quantum Trader AI.

Handles market data retrieval, caching, and aggregation.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta
import polars as pl
from structlog import get_logger

from quantum_trader.api.exceptions import (
    DataNotFoundException,
    DataValidationException,
    CacheException
)

logger = get_logger(__name__)


class DataService:
    """Production-ready data service for market data.

    Provides high-performance market data operations:
    - Real-time market data retrieval
    - Historical data aggregation
    - Multi-exchange data normalization
    - Caching with Redis
    - OHLCV data processing

    Attributes:
        config: Configuration dictionary
        db_pool: Database connection pool
        redis_client: Redis client for caching
        cache_ttl: Cache time-to-live in seconds
    """

    def __init__(
        self,
        config: Dict[str, Any],
        db_pool: Any,
        redis_client: Any
    ) -> None:
        """Initialize data service.

        Args:
            config: Configuration from config files
            db_pool: Database connection pool
            redis_client: Redis client

        Raises:
            ValueError: If config validation fails
        """
        self.config = config
        self.db_pool = db_pool
        self.redis_client = redis_client

        # Load from config
        data_config = config.get("data_service", {})
        self.cache_ttl = data_config.get("cache_ttl_seconds", 60)
        self.max_results = data_config.get("max_results", 10000)
        self.supported_timeframes = data_config.get(
            "supported_timeframes",
            ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
        )

        self._validate_config()
        logger.info("Data service initialized")

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.cache_ttl <= 0:
            raise ValueError("cache_ttl must be positive")
        if self.max_results <= 0:
            raise ValueError("max_results must be positive")
        if not self.supported_timeframes:
            raise ValueError("supported_timeframes cannot be empty")

    async def get_ticker(
        self,
        symbol: str,
        exchange: str
    ) -> Dict[str, Any]:
        """Get current ticker data for symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            exchange: Exchange name

        Returns:
            Dictionary with ticker data:
                - symbol: Trading symbol
                - exchange: Exchange name
                - last_price: Last traded price
                - bid: Best bid price
                - ask: Best ask price
                - volume_24h: 24h volume
                - timestamp: Data timestamp

        Raises:
            DataNotFoundException: If ticker not found
            DataValidationException: If parameters invalid
        """
        if not symbol or not exchange:
            raise DataValidationException("Symbol and exchange are required")

        # Check cache
        cache_key = f"ticker:{exchange}:{symbol}"
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                logger.debug("Cache hit for ticker", symbol=symbol, exchange=exchange)
                return cached_data
        except Exception as e:
            logger.warning("Cache read failed", error=str(e))

        try:
            # Query database
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT
                        symbol, exchange, last_price, bid, ask,
                        volume_24h, timestamp
                    FROM market_tickers
                    WHERE symbol = $1 AND exchange = $2
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    symbol, exchange
                )

                if not row:
                    raise DataNotFoundException(
                        f"Ticker not found for {symbol} on {exchange}"
                    )

                ticker_data = {
                    "symbol": row["symbol"],
                    "exchange": row["exchange"],
                    "last_price": str(Decimal(str(row["last_price"]))),
                    "bid": str(Decimal(str(row["bid"]))) if row["bid"] else None,
                    "ask": str(Decimal(str(row["ask"]))) if row["ask"] else None,
                    "volume_24h": str(Decimal(str(row["volume_24h"]))),
                    "timestamp": row["timestamp"].isoformat()
                }

            # Cache result
            try:
                await self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    ticker_data
                )
            except Exception as e:
                logger.warning("Cache write failed", error=str(e))

            return ticker_data

        except DataNotFoundException:
            raise
        except Exception as e:
            logger.error("Failed to get ticker", error=str(e))
            raise

    async def get_orderbook(
        self,
        symbol: str,
        exchange: str,
        depth: int = 20
    ) -> Dict[str, Any]:
        """Get orderbook for symbol.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            depth: Orderbook depth (default 20)

        Returns:
            Dictionary with orderbook data:
                - symbol: Trading symbol
                - exchange: Exchange name
                - bids: List of [price, quantity] bids
                - asks: List of [price, quantity] asks
                - timestamp: Data timestamp

        Raises:
            DataNotFoundException: If orderbook not found
            DataValidationException: If parameters invalid
        """
        if not symbol or not exchange:
            raise DataValidationException("Symbol and exchange are required")

        if depth <= 0 or depth > 100:
            raise DataValidationException("Depth must be between 1 and 100")

        # Check cache
        cache_key = f"orderbook:{exchange}:{symbol}:{depth}"
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                logger.debug("Cache hit for orderbook", symbol=symbol, exchange=exchange)
                return cached_data
        except Exception as e:
            logger.warning("Cache read failed", error=str(e))

        try:
            async with self.db_pool.acquire() as conn:
                # Get latest orderbook
                row = await conn.fetchrow(
                    """
                    SELECT bids, asks, timestamp
                    FROM market_orderbook
                    WHERE symbol = $1 AND exchange = $2
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    symbol, exchange
                )

                if not row:
                    raise DataNotFoundException(
                        f"Orderbook not found for {symbol} on {exchange}"
                    )

                # Parse and limit depth
                bids = row["bids"][:depth] if row["bids"] else []
                asks = row["asks"][:depth] if row["asks"] else []

                orderbook_data = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "bids": [[str(Decimal(str(p))), str(Decimal(str(q)))] for p, q in bids],
                    "asks": [[str(Decimal(str(p))), str(Decimal(str(q)))] for p, q in asks],
                    "timestamp": row["timestamp"].isoformat()
                }

            # Cache result
            try:
                await self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    orderbook_data
                )
            except Exception as e:
                logger.warning("Cache write failed", error=str(e))

            return orderbook_data

        except DataNotFoundException:
            raise
        except Exception as e:
            logger.error("Failed to get orderbook", error=str(e))
            raise

    async def get_ohlcv(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pl.DataFrame:
        """Get OHLCV data for symbol.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe (e.g., '1m', '5m', '1h', '1d')
            start_time: Start time (UTC)
            end_time: End time (UTC)
            limit: Maximum number of candles

        Returns:
            Polars DataFrame with OHLCV data

        Raises:
            DataValidationException: If parameters invalid
            DataNotFoundException: If no data found
        """
        if timeframe not in self.supported_timeframes:
            raise DataValidationException(
                f"Unsupported timeframe. Supported: {self.supported_timeframes}"
            )

        if limit and (limit <= 0 or limit > self.max_results):
            raise DataValidationException(
                f"Limit must be between 1 and {self.max_results}"
            )

        try:
            # Build query
            query = """
                SELECT
                    timestamp, open, high, low, close, volume
                FROM market_ohlcv
                WHERE symbol = $1 AND exchange = $2 AND timeframe = $3
            """
            params: List[Any] = [symbol, exchange, timeframe]
            param_count = 3

            if start_time:
                param_count += 1
                query += f" AND timestamp >= ${param_count}"
                params.append(start_time)

            if end_time:
                param_count += 1
                query += f" AND timestamp <= ${param_count}"
                params.append(end_time)

            query += " ORDER BY timestamp DESC"

            if limit:
                param_count += 1
                query += f" LIMIT ${param_count}"
                params.append(limit)

            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

            if not rows:
                raise DataNotFoundException(
                    f"No OHLCV data found for {symbol} on {exchange}"
                )

            # Convert to Polars DataFrame
            data = {
                "timestamp": [r["timestamp"] for r in rows],
                "open": [Decimal(str(r["open"])) for r in rows],
                "high": [Decimal(str(r["high"])) for r in rows],
                "low": [Decimal(str(r["low"])) for r in rows],
                "close": [Decimal(str(r["close"])) for r in rows],
                "volume": [Decimal(str(r["volume"])) for r in rows],
            }

            df = pl.DataFrame(data)

            logger.info(
                "OHLCV data retrieved",
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                rows=df.height
            )

            return df

        except DataNotFoundException:
            raise
        except DataValidationException:
            raise
        except Exception as e:
            logger.error("Failed to get OHLCV data", error=str(e))
            raise

    async def get_recent_trades(
        self,
        symbol: str,
        exchange: str,
        limit: int = 100
    ) -> pl.DataFrame:
        """Get recent trades for symbol.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            limit: Maximum number of trades

        Returns:
            Polars DataFrame with trade data

        Raises:
            DataValidationException: If parameters invalid
        """
        if limit <= 0 or limit > self.max_results:
            raise DataValidationException(
                f"Limit must be between 1 and {self.max_results}"
            )

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT
                        trade_id, timestamp, price, quantity, side
                    FROM market_trades
                    WHERE symbol = $1 AND exchange = $2
                    ORDER BY timestamp DESC
                    LIMIT $3
                    """,
                    symbol, exchange, limit
                )

            if not rows:
                return pl.DataFrame()

            # Convert to Polars DataFrame
            data = {
                "trade_id": [r["trade_id"] for r in rows],
                "timestamp": [r["timestamp"] for r in rows],
                "price": [Decimal(str(r["price"])) for r in rows],
                "quantity": [Decimal(str(r["quantity"])) for r in rows],
                "side": [r["side"] for r in rows],
            }

            return pl.DataFrame(data)

        except Exception as e:
            logger.error("Failed to get recent trades", error=str(e))
            raise

    async def get_symbols(self, exchange: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of available trading symbols.

        Args:
            exchange: Optional exchange filter

        Returns:
            List of symbol dictionaries

        Raises:
            Exception: If query fails
        """
        try:
            query = """
                SELECT DISTINCT symbol, exchange, base_asset, quote_asset, is_active
                FROM trading_symbols
            """
            params = []

            if exchange:
                query += " WHERE exchange = $1"
                params.append(exchange)

            query += " ORDER BY symbol, exchange"

            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

            symbols = [
                {
                    "symbol": r["symbol"],
                    "exchange": r["exchange"],
                    "base_asset": r["base_asset"],
                    "quote_asset": r["quote_asset"],
                    "is_active": r["is_active"]
                }
                for r in rows
            ]

            return symbols

        except Exception as e:
            logger.error("Failed to get symbols", error=str(e))
            raise

    async def get_aggregated_ohlcv(
        self,
        symbol: str,
        exchanges: List[str],
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pl.DataFrame:
        """Get volume-weighted aggregated OHLCV across multiple exchanges.

        Args:
            symbol: Trading symbol
            exchanges: List of exchanges
            timeframe: Timeframe
            start_time: Start time
            end_time: End time
            limit: Maximum number of candles

        Returns:
            Polars DataFrame with aggregated OHLCV data

        Raises:
            DataValidationException: If parameters invalid
        """
        if not exchanges:
            raise DataValidationException("At least one exchange required")

        # Fetch data from each exchange
        dfs = []
        for exchange in exchanges:
            try:
                df = await self.get_ohlcv(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit
                )
                dfs.append(df)
            except DataNotFoundException:
                logger.warning(
                    "No data for exchange",
                    symbol=symbol,
                    exchange=exchange
                )
                continue

        if not dfs:
            raise DataNotFoundException(
                f"No OHLCV data found for {symbol} across exchanges"
            )

        # Concatenate and aggregate
        combined_df = pl.concat(dfs)

        # Volume-weighted aggregation
        aggregated = combined_df.groupby("timestamp").agg([
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume")
        ]).sort("timestamp", descending=True)

        if limit:
            aggregated = aggregated.head(limit)

        logger.info(
            "Aggregated OHLCV data",
            symbol=symbol,
            exchanges=exchanges,
            rows=aggregated.height
        )

        return aggregated

    async def invalidate_cache(
        self,
        pattern: str
    ) -> int:
        """Invalidate cache entries matching pattern.

        Args:
            pattern: Redis key pattern (e.g., 'ticker:*')

        Returns:
            Number of keys deleted

        Raises:
            CacheException: If cache operation fails
        """
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                deleted = await self.redis_client.delete(*keys)
                logger.info("Cache invalidated", pattern=pattern, deleted=deleted)
                return deleted
            return 0

        except Exception as e:
            logger.error("Cache invalidation failed", error=str(e))
            raise CacheException("Cache invalidation failed")

    async def close(self) -> None:
        """Cleanup resources."""
        logger.info("Data service shutting down")
