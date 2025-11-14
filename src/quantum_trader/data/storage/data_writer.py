"""High-performance data writer for market data storage.

This module provides asynchronous data writing capabilities to both TimescaleDB
for historical data and Redis for real-time access. Implements batching,
connection pooling, and retry logic for production reliability.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
from asyncpg import Pool, create_pool
from redis.asyncio import Redis, ConnectionPool
from structlog import get_logger

from quantum_trader.models import OrderBook, Ticker

logger = get_logger(__name__)


class DataWriter:
    """High-performance data writer for market data.

    Manages concurrent writes to TimescaleDB and Redis with batching,
    connection pooling, and automatic retry logic.

    Attributes:
        config: Configuration dictionary containing database and cache settings
        pg_pool: PostgreSQL connection pool
        redis_pool: Redis connection pool
        redis_client: Redis async client
        batch_size: Maximum batch size for bulk writes
        flush_interval: Interval in seconds for automatic batch flushing
        _write_buffer: Internal buffer for batching writes
        _flush_task: Background task for periodic flushing
        _running: Flag indicating if writer is active

    Example:
        >>> config = {
        ...     "database": {"host": "localhost", "port": 5432, ...},
        ...     "cache": {"host": "localhost", "port": 6379, ...},
        ...     "data_writer": {"batch_size": 1000, "flush_interval": 1.0}
        ... }
        >>> async with DataWriter(config) as writer:
        ...     await writer.write_ticker(ticker_data)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize data writer.

        Args:
            config: Configuration dictionary with database, cache, and writer settings

        Raises:
            ValueError: If required configuration keys are missing
        """
        self.config = config
        self._validate_config()

        self.pg_pool: Optional[Pool] = None
        self.redis_pool: Optional[ConnectionPool] = None
        self.redis_client: Optional[Redis] = None

        # Writer configuration
        writer_config = config.get("data_writer", {})
        self.batch_size = writer_config.get("batch_size", 1000)
        self.flush_interval = writer_config.get("flush_interval", 1.0)
        self.retry_attempts = writer_config.get("retry_attempts", 3)
        self.retry_delay = writer_config.get("retry_delay_ms", 1000) / 1000.0
        self.retry_backoff = writer_config.get("retry_backoff_multiplier", 2.0)

        # Internal state
        self._write_buffer: Dict[str, List[Dict[str, Any]]] = {
            "tickers": [],
            "orderbooks": [],
            "trades": [],
            "funding": [],
            "liquidations": []
        }
        self._buffer_lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False

    def _validate_config(self) -> None:
        """Validate required configuration parameters.

        Raises:
            ValueError: If required configuration is missing
        """
        required_keys = ["database", "cache"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")

        db_config = self.config["database"]
        required_db_keys = ["host", "port", "name", "user", "password"]
        for key in required_db_keys:
            if key not in db_config:
                raise ValueError(f"Missing required database configuration: {key}")

        cache_config = self.config["cache"]
        required_cache_keys = ["host", "port"]
        for key in required_cache_keys:
            if key not in cache_config:
                raise ValueError(f"Missing required cache configuration: {key}")

    async def connect(self) -> None:
        """Establish connections to database and cache.

        Creates connection pools and starts background flush task.

        Raises:
            Exception: If connection establishment fails
        """
        try:
            # Create PostgreSQL connection pool
            db_config = self.config["database"]
            self.pg_pool = await create_pool(
                host=db_config["host"],
                port=db_config["port"],
                database=db_config["name"],
                user=db_config["user"],
                password=db_config["password"],
                min_size=db_config.get("pool_size", 20),
                max_size=db_config.get("max_overflow", 10),
                timeout=db_config.get("pool_timeout_seconds", 30)
            )

            # Create Redis connection pool
            cache_config = self.config["cache"]
            self.redis_pool = ConnectionPool(
                host=cache_config["host"],
                port=cache_config["port"],
                db=cache_config.get("db", 0),
                password=cache_config.get("password"),
                decode_responses=False,
                max_connections=cache_config.get("max_connections", 50),
                socket_timeout=cache_config.get("socket_timeout_seconds", 5),
                socket_connect_timeout=cache_config.get("socket_connect_timeout_seconds", 5)
            )
            self.redis_client = Redis(connection_pool=self.redis_pool)

            # Test connections
            async with self.pg_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")

            await self.redis_client.ping()

            # Start background flush task
            self._running = True
            self._flush_task = asyncio.create_task(self._periodic_flush())

            logger.info(
                "data_writer_connected",
                pg_pool_size=self.pg_pool.get_size(),
                redis_pool_size=self.redis_pool.max_connections
            )

        except Exception as e:
            logger.error("data_writer_connection_failed", error=str(e))
            await self.disconnect()
            raise

    async def disconnect(self) -> None:
        """Gracefully shutdown connections and flush remaining data.

        Stops background tasks, flushes buffers, and closes all connections.
        """
        try:
            self._running = False

            # Stop flush task
            if self._flush_task and not self._flush_task.done():
                self._flush_task.cancel()
                try:
                    await self._flush_task
                except asyncio.CancelledError:
                    pass

            # Flush remaining data
            await self._flush_all_buffers()

            # Close connections
            if self.redis_client:
                await self.redis_client.aclose()

            if self.redis_pool:
                await self.redis_pool.disconnect()

            if self.pg_pool:
                await self.pg_pool.close()

            logger.info("data_writer_disconnected")

        except Exception as e:
            logger.error("data_writer_disconnect_failed", error=str(e))

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def write_ticker(self, ticker: Ticker) -> None:
        """Write ticker data to storage.

        Args:
            ticker: Ticker data to write

        Example:
            >>> await writer.write_ticker(ticker)
        """
        try:
            # Add to buffer
            async with self._buffer_lock:
                self._write_buffer["tickers"].append({
                    "symbol": ticker.symbol,
                    "bid": str(ticker.bid),
                    "ask": str(ticker.ask),
                    "last": str(ticker.last),
                    "volume": str(ticker.volume),
                    "timestamp": ticker.timestamp
                })

                # Flush if buffer full
                if len(self._write_buffer["tickers"]) >= self.batch_size:
                    await self._flush_tickers()

            # Write to Redis for real-time access
            await self._write_ticker_to_redis(ticker)

        except Exception as e:
            logger.error("write_ticker_failed", symbol=ticker.symbol, error=str(e))
            raise

    async def write_orderbook(self, orderbook: OrderBook) -> None:
        """Write order book snapshot to storage.

        Args:
            orderbook: OrderBook snapshot to write

        Example:
            >>> await writer.write_orderbook(orderbook)
        """
        try:
            # Add to buffer
            async with self._buffer_lock:
                self._write_buffer["orderbooks"].append({
                    "symbol": orderbook.symbol,
                    "bids": [(str(p), str(s)) for p, s in orderbook.bids],
                    "asks": [(str(p), str(s)) for p, s in orderbook.asks],
                    "timestamp": orderbook.timestamp
                })

                # Flush if buffer full
                if len(self._write_buffer["orderbooks"]) >= self.batch_size:
                    await self._flush_orderbooks()

            # Write to Redis for real-time access
            await self._write_orderbook_to_redis(orderbook)

        except Exception as e:
            logger.error("write_orderbook_failed", symbol=orderbook.symbol, error=str(e))
            raise

    async def write_batch(self, data_type: str, data: pl.DataFrame) -> None:
        """Write batch of data to TimescaleDB.

        Args:
            data_type: Type of data (tickers, orderbooks, trades, etc.)
            data: Polars DataFrame containing batch data

        Raises:
            ValueError: If data_type is not supported
        """
        if data.is_empty():
            return

        try:
            if data_type == "tickers":
                await self._write_tickers_batch(data)
            elif data_type == "orderbooks":
                await self._write_orderbooks_batch(data)
            elif data_type == "trades":
                await self._write_trades_batch(data)
            elif data_type == "funding":
                await self._write_funding_batch(data)
            elif data_type == "liquidations":
                await self._write_liquidations_batch(data)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")

            logger.debug(
                "batch_written",
                data_type=data_type,
                rows=len(data)
            )

        except Exception as e:
            logger.error(
                "write_batch_failed",
                data_type=data_type,
                rows=len(data),
                error=str(e)
            )
            raise

    async def _write_ticker_to_redis(self, ticker: Ticker) -> None:
        """Write ticker to Redis for real-time access."""
        try:
            key = f"ticker:{ticker.symbol}"
            value = {
                "bid": str(ticker.bid),
                "ask": str(ticker.ask),
                "last": str(ticker.last),
                "volume": str(ticker.volume),
                "timestamp": ticker.timestamp.isoformat()
            }

            # Use pipeline for atomic operations
            async with self.redis_client.pipeline(transaction=True) as pipe:
                pipe.hset(key, mapping=value)
                pipe.expire(key, 3600)  # 1 hour TTL
                await pipe.execute()

        except Exception as e:
            logger.error("redis_ticker_write_failed", symbol=ticker.symbol, error=str(e))

    async def _write_orderbook_to_redis(self, orderbook: OrderBook) -> None:
        """Write orderbook to Redis for real-time access."""
        try:
            key = f"orderbook:{orderbook.symbol}"

            # Store top 10 levels
            bids_data = [(str(p), str(s)) for p, s in orderbook.bids[:10]]
            asks_data = [(str(p), str(s)) for p, s in orderbook.asks[:10]]

            value = {
                "bids": str(bids_data),
                "asks": str(asks_data),
                "timestamp": orderbook.timestamp.isoformat()
            }

            async with self.redis_client.pipeline(transaction=True) as pipe:
                pipe.hset(key, mapping=value)
                pipe.expire(key, 300)  # 5 minutes TTL
                await pipe.execute()

        except Exception as e:
            logger.error("redis_orderbook_write_failed", symbol=orderbook.symbol, error=str(e))

    async def _flush_tickers(self) -> None:
        """Flush ticker buffer to database."""
        if not self._write_buffer["tickers"]:
            return

        tickers = self._write_buffer["tickers"].copy()
        self._write_buffer["tickers"].clear()

        df = pl.DataFrame(tickers)
        await self._write_tickers_batch(df)

    async def _flush_orderbooks(self) -> None:
        """Flush orderbook buffer to database."""
        if not self._write_buffer["orderbooks"]:
            return

        orderbooks = self._write_buffer["orderbooks"].copy()
        self._write_buffer["orderbooks"].clear()

        df = pl.DataFrame(orderbooks)
        await self._write_orderbooks_batch(df)

    async def _flush_all_buffers(self) -> None:
        """Flush all data buffers."""
        async with self._buffer_lock:
            await self._flush_tickers()
            await self._flush_orderbooks()

    async def _periodic_flush(self) -> None:
        """Background task for periodic buffer flushing."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_all_buffers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("periodic_flush_failed", error=str(e))

    async def _write_tickers_batch(self, df: pl.DataFrame) -> None:
        """Write batch of tickers to TimescaleDB with retry logic."""
        query = """
            INSERT INTO tickers (symbol, bid, ask, last, volume, timestamp)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (symbol, timestamp) DO UPDATE
            SET bid = EXCLUDED.bid,
                ask = EXCLUDED.ask,
                last = EXCLUDED.last,
                volume = EXCLUDED.volume
        """

        for attempt in range(self.retry_attempts):
            try:
                async with self.pg_pool.acquire() as conn:
                    async with conn.transaction():
                        await conn.executemany(
                            query,
                            [
                                (
                                    row["symbol"],
                                    Decimal(row["bid"]),
                                    Decimal(row["ask"]),
                                    Decimal(row["last"]),
                                    Decimal(row["volume"]),
                                    row["timestamp"]
                                )
                                for row in df.to_dicts()
                            ]
                        )
                return

            except Exception as e:
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay * (self.retry_backoff ** attempt)
                    logger.warning(
                        "ticker_write_retry",
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e)
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("ticker_write_failed", error=str(e))
                    raise

    async def _write_orderbooks_batch(self, df: pl.DataFrame) -> None:
        """Write batch of orderbooks to TimescaleDB with retry logic."""
        query = """
            INSERT INTO orderbooks (symbol, bids, asks, timestamp)
            VALUES ($1, $2, $3, $4)
        """

        for attempt in range(self.retry_attempts):
            try:
                async with self.pg_pool.acquire() as conn:
                    async with conn.transaction():
                        await conn.executemany(
                            query,
                            [
                                (
                                    row["symbol"],
                                    row["bids"],
                                    row["asks"],
                                    row["timestamp"]
                                )
                                for row in df.to_dicts()
                            ]
                        )
                return

            except Exception as e:
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay * (self.retry_backoff ** attempt)
                    logger.warning(
                        "orderbook_write_retry",
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e)
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("orderbook_write_failed", error=str(e))
                    raise

    async def _write_trades_batch(self, df: pl.DataFrame) -> None:
        """Write batch of trades to TimescaleDB."""
        query = """
            INSERT INTO trades (symbol, trade_id, price, size, side, timestamp)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (symbol, trade_id) DO NOTHING
        """

        async with self.pg_pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    query,
                    [
                        (
                            row["symbol"],
                            row["trade_id"],
                            Decimal(str(row["price"])),
                            Decimal(str(row["size"])),
                            row["side"],
                            row["timestamp"]
                        )
                        for row in df.to_dicts()
                    ]
                )

    async def _write_funding_batch(self, df: pl.DataFrame) -> None:
        """Write batch of funding rates to TimescaleDB."""
        query = """
            INSERT INTO funding_rates (symbol, rate, timestamp)
            VALUES ($1, $2, $3)
            ON CONFLICT (symbol, timestamp) DO UPDATE
            SET rate = EXCLUDED.rate
        """

        async with self.pg_pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    query,
                    [
                        (
                            row["symbol"],
                            Decimal(str(row["rate"])),
                            row["timestamp"]
                        )
                        for row in df.to_dicts()
                    ]
                )

    async def _write_liquidations_batch(self, df: pl.DataFrame) -> None:
        """Write batch of liquidations to TimescaleDB."""
        query = """
            INSERT INTO liquidations (symbol, side, price, quantity, timestamp)
            VALUES ($1, $2, $3, $4, $5)
        """

        async with self.pg_pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    query,
                    [
                        (
                            row["symbol"],
                            row["side"],
                            Decimal(str(row["price"])),
                            Decimal(str(row["quantity"])),
                            row["timestamp"]
                        )
                        for row in df.to_dicts()
                    ]
                )
