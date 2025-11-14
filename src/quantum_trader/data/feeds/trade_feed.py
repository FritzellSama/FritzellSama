"""Real-time trade feed processor for institutional trading.

Handles high-frequency trade data ingestion from multiple exchanges with
sub-10ms latency requirements. Processes 10K+ trades/day with full audit trail.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Any, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class Trade:
    """Individual trade execution record."""
    symbol: str
    price: Decimal
    quantity: Decimal
    side: str  # 'BUY' or 'SELL'
    exchange: str
    trade_id: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class TradeFeed:
    """High-performance real-time trade feed with Redis caching and TimescaleDB persistence.

    Attributes:
        config: Configuration dictionary with connection parameters
        redis_client: Async Redis client for sub-ms caching
        db_pool: AsyncPG connection pool for persistence
        _running: Feed status flag
        _buffer: Trade buffer for batch processing
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize trade feed with configuration.

        Args:
            config: Must contain:
                - redis_url: Redis connection string
                - db_url: PostgreSQL connection string
                - buffer_size: Batch size for DB writes
                - flush_interval_ms: Max time between flushes
                - max_retries: Retry attempts for failed operations
                - retry_delay_ms: Initial retry delay (exponential backoff)

        Raises:
            ValueError: If required config parameters missing
        """
        self.config = config
        self._validate_config()

        self.redis_client: Optional[Any] = None
        self.db_pool: Optional[Any] = None
        self._running = False
        self._buffer: List[Trade] = []
        self._lock = asyncio.Lock()

        self.buffer_size = config['buffer_size']
        self.flush_interval = config['flush_interval_ms'] / 1000.0
        self.max_retries = config['max_retries']
        self.retry_delay = config['retry_delay_ms'] / 1000.0

        logger.info("TradeFeed initialized", config_keys=list(config.keys()))

    def _validate_config(self) -> None:
        """Validate required configuration parameters."""
        required = [
            'redis_url', 'db_url', 'buffer_size',
            'flush_interval_ms', 'max_retries', 'retry_delay_ms'
        ]
        missing = [key for key in required if key not in self.config]
        if missing:
            raise ValueError(f"Missing required config parameters: {missing}")

    async def connect(self) -> None:
        """Establish connections to Redis and PostgreSQL with retry logic."""
        import redis.asyncio as redis
        import asyncpg

        for attempt in range(self.max_retries):
            try:
                # Connect to Redis
                self.redis_client = await redis.from_url(
                    self.config['redis_url'],
                    encoding="utf-8",
                    decode_responses=False
                )
                await self.redis_client.ping()
                logger.info("Connected to Redis", url=self.config['redis_url'].split('@')[-1])

                # Connect to PostgreSQL
                self.db_pool = await asyncpg.create_pool(
                    self.config['db_url'],
                    min_size=self.config.get('db_pool_min', 5),
                    max_size=self.config.get('db_pool_max', 20),
                    command_timeout=self.config.get('db_timeout', 30)
                )
                logger.info("Connected to PostgreSQL", pool_size=self.config.get('db_pool_max', 20))

                self._running = True
                return

            except Exception as e:
                logger.error(
                    "Connection failed",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=str(e)
                )
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    raise

    async def disconnect(self) -> None:
        """Gracefully disconnect from all services."""
        self._running = False

        # Flush remaining trades
        if self._buffer:
            await self._flush_buffer()

        # Close connections
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")

        if self.db_pool:
            await self.db_pool.close()
            logger.info("PostgreSQL connection closed")

    async def publish_trade(self, trade: Trade) -> None:
        """Publish trade to feed with caching and persistence.

        Args:
            trade: Trade object to publish

        Raises:
            ConnectionError: If Redis/DB unavailable after retries
        """
        if not self._running:
            raise RuntimeError("TradeFeed not connected")

        try:
            # Cache in Redis for fast access
            await self._cache_trade(trade)

            # Buffer for batch DB insert
            async with self._lock:
                self._buffer.append(trade)
                if len(self._buffer) >= self.buffer_size:
                    await self._flush_buffer()

            logger.debug(
                "Trade published",
                symbol=trade.symbol,
                price=str(trade.price),
                quantity=str(trade.quantity)
            )

        except Exception as e:
            logger.error("Failed to publish trade", trade_id=trade.trade_id, error=str(e))
            raise

    async def _cache_trade(self, trade: Trade) -> None:
        """Cache trade in Redis with exponential backoff retry."""
        import json

        cache_key = f"trade:{trade.exchange}:{trade.symbol}:{trade.trade_id}"
        cache_data = {
            'symbol': trade.symbol,
            'price': str(trade.price),
            'quantity': str(trade.quantity),
            'side': trade.side,
            'exchange': trade.exchange,
            'trade_id': trade.trade_id,
            'timestamp': trade.timestamp.isoformat(),
            'metadata': trade.metadata
        }

        for attempt in range(self.max_retries):
            try:
                await self.redis_client.setex(
                    cache_key,
                    self.config.get('cache_ttl_seconds', 3600),
                    json.dumps(cache_data)
                )
                return
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    logger.error("Redis cache failed", key=cache_key, error=str(e))
                    raise

    async def _flush_buffer(self) -> None:
        """Flush trade buffer to PostgreSQL with batch insert."""
        if not self._buffer:
            return

        async with self._lock:
            trades_to_insert = self._buffer[:]
            self._buffer.clear()

        # Convert to Polars DataFrame for efficient processing
        df = pl.DataFrame({
            'symbol': [t.symbol for t in trades_to_insert],
            'price': [str(t.price) for t in trades_to_insert],
            'quantity': [str(t.quantity) for t in trades_to_insert],
            'side': [t.side for t in trades_to_insert],
            'exchange': [t.exchange for t in trades_to_insert],
            'trade_id': [t.trade_id for t in trades_to_insert],
            'timestamp': [t.timestamp for t in trades_to_insert],
        })

        for attempt in range(self.max_retries):
            try:
                async with self.db_pool.acquire() as conn:
                    # Batch insert using COPY for maximum performance
                    records = [
                        (
                            row['symbol'], row['price'], row['quantity'],
                            row['side'], row['exchange'], row['trade_id'],
                            row['timestamp']
                        )
                        for row in df.iter_rows(named=True)
                    ]

                    await conn.executemany(
                        """
                        INSERT INTO trades (symbol, price, quantity, side, exchange, trade_id, timestamp)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (trade_id) DO NOTHING
                        """,
                        records
                    )

                logger.info("Trade buffer flushed", count=len(trades_to_insert))
                return

            except Exception as e:
                logger.error(
                    "Buffer flush failed",
                    attempt=attempt + 1,
                    count=len(trades_to_insert),
                    error=str(e)
                )
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    # Re-buffer trades on final failure
                    async with self._lock:
                        self._buffer.extend(trades_to_insert)
                    raise

    async def get_recent_trades(
        self,
        symbol: str,
        exchange: str,
        limit: int = 100
    ) -> pl.DataFrame:
        """Retrieve recent trades for a symbol from PostgreSQL.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            exchange: Exchange name
            limit: Maximum number of trades to return

        Returns:
            Polars DataFrame with trade history

        Raises:
            ConnectionError: If database unavailable
        """
        if not self._running:
            raise RuntimeError("TradeFeed not connected")

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT symbol, price, quantity, side, exchange, trade_id, timestamp
                    FROM trades
                    WHERE symbol = $1 AND exchange = $2
                    ORDER BY timestamp DESC
                    LIMIT $3
                    """,
                    symbol, exchange, limit
                )

            if not rows:
                return pl.DataFrame()

            df = pl.DataFrame({
                'symbol': [r['symbol'] for r in rows],
                'price': [r['price'] for r in rows],
                'quantity': [r['quantity'] for r in rows],
                'side': [r['side'] for r in rows],
                'exchange': [r['exchange'] for r in rows],
                'trade_id': [r['trade_id'] for r in rows],
                'timestamp': [r['timestamp'] for r in rows],
            })

            logger.debug("Retrieved recent trades", symbol=symbol, count=len(df))
            return df

        except Exception as e:
            logger.error("Failed to retrieve trades", symbol=symbol, exchange=exchange, error=str(e))
            raise

    async def stream_trades(
        self,
        symbols: List[str],
        exchanges: List[str]
    ) -> AsyncIterator[Trade]:
        """Stream real-time trades from Redis pub/sub.

        Args:
            symbols: List of trading pairs to monitor
            exchanges: List of exchanges to monitor

        Yields:
            Trade objects as they arrive

        Raises:
            ConnectionError: If Redis unavailable
        """
        import json

        if not self._running:
            raise RuntimeError("TradeFeed not connected")

        pubsub = self.redis_client.pubsub()

        try:
            # Subscribe to trade channels
            channels = [
                f"trades:{exchange}:{symbol}"
                for exchange in exchanges
                for symbol in symbols
            ]
            await pubsub.subscribe(*channels)
            logger.info("Subscribed to trade channels", channels=channels)

            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        trade = Trade(
                            symbol=data['symbol'],
                            price=Decimal(data['price']),
                            quantity=Decimal(data['quantity']),
                            side=data['side'],
                            exchange=data['exchange'],
                            trade_id=data['trade_id'],
                            timestamp=datetime.fromisoformat(data['timestamp']),
                            metadata=data.get('metadata', {})
                        )
                        yield trade
                    except Exception as e:
                        logger.error("Failed to parse trade message", error=str(e))
                        continue
        finally:
            await pubsub.unsubscribe()
            await pubsub.close()
