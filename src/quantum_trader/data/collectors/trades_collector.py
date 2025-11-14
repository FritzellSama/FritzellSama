"""Multi-exchange trade data collector with intelligent retry and rate limiting.

Aggregates trades from multiple exchanges simultaneously while respecting
rate limits and handling connection failures gracefully.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Any, Set
from datetime import datetime, timezone
from dataclasses import dataclass, field
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class CollectorStats:
    """Trade collector statistics."""
    trades_collected: int = 0
    trades_failed: int = 0
    exchanges_active: Set[str] = field(default_factory=set)
    last_trade_time: Optional[datetime] = None
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TradesCollector:
    """Multi-exchange trade collector with rate limiting and circuit breaker pattern.

    Attributes:
        config: Configuration dictionary
        exchange_clients: Map of exchange name to client instance
        _running: Collection status flag
        _stats: Collector statistics
        _circuit_breakers: Circuit breaker state per exchange
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize trades collector with configuration.

        Args:
            config: Must contain:
                - exchanges: List of exchange configurations
                - collection_interval_ms: Polling interval
                - rate_limit_per_second: Max requests per second per exchange
                - circuit_breaker_threshold: Failures before circuit opens
                - circuit_breaker_timeout_seconds: Circuit breaker reset time
                - max_retries: Maximum retry attempts
                - retry_delay_ms: Initial retry delay

        Raises:
            ValueError: If required config parameters missing
        """
        self.config = config
        self._validate_config()

        self.exchange_clients: Dict[str, Any] = {}
        self._running = False
        self._stats = CollectorStats()
        self._circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self._rate_limiters: Dict[str, asyncio.Semaphore] = {}
        self._collection_tasks: List[asyncio.Task] = []

        self.collection_interval = config['collection_interval_ms'] / 1000.0
        self.rate_limit = config['rate_limit_per_second']
        self.circuit_threshold = config['circuit_breaker_threshold']
        self.circuit_timeout = config['circuit_breaker_timeout_seconds']
        self.max_retries = config['max_retries']
        self.retry_delay = config['retry_delay_ms'] / 1000.0

        logger.info("TradesCollector initialized", exchanges=len(config.get('exchanges', [])))

    def _validate_config(self) -> None:
        """Validate required configuration parameters."""
        required = [
            'exchanges', 'collection_interval_ms', 'rate_limit_per_second',
            'circuit_breaker_threshold', 'circuit_breaker_timeout_seconds',
            'max_retries', 'retry_delay_ms'
        ]
        missing = [key for key in required if key not in self.config]
        if missing:
            raise ValueError(f"Missing required config parameters: {missing}")

        if not isinstance(self.config['exchanges'], list):
            raise ValueError("'exchanges' must be a list")

    async def connect(self, exchange_clients: Dict[str, Any]) -> None:
        """Connect to exchange clients and initialize collectors.

        Args:
            exchange_clients: Map of exchange name to initialized client instance

        Raises:
            ValueError: If no exchange clients provided
        """
        if not exchange_clients:
            raise ValueError("No exchange clients provided")

        self.exchange_clients = exchange_clients

        # Initialize rate limiters and circuit breakers for each exchange
        for exchange_name in exchange_clients.keys():
            self._rate_limiters[exchange_name] = asyncio.Semaphore(self.rate_limit)
            self._circuit_breakers[exchange_name] = {
                'failures': 0,
                'state': 'closed',  # closed, open, half_open
                'last_failure_time': None
            }

        logger.info("Connected to exchanges", exchanges=list(exchange_clients.keys()))
        self._running = True

    async def disconnect(self) -> None:
        """Gracefully stop all collection tasks."""
        self._running = False

        # Cancel all collection tasks
        for task in self._collection_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._collection_tasks:
            await asyncio.gather(*self._collection_tasks, return_exceptions=True)

        logger.info(
            "TradesCollector disconnected",
            total_trades=self._stats.trades_collected,
            failures=self._stats.trades_failed
        )

    async def start_collection(
        self,
        symbols: List[str],
        callback: Optional[Any] = None
    ) -> None:
        """Start collecting trades from all exchanges.

        Args:
            symbols: List of trading pairs to collect (e.g., ['BTC/USDT', 'ETH/USDT'])
            callback: Optional async callback(trade_data) for each trade

        Raises:
            RuntimeError: If collector not connected
        """
        if not self._running:
            raise RuntimeError("Collector not connected")

        logger.info("Starting trade collection", symbols=symbols, exchanges=len(self.exchange_clients))

        # Start collection task for each exchange
        for exchange_name, client in self.exchange_clients.items():
            task = asyncio.create_task(
                self._collect_from_exchange(exchange_name, client, symbols, callback)
            )
            self._collection_tasks.append(task)

        # Start stats logging task
        stats_task = asyncio.create_task(self._log_stats())
        self._collection_tasks.append(stats_task)

    async def _collect_from_exchange(
        self,
        exchange_name: str,
        client: Any,
        symbols: List[str],
        callback: Optional[Any]
    ) -> None:
        """Collect trades from a single exchange with circuit breaker pattern.

        Args:
            exchange_name: Name of the exchange
            client: Exchange client instance
            symbols: Symbols to collect
            callback: Optional callback for each trade
        """
        logger.info("Starting collection", exchange=exchange_name, symbols=symbols)

        while self._running:
            try:
                # Check circuit breaker
                if not await self._check_circuit_breaker(exchange_name):
                    await asyncio.sleep(self.collection_interval)
                    continue

                # Rate limiting
                async with self._rate_limiters[exchange_name]:
                    # Collect trades for all symbols
                    for symbol in symbols:
                        try:
                            trades_df = await self._fetch_trades_with_retry(
                                exchange_name, client, symbol
                            )

                            if trades_df is not None and len(trades_df) > 0:
                                # Update stats
                                self._stats.trades_collected += len(trades_df)
                                self._stats.exchanges_active.add(exchange_name)
                                self._stats.last_trade_time = datetime.now(timezone.utc)

                                # Call callback if provided
                                if callback:
                                    await callback(exchange_name, symbol, trades_df)

                                # Reset circuit breaker on success
                                self._circuit_breakers[exchange_name]['failures'] = 0

                                logger.debug(
                                    "Trades collected",
                                    exchange=exchange_name,
                                    symbol=symbol,
                                    count=len(trades_df)
                                )

                        except Exception as e:
                            logger.error(
                                "Failed to collect trades",
                                exchange=exchange_name,
                                symbol=symbol,
                                error=str(e)
                            )
                            self._stats.trades_failed += 1
                            await self._record_failure(exchange_name)

                await asyncio.sleep(self.collection_interval)

            except asyncio.CancelledError:
                logger.info("Collection cancelled", exchange=exchange_name)
                break
            except Exception as e:
                logger.error("Collection error", exchange=exchange_name, error=str(e))
                await asyncio.sleep(self.collection_interval)

    async def _fetch_trades_with_retry(
        self,
        exchange_name: str,
        client: Any,
        symbol: str
    ) -> Optional[pl.DataFrame]:
        """Fetch trades with exponential backoff retry.

        Args:
            exchange_name: Exchange name for logging
            client: Exchange client
            symbol: Trading pair

        Returns:
            Polars DataFrame with trade data or None on failure
        """
        for attempt in range(self.max_retries):
            try:
                # Fetch recent trades from exchange
                trades = await client.get_trades(symbol)

                if not trades:
                    return None

                # Convert to Polars DataFrame
                df = pl.DataFrame({
                    'symbol': [symbol] * len(trades),
                    'price': [str(Decimal(str(t['price']))) for t in trades],
                    'quantity': [str(Decimal(str(t['quantity']))) for t in trades],
                    'side': [t['side'] for t in trades],
                    'exchange': [exchange_name] * len(trades),
                    'trade_id': [t['id'] for t in trades],
                    'timestamp': [
                        datetime.fromtimestamp(t['timestamp'] / 1000, tz=timezone.utc)
                        if t['timestamp'] > 1e12
                        else datetime.fromtimestamp(t['timestamp'], tz=timezone.utc)
                        for t in trades
                    ]
                })

                return df

            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        "Fetch failed, retrying",
                        exchange=exchange_name,
                        symbol=symbol,
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e)
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Fetch failed after retries",
                        exchange=exchange_name,
                        symbol=symbol,
                        error=str(e)
                    )
                    raise

        return None

    async def _check_circuit_breaker(self, exchange_name: str) -> bool:
        """Check if circuit breaker allows requests.

        Args:
            exchange_name: Exchange to check

        Returns:
            True if requests allowed, False if circuit open
        """
        breaker = self._circuit_breakers[exchange_name]

        if breaker['state'] == 'closed':
            return True

        if breaker['state'] == 'open':
            # Check if timeout elapsed
            if breaker['last_failure_time']:
                elapsed = (datetime.now(timezone.utc) - breaker['last_failure_time']).total_seconds()
                if elapsed >= self.circuit_timeout:
                    breaker['state'] = 'half_open'
                    logger.info("Circuit breaker half-open", exchange=exchange_name)
                    return True

            return False

        # half_open state - allow one request to test
        return True

    async def _record_failure(self, exchange_name: str) -> None:
        """Record a failure and potentially open circuit breaker.

        Args:
            exchange_name: Exchange that failed
        """
        breaker = self._circuit_breakers[exchange_name]
        breaker['failures'] += 1
        breaker['last_failure_time'] = datetime.now(timezone.utc)

        if breaker['failures'] >= self.circuit_threshold:
            breaker['state'] = 'open'
            logger.warning(
                "Circuit breaker opened",
                exchange=exchange_name,
                failures=breaker['failures']
            )

    async def _log_stats(self) -> None:
        """Periodically log collector statistics."""
        log_interval = self.config.get('stats_log_interval_seconds', 60)

        while self._running:
            try:
                await asyncio.sleep(log_interval)

                runtime = (datetime.now(timezone.utc) - self._stats.start_time).total_seconds()
                rate = self._stats.trades_collected / runtime if runtime > 0 else 0

                logger.info(
                    "Collector statistics",
                    trades_collected=self._stats.trades_collected,
                    trades_failed=self._stats.trades_failed,
                    exchanges_active=len(self._stats.exchanges_active),
                    trades_per_second=f"{rate:.2f}",
                    runtime_seconds=f"{runtime:.0f}"
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Stats logging failed", error=str(e))

    def get_stats(self) -> Dict[str, Any]:
        """Get current collector statistics.

        Returns:
            Dictionary with collection stats
        """
        runtime = (datetime.now(timezone.utc) - self._stats.start_time).total_seconds()

        return {
            'trades_collected': self._stats.trades_collected,
            'trades_failed': self._stats.trades_failed,
            'exchanges_active': list(self._stats.exchanges_active),
            'last_trade_time': self._stats.last_trade_time.isoformat() if self._stats.last_trade_time else None,
            'runtime_seconds': runtime,
            'trades_per_second': self._stats.trades_collected / runtime if runtime > 0 else 0,
            'circuit_breakers': {
                name: {
                    'state': breaker['state'],
                    'failures': breaker['failures']
                }
                for name, breaker in self._circuit_breakers.items()
            }
        }
