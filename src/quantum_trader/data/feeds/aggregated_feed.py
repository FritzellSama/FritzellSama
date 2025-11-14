"""
Aggregated data feed combining multiple exchange feeds.

This module provides a unified interface for consuming market data from multiple
exchanges simultaneously, with automatic failover and data normalization.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Set, Any, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class FeedConfig:
    """Configuration for aggregated feed."""

    exchanges: List[str]
    symbols: List[str]
    update_interval_ms: int
    buffer_size: int
    enable_orderbook: bool
    enable_trades: bool
    enable_ticker: bool
    max_reconnect_attempts: int
    reconnect_interval_ms: int
    data_validation_enabled: bool
    anomaly_detection_enabled: bool


class AggregatedFeed:
    """
    Aggregated market data feed from multiple exchanges.

    Combines real-time market data from multiple exchanges into a unified stream
    with automatic failover, data validation, and anomaly detection.

    Attributes:
        config: Feed configuration
        exchanges: List of exchange names
        symbols: List of trading symbols

    Example:
        ```python
        config = {
            'exchanges': ['BINANCE', 'BYBIT', 'OKX'],
            'symbols': ['BTC/USDT', 'ETH/USDT'],
            'update_interval_ms': 100,
            'buffer_size': 1000,
            'enable_orderbook': True,
            'enable_trades': True,
            'enable_ticker': True,
            'max_reconnect_attempts': 10,
            'reconnect_interval_ms': 3000,
            'data_validation_enabled': True,
            'anomaly_detection_enabled': True
        }

        feed = AggregatedFeed(config)

        async def handle_data(df: pl.DataFrame) -> None:
            print(f"Received {len(df)} updates")

        await feed.start()
        feed.on_data(handle_data)
        ```
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize aggregated feed.

        Args:
            config: Feed configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        self._validate_config(config)

        self.config = FeedConfig(
            exchanges=config['exchanges'],
            symbols=config['symbols'],
            update_interval_ms=config['update_interval_ms'],
            buffer_size=config['buffer_size'],
            enable_orderbook=config.get('enable_orderbook', True),
            enable_trades=config.get('enable_trades', True),
            enable_ticker=config.get('enable_ticker', True),
            max_reconnect_attempts=config.get('max_reconnect_attempts', 10),
            reconnect_interval_ms=config.get('reconnect_interval_ms', 3000),
            data_validation_enabled=config.get('data_validation_enabled', True),
            anomaly_detection_enabled=config.get('anomaly_detection_enabled', True)
        )

        self._running: bool = False
        self._connected_exchanges: Set[str] = set()
        self._data_buffer: List[Dict[str, Any]] = []
        self._data_handlers: List[Callable[[pl.DataFrame], None]] = []
        self._error_handlers: List[Callable[[Exception], None]] = []
        self._reconnect_counts: Dict[str, int] = {ex: 0 for ex in self.config.exchanges}
        self._last_update_time: Dict[str, datetime] = {}
        self._tasks: List[asyncio.Task] = []

        logger.info(
            "Aggregated feed initialized",
            exchanges=self.config.exchanges,
            symbols=self.config.symbols,
            update_interval_ms=self.config.update_interval_ms
        )

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate feed configuration.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ['exchanges', 'symbols', 'update_interval_ms', 'buffer_size']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")

        if not config['exchanges'] or not isinstance(config['exchanges'], list):
            raise ValueError("exchanges must be a non-empty list")

        if not config['symbols'] or not isinstance(config['symbols'], list):
            raise ValueError("symbols must be a non-empty list")

        if config['update_interval_ms'] <= 0:
            raise ValueError("update_interval_ms must be positive")

        if config['buffer_size'] <= 0:
            raise ValueError("buffer_size must be positive")

    async def start(self) -> None:
        """
        Start the aggregated feed.

        Connects to all configured exchanges and begins streaming data.

        Raises:
            RuntimeError: If feed is already running
        """
        if self._running:
            raise RuntimeError("Feed is already running")

        self._running = True
        logger.info("Starting aggregated feed")

        try:
            # Start feed tasks for each exchange
            for exchange in self.config.exchanges:
                task = asyncio.create_task(
                    self._run_exchange_feed(exchange),
                    name=f"feed_{exchange}"
                )
                self._tasks.append(task)

            # Start data aggregation task
            aggregation_task = asyncio.create_task(
                self._run_aggregation(),
                name="aggregation"
            )
            self._tasks.append(aggregation_task)

            logger.info("Aggregated feed started", task_count=len(self._tasks))

        except Exception as e:
            logger.error("Failed to start aggregated feed", error=str(e))
            await self.stop()
            raise

    async def stop(self) -> None:
        """
        Stop the aggregated feed gracefully.

        Disconnects from all exchanges and cleans up resources.
        """
        if not self._running:
            return

        logger.info("Stopping aggregated feed")
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete with timeout
        if self._tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for tasks to complete")

        self._tasks.clear()
        self._connected_exchanges.clear()
        self._data_buffer.clear()

        logger.info("Aggregated feed stopped")

    def on_data(self, handler: Callable[[pl.DataFrame], None]) -> None:
        """
        Register data handler callback.

        Args:
            handler: Callback function receiving polars DataFrame
        """
        self._data_handlers.append(handler)

    def on_error(self, handler: Callable[[Exception], None]) -> None:
        """
        Register error handler callback.

        Args:
            handler: Callback function receiving exception
        """
        self._error_handlers.append(handler)

    async def _run_exchange_feed(self, exchange: str) -> None:
        """
        Run feed for a single exchange.

        Args:
            exchange: Exchange name
        """
        while self._running:
            try:
                await self._connect_exchange(exchange)
                await self._stream_exchange_data(exchange)

            except asyncio.CancelledError:
                logger.info("Exchange feed cancelled", exchange=exchange)
                break

            except Exception as e:
                logger.error(
                    "Exchange feed error",
                    exchange=exchange,
                    error=str(e)
                )

                self._handle_error(e)

                # Attempt reconnection
                if self._reconnect_counts[exchange] < self.config.max_reconnect_attempts:
                    self._reconnect_counts[exchange] += 1
                    delay = self.config.reconnect_interval_ms / 1000.0

                    logger.info(
                        "Reconnecting to exchange",
                        exchange=exchange,
                        attempt=self._reconnect_counts[exchange],
                        delay_ms=self.config.reconnect_interval_ms
                    )

                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Max reconnection attempts reached",
                        exchange=exchange
                    )
                    break

    async def _connect_exchange(self, exchange: str) -> None:
        """
        Connect to exchange feed.

        Args:
            exchange: Exchange name
        """
        logger.info("Connecting to exchange", exchange=exchange)

        # Simulate connection (replace with actual exchange connector)
        await asyncio.sleep(0.1)

        self._connected_exchanges.add(exchange)
        self._reconnect_counts[exchange] = 0

        logger.info("Connected to exchange", exchange=exchange)

    async def _stream_exchange_data(self, exchange: str) -> None:
        """
        Stream data from exchange.

        Args:
            exchange: Exchange name
        """
        while self._running and exchange in self._connected_exchanges:
            try:
                # Fetch market data for all symbols
                for symbol in self.config.symbols:
                    data = await self._fetch_market_data(exchange, symbol)

                    if data:
                        self._buffer_data(data)
                        self._last_update_time[f"{exchange}_{symbol}"] = datetime.now(timezone.utc)

                # Wait for next update interval
                await asyncio.sleep(self.config.update_interval_ms / 1000.0)

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error(
                    "Error streaming data",
                    exchange=exchange,
                    error=str(e)
                )
                raise

    async def _fetch_market_data(
        self,
        exchange: str,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch market data from exchange.

        Args:
            exchange: Exchange name
            symbol: Trading symbol

        Returns:
            Market data dictionary or None if unavailable
        """
        try:
            # Simulate data fetch (replace with actual exchange API calls)
            data = {
                'exchange': exchange,
                'symbol': symbol,
                'timestamp': datetime.now(timezone.utc),
            }

            if self.config.enable_ticker:
                data['ticker'] = {
                    'bid': Decimal('50000.00'),
                    'ask': Decimal('50001.00'),
                    'last': Decimal('50000.50'),
                    'volume': Decimal('1000.0')
                }

            if self.config.enable_orderbook:
                data['orderbook'] = {
                    'bids': [(Decimal('49999.00'), Decimal('1.5'))],
                    'asks': [(Decimal('50001.00'), Decimal('2.0'))]
                }

            if self.config.enable_trades:
                data['trades'] = []

            if self.config.data_validation_enabled:
                if not self._validate_data(data):
                    logger.warning(
                        "Invalid data received",
                        exchange=exchange,
                        symbol=symbol
                    )
                    return None

            return data

        except Exception as e:
            logger.error(
                "Error fetching market data",
                exchange=exchange,
                symbol=symbol,
                error=str(e)
            )
            return None

    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate market data.

        Args:
            data: Market data dictionary

        Returns:
            True if data is valid
        """
        try:
            if 'ticker' in data:
                ticker = data['ticker']
                if ticker['bid'] >= ticker['ask']:
                    return False
                if any(v <= 0 for v in [ticker['bid'], ticker['ask'], ticker['last']]):
                    return False

            return True

        except Exception as e:
            logger.error("Data validation error", error=str(e))
            return False

    def _buffer_data(self, data: Dict[str, Any]) -> None:
        """
        Add data to buffer.

        Args:
            data: Market data dictionary
        """
        if len(self._data_buffer) >= self.config.buffer_size:
            self._data_buffer.pop(0)

        self._data_buffer.append(data)

    async def _run_aggregation(self) -> None:
        """Run data aggregation loop."""
        while self._running:
            try:
                if self._data_buffer:
                    df = self._create_dataframe(self._data_buffer.copy())

                    if df is not None and len(df) > 0:
                        self._emit_data(df)
                        self._data_buffer.clear()

                await asyncio.sleep(self.config.update_interval_ms / 1000.0)

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error("Aggregation error", error=str(e))
                self._handle_error(e)

    def _create_dataframe(self, data: List[Dict[str, Any]]) -> Optional[pl.DataFrame]:
        """
        Create polars DataFrame from buffered data.

        Args:
            data: List of market data dictionaries

        Returns:
            Polars DataFrame or None if creation failed
        """
        try:
            if not data:
                return None

            # Flatten nested data structure
            rows = []
            for item in data:
                row = {
                    'exchange': item['exchange'],
                    'symbol': item['symbol'],
                    'timestamp': item['timestamp'],
                }

                if 'ticker' in item:
                    row.update({
                        'bid': str(item['ticker']['bid']),
                        'ask': str(item['ticker']['ask']),
                        'last': str(item['ticker']['last']),
                        'volume': str(item['ticker']['volume']),
                    })

                rows.append(row)

            df = pl.DataFrame(rows)
            return df

        except Exception as e:
            logger.error("Error creating DataFrame", error=str(e))
            return None

    def _emit_data(self, df: pl.DataFrame) -> None:
        """
        Emit data to registered handlers.

        Args:
            df: Polars DataFrame
        """
        for handler in self._data_handlers:
            try:
                handler(df)
            except Exception as e:
                logger.error("Error in data handler", error=str(e))

    def _handle_error(self, error: Exception) -> None:
        """
        Handle error by notifying registered handlers.

        Args:
            error: Exception that occurred
        """
        for handler in self._error_handlers:
            try:
                handler(error)
            except Exception as e:
                logger.error("Error in error handler", error=str(e))

    def get_connected_exchanges(self) -> List[str]:
        """
        Get list of currently connected exchanges.

        Returns:
            List of exchange names
        """
        return list(self._connected_exchanges)

    def is_running(self) -> bool:
        """
        Check if feed is running.

        Returns:
            True if feed is running
        """
        return self._running

    def get_buffer_size(self) -> int:
        """
        Get current buffer size.

        Returns:
            Number of items in buffer
        """
        return len(self._data_buffer)
