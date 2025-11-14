"""
Base data collector for market data collection.

This module provides an abstract base class for implementing data collectors
that fetch market data from various sources (exchanges, APIs, etc.).
"""

import asyncio
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timezone
from dataclasses import dataclass
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class CollectorConfig:
    """Configuration for data collector."""

    name: str
    symbols: List[str]
    collection_interval_ms: int
    max_retries: int
    retry_delay_ms: int
    timeout_ms: int
    enable_validation: bool
    buffer_size: int
    batch_size: int


class CollectorError(Exception):
    """Base exception for collector errors."""

    pass


class ConnectionError(CollectorError):
    """Exception raised for connection errors."""

    pass


class DataError(CollectorError):
    """Exception raised for data-related errors."""

    pass


class BaseCollector(ABC):
    """
    Abstract base class for market data collectors.

    Provides common functionality for collecting, validating, and buffering
    market data from various sources.

    Attributes:
        config: Collector configuration
        name: Collector name
        symbols: List of symbols to collect

    Example:
        ```python
        class BinanceCollector(BaseCollector):
            async def _fetch_data(self, symbol: str) -> Optional[Dict[str, Any]]:
                # Implement Binance-specific data fetching
                pass

            async def _connect(self) -> None:
                # Implement Binance connection logic
                pass

            async def _disconnect(self) -> None:
                # Implement Binance disconnection logic
                pass

        config = {
            'name': 'binance',
            'symbols': ['BTC/USDT', 'ETH/USDT'],
            'collection_interval_ms': 1000,
            'max_retries': 3,
            'retry_delay_ms': 1000,
            'timeout_ms': 5000,
            'enable_validation': True,
            'buffer_size': 1000,
            'batch_size': 100
        }

        collector = BinanceCollector(config)
        await collector.start()
        ```
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize base collector.

        Args:
            config: Collector configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        self._validate_config(config)

        self.config = CollectorConfig(
            name=config['name'],
            symbols=config['symbols'],
            collection_interval_ms=config['collection_interval_ms'],
            max_retries=config.get('max_retries', 3),
            retry_delay_ms=config.get('retry_delay_ms', 1000),
            timeout_ms=config.get('timeout_ms', 5000),
            enable_validation=config.get('enable_validation', True),
            buffer_size=config.get('buffer_size', 1000),
            batch_size=config.get('batch_size', 100)
        )

        self._running: bool = False
        self._connected: bool = False
        self._data_buffer: List[Dict[str, Any]] = []
        self._failed_symbols: Set[str] = set()
        self._collection_task: Optional[asyncio.Task] = None
        self._stats: Dict[str, int] = {
            'total_collected': 0,
            'total_errors': 0,
            'total_retries': 0,
            'validation_failures': 0
        }

        logger.info(
            "Collector initialized",
            name=self.config.name,
            symbols=self.config.symbols,
            interval_ms=self.config.collection_interval_ms
        )

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate collector configuration.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ['name', 'symbols', 'collection_interval_ms']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")

        if not config['name'] or not isinstance(config['name'], str):
            raise ValueError("name must be a non-empty string")

        if not config['symbols'] or not isinstance(config['symbols'], list):
            raise ValueError("symbols must be a non-empty list")

        if config['collection_interval_ms'] <= 0:
            raise ValueError("collection_interval_ms must be positive")

    async def start(self) -> None:
        """
        Start the data collector.

        Connects to data source and begins collecting data.

        Raises:
            RuntimeError: If collector is already running
            ConnectionError: If connection fails
        """
        if self._running:
            raise RuntimeError("Collector is already running")

        logger.info("Starting collector", name=self.config.name)

        try:
            await self._connect()
            self._connected = True
            self._running = True

            self._collection_task = asyncio.create_task(
                self._collection_loop(),
                name=f"collector_{self.config.name}"
            )

            logger.info("Collector started", name=self.config.name)

        except Exception as e:
            logger.error(
                "Failed to start collector",
                name=self.config.name,
                error=str(e)
            )
            raise ConnectionError(f"Failed to start collector: {e}") from e

    async def stop(self) -> None:
        """
        Stop the data collector gracefully.

        Disconnects from data source and cleans up resources.
        """
        if not self._running:
            return

        logger.info("Stopping collector", name=self.config.name)
        self._running = False

        if self._collection_task and not self._collection_task.done():
            self._collection_task.cancel()

            try:
                await asyncio.wait_for(self._collection_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "Timeout waiting for collection task",
                    name=self.config.name
                )
            except asyncio.CancelledError:
                pass

        if self._connected:
            try:
                await self._disconnect()
            except Exception as e:
                logger.error(
                    "Error disconnecting",
                    name=self.config.name,
                    error=str(e)
                )
            finally:
                self._connected = False

        self._data_buffer.clear()

        logger.info(
            "Collector stopped",
            name=self.config.name,
            stats=self._stats
        )

    async def _collection_loop(self) -> None:
        """Run the main collection loop."""
        while self._running:
            try:
                batch_start = datetime.now(timezone.utc)

                # Collect data for all symbols
                for symbol in self.config.symbols:
                    if symbol in self._failed_symbols:
                        continue

                    try:
                        data = await self._fetch_with_retry(symbol)

                        if data:
                            if self.config.enable_validation:
                                if not self._validate_data(data):
                                    self._stats['validation_failures'] += 1
                                    logger.warning(
                                        "Data validation failed",
                                        name=self.config.name,
                                        symbol=symbol
                                    )
                                    continue

                            self._buffer_data(data)
                            self._stats['total_collected'] += 1

                    except Exception as e:
                        logger.error(
                            "Error collecting data for symbol",
                            name=self.config.name,
                            symbol=symbol,
                            error=str(e)
                        )
                        self._stats['total_errors'] += 1

                # Process batches if buffer is large enough
                if len(self._data_buffer) >= self.config.batch_size:
                    await self._process_batch()

                # Calculate sleep time to maintain interval
                elapsed = (datetime.now(timezone.utc) - batch_start).total_seconds() * 1000
                sleep_ms = max(0, self.config.collection_interval_ms - elapsed)

                if sleep_ms > 0:
                    await asyncio.sleep(sleep_ms / 1000.0)

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error(
                    "Error in collection loop",
                    name=self.config.name,
                    error=str(e)
                )
                self._stats['total_errors'] += 1
                await asyncio.sleep(1.0)

    async def _fetch_with_retry(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch data with retry logic.

        Args:
            symbol: Trading symbol

        Returns:
            Data dictionary or None if all retries failed
        """
        for attempt in range(self.config.max_retries):
            try:
                data = await asyncio.wait_for(
                    self._fetch_data(symbol),
                    timeout=self.config.timeout_ms / 1000.0
                )
                return data

            except asyncio.TimeoutError:
                logger.warning(
                    "Timeout fetching data",
                    name=self.config.name,
                    symbol=symbol,
                    attempt=attempt + 1
                )
                self._stats['total_retries'] += 1

            except Exception as e:
                logger.error(
                    "Error fetching data",
                    name=self.config.name,
                    symbol=symbol,
                    attempt=attempt + 1,
                    error=str(e)
                )
                self._stats['total_retries'] += 1

            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay_ms / 1000.0)

        # Mark symbol as failed after all retries exhausted
        self._failed_symbols.add(symbol)
        logger.error(
            "All retries exhausted for symbol",
            name=self.config.name,
            symbol=symbol
        )
        return None

    def _buffer_data(self, data: Dict[str, Any]) -> None:
        """
        Add data to buffer.

        Args:
            data: Data dictionary to buffer
        """
        if len(self._data_buffer) >= self.config.buffer_size:
            self._data_buffer.pop(0)

        self._data_buffer.append(data)

    async def _process_batch(self) -> None:
        """
        Process buffered data batch.

        Converts buffered data to DataFrame and stores it.
        """
        try:
            if not self._data_buffer:
                return

            batch = self._data_buffer[:self.config.batch_size]
            self._data_buffer = self._data_buffer[self.config.batch_size:]

            df = self._create_dataframe(batch)

            if df is not None and len(df) > 0:
                await self._store_data(df)

        except Exception as e:
            logger.error(
                "Error processing batch",
                name=self.config.name,
                error=str(e)
            )

    def _create_dataframe(self, data: List[Dict[str, Any]]) -> Optional[pl.DataFrame]:
        """
        Create DataFrame from data list.

        Args:
            data: List of data dictionaries

        Returns:
            Polars DataFrame or None if creation failed
        """
        try:
            if not data:
                return None

            df = pl.DataFrame(data)
            return df

        except Exception as e:
            logger.error(
                "Error creating DataFrame",
                name=self.config.name,
                error=str(e)
            )
            return None

    @abstractmethod
    async def _fetch_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch data for a symbol.

        Must be implemented by subclasses.

        Args:
            symbol: Trading symbol

        Returns:
            Data dictionary or None if unavailable
        """
        pass

    @abstractmethod
    async def _connect(self) -> None:
        """
        Connect to data source.

        Must be implemented by subclasses.

        Raises:
            ConnectionError: If connection fails
        """
        pass

    @abstractmethod
    async def _disconnect(self) -> None:
        """
        Disconnect from data source.

        Must be implemented by subclasses.
        """
        pass

    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate collected data.

        Can be overridden by subclasses for custom validation.

        Args:
            data: Data dictionary to validate

        Returns:
            True if data is valid
        """
        try:
            required_fields = ['symbol', 'timestamp']
            for field in required_fields:
                if field not in data:
                    return False

            return True

        except Exception as e:
            logger.error(
                "Data validation error",
                name=self.config.name,
                error=str(e)
            )
            return False

    async def _store_data(self, df: pl.DataFrame) -> None:
        """
        Store collected data.

        Can be overridden by subclasses for custom storage logic.

        Args:
            df: DataFrame to store
        """
        logger.debug(
            "Data stored",
            name=self.config.name,
            rows=len(df)
        )

    def get_stats(self) -> Dict[str, int]:
        """
        Get collector statistics.

        Returns:
            Dictionary of statistics
        """
        return self._stats.copy()

    def is_running(self) -> bool:
        """
        Check if collector is running.

        Returns:
            True if collector is running
        """
        return self._running

    def is_connected(self) -> bool:
        """
        Check if collector is connected.

        Returns:
            True if collector is connected
        """
        return self._connected

    def get_buffer_size(self) -> int:
        """
        Get current buffer size.

        Returns:
            Number of items in buffer
        """
        return len(self._data_buffer)
