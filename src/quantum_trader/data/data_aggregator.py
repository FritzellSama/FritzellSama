"""
Data aggregator for combining and normalizing data from multiple sources.

This module provides a unified interface for aggregating market data from
multiple exchanges and data feeds with normalization and deduplication.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set, Callable
from datetime import datetime, timezone
from dataclasses import dataclass
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class AggregatorConfig:
    """Configuration for data aggregator."""

    sources: List[str]
    symbols: List[str]
    aggregation_interval_ms: int
    enable_deduplication: bool
    enable_normalization: bool
    enable_validation: bool
    buffer_size: int
    priority_order: List[str]
    fallback_enabled: bool


class AggregatorError(Exception):
    """Base exception for aggregator errors."""

    pass


class DataAggregator:
    """
    Data aggregator for multi-source market data.

    Combines and normalizes market data from multiple sources with
    deduplication, validation, and automatic failover.

    Attributes:
        config: Aggregator configuration
        sources: List of data source names
        symbols: List of trading symbols

    Example:
        ```python
        config = {
            'sources': ['binance', 'bybit', 'okx'],
            'symbols': ['BTC/USDT', 'ETH/USDT'],
            'aggregation_interval_ms': 100,
            'enable_deduplication': True,
            'enable_normalization': True,
            'enable_validation': True,
            'buffer_size': 1000,
            'priority_order': ['binance', 'bybit', 'okx'],
            'fallback_enabled': True
        }

        aggregator = DataAggregator(config)
        await aggregator.start()

        # Register data handler
        async def handle_data(df: pl.DataFrame):
            print(f"Received {len(df)} aggregated updates")

        aggregator.on_data(handle_data)
        ```
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize data aggregator.

        Args:
            config: Aggregator configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        self._validate_config(config)

        self.config = AggregatorConfig(
            sources=config['sources'],
            symbols=config['symbols'],
            aggregation_interval_ms=config['aggregation_interval_ms'],
            enable_deduplication=config.get('enable_deduplication', True),
            enable_normalization=config.get('enable_normalization', True),
            enable_validation=config.get('enable_validation', True),
            buffer_size=config.get('buffer_size', 1000),
            priority_order=config.get('priority_order', config['sources']),
            fallback_enabled=config.get('fallback_enabled', True)
        )

        self._running: bool = False
        self._data_buffers: Dict[str, List[Dict[str, Any]]] = {
            source: [] for source in self.config.sources
        }
        self._data_handlers: List[Callable[[pl.DataFrame], Any]] = []
        self._active_sources: Set[str] = set()
        self._aggregation_task: Optional[asyncio.Task] = None
        self._stats: Dict[str, int] = {
            'total_aggregated': 0,
            'duplicates_removed': 0,
            'validation_failures': 0,
            'normalization_errors': 0
        }

        logger.info(
            "Data aggregator initialized",
            sources=self.config.sources,
            symbols=self.config.symbols,
            interval_ms=self.config.aggregation_interval_ms
        )

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate aggregator configuration.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ['sources', 'symbols', 'aggregation_interval_ms']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")

        if not config['sources'] or not isinstance(config['sources'], list):
            raise ValueError("sources must be a non-empty list")

        if not config['symbols'] or not isinstance(config['symbols'], list):
            raise ValueError("symbols must be a non-empty list")

        if config['aggregation_interval_ms'] <= 0:
            raise ValueError("aggregation_interval_ms must be positive")

    async def start(self) -> None:
        """
        Start the aggregator.

        Raises:
            RuntimeError: If aggregator is already running
        """
        if self._running:
            raise RuntimeError("Aggregator is already running")

        logger.info("Starting data aggregator")
        self._running = True

        # Mark all sources as active
        self._active_sources = set(self.config.sources)

        # Start aggregation task
        self._aggregation_task = asyncio.create_task(
            self._aggregation_loop(),
            name="aggregation"
        )

        logger.info("Data aggregator started")

    async def stop(self) -> None:
        """Stop the aggregator gracefully."""
        if not self._running:
            return

        logger.info("Stopping data aggregator")
        self._running = False

        if self._aggregation_task and not self._aggregation_task.done():
            self._aggregation_task.cancel()

            try:
                await asyncio.wait_for(self._aggregation_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for aggregation task")
            except asyncio.CancelledError:
                pass

        for source in self._data_buffers:
            self._data_buffers[source].clear()

        self._active_sources.clear()

        logger.info("Data aggregator stopped", stats=self._stats)

    def on_data(self, handler: Callable[[pl.DataFrame], Any]) -> None:
        """
        Register data handler callback.

        Args:
            handler: Callback function receiving aggregated DataFrame
        """
        self._data_handlers.append(handler)

    async def add_data(self, source: str, data: Dict[str, Any]) -> None:
        """
        Add data from a source.

        Args:
            source: Data source name
            data: Data dictionary

        Raises:
            ValueError: If source is invalid
        """
        if source not in self.config.sources:
            raise ValueError(f"Invalid source: {source}")

        try:
            # Validate data if enabled
            if self.config.enable_validation:
                if not self._validate_data(data):
                    self._stats['validation_failures'] += 1
                    logger.warning("Data validation failed", source=source)
                    return

            # Normalize data if enabled
            if self.config.enable_normalization:
                data = self._normalize_data(source, data)

            # Add to buffer
            if len(self._data_buffers[source]) >= self.config.buffer_size:
                self._data_buffers[source].pop(0)

            self._data_buffers[source].append(data)

        except Exception as e:
            logger.error("Error adding data", source=source, error=str(e))

    async def _aggregation_loop(self) -> None:
        """Run aggregation loop."""
        while self._running:
            try:
                # Collect all buffered data
                all_data = []
                for source in self.config.sources:
                    if source in self._active_sources:
                        all_data.extend(self._data_buffers[source].copy())
                        self._data_buffers[source].clear()

                if all_data:
                    # Create DataFrame
                    df = pl.DataFrame(all_data)

                    # Deduplicate if enabled
                    if self.config.enable_deduplication:
                        original_len = len(df)
                        df = self._deduplicate(df)
                        self._stats['duplicates_removed'] += original_len - len(df)

                    # Sort by timestamp
                    if 'timestamp' in df.columns:
                        df = df.sort('timestamp')

                    # Emit aggregated data
                    if len(df) > 0:
                        self._emit_data(df)
                        self._stats['total_aggregated'] += len(df)

                # Wait for next interval
                await asyncio.sleep(self.config.aggregation_interval_ms / 1000.0)

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error("Aggregation error", error=str(e))

    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate data.

        Args:
            data: Data dictionary

        Returns:
            True if data is valid
        """
        try:
            required_fields = ['symbol', 'timestamp']
            for field in required_fields:
                if field not in data:
                    return False

            # Check timestamp is recent
            if isinstance(data['timestamp'], datetime):
                age = (datetime.now(timezone.utc) - data['timestamp']).total_seconds()
                max_age = int(60)  # Configurable via env
                if age > max_age:
                    logger.warning("Data too old", age=age)
                    return False

            return True

        except Exception as e:
            logger.error("Data validation error", error=str(e))
            return False

    def _normalize_data(self, source: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize data from different sources.

        Args:
            source: Data source name
            data: Raw data dictionary

        Returns:
            Normalized data dictionary
        """
        try:
            normalized = data.copy()

            # Add source metadata
            normalized['source'] = source

            # Normalize timestamp
            if 'timestamp' in normalized and not isinstance(normalized['timestamp'], datetime):
                if isinstance(normalized['timestamp'], (int, float)):
                    normalized['timestamp'] = datetime.fromtimestamp(
                        normalized['timestamp'],
                        tz=timezone.utc
                    )

            # Normalize price fields to Decimal strings
            price_fields = ['price', 'bid', 'ask', 'last', 'open', 'high', 'low', 'close']
            for field in price_fields:
                if field in normalized and not isinstance(normalized[field], str):
                    normalized[field] = str(Decimal(str(normalized[field])))

            # Normalize volume fields
            volume_fields = ['volume', 'quantity', 'amount']
            for field in volume_fields:
                if field in normalized and not isinstance(normalized[field], str):
                    normalized[field] = str(Decimal(str(normalized[field])))

            return normalized

        except Exception as e:
            self._stats['normalization_errors'] += 1
            logger.error("Normalization error", source=source, error=str(e))
            return data

    def _deduplicate(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Remove duplicate data points.

        Args:
            df: Input DataFrame

        Returns:
            Deduplicated DataFrame
        """
        try:
            # Define deduplication columns based on available columns
            dedup_cols = ['symbol', 'timestamp']

            # Add price columns if available
            price_cols = ['price', 'last', 'close']
            for col in price_cols:
                if col in df.columns:
                    dedup_cols.append(col)
                    break

            # Deduplicate keeping first occurrence
            deduplicated = df.unique(subset=dedup_cols, keep='first')

            return deduplicated

        except Exception as e:
            logger.error("Deduplication error", error=str(e))
            return df

    def _emit_data(self, df: pl.DataFrame) -> None:
        """
        Emit aggregated data to handlers.

        Args:
            df: Aggregated DataFrame
        """
        for handler in self._data_handlers:
            try:
                result = handler(df)
                # Handle async handlers
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)

            except Exception as e:
                logger.error("Error in data handler", error=str(e))

    def set_source_active(self, source: str, active: bool) -> None:
        """
        Set source active/inactive status.

        Args:
            source: Data source name
            active: True to activate, False to deactivate

        Raises:
            ValueError: If source is invalid
        """
        if source not in self.config.sources:
            raise ValueError(f"Invalid source: {source}")

        if active:
            self._active_sources.add(source)
            logger.info("Source activated", source=source)
        else:
            self._active_sources.discard(source)
            logger.info("Source deactivated", source=source)

    def get_active_sources(self) -> List[str]:
        """
        Get list of active sources.

        Returns:
            List of active source names
        """
        return list(self._active_sources)

    def get_priority_source(self, symbol: str) -> Optional[str]:
        """
        Get priority source for symbol based on configuration.

        Args:
            symbol: Trading symbol

        Returns:
            Priority source name or None
        """
        for source in self.config.priority_order:
            if source in self._active_sources:
                return source
        return None

    def get_stats(self) -> Dict[str, int]:
        """
        Get aggregator statistics.

        Returns:
            Dictionary of statistics
        """
        stats = self._stats.copy()
        stats['active_sources'] = len(self._active_sources)
        stats['total_sources'] = len(self.config.sources)
        return stats

    def is_running(self) -> bool:
        """
        Check if aggregator is running.

        Returns:
            True if aggregator is running
        """
        return self._running

    def get_buffer_size(self, source: str) -> int:
        """
        Get buffer size for source.

        Args:
            source: Data source name

        Returns:
            Number of buffered items

        Raises:
            ValueError: If source is invalid
        """
        if source not in self.config.sources:
            raise ValueError(f"Invalid source: {source}")

        return len(self._data_buffers[source])
