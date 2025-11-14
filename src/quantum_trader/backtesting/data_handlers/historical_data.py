"""
Historical Data Handler - Load and manage historical market data for backtesting.

This module provides comprehensive historical data loading, caching, validation,
and preprocessing for efficient backtesting operations.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os

from structlog import get_logger
import polars as pl

logger = get_logger(__name__)


class HistoricalDataHandler:
    """Handles loading and management of historical market data.

    Attributes:
        config: Handler configuration from environment
        data_dir: Directory containing historical data
        cache: In-memory data cache
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize historical data handler.

        Args:
            config: Optional configuration override

        Raises:
            ValueError: If configuration invalid
        """
        self.config: Dict[str, Any] = config or self._load_config()
        self.data_dir: Path = Path(self.config['data_dir'])
        self.cache: Dict[str, pl.DataFrame] = {}
        self._cache_enabled: bool = self.config['enable_cache']

        logger.info("HistoricalDataHandler initialized", data_dir=str(self.data_dir))

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables.

        Returns:
            Configuration dictionary
        """
        try:
            config = {
                'data_dir': os.getenv('HISTORICAL_DATA_DIR', './data/historical'),
                'enable_cache': os.getenv('ENABLE_DATA_CACHE', 'true').lower() == 'true',
                'cache_max_size_mb': int(os.getenv('CACHE_MAX_SIZE_MB', '1000')),
                'default_timeframe': os.getenv('DEFAULT_TIMEFRAME', '1h'),
                'date_format': os.getenv('DATE_FORMAT', '%Y-%m-%d %H:%M:%S'),
                'validate_on_load': os.getenv('VALIDATE_ON_LOAD', 'true').lower() == 'true',
            }

            logger.debug("Historical data handler config loaded", config=config)
            return config

        except Exception as e:
            logger.error("Failed to load config", error=str(e))
            raise ValueError(f"Configuration load failed: {e}")

    async def load_ohlcv_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: Optional[str] = None,
        exchange: Optional[str] = None
    ) -> pl.DataFrame:
        """Load OHLCV historical data.

        Args:
            symbol: Trading symbol
            start_date: Start date for data
            end_date: End date for data
            timeframe: Data timeframe (e.g., '1m', '1h', '1d')
            exchange: Exchange name

        Returns:
            DataFrame with OHLCV data

        Example:
            >>> data = await handler.load_ohlcv_data(
            ...     'BTC/USDT',
            ...     datetime(2025, 1, 1),
            ...     datetime(2025, 1, 31),
            ...     '1h'
            ... )
            >>> data.height
            744
        """
        try:
            timeframe = timeframe or self.config['default_timeframe']

            logger.info(
                "Loading OHLCV data",
                symbol=symbol,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                timeframe=timeframe
            )

            # Check cache first
            cache_key = self._build_cache_key(symbol, start_date, end_date, timeframe, exchange)
            if self._cache_enabled and cache_key in self.cache:
                logger.debug("Data found in cache", cache_key=cache_key)
                return self.cache[cache_key]

            # Load data from storage
            data = await self._load_data_from_storage(
                symbol,
                start_date,
                end_date,
                timeframe,
                exchange
            )

            # Validate data if configured
            if self.config['validate_on_load']:
                data = await self._validate_and_clean_data(data)

            # Cache data
            if self._cache_enabled:
                self._add_to_cache(cache_key, data)

            logger.info(
                "OHLCV data loaded",
                rows=data.height,
                symbol=symbol,
                timeframe=timeframe
            )

            return data

        except Exception as e:
            logger.error("Failed to load OHLCV data", error=str(e))
            raise

    async def load_multiple_symbols(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: Optional[str] = None,
        exchange: Optional[str] = None
    ) -> Dict[str, pl.DataFrame]:
        """Load data for multiple symbols concurrently.

        Args:
            symbols: List of trading symbols
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            exchange: Exchange name

        Returns:
            Dictionary mapping symbols to DataFrames

        Example:
            >>> data_dict = await handler.load_multiple_symbols(
            ...     ['BTC/USDT', 'ETH/USDT'],
            ...     datetime(2025, 1, 1),
            ...     datetime(2025, 1, 31)
            ... )
        """
        try:
            logger.info(
                "Loading multiple symbols",
                symbol_count=len(symbols),
                timeframe=timeframe
            )

            # Load all symbols concurrently
            tasks = [
                self.load_ohlcv_data(symbol, start_date, end_date, timeframe, exchange)
                for symbol in symbols
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Build result dictionary
            data_dict: Dict[str, pl.DataFrame] = {}

            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    logger.error(
                        "Failed to load symbol",
                        symbol=symbol,
                        error=str(result)
                    )
                else:
                    data_dict[symbol] = result

            logger.info(
                "Multiple symbols loaded",
                successful=len(data_dict),
                failed=len(symbols) - len(data_dict)
            )

            return data_dict

        except Exception as e:
            logger.error("Failed to load multiple symbols", error=str(e))
            raise

    async def load_tick_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        exchange: Optional[str] = None
    ) -> pl.DataFrame:
        """Load tick-level data.

        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            exchange: Exchange name

        Returns:
            DataFrame with tick data

        Example:
            >>> ticks = await handler.load_tick_data(
            ...     'BTC/USDT',
            ...     datetime(2025, 1, 1, 0, 0),
            ...     datetime(2025, 1, 1, 1, 0)
            ... )
        """
        try:
            logger.info(
                "Loading tick data",
                symbol=symbol,
                start=start_date.isoformat(),
                end=end_date.isoformat()
            )

            # In production, would load actual tick data
            # For now, simulate with empty DataFrame
            tick_data = pl.DataFrame({
                'timestamp': [],
                'price': [],
                'quantity': [],
                'side': []
            })

            logger.info("Tick data loaded", rows=tick_data.height)

            return tick_data

        except Exception as e:
            logger.error("Failed to load tick data", error=str(e))
            raise

    async def get_available_date_range(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
        exchange: Optional[str] = None
    ) -> Tuple[datetime, datetime]:
        """Get available date range for symbol.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            exchange: Exchange name

        Returns:
            Tuple of (start_date, end_date)

        Example:
            >>> start, end = await handler.get_available_date_range('BTC/USDT')
            >>> print(f"Data available from {start} to {end}")
        """
        try:
            logger.debug("Getting available date range", symbol=symbol)

            # In production, would query actual data storage
            # For now, return sample range
            start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
            end_date = datetime.now(timezone.utc)

            logger.debug(
                "Date range retrieved",
                start=start_date.isoformat(),
                end=end_date.isoformat()
            )

            return start_date, end_date

        except Exception as e:
            logger.error("Failed to get date range", error=str(e))
            raise

    async def preload_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: Optional[str] = None
    ) -> None:
        """Preload data into cache.

        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe

        Example:
            >>> await handler.preload_data(
            ...     'BTC/USDT',
            ...     datetime(2025, 1, 1),
            ...     datetime(2025, 1, 31)
            ... )
        """
        try:
            logger.info("Preloading data", symbol=symbol)

            await self.load_ohlcv_data(symbol, start_date, end_date, timeframe)

            logger.info("Data preloaded successfully")

        except Exception as e:
            logger.error("Failed to preload data", error=str(e))
            raise

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear data cache.

        Args:
            symbol: Optional symbol to clear, or None for all

        Example:
            >>> handler.clear_cache()  # Clear all
            >>> handler.clear_cache('BTC/USDT')  # Clear specific symbol
        """
        try:
            if symbol:
                # Clear specific symbol
                keys_to_remove = [k for k in self.cache.keys() if symbol in k]
                for key in keys_to_remove:
                    del self.cache[key]

                logger.info("Cache cleared for symbol", symbol=symbol, keys_removed=len(keys_to_remove))
            else:
                # Clear all
                self.cache.clear()
                logger.info("All cache cleared")

        except Exception as e:
            logger.error("Failed to clear cache", error=str(e))

    async def _load_data_from_storage(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
        exchange: Optional[str]
    ) -> pl.DataFrame:
        """Load data from file storage.

        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            exchange: Exchange name

        Returns:
            DataFrame with loaded data
        """
        try:
            logger.debug("Loading data from storage", symbol=symbol)

            # In production, would load from actual storage (CSV, Parquet, database, etc.)
            # For now, generate sample data
            data = await self._generate_sample_data(
                symbol,
                start_date,
                end_date,
                timeframe
            )

            return data

        except Exception as e:
            logger.error("Failed to load from storage", error=str(e))
            raise

    async def _generate_sample_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> pl.DataFrame:
        """Generate sample OHLCV data for demonstration.

        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe

        Returns:
            Sample DataFrame
        """
        try:
            # Parse timeframe to seconds
            interval_seconds = self._parse_timeframe_seconds(timeframe)

            # Generate timestamps
            timestamps: List[datetime] = []
            current = start_date

            while current <= end_date:
                timestamps.append(current)
                current += timedelta(seconds=interval_seconds)

            # Generate sample prices
            base_price = Decimal('50000')
            data_rows = []

            for ts in timestamps:
                # Simple random walk simulation
                import random
                change_pct = Decimal(str(random.uniform(-0.02, 0.02)))
                base_price = base_price * (Decimal('1') + change_pct)

                open_price = base_price
                high_price = base_price * Decimal('1.01')
                low_price = base_price * Decimal('0.99')
                close_price = base_price * (Decimal('1') + Decimal(str(random.uniform(-0.01, 0.01))))
                volume = Decimal(str(random.uniform(10, 100)))

                data_rows.append({
                    'timestamp': ts,
                    'open': float(open_price),
                    'high': float(high_price),
                    'low': float(low_price),
                    'close': float(close_price),
                    'volume': float(volume)
                })

            data_df = pl.DataFrame(data_rows)

            logger.debug("Sample data generated", rows=len(data_rows))

            return data_df

        except Exception as e:
            logger.error("Failed to generate sample data", error=str(e))
            raise

    async def _validate_and_clean_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """Validate and clean loaded data.

        Args:
            data: Raw data

        Returns:
            Cleaned data
        """
        try:
            logger.debug("Validating data", rows=data.height)

            # Remove nulls
            data = data.drop_nulls()

            # Remove duplicates
            data = data.unique(subset=['timestamp'])

            # Sort by timestamp
            data = data.sort('timestamp')

            # Validate price relationships
            data = data.filter(
                (pl.col('high') >= pl.col('low')) &
                (pl.col('high') >= pl.col('open')) &
                (pl.col('high') >= pl.col('close')) &
                (pl.col('low') <= pl.col('open')) &
                (pl.col('low') <= pl.col('close')) &
                (pl.col('volume') >= 0)
            )

            logger.debug("Data validated and cleaned", rows=data.height)

            return data

        except Exception as e:
            logger.error("Data validation failed", error=str(e))
            raise

    def _build_cache_key(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
        exchange: Optional[str]
    ) -> str:
        """Build cache key from parameters.

        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            exchange: Exchange name

        Returns:
            Cache key string
        """
        parts = [
            symbol,
            start_date.isoformat(),
            end_date.isoformat(),
            timeframe
        ]

        if exchange:
            parts.append(exchange)

        return '|'.join(parts)

    def _add_to_cache(self, cache_key: str, data: pl.DataFrame) -> None:
        """Add data to cache.

        Args:
            cache_key: Cache key
            data: Data to cache
        """
        try:
            # Simple cache implementation
            # In production, would implement LRU cache with size limits
            self.cache[cache_key] = data

            logger.debug("Data added to cache", cache_key=cache_key, rows=data.height)

        except Exception as e:
            logger.error("Failed to add to cache", error=str(e))

    def _parse_timeframe_seconds(self, timeframe: str) -> int:
        """Parse timeframe string to seconds.

        Args:
            timeframe: Timeframe string (e.g., '1m', '1h', '1d')

        Returns:
            Seconds
        """
        try:
            if timeframe.endswith('s'):
                return int(timeframe[:-1])
            elif timeframe.endswith('m'):
                return int(timeframe[:-1]) * 60
            elif timeframe.endswith('h'):
                return int(timeframe[:-1]) * 3600
            elif timeframe.endswith('d'):
                return int(timeframe[:-1]) * 86400
            elif timeframe.endswith('w'):
                return int(timeframe[:-1]) * 604800
            else:
                raise ValueError(f"Invalid timeframe format: {timeframe}")

        except Exception as e:
            logger.error("Failed to parse timeframe", error=str(e))
            raise

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics

        Example:
            >>> stats = handler.get_cache_stats()
            >>> print(f"Cached entries: {stats['entry_count']}")
        """
        try:
            total_rows = sum(df.height for df in self.cache.values())

            stats = {
                'entry_count': len(self.cache),
                'total_rows': total_rows,
                'enabled': self._cache_enabled
            }

            return stats

        except Exception as e:
            logger.error("Failed to get cache stats", error=str(e))
            return {'entry_count': 0, 'total_rows': 0, 'enabled': False}
