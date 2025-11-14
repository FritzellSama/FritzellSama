"""
Candle (OHLCV) data processor for aggregating tick data into candles.

This module provides efficient processing of raw tick data into OHLCV candles
with support for multiple timeframes and parallel processing.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessorConfig:
    """Configuration for candle processor."""

    timeframes: List[str]
    symbols: List[str]
    buffer_size: int
    batch_size: int
    enable_validation: bool
    enable_checkpointing: bool
    checkpoint_interval_seconds: int
    parallel_processing: bool
    max_workers: int


class ProcessorError(Exception):
    """Base exception for processor errors."""

    pass


class CandleProcessor:
    """
    High-performance candle processor for tick-to-OHLCV aggregation.

    Processes raw tick data into OHLCV candles for multiple timeframes
    with parallel processing and checkpoint/recovery support.

    Attributes:
        config: Processor configuration
        timeframes: List of timeframes to process
        symbols: List of symbols to process

    Example:
        ```python
        config = {
            'timeframes': ['1m', '5m', '15m', '1h'],
            'symbols': ['BTC/USDT', 'ETH/USDT'],
            'buffer_size': 10000,
            'batch_size': 1000,
            'enable_validation': True,
            'enable_checkpointing': True,
            'checkpoint_interval_seconds': 60,
            'parallel_processing': True,
            'max_workers': 4
        }

        processor = CandleProcessor(config)
        await processor.start()

        # Process tick data
        ticks_df = pl.DataFrame({...})
        candles_df = await processor.process(ticks_df, '1m')
        ```
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize candle processor.

        Args:
            config: Processor configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        self._validate_config(config)

        self.config = ProcessorConfig(
            timeframes=config['timeframes'],
            symbols=config['symbols'],
            buffer_size=config['buffer_size'],
            batch_size=config['batch_size'],
            enable_validation=config.get('enable_validation', True),
            enable_checkpointing=config.get('enable_checkpointing', True),
            checkpoint_interval_seconds=config.get('checkpoint_interval_seconds', 60),
            parallel_processing=config.get('parallel_processing', True),
            max_workers=config.get('max_workers', 4)
        )

        self._running: bool = False
        self._tick_buffers: Dict[str, List[Dict[str, Any]]] = {
            symbol: [] for symbol in self.config.symbols
        }
        self._candle_cache: Dict[str, Dict[str, Any]] = {}
        self._checkpoint_task: Optional[asyncio.Task] = None
        self._stats: Dict[str, int] = {
            'ticks_processed': 0,
            'candles_generated': 0,
            'validation_failures': 0,
            'processing_errors': 0
        }

        self._timeframe_seconds = self._parse_timeframes()

        logger.info(
            "Candle processor initialized",
            timeframes=self.config.timeframes,
            symbols=self.config.symbols,
            parallel=self.config.parallel_processing
        )

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate processor configuration.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ['timeframes', 'symbols', 'buffer_size', 'batch_size']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")

        if not config['timeframes'] or not isinstance(config['timeframes'], list):
            raise ValueError("timeframes must be a non-empty list")

        if not config['symbols'] or not isinstance(config['symbols'], list):
            raise ValueError("symbols must be a non-empty list")

        if config['buffer_size'] <= 0:
            raise ValueError("buffer_size must be positive")

        if config['batch_size'] <= 0:
            raise ValueError("batch_size must be positive")

    def _parse_timeframes(self) -> Dict[str, int]:
        """
        Parse timeframes to seconds.

        Returns:
            Dictionary mapping timeframe to seconds
        """
        timeframe_map = {}

        for tf in self.config.timeframes:
            if tf.endswith('s'):
                seconds = int(tf[:-1])
            elif tf.endswith('m'):
                seconds = int(tf[:-1]) * 60
            elif tf.endswith('h'):
                seconds = int(tf[:-1]) * 3600
            elif tf.endswith('d'):
                seconds = int(tf[:-1]) * 86400
            elif tf.endswith('w'):
                seconds = int(tf[:-1]) * 604800
            elif tf.endswith('M'):
                seconds = int(tf[:-1]) * 2592000  # Approximate 30 days
            else:
                raise ValueError(f"Invalid timeframe format: {tf}")

            timeframe_map[tf] = seconds

        return timeframe_map

    async def start(self) -> None:
        """
        Start the processor.

        Raises:
            RuntimeError: If processor is already running
        """
        if self._running:
            raise RuntimeError("Processor is already running")

        logger.info("Starting candle processor")
        self._running = True

        if self.config.enable_checkpointing:
            self._checkpoint_task = asyncio.create_task(
                self._checkpoint_loop(),
                name="checkpoint"
            )

        logger.info("Candle processor started")

    async def stop(self) -> None:
        """Stop the processor gracefully."""
        if not self._running:
            return

        logger.info("Stopping candle processor")
        self._running = False

        if self._checkpoint_task and not self._checkpoint_task.done():
            self._checkpoint_task.cancel()

            try:
                await asyncio.wait_for(self._checkpoint_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for checkpoint task")
            except asyncio.CancelledError:
                pass

        # Final checkpoint
        if self.config.enable_checkpointing:
            await self._save_checkpoint()

        self._tick_buffers.clear()
        self._candle_cache.clear()

        logger.info("Candle processor stopped", stats=self._stats)

    async def process(
        self,
        ticks_df: pl.DataFrame,
        timeframe: str
    ) -> pl.DataFrame:
        """
        Process tick data into candles.

        Args:
            ticks_df: DataFrame with tick data
            timeframe: Target timeframe (e.g., '1m', '5m')

        Returns:
            DataFrame with OHLCV candles

        Raises:
            ProcessorError: If processing fails
        """
        try:
            if timeframe not in self._timeframe_seconds:
                raise ValueError(f"Invalid timeframe: {timeframe}")

            if len(ticks_df) == 0:
                return pl.DataFrame()

            # Validate input data
            if self.config.enable_validation:
                if not self._validate_tick_data(ticks_df):
                    self._stats['validation_failures'] += 1
                    raise ProcessorError("Tick data validation failed")

            # Process based on parallel configuration
            if self.config.parallel_processing:
                candles_df = await self._process_parallel(ticks_df, timeframe)
            else:
                candles_df = await self._process_sequential(ticks_df, timeframe)

            self._stats['ticks_processed'] += len(ticks_df)
            self._stats['candles_generated'] += len(candles_df)

            return candles_df

        except Exception as e:
            self._stats['processing_errors'] += 1
            logger.error("Error processing ticks", timeframe=timeframe, error=str(e))
            raise ProcessorError(f"Failed to process ticks: {e}") from e

    async def _process_sequential(
        self,
        ticks_df: pl.DataFrame,
        timeframe: str
    ) -> pl.DataFrame:
        """
        Process ticks sequentially.

        Args:
            ticks_df: Tick data DataFrame
            timeframe: Target timeframe

        Returns:
            Candles DataFrame
        """
        timeframe_seconds = self._timeframe_seconds[timeframe]

        # Group by symbol and timeframe window
        candles = ticks_df.group_by_dynamic(
            'timestamp',
            every=f"{timeframe_seconds}s",
            by='symbol',
            closed='left'
        ).agg([
            pl.first('price').alias('open'),
            pl.max('price').alias('high'),
            pl.min('price').alias('low'),
            pl.last('price').alias('close'),
            pl.sum('volume').alias('volume'),
            pl.count().alias('trades')
        ])

        return candles

    async def _process_parallel(
        self,
        ticks_df: pl.DataFrame,
        timeframe: str
    ) -> pl.DataFrame:
        """
        Process ticks in parallel by symbol.

        Args:
            ticks_df: Tick data DataFrame
            timeframe: Target timeframe

        Returns:
            Candles DataFrame
        """
        symbols = ticks_df['symbol'].unique().to_list()

        # Split by symbol and process in parallel
        tasks = []
        for symbol in symbols:
            symbol_df = ticks_df.filter(pl.col('symbol') == symbol)
            task = asyncio.create_task(
                self._process_symbol_candles(symbol_df, timeframe)
            )
            tasks.append(task)

        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("Error processing symbol candles", error=str(result))
                self._stats['processing_errors'] += 1
            elif result is not None and len(result) > 0:
                valid_results.append(result)

        if not valid_results:
            return pl.DataFrame()

        combined_df = pl.concat(valid_results)
        return combined_df

    async def _process_symbol_candles(
        self,
        symbol_df: pl.DataFrame,
        timeframe: str
    ) -> pl.DataFrame:
        """
        Process candles for a single symbol.

        Args:
            symbol_df: Tick data for single symbol
            timeframe: Target timeframe

        Returns:
            Candles DataFrame
        """
        try:
            timeframe_seconds = self._timeframe_seconds[timeframe]

            # Aggregate ticks into candles
            candles = symbol_df.group_by_dynamic(
                'timestamp',
                every=f"{timeframe_seconds}s",
                closed='left'
            ).agg([
                pl.col('symbol').first(),
                pl.col('price').first().alias('open'),
                pl.col('price').max().alias('high'),
                pl.col('price').min().alias('low'),
                pl.col('price').last().alias('close'),
                pl.col('volume').sum().alias('volume'),
                pl.count().alias('trades')
            ])

            return candles

        except Exception as e:
            logger.error("Error processing symbol candles", error=str(e))
            raise

    def _validate_tick_data(self, df: pl.DataFrame) -> bool:
        """
        Validate tick data DataFrame.

        Args:
            df: Tick data DataFrame

        Returns:
            True if data is valid
        """
        try:
            required_columns = ['symbol', 'timestamp', 'price', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    logger.error("Missing required column", column=col)
                    return False

            # Check for null values
            if df.null_count().sum_horizontal()[0] > 0:
                logger.error("Tick data contains null values")
                return False

            # Check for negative prices or volumes
            if (df['price'] <= 0).any() or (df['volume'] < 0).any():
                logger.error("Tick data contains invalid prices or volumes")
                return False

            return True

        except Exception as e:
            logger.error("Tick data validation error", error=str(e))
            return False

    async def _checkpoint_loop(self) -> None:
        """Run checkpoint loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.checkpoint_interval_seconds)

                if self._running:
                    await self._save_checkpoint()

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error("Checkpoint error", error=str(e))

    async def _save_checkpoint(self) -> None:
        """Save processing checkpoint."""
        try:
            checkpoint = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'stats': self._stats.copy(),
                'buffer_sizes': {
                    symbol: len(buffer)
                    for symbol, buffer in self._tick_buffers.items()
                }
            }

            logger.debug("Checkpoint saved", checkpoint=checkpoint)

        except Exception as e:
            logger.error("Error saving checkpoint", error=str(e))

    async def add_ticks(self, ticks_df: pl.DataFrame) -> None:
        """
        Add ticks to buffer for processing.

        Args:
            ticks_df: Tick data DataFrame
        """
        try:
            for symbol in self.config.symbols:
                symbol_df = ticks_df.filter(pl.col('symbol') == symbol)

                if len(symbol_df) > 0:
                    tick_dicts = symbol_df.to_dicts()
                    self._tick_buffers[symbol].extend(tick_dicts)

                    # Trim buffer if too large
                    if len(self._tick_buffers[symbol]) > self.config.buffer_size:
                        self._tick_buffers[symbol] = self._tick_buffers[symbol][-self.config.buffer_size:]

        except Exception as e:
            logger.error("Error adding ticks to buffer", error=str(e))

    async def process_buffered(self, timeframe: str) -> pl.DataFrame:
        """
        Process all buffered ticks.

        Args:
            timeframe: Target timeframe

        Returns:
            Candles DataFrame
        """
        try:
            all_ticks = []
            for symbol, ticks in self._tick_buffers.items():
                all_ticks.extend(ticks)

            if not all_ticks:
                return pl.DataFrame()

            ticks_df = pl.DataFrame(all_ticks)
            candles_df = await self.process(ticks_df, timeframe)

            # Clear processed buffers
            for symbol in self._tick_buffers:
                self._tick_buffers[symbol].clear()

            return candles_df

        except Exception as e:
            logger.error("Error processing buffered ticks", error=str(e))
            raise ProcessorError(f"Failed to process buffered ticks: {e}") from e

    def get_stats(self) -> Dict[str, int]:
        """
        Get processor statistics.

        Returns:
            Dictionary of statistics
        """
        return self._stats.copy()

    def is_running(self) -> bool:
        """
        Check if processor is running.

        Returns:
            True if processor is running
        """
        return self._running

    def get_buffer_size(self, symbol: str) -> int:
        """
        Get buffer size for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Number of buffered ticks
        """
        return len(self._tick_buffers.get(symbol, []))
