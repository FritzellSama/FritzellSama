"""
Data Replay - Simulate real-time data streaming for backtesting.

This module provides tools to replay historical data as if it were arriving
in real-time, enabling accurate backtesting of streaming strategies.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Any, Optional, AsyncIterator, Callable
from datetime import datetime, timezone, timedelta
import os

from structlog import get_logger
import polars as pl

logger = get_logger(__name__)


class DataReplayEngine:
    """Replays historical data to simulate real-time streaming.

    Attributes:
        config: Replay configuration from environment
        data_buffer: Buffer of historical data to replay
        replay_speed: Speed multiplier for replay
        current_position: Current position in data stream
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize data replay engine.

        Args:
            config: Optional configuration override

        Raises:
            ValueError: If configuration invalid
        """
        self.config: Dict[str, Any] = config or self._load_config()
        self.data_buffer: Optional[pl.DataFrame] = None
        self.replay_speed: Decimal = self.config['replay_speed']
        self.current_position: int = 0
        self._is_replaying: bool = False

        logger.info("DataReplayEngine initialized", replay_speed=str(self.replay_speed))

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables.

        Returns:
            Configuration dictionary
        """
        try:
            config = {
                'replay_speed': Decimal(os.getenv('REPLAY_SPEED', '1.0')),
                'buffer_size': int(os.getenv('REPLAY_BUFFER_SIZE', '10000')),
                'enable_jitter': os.getenv('REPLAY_ENABLE_JITTER', 'false').lower() == 'true',
                'jitter_max_ms': int(os.getenv('REPLAY_JITTER_MAX_MS', '100')),
                'emit_on_timestamp': os.getenv('REPLAY_EMIT_ON_TIMESTAMP', 'true').lower() == 'true',
            }

            logger.debug("Data replay config loaded", config=config)
            return config

        except Exception as e:
            logger.error("Failed to load config", error=str(e))
            raise ValueError(f"Configuration load failed: {e}")

    async def load_data(
        self,
        data: pl.DataFrame,
        timestamp_column: str = 'timestamp'
    ) -> None:
        """Load historical data for replay.

        Args:
            data: Historical data to replay
            timestamp_column: Name of timestamp column

        Raises:
            ValueError: If data validation fails

        Example:
            >>> await engine.load_data(historical_df)
        """
        try:
            logger.info("Loading data for replay", rows=data.height)

            if data.height == 0:
                raise ValueError("Cannot load empty data")

            if timestamp_column not in data.columns:
                raise ValueError(f"Missing timestamp column: {timestamp_column}")

            # Sort by timestamp
            self.data_buffer = data.sort(timestamp_column)
            self.current_position = 0

            logger.info("Data loaded successfully", rows=self.data_buffer.height)

        except Exception as e:
            logger.error("Failed to load data", error=str(e))
            raise

    async def replay_stream(
        self,
        callback: Callable[[Dict[str, Any]], None],
        start_index: int = 0,
        end_index: Optional[int] = None
    ) -> None:
        """Replay data as a stream, calling callback for each event.

        Args:
            callback: Function to call for each data event
            start_index: Starting index in data
            end_index: Ending index (None for end of data)

        Raises:
            ValueError: If data not loaded

        Example:
            >>> async def process_tick(data):
            ...     print(data)
            >>> await engine.replay_stream(process_tick)
        """
        try:
            if self.data_buffer is None:
                raise ValueError("No data loaded for replay")

            logger.info("Starting data replay", start=start_index, end=end_index)

            self._is_replaying = True
            self.current_position = start_index

            end = end_index if end_index is not None else self.data_buffer.height

            prev_timestamp: Optional[datetime] = None

            for i in range(start_index, end):
                if not self._is_replaying:
                    logger.info("Replay stopped")
                    break

                # Get current row
                row = self.data_buffer.row(i, named=True)
                current_timestamp = row['timestamp']

                # Calculate delay based on timestamp difference
                if prev_timestamp and self.config['emit_on_timestamp']:
                    delay = await self._calculate_delay(prev_timestamp, current_timestamp)
                    if delay > 0:
                        await asyncio.sleep(delay)

                # Emit event
                await self._emit_event(callback, row)

                prev_timestamp = current_timestamp
                self.current_position = i + 1

            self._is_replaying = False
            logger.info("Replay completed", events_replayed=end - start_index)

        except Exception as e:
            self._is_replaying = False
            logger.error("Replay failed", error=str(e))
            raise

    async def replay_iterator(
        self,
        start_index: int = 0,
        end_index: Optional[int] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Replay data as an async iterator.

        Args:
            start_index: Starting index
            end_index: Ending index (None for end)

        Yields:
            Data events

        Example:
            >>> async for event in engine.replay_iterator():
            ...     process(event)
        """
        try:
            if self.data_buffer is None:
                raise ValueError("No data loaded for replay")

            logger.info("Starting iterator replay", start=start_index, end=end_index)

            self._is_replaying = True
            self.current_position = start_index

            end = end_index if end_index is not None else self.data_buffer.height

            prev_timestamp: Optional[datetime] = None

            for i in range(start_index, end):
                if not self._is_replaying:
                    break

                row = self.data_buffer.row(i, named=True)
                current_timestamp = row['timestamp']

                # Calculate delay
                if prev_timestamp and self.config['emit_on_timestamp']:
                    delay = await self._calculate_delay(prev_timestamp, current_timestamp)
                    if delay > 0:
                        await asyncio.sleep(delay)

                # Yield event
                yield row

                prev_timestamp = current_timestamp
                self.current_position = i + 1

            self._is_replaying = False

        except Exception as e:
            self._is_replaying = False
            logger.error("Iterator replay failed", error=str(e))
            raise

    async def replay_batch(
        self,
        batch_size: int,
        start_index: int = 0
    ) -> AsyncIterator[pl.DataFrame]:
        """Replay data in batches.

        Args:
            batch_size: Number of events per batch
            start_index: Starting index

        Yields:
            DataFrames containing batches of events

        Example:
            >>> async for batch in engine.replay_batch(100):
            ...     process_batch(batch)
        """
        try:
            if self.data_buffer is None:
                raise ValueError("No data loaded for replay")

            logger.info("Starting batch replay", batch_size=batch_size, start=start_index)

            self._is_replaying = True
            self.current_position = start_index

            total_rows = self.data_buffer.height

            while self.current_position < total_rows and self._is_replaying:
                # Get next batch
                end_pos = min(self.current_position + batch_size, total_rows)
                batch = self.data_buffer.slice(self.current_position, end_pos - self.current_position)

                # Calculate delay based on timestamps
                if self.config['emit_on_timestamp'] and batch.height > 0:
                    first_ts = batch['timestamp'][0]
                    last_ts = batch['timestamp'][-1]

                    if isinstance(first_ts, datetime) and isinstance(last_ts, datetime):
                        time_diff = (last_ts - first_ts).total_seconds()
                        delay = float(Decimal(str(time_diff)) / self.replay_speed)
                        if delay > 0:
                            await asyncio.sleep(delay)

                # Yield batch
                yield batch

                self.current_position = end_pos

            self._is_replaying = False
            logger.info("Batch replay completed")

        except Exception as e:
            self._is_replaying = False
            logger.error("Batch replay failed", error=str(e))
            raise

    async def replay_time_window(
        self,
        start_time: datetime,
        end_time: datetime,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Replay data within a specific time window.

        Args:
            start_time: Window start timestamp
            end_time: Window end timestamp
            callback: Event callback function

        Example:
            >>> await engine.replay_time_window(
            ...     datetime(2025, 1, 1),
            ...     datetime(2025, 1, 2),
            ...     process_tick
            ... )
        """
        try:
            if self.data_buffer is None:
                raise ValueError("No data loaded for replay")

            logger.info(
                "Replaying time window",
                start=start_time.isoformat(),
                end=end_time.isoformat()
            )

            # Filter data to time window
            filtered = self.data_buffer.filter(
                (pl.col('timestamp') >= start_time) &
                (pl.col('timestamp') <= end_time)
            )

            if filtered.height == 0:
                logger.warning("No data in specified time window")
                return

            # Replay filtered data
            prev_timestamp: Optional[datetime] = None

            for row in filtered.iter_rows(named=True):
                current_timestamp = row['timestamp']

                # Calculate delay
                if prev_timestamp and self.config['emit_on_timestamp']:
                    delay = await self._calculate_delay(prev_timestamp, current_timestamp)
                    if delay > 0:
                        await asyncio.sleep(delay)

                # Emit event
                await self._emit_event(callback, row)

                prev_timestamp = current_timestamp

            logger.info("Time window replay completed", events=filtered.height)

        except Exception as e:
            logger.error("Time window replay failed", error=str(e))
            raise

    def stop_replay(self) -> None:
        """Stop ongoing replay."""
        self._is_replaying = False
        logger.info("Replay stop requested")

    def reset(self) -> None:
        """Reset replay position to beginning."""
        self.current_position = 0
        self._is_replaying = False
        logger.info("Replay reset")

    def set_speed(self, speed: Decimal) -> None:
        """Set replay speed multiplier.

        Args:
            speed: Speed multiplier (1.0 = real-time, 2.0 = 2x speed)

        Example:
            >>> engine.set_speed(Decimal('10.0'))  # 10x speed
        """
        try:
            if speed <= Decimal('0'):
                raise ValueError("Speed must be positive")

            self.replay_speed = speed
            logger.info("Replay speed set", speed=str(speed))

        except Exception as e:
            logger.error("Failed to set speed", error=str(e))
            raise

    async def _calculate_delay(
        self,
        prev_timestamp: datetime,
        current_timestamp: datetime
    ) -> float:
        """Calculate delay between events based on timestamps and replay speed.

        Args:
            prev_timestamp: Previous event timestamp
            current_timestamp: Current event timestamp

        Returns:
            Delay in seconds
        """
        try:
            # Calculate time difference
            time_diff = (current_timestamp - prev_timestamp).total_seconds()

            if time_diff <= 0:
                return 0.0

            # Apply replay speed
            delay = float(Decimal(str(time_diff)) / self.replay_speed)

            # Add jitter if enabled
            if self.config['enable_jitter']:
                import random
                jitter_ms = random.randint(0, self.config['jitter_max_ms'])
                delay += jitter_ms / 1000.0

            return delay

        except Exception as e:
            logger.error("Failed to calculate delay", error=str(e))
            return 0.0

    async def _emit_event(
        self,
        callback: Callable[[Dict[str, Any]], None],
        event_data: Dict[str, Any]
    ) -> None:
        """Emit event to callback.

        Args:
            callback: Callback function
            event_data: Event data to emit
        """
        try:
            # Convert Decimal values to strings for serialization
            serialized_event = {}
            for key, value in event_data.items():
                if isinstance(value, Decimal):
                    serialized_event[key] = str(value)
                else:
                    serialized_event[key] = value

            # Call callback (support both sync and async)
            if asyncio.iscoroutinefunction(callback):
                await callback(serialized_event)
            else:
                callback(serialized_event)

        except Exception as e:
            logger.error("Failed to emit event", error=str(e))
            raise

    def get_progress(self) -> Dict[str, Any]:
        """Get current replay progress.

        Returns:
            Progress information dictionary

        Example:
            >>> progress = engine.get_progress()
            >>> print(f"Progress: {progress['percent_complete']}%")
        """
        try:
            if self.data_buffer is None:
                return {
                    'current_position': 0,
                    'total_events': 0,
                    'percent_complete': Decimal('0'),
                    'is_replaying': self._is_replaying
                }

            total = self.data_buffer.height
            percent = (Decimal(self.current_position) / Decimal(total)) * Decimal('100') if total > 0 else Decimal('0')

            return {
                'current_position': self.current_position,
                'total_events': total,
                'percent_complete': percent,
                'is_replaying': self._is_replaying
            }

        except Exception as e:
            logger.error("Failed to get progress", error=str(e))
            return {
                'current_position': 0,
                'total_events': 0,
                'percent_complete': Decimal('0'),
                'is_replaying': False
            }
