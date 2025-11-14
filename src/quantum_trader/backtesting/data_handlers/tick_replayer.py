"""Tick Data Replayer for High-Fidelity Backtesting.

Replays individual tick-level market data for ultra-high frequency
trading strategy backtesting.
"""

import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Iterator, Tuple
from dataclasses import dataclass
from enum import Enum
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class TickType(Enum):
    """Type of market tick."""
    TRADE = "TRADE"
    QUOTE = "QUOTE"
    ORDERBOOK_UPDATE = "ORDERBOOK_UPDATE"


@dataclass
class Tick:
    """Individual market tick.

    Attributes:
        timestamp: Tick timestamp with microsecond precision
        symbol: Trading symbol
        tick_type: Type of tick
        price: Trade price (for TRADE ticks)
        quantity: Trade quantity (for TRADE ticks)
        bid_price: Best bid price (for QUOTE ticks)
        ask_price: Best ask price (for QUOTE ticks)
        bid_size: Bid size (for QUOTE ticks)
        ask_size: Ask size (for QUOTE ticks)
        sequence: Sequence number for ordering
        metadata: Additional tick data
    """
    timestamp: datetime
    symbol: str
    tick_type: TickType
    price: Optional[Decimal] = None
    quantity: Optional[Decimal] = None
    bid_price: Optional[Decimal] = None
    ask_price: Optional[Decimal] = None
    bid_size: Optional[Decimal] = None
    ask_size: Optional[Decimal] = None
    sequence: int = 0
    metadata: Dict[str, Any] = None


class TickReplayer:
    """Replay tick-level market data for backtesting.

    Provides precise tick-by-tick replay of historical market data
    for high-frequency trading strategy testing.

    Attributes:
        config: Replayer configuration from config files
        symbol: Trading symbol
        tick_data: Cached tick data
        current_index: Current replay position
    """

    def __init__(self, config: Dict[str, Any], symbol: str) -> None:
        """Initialize tick replayer.

        Args:
            config: Configuration dictionary
            symbol: Trading symbol

        Raises:
            ValueError: If configuration is invalid

        Example:
            >>> config = {
            ...     "replay_speed": 1.0,
            ...     "enable_microsecond_precision": True
            ... }
            >>> replayer = TickReplayer(config, "BTC/USDT")
        """
        self.config = config
        self.symbol = symbol
        self._validate_config()

        self.replay_speed = Decimal(str(config.get("replay_speed", 1.0)))
        self.enable_microsecond_precision = config.get("enable_microsecond_precision", True)
        self.buffer_size = config.get("buffer_size", 10000)

        # Data storage
        self.tick_data: List[Tick] = []
        self.current_index = 0

        # Statistics
        self.replay_stats = {
            "total_ticks_loaded": 0,
            "total_ticks_replayed": 0,
            "trades_replayed": 0,
            "quotes_replayed": 0
        }

        logger.info(
            "TickReplayer initialized",
            symbol=symbol,
            replay_speed=float(self.replay_speed),
            buffer_size=self.buffer_size
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("config must be a dictionary")

    async def load_tick_data(self, data: pl.DataFrame) -> None:
        """Load historical tick data.

        Args:
            data: DataFrame with tick data

        Raises:
            ValueError: If data is invalid

        Example:
            >>> replayer = TickReplayer(config, "BTC/USDT")
            >>> await replayer.load_tick_data(tick_data_df)
        """
        try:
            required_cols = ["timestamp", "tick_type"]
            missing_cols = [col for col in required_cols if col not in data.columns]

            if missing_cols:
                error_msg = f"Missing required columns: {missing_cols}"
                logger.error("Invalid tick data", missing_cols=missing_cols)
                raise ValueError(error_msg)

            # Parse tick data
            await self._parse_tick_data(data)

            # Sort by timestamp
            self.tick_data.sort(key=lambda t: (t.timestamp, t.sequence))

            self.replay_stats["total_ticks_loaded"] = len(self.tick_data)

            logger.info(
                "Tick data loaded",
                symbol=self.symbol,
                total_ticks=len(self.tick_data),
                start_time=self.tick_data[0].timestamp if self.tick_data else None,
                end_time=self.tick_data[-1].timestamp if self.tick_data else None
            )

        except Exception as e:
            logger.error("Failed to load tick data", error=str(e))
            raise

    async def _parse_tick_data(self, data: pl.DataFrame) -> None:
        """Parse raw tick data into Tick objects.

        Args:
            data: Raw tick DataFrame
        """
        for idx, row in enumerate(data.iter_rows(named=True)):
            tick_type_str = row.get("tick_type", "TRADE").upper()
            tick_type = TickType[tick_type_str]

            tick = Tick(
                timestamp=row["timestamp"],
                symbol=self.symbol,
                tick_type=tick_type,
                price=Decimal(str(row["price"])) if "price" in row and row["price"] is not None else None,
                quantity=Decimal(str(row["quantity"])) if "quantity" in row and row["quantity"] is not None else None,
                bid_price=Decimal(str(row["bid_price"])) if "bid_price" in row and row["bid_price"] is not None else None,
                ask_price=Decimal(str(row["ask_price"])) if "ask_price" in row and row["ask_price"] is not None else None,
                bid_size=Decimal(str(row["bid_size"])) if "bid_size" in row and row["bid_size"] is not None else None,
                ask_size=Decimal(str(row["ask_size"])) if "ask_size" in row and row["ask_size"] is not None else None,
                sequence=row.get("sequence", idx),
                metadata={}
            )

            self.tick_data.append(tick)

    async def get_next_tick(self) -> Optional[Tick]:
        """Get next tick in sequence.

        Returns:
            Next tick or None if replay finished

        Example:
            >>> replayer = TickReplayer(config, "BTC/USDT")
            >>> await replayer.load_tick_data(tick_df)
            >>> while tick := await replayer.get_next_tick():
            ...     process_tick(tick)
        """
        if self.current_index >= len(self.tick_data):
            return None

        tick = self.tick_data[self.current_index]
        self.current_index += 1

        # Update statistics
        self.replay_stats["total_ticks_replayed"] += 1

        if tick.tick_type == TickType.TRADE:
            self.replay_stats["trades_replayed"] += 1
        elif tick.tick_type == TickType.QUOTE:
            self.replay_stats["quotes_replayed"] += 1

        return tick

    async def get_ticks_until(self, end_timestamp: datetime) -> List[Tick]:
        """Get all ticks until specified timestamp.

        Args:
            end_timestamp: Target timestamp

        Returns:
            List of ticks up to timestamp

        Example:
            >>> replayer = TickReplayer(config, "BTC/USDT")
            >>> ticks = await replayer.get_ticks_until(datetime.utcnow())
        """
        ticks = []

        while self.current_index < len(self.tick_data):
            tick = self.tick_data[self.current_index]

            if tick.timestamp > end_timestamp:
                break

            ticks.append(tick)
            self.current_index += 1

            # Update statistics
            self.replay_stats["total_ticks_replayed"] += 1

            if tick.tick_type == TickType.TRADE:
                self.replay_stats["trades_replayed"] += 1
            elif tick.tick_type == TickType.QUOTE:
                self.replay_stats["quotes_replayed"] += 1

        return ticks

    async def get_ticks_in_window(
        self,
        start_timestamp: datetime,
        end_timestamp: datetime
    ) -> List[Tick]:
        """Get all ticks within time window.

        Args:
            start_timestamp: Window start
            end_timestamp: Window end

        Returns:
            List of ticks in window

        Example:
            >>> replayer = TickReplayer(config, "BTC/USDT")
            >>> start = datetime(2024, 1, 1, 12, 0)
            >>> end = datetime(2024, 1, 1, 12, 1)
            >>> ticks = await replayer.get_ticks_in_window(start, end)
        """
        ticks = []

        for tick in self.tick_data:
            if start_timestamp <= tick.timestamp <= end_timestamp:
                ticks.append(tick)

        return ticks

    async def replay_with_timing(self) -> Iterator[Tick]:
        """Replay ticks with realistic timing.

        Yields ticks with delays matching original timing,
        adjusted by replay_speed.

        Yields:
            Ticks with realistic delays

        Example:
            >>> replayer = TickReplayer(config, "BTC/USDT")
            >>> async for tick in replayer.replay_with_timing():
            ...     process_tick(tick)
        """
        if not self.tick_data:
            logger.warning("No tick data loaded")
            return

        prev_timestamp = self.tick_data[0].timestamp

        for tick in self.tick_data:
            # Calculate delay
            time_diff = tick.timestamp - prev_timestamp

            if time_diff.total_seconds() > 0:
                # Apply replay speed adjustment
                adjusted_delay = time_diff.total_seconds() / float(self.replay_speed)

                # Sleep for adjusted delay
                await asyncio.sleep(adjusted_delay)

            prev_timestamp = tick.timestamp

            # Update statistics
            self.replay_stats["total_ticks_replayed"] += 1

            if tick.tick_type == TickType.TRADE:
                self.replay_stats["trades_replayed"] += 1
            elif tick.tick_type == TickType.QUOTE:
                self.replay_stats["quotes_replayed"] += 1

            yield tick

    def reset(self) -> None:
        """Reset replay to beginning.

        Example:
            >>> replayer = TickReplayer(config, "BTC/USDT")
            >>> replayer.reset()
        """
        self.current_index = 0
        logger.debug("Tick replayer reset", symbol=self.symbol)

    def seek_to_timestamp(self, timestamp: datetime) -> None:
        """Seek to specific timestamp.

        Args:
            timestamp: Target timestamp

        Example:
            >>> replayer = TickReplayer(config, "BTC/USDT")
            >>> replayer.seek_to_timestamp(datetime(2024, 1, 1, 12, 0))
        """
        # Binary search for timestamp
        left, right = 0, len(self.tick_data) - 1
        result_idx = 0

        while left <= right:
            mid = (left + right) // 2
            mid_time = self.tick_data[mid].timestamp

            if mid_time <= timestamp:
                result_idx = mid
                left = mid + 1
            else:
                right = mid - 1

        self.current_index = result_idx

        logger.debug(
            "Seeked to timestamp",
            symbol=self.symbol,
            timestamp=timestamp.isoformat(),
            index=result_idx
        )

    def get_current_position(self) -> Dict[str, Any]:
        """Get current replay position.

        Returns:
            Dictionary with position info

        Example:
            >>> replayer = TickReplayer(config, "BTC/USDT")
            >>> position = replayer.get_current_position()
            >>> print(f"At tick {position['current_index']} of {position['total_ticks']}")
        """
        if self.current_index < len(self.tick_data):
            current_tick = self.tick_data[self.current_index]
            current_timestamp = current_tick.timestamp
        else:
            current_timestamp = None

        return {
            "current_index": self.current_index,
            "total_ticks": len(self.tick_data),
            "current_timestamp": current_timestamp,
            "progress_pct": (self.current_index / len(self.tick_data) * 100) if self.tick_data else 0
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get replay statistics.

        Returns:
            Dictionary with replay stats

        Example:
            >>> replayer = TickReplayer(config, "BTC/USDT")
            >>> # ... replay ticks ...
            >>> stats = replayer.get_statistics()
            >>> print(f"Replayed {stats['total_ticks_replayed']} ticks")
        """
        return self.replay_stats.copy()

    async def calculate_tick_statistics(
        self,
        start_timestamp: Optional[datetime] = None,
        end_timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Calculate statistics for tick data.

        Args:
            start_timestamp: Optional start time for analysis
            end_timestamp: Optional end time for analysis

        Returns:
            Dictionary with tick statistics

        Example:
            >>> replayer = TickReplayer(config, "BTC/USDT")
            >>> stats = await replayer.calculate_tick_statistics()
            >>> print(f"Average ticks per second: {stats['avg_ticks_per_second']}")
        """
        # Filter ticks if timestamps provided
        if start_timestamp or end_timestamp:
            filtered_ticks = [
                t for t in self.tick_data
                if (not start_timestamp or t.timestamp >= start_timestamp) and
                   (not end_timestamp or t.timestamp <= end_timestamp)
            ]
        else:
            filtered_ticks = self.tick_data

        if not filtered_ticks:
            return {
                "total_ticks": 0,
                "trade_ticks": 0,
                "quote_ticks": 0,
                "avg_ticks_per_second": 0
            }

        # Count by type
        trade_ticks = len([t for t in filtered_ticks if t.tick_type == TickType.TRADE])
        quote_ticks = len([t for t in filtered_ticks if t.tick_type == TickType.QUOTE])

        # Calculate time span
        if len(filtered_ticks) > 1:
            time_span = (filtered_ticks[-1].timestamp - filtered_ticks[0].timestamp).total_seconds()
            if time_span > 0:
                avg_ticks_per_second = len(filtered_ticks) / time_span
            else:
                avg_ticks_per_second = 0
        else:
            avg_ticks_per_second = 0

        return {
            "total_ticks": len(filtered_ticks),
            "trade_ticks": trade_ticks,
            "quote_ticks": quote_ticks,
            "orderbook_updates": len(filtered_ticks) - trade_ticks - quote_ticks,
            "avg_ticks_per_second": avg_ticks_per_second,
            "start_timestamp": filtered_ticks[0].timestamp if filtered_ticks else None,
            "end_timestamp": filtered_ticks[-1].timestamp if filtered_ticks else None
        }
