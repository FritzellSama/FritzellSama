"""Tick data processor for high-frequency analytics.

This module processes tick-level price updates to compute OHLCV bars,
volume profiles, and tick-based indicators. Optimized for stream processing
with memory-efficient batching and parallel computation.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class TickProcessor:
    """Processes tick data for bar generation and analytics.

    Aggregates tick-level data into time-based and volume-based bars.
    Computes tick-level statistics and microstructure indicators.

    Attributes:
        config: Configuration dictionary
        bar_intervals: Time intervals for OHLCV bars
        volume_bar_size: Size for volume-based bars
        checkpoint_interval: Interval for checkpointing state
        _active_bars: Active bar builders per symbol
        _processed_count: Count of processed ticks

    Example:
        >>> processor = TickProcessor(config)
        >>> bars = await processor.process_tick(tick_data)
        >>> print(f"OHLCV: {bars}")
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize tick processor.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If required configuration is missing
        """
        self.config = config
        self._validate_config()

        processor_config = config.get("tick_processor", {})
        self.bar_intervals = processor_config.get(
            "bar_intervals",
            ["1m", "5m", "15m", "1h"]
        )
        self.volume_bar_size = Decimal(
            str(processor_config.get("volume_bar_size_usdt", 100000))
        )
        self.checkpoint_interval = processor_config.get("checkpoint_interval", 1000)
        self.enable_volume_bars = processor_config.get("enable_volume_bars", True)
        self.enable_tick_stats = processor_config.get("enable_tick_stats", True)

        # Processing state
        self._processed_count = 0
        self._active_bars: Dict[str, Dict[str, Any]] = {}
        self._volume_bars: Dict[str, Dict[str, Any]] = {}
        self._tick_buffer: List[Dict[str, Any]] = []
        self._checkpoint_data: Dict[str, Any] = {}

        # Metrics
        self._metrics: Dict[str, int] = {
            "total_processed": 0,
            "total_errors": 0,
            "bars_generated": 0,
            "volume_bars_generated": 0,
            "checkpoints_saved": 0
        }

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required configuration is missing
        """
        if "tick_processor" not in self.config:
            raise ValueError("Missing tick_processor configuration")

    async def process_tick(
        self,
        symbol: str,
        price: Decimal,
        size: Decimal,
        timestamp: datetime,
        side: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Process single tick and generate completed bars.

        Args:
            symbol: Trading symbol
            price: Tick price
            size: Tick size/volume
            timestamp: Tick timestamp
            side: Optional side (buy/sell)

        Returns:
            List of completed bars

        Example:
            >>> bars = await processor.process_tick(
            ...     "BTC/USDT", Decimal("50000"), Decimal("0.5"),
            ...     datetime.now(timezone.utc)
            ... )
        """
        try:
            completed_bars = []

            # Initialize bars for symbol if needed
            if symbol not in self._active_bars:
                self._initialize_symbol_bars(symbol)

            # Update time-based bars
            for interval in self.bar_intervals:
                bar_key = f"{symbol}:{interval}"

                # Check if we need to complete current bar
                completed_bar = self._update_time_bar(
                    bar_key,
                    interval,
                    price,
                    size,
                    timestamp
                )

                if completed_bar:
                    completed_bars.append(completed_bar)
                    self._metrics["bars_generated"] += 1

            # Update volume-based bars if enabled
            if self.enable_volume_bars:
                volume_bar = await self._update_volume_bar(
                    symbol,
                    price,
                    size,
                    timestamp
                )

                if volume_bar:
                    completed_bars.append(volume_bar)
                    self._metrics["volume_bars_generated"] += 1

            # Update tick statistics if enabled
            if self.enable_tick_stats:
                self._update_tick_stats(symbol, price, size, side)

            # Update metrics
            self._metrics["total_processed"] += 1
            self._processed_count += 1

            # Checkpoint if needed
            if self._processed_count % self.checkpoint_interval == 0:
                await self._save_checkpoint()

            return completed_bars

        except Exception as e:
            logger.error(
                "tick_processing_failed",
                symbol=symbol,
                error=str(e)
            )
            self._metrics["total_errors"] += 1
            raise

    def _initialize_symbol_bars(self, symbol: str) -> None:
        """Initialize bar builders for a symbol.

        Args:
            symbol: Trading symbol
        """
        self._active_bars[symbol] = {}
        self._volume_bars[symbol] = {
            "open": None,
            "high": None,
            "low": None,
            "close": None,
            "volume": Decimal("0"),
            "notional": Decimal("0"),
            "start_time": None,
            "trades": 0
        }

    def _update_time_bar(
        self,
        bar_key: str,
        interval: str,
        price: Decimal,
        size: Decimal,
        timestamp: datetime
    ) -> Optional[Dict[str, Any]]:
        """Update time-based bar and return if completed.

        Args:
            bar_key: Bar identifier key
            interval: Time interval (e.g., "1m", "5m")
            price: Tick price
            size: Tick size
            timestamp: Tick timestamp

        Returns:
            Completed bar or None
        """
        # Parse interval
        interval_seconds = self._parse_interval(interval)
        bar_start = self._get_bar_start(timestamp, interval_seconds)

        # Get or create bar
        if bar_key not in self._active_bars:
            symbol = bar_key.split(":")[0]
            if symbol not in self._active_bars:
                self._initialize_symbol_bars(symbol)

        symbol_bars = self._active_bars[bar_key.split(":")[0]]

        # Check if we need to close current bar
        if bar_key in symbol_bars:
            current_bar = symbol_bars[bar_key]
            current_start = current_bar["bar_start"]

            if bar_start > current_start:
                # Complete current bar
                completed_bar = {
                    "symbol": current_bar["symbol"],
                    "interval": interval,
                    "open": current_bar["open"],
                    "high": current_bar["high"],
                    "low": current_bar["low"],
                    "close": current_bar["close"],
                    "volume": current_bar["volume"],
                    "notional": current_bar["notional"],
                    "trades": current_bar["trades"],
                    "timestamp": current_start,
                    "bar_type": "time"
                }

                # Start new bar
                symbol_bars[bar_key] = self._create_new_bar(
                    bar_key.split(":")[0],
                    bar_start,
                    price,
                    size
                )

                return completed_bar

        # Update or create current bar
        if bar_key not in symbol_bars:
            symbol_bars[bar_key] = self._create_new_bar(
                bar_key.split(":")[0],
                bar_start,
                price,
                size
            )
        else:
            bar = symbol_bars[bar_key]
            bar["high"] = max(bar["high"], price)
            bar["low"] = min(bar["low"], price)
            bar["close"] = price
            bar["volume"] += size
            bar["notional"] += price * size
            bar["trades"] += 1

        return None

    async def _update_volume_bar(
        self,
        symbol: str,
        price: Decimal,
        size: Decimal,
        timestamp: datetime
    ) -> Optional[Dict[str, Any]]:
        """Update volume-based bar.

        Args:
            symbol: Trading symbol
            price: Tick price
            size: Tick size
            timestamp: Tick timestamp

        Returns:
            Completed volume bar or None
        """
        if symbol not in self._volume_bars:
            self._initialize_symbol_bars(symbol)

        vol_bar = self._volume_bars[symbol]

        # Initialize if needed
        if vol_bar["open"] is None:
            vol_bar["open"] = price
            vol_bar["high"] = price
            vol_bar["low"] = price
            vol_bar["start_time"] = timestamp

        # Update bar
        vol_bar["high"] = max(vol_bar["high"], price)
        vol_bar["low"] = min(vol_bar["low"], price)
        vol_bar["close"] = price
        vol_bar["volume"] += size
        vol_bar["notional"] += price * size
        vol_bar["trades"] += 1

        # Check if bar is complete
        if vol_bar["notional"] >= self.volume_bar_size:
            completed_bar = {
                "symbol": symbol,
                "interval": f"{self.volume_bar_size}usd",
                "open": vol_bar["open"],
                "high": vol_bar["high"],
                "low": vol_bar["low"],
                "close": vol_bar["close"],
                "volume": vol_bar["volume"],
                "notional": vol_bar["notional"],
                "trades": vol_bar["trades"],
                "timestamp": vol_bar["start_time"],
                "bar_type": "volume"
            }

            # Reset bar
            self._volume_bars[symbol] = {
                "open": None,
                "high": None,
                "low": None,
                "close": None,
                "volume": Decimal("0"),
                "notional": Decimal("0"),
                "start_time": None,
                "trades": 0
            }

            return completed_bar

        return None

    def _update_tick_stats(
        self,
        symbol: str,
        price: Decimal,
        size: Decimal,
        side: Optional[str]
    ) -> None:
        """Update tick-level statistics.

        Args:
            symbol: Trading symbol
            price: Tick price
            size: Tick size
            side: Trade side
        """
        # Store in buffer for batch processing
        self._tick_buffer.append({
            "symbol": symbol,
            "price": price,
            "size": size,
            "side": side,
            "timestamp": datetime.now(timezone.utc)
        })

        # Process buffer if it gets large
        if len(self._tick_buffer) > 1000:
            asyncio.create_task(self._process_tick_buffer())

    async def _process_tick_buffer(self) -> None:
        """Process buffered ticks for statistics."""
        if not self._tick_buffer:
            return

        buffer = self._tick_buffer.copy()
        self._tick_buffer.clear()

        # Convert to Polars for efficient processing
        df = pl.DataFrame(buffer)

        # Could compute various tick statistics here
        # For now, just log the count
        logger.debug(
            "tick_buffer_processed",
            count=len(buffer)
        )

    def _create_new_bar(
        self,
        symbol: str,
        bar_start: datetime,
        price: Decimal,
        size: Decimal
    ) -> Dict[str, Any]:
        """Create new bar.

        Args:
            symbol: Trading symbol
            bar_start: Bar start timestamp
            price: Initial price
            size: Initial size

        Returns:
            New bar dictionary
        """
        return {
            "symbol": symbol,
            "bar_start": bar_start,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": size,
            "notional": price * size,
            "trades": 1
        }

    def _parse_interval(self, interval: str) -> int:
        """Parse interval string to seconds.

        Args:
            interval: Interval string (e.g., "1m", "5m", "1h")

        Returns:
            Interval in seconds
        """
        interval_map = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400
        }

        return interval_map.get(interval, 60)

    def _get_bar_start(self, timestamp: datetime, interval_seconds: int) -> datetime:
        """Get bar start timestamp.

        Args:
            timestamp: Current timestamp
            interval_seconds: Bar interval in seconds

        Returns:
            Bar start timestamp
        """
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        seconds_since_epoch = int((timestamp - epoch).total_seconds())
        bar_start_seconds = (seconds_since_epoch // interval_seconds) * interval_seconds

        return epoch + timedelta(seconds=bar_start_seconds)

    async def process_batch(
        self,
        ticks_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Process batch of ticks efficiently.

        Args:
            ticks_df: Polars DataFrame with tick data

        Returns:
            DataFrame with completed bars

        Example:
            >>> df = pl.DataFrame([...])
            >>> bars_df = await processor.process_batch(df)
        """
        try:
            if ticks_df.is_empty():
                return pl.DataFrame()

            all_bars = []

            # Process each tick
            for row in ticks_df.to_dicts():
                bars = await self.process_tick(
                    symbol=row["symbol"],
                    price=Decimal(str(row["price"])),
                    size=Decimal(str(row["size"])),
                    timestamp=row["timestamp"],
                    side=row.get("side")
                )

                all_bars.extend(bars)

            if all_bars:
                result_df = pl.DataFrame(all_bars)
                logger.debug(
                    "tick_batch_processed",
                    ticks=len(ticks_df),
                    bars=len(all_bars)
                )
                return result_df
            else:
                return pl.DataFrame()

        except Exception as e:
            logger.error(
                "tick_batch_processing_failed",
                rows=len(ticks_df),
                error=str(e)
            )
            self._metrics["total_errors"] += 1
            raise

    async def _save_checkpoint(self) -> None:
        """Save processing checkpoint for recovery."""
        try:
            self._checkpoint_data = {
                "processed_count": self._processed_count,
                "timestamp": datetime.now(timezone.utc),
                "metrics": self._metrics.copy(),
                "active_bars_count": len(self._active_bars)
            }

            self._metrics["checkpoints_saved"] += 1

            logger.debug(
                "tick_checkpoint_saved",
                processed=self._processed_count
            )

        except Exception as e:
            logger.error("checkpoint_save_failed", error=str(e))

    def get_checkpoint(self) -> Dict[str, Any]:
        """Get current checkpoint data.

        Returns:
            Checkpoint dictionary
        """
        return self._checkpoint_data.copy()

    def restore_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Restore from checkpoint.

        Args:
            checkpoint: Checkpoint data to restore
        """
        self._processed_count = checkpoint.get("processed_count", 0)
        self._metrics = checkpoint.get("metrics", {}).copy()

        logger.info(
            "tick_checkpoint_restored",
            processed=self._processed_count
        )

    def get_metrics(self) -> Dict[str, int]:
        """Get processor metrics.

        Returns:
            Dictionary of metrics
        """
        return self._metrics.copy()
