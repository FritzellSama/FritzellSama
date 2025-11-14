"""
Order Flow Indicator - Analyzes institutional order flow and market depth.

This indicator tracks:
- Volume delta (buy volume - sell volume)
- Cumulative delta
- Volume at price levels
- Order flow imbalance
- Absorption and exhaustion patterns
"""

import asyncio
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any, Tuple, DefaultDict
import polars as pl
from structlog import get_logger
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

logger = get_logger(__name__)

# Set high precision for Decimal calculations
getcontext().prec = 28


class OrderFlowSignal(Enum):
    """Order flow trading signals."""
    STRONG_BUYING = "strong_buying"
    BUYING = "buying"
    NEUTRAL = "neutral"
    SELLING = "selling"
    STRONG_SELLING = "strong_selling"


class OrderFlowPattern(Enum):
    """Order flow patterns."""
    ABSORPTION = "absorption"  # Large volume without price movement
    EXHAUSTION = "exhaustion"  # High volume at reversal
    BREAKOUT = "breakout"  # Volume surge with price move
    IMBALANCE = "imbalance"  # Significant buy/sell imbalance
    NORMAL = "normal"


@dataclass
class OrderFlowMetrics:
    """Order flow metrics."""
    volume_delta: Decimal
    cumulative_delta: Decimal
    buy_volume: Decimal
    sell_volume: Decimal
    delta_percentage: Decimal
    imbalance_ratio: Decimal
    pattern: OrderFlowPattern


class OrderFlowIndicator:
    """
    Order flow analysis indicator for institutional trading.

    Analyzes volume dynamics to identify:
    - Institutional accumulation/distribution
    - Support/resistance absorption
    - Momentum shifts
    - Market maker activity

    Attributes:
        config: Configuration dictionary
        cumulative_period: Period for cumulative delta
        imbalance_threshold: Threshold for imbalance detection
        absorption_threshold: Threshold for absorption pattern

    Example:
        >>> config = {
        ...     "cumulative_period": 100,
        ...     "imbalance_threshold": "0.6",
        ...     "absorption_threshold": "0.3"
        ... }
        >>> order_flow = OrderFlowIndicator(config)
        >>> result = await order_flow.calculate(volume_data)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize order flow indicator.

        Args:
            config: Configuration dictionary containing:
                - cumulative_period: Period for cumulative calculations
                - imbalance_threshold: Threshold for imbalance (0-1)
                - absorption_threshold: Threshold for absorption detection
                - price_bins: Number of price bins for volume profile
                - tick_size: Minimum price increment

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.cumulative_period: int = int(config.get("cumulative_period", 100))
        self.imbalance_threshold: Decimal = Decimal(str(
            config.get("imbalance_threshold", "0.6")
        ))
        self.absorption_threshold: Decimal = Decimal(str(
            config.get("absorption_threshold", "0.3")
        ))
        self.price_bins: int = int(config.get("price_bins", 50))
        self.tick_size: Decimal = Decimal(str(config.get("tick_size", "0.01")))

        logger.info(
            "Order Flow indicator initialized",
            cumulative_period=self.cumulative_period,
            imbalance_threshold=str(self.imbalance_threshold)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if "cumulative_period" in self.config:
            if int(self.config["cumulative_period"]) < 1:
                raise ValueError("cumulative_period must be positive")

        if "imbalance_threshold" in self.config:
            threshold = Decimal(str(self.config["imbalance_threshold"]))
            if threshold < Decimal("0") or threshold > Decimal("1"):
                raise ValueError("imbalance_threshold must be between 0 and 1")

        if "tick_size" in self.config:
            if Decimal(str(self.config["tick_size"])) <= Decimal("0"):
                raise ValueError("tick_size must be positive")

        logger.debug("Order Flow configuration validated")

    async def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate order flow indicators.

        Args:
            data: Polars DataFrame with columns:
                - timestamp
                - open, high, low, close
                - volume
                - buy_volume (optional)
                - sell_volume (optional)

        Returns:
            DataFrame with order flow metrics and signals

        Raises:
            ValueError: If data is invalid or insufficient
        """
        try:
            # Validate input data
            self._validate_data(data)

            # Classify volume as buy/sell if not provided
            result = await self._classify_volume(data)

            # Calculate volume delta
            result = await self._calculate_volume_delta(result)

            # Calculate cumulative delta
            result = await self._calculate_cumulative_delta(result)

            # Calculate delta percentage
            result = await self._calculate_delta_percentage(result)

            # Calculate imbalance ratio
            result = await self._calculate_imbalance_ratio(result)

            # Detect order flow patterns
            result = await self._detect_patterns(result)

            # Generate signals
            result = await self._generate_signals(result)

            # Calculate volume at price (optional enhanced analysis)
            result = await self._analyze_volume_at_price(result)

            logger.info(
                "Order flow indicators calculated",
                rows=len(result)
            )

            return result

        except Exception as e:
            logger.error("Order flow calculation failed", error=str(e))
            raise

    async def _classify_volume(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Classify volume as buy or sell if not already provided.

        Uses tick rule and price action to estimate buy/sell volume.

        Args:
            data: Input data

        Returns:
            DataFrame with buy_volume and sell_volume columns
        """
        try:
            # Check if buy/sell volume already provided
            has_buy_sell = "buy_volume" in data.columns and "sell_volume" in data.columns

            if has_buy_sell:
                return data

            # Estimate buy/sell volume using price action
            buy_volumes = []
            sell_volumes = []

            closes = [Decimal(str(x)) for x in data["close"].to_list()]
            opens = [Decimal(str(x)) for x in data["open"].to_list()]
            highs = [Decimal(str(x)) for x in data["high"].to_list()]
            lows = [Decimal(str(x)) for x in data["low"].to_list()]
            volumes = [Decimal(str(x)) for x in data["volume"].to_list()]

            for i in range(len(data)):
                close = closes[i]
                open_price = opens[i]
                high = highs[i]
                low = lows[i]
                volume = volumes[i]

                # Calculate price position within bar (0-1)
                price_range = high - low
                if price_range == Decimal("0"):
                    # No range: split volume evenly
                    buy_vol = volume / Decimal("2")
                    sell_vol = volume / Decimal("2")
                else:
                    # Position where close is relative to range
                    close_position = (close - low) / price_range

                    # Allocate volume based on close position
                    buy_vol = volume * close_position
                    sell_vol = volume * (Decimal("1") - close_position)

                    # Adjust for bar direction
                    if close > open_price:
                        # Bullish bar: more buy volume
                        buy_vol = buy_vol * Decimal("1.2")
                        sell_vol = sell_vol * Decimal("0.8")
                    elif close < open_price:
                        # Bearish bar: more sell volume
                        buy_vol = buy_vol * Decimal("0.8")
                        sell_vol = sell_vol * Decimal("1.2")

                buy_volumes.append(str(buy_vol))
                sell_volumes.append(str(sell_vol))

            result = data.with_columns([
                pl.Series("buy_volume", buy_volumes),
                pl.Series("sell_volume", sell_volumes)
            ])

            return result

        except Exception as e:
            logger.error("Volume classification failed", error=str(e))
            raise

    async def _calculate_volume_delta(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate volume delta (buy volume - sell volume).

        Args:
            data: DataFrame with buy_volume and sell_volume

        Returns:
            DataFrame with volume_delta column
        """
        try:
            deltas = []

            buy_volumes = [Decimal(str(x)) for x in data["buy_volume"].to_list()]
            sell_volumes = [Decimal(str(x)) for x in data["sell_volume"].to_list()]

            for buy_vol, sell_vol in zip(buy_volumes, sell_volumes):
                delta = buy_vol - sell_vol
                deltas.append(str(delta))

            result = data.with_columns([
                pl.Series("volume_delta", deltas)
            ])

            return result

        except Exception as e:
            logger.error("Volume delta calculation failed", error=str(e))
            raise

    async def _calculate_cumulative_delta(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate cumulative volume delta.

        Args:
            data: DataFrame with volume_delta

        Returns:
            DataFrame with cumulative_delta column
        """
        try:
            cumulative_deltas = []

            deltas = [Decimal(str(x)) for x in data["volume_delta"].to_list()]

            cumulative = Decimal("0")
            for delta in deltas:
                cumulative += delta
                cumulative_deltas.append(str(cumulative))

            result = data.with_columns([
                pl.Series("cumulative_delta", cumulative_deltas)
            ])

            # Also calculate cumulative delta over rolling period
            rolling_cumulative = []

            for i in range(len(deltas)):
                if i < self.cumulative_period:
                    window_sum = sum(deltas[:i+1])
                else:
                    window_sum = sum(deltas[i-self.cumulative_period+1:i+1])

                rolling_cumulative.append(str(window_sum))

            result = result.with_columns([
                pl.Series("rolling_cumulative_delta", rolling_cumulative)
            ])

            return result

        except Exception as e:
            logger.error("Cumulative delta calculation failed", error=str(e))
            raise

    async def _calculate_delta_percentage(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate delta as percentage of total volume.

        Args:
            data: DataFrame with volume_delta and volume

        Returns:
            DataFrame with delta_percentage column
        """
        try:
            percentages = []

            deltas = [Decimal(str(x)) for x in data["volume_delta"].to_list()]
            volumes = [Decimal(str(x)) for x in data["volume"].to_list()]

            for delta, volume in zip(deltas, volumes):
                if volume == Decimal("0"):
                    percentage = Decimal("0")
                else:
                    percentage = (delta / volume) * Decimal("100")

                percentages.append(str(percentage))

            result = data.with_columns([
                pl.Series("delta_percentage", percentages)
            ])

            return result

        except Exception as e:
            logger.error("Delta percentage calculation failed", error=str(e))
            raise

    async def _calculate_imbalance_ratio(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate buy/sell imbalance ratio.

        Ratio = buy_volume / (buy_volume + sell_volume)
        > 0.5 = more buying
        < 0.5 = more selling

        Args:
            data: DataFrame with buy_volume and sell_volume

        Returns:
            DataFrame with imbalance_ratio column
        """
        try:
            ratios = []

            buy_volumes = [Decimal(str(x)) for x in data["buy_volume"].to_list()]
            sell_volumes = [Decimal(str(x)) for x in data["sell_volume"].to_list()]

            for buy_vol, sell_vol in zip(buy_volumes, sell_volumes):
                total_vol = buy_vol + sell_vol

                if total_vol == Decimal("0"):
                    ratio = Decimal("0.5")  # Neutral
                else:
                    ratio = buy_vol / total_vol

                ratios.append(str(ratio))

            result = data.with_columns([
                pl.Series("imbalance_ratio", ratios)
            ])

            return result

        except Exception as e:
            logger.error("Imbalance ratio calculation failed", error=str(e))
            raise

    async def _detect_patterns(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Detect order flow patterns.

        Patterns:
        - ABSORPTION: High volume with minimal price movement
        - EXHAUSTION: High volume at price extreme with reversal
        - BREAKOUT: Volume surge with significant price move
        - IMBALANCE: Strong buy or sell imbalance
        - NORMAL: No special pattern

        Args:
            data: DataFrame with order flow metrics

        Returns:
            DataFrame with pattern column
        """
        try:
            patterns = []

            volumes = [Decimal(str(x)) for x in data["volume"].to_list()]
            closes = [Decimal(str(x)) for x in data["close"].to_list()]
            highs = [Decimal(str(x)) for x in data["high"].to_list()]
            lows = [Decimal(str(x)) for x in data["low"].to_list()]
            imbalance_ratios = [Decimal(str(x)) for x in data["imbalance_ratio"].to_list()]

            # Calculate average volume for comparison
            avg_volume_window = min(20, len(volumes))

            for i in range(len(data)):
                if i < 5:
                    patterns.append(OrderFlowPattern.NORMAL.value)
                    continue

                # Calculate recent metrics
                recent_volumes = volumes[max(0, i-avg_volume_window):i+1]
                avg_volume = sum(recent_volumes) / Decimal(str(len(recent_volumes)))

                current_volume = volumes[i]
                price_range = highs[i] - lows[i]
                imbalance = imbalance_ratios[i]

                # Calculate price movement
                if i > 0:
                    price_change = abs(closes[i] - closes[i-1])
                else:
                    price_change = Decimal("0")

                # Detect patterns
                pattern = OrderFlowPattern.NORMAL

                # High volume threshold
                high_volume = current_volume > avg_volume * Decimal("2")

                # ABSORPTION: High volume, low price movement
                if high_volume and price_change < price_range * self.absorption_threshold:
                    pattern = OrderFlowPattern.ABSORPTION

                # IMBALANCE: Strong directional imbalance
                elif (imbalance > Decimal("1") - self.imbalance_threshold or
                      imbalance < self.imbalance_threshold):
                    pattern = OrderFlowPattern.IMBALANCE

                # BREAKOUT: High volume with significant price move
                elif high_volume and price_change > price_range * Decimal("0.7"):
                    pattern = OrderFlowPattern.BREAKOUT

                # EXHAUSTION: Check for reversal after high volume
                elif i >= 2 and high_volume:
                    # Check if price reversed after this bar
                    prev_trend = closes[i-1] - closes[i-2]
                    curr_move = closes[i] - closes[i-1]

                    if prev_trend * curr_move < Decimal("0"):  # Sign change = reversal
                        pattern = OrderFlowPattern.EXHAUSTION

                patterns.append(pattern.value)

            result = data.with_columns([
                pl.Series("pattern", patterns)
            ])

            return result

        except Exception as e:
            logger.error("Pattern detection failed", error=str(e))
            raise

    async def _generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Generate order flow trading signals.

        Signal based on:
        - Cumulative delta trend
        - Imbalance ratio
        - Order flow patterns

        Args:
            data: DataFrame with order flow metrics

        Returns:
            DataFrame with signal column
        """
        try:
            def determine_signal(row, prev_row) -> str:
                try:
                    cum_delta = Decimal(str(row["rolling_cumulative_delta"]))
                    imbalance = Decimal(str(row["imbalance_ratio"]))
                    pattern = row["pattern"]

                    # Previous cumulative delta for trend
                    if prev_row:
                        prev_cum_delta = Decimal(str(prev_row["rolling_cumulative_delta"]))
                        delta_trend = cum_delta - prev_cum_delta
                    else:
                        delta_trend = Decimal("0")

                    # Strong signals based on patterns
                    if pattern == OrderFlowPattern.BREAKOUT.value:
                        if imbalance > Decimal("0.7"):
                            return OrderFlowSignal.STRONG_BUYING.value
                        elif imbalance < Decimal("0.3"):
                            return OrderFlowSignal.STRONG_SELLING.value

                    if pattern == OrderFlowPattern.EXHAUSTION.value:
                        # Exhaustion suggests reversal
                        if imbalance > Decimal("0.6"):
                            return OrderFlowSignal.STRONG_SELLING.value  # Reversal from buying
                        elif imbalance < Decimal("0.4"):
                            return OrderFlowSignal.STRONG_BUYING.value  # Reversal from selling

                    # Normal signals based on cumulative delta and imbalance
                    if delta_trend > Decimal("0") and imbalance > Decimal("0.6"):
                        return OrderFlowSignal.BUYING.value
                    elif delta_trend > Decimal("0") and imbalance > Decimal("0.7"):
                        return OrderFlowSignal.STRONG_BUYING.value
                    elif delta_trend < Decimal("0") and imbalance < Decimal("0.4"):
                        return OrderFlowSignal.SELLING.value
                    elif delta_trend < Decimal("0") and imbalance < Decimal("0.3"):
                        return OrderFlowSignal.STRONG_SELLING.value
                    else:
                        return OrderFlowSignal.NEUTRAL.value

                except Exception:
                    return OrderFlowSignal.NEUTRAL.value

            signals = []
            rows = list(data.iter_rows(named=True))

            for i, row in enumerate(rows):
                prev_row = rows[i-1] if i > 0 else None
                signals.append(determine_signal(row, prev_row))

            result = data.with_columns([
                pl.Series("signal", signals)
            ])

            # Add signal strength
            signal_strength_map = {
                OrderFlowSignal.STRONG_BUYING.value: Decimal("2"),
                OrderFlowSignal.BUYING.value: Decimal("1"),
                OrderFlowSignal.NEUTRAL.value: Decimal("0"),
                OrderFlowSignal.SELLING.value: Decimal("-1"),
                OrderFlowSignal.STRONG_SELLING.value: Decimal("-2")
            }

            result = result.with_columns([
                pl.col("signal").map_elements(
                    lambda s: str(signal_strength_map.get(s, Decimal("0"))),
                    return_dtype=pl.Utf8
                ).alias("signal_strength")
            ])

            return result

        except Exception as e:
            logger.error("Signal generation failed", error=str(e))
            raise

    async def _analyze_volume_at_price(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Analyze volume distribution at price levels.

        Creates a volume profile showing where volume occurred.

        Args:
            data: DataFrame with price and volume data

        Returns:
            DataFrame with volume_at_price metrics
        """
        try:
            # This is a simplified version
            # Full implementation would create a price histogram

            # Calculate volume-weighted average price
            vwaps = []

            highs = [Decimal(str(x)) for x in data["high"].to_list()]
            lows = [Decimal(str(x)) for x in data["low"].to_list()]
            closes = [Decimal(str(x)) for x in data["close"].to_list()]
            volumes = [Decimal(str(x)) for x in data["volume"].to_list()]

            cumulative_pv = Decimal("0")
            cumulative_vol = Decimal("0")

            for i in range(len(data)):
                # Typical price
                typical_price = (highs[i] + lows[i] + closes[i]) / Decimal("3")
                volume = volumes[i]

                cumulative_pv += typical_price * volume
                cumulative_vol += volume

                if cumulative_vol == Decimal("0"):
                    vwap = typical_price
                else:
                    vwap = cumulative_pv / cumulative_vol

                vwaps.append(str(vwap))

            result = data.with_columns([
                pl.Series("vwap", vwaps)
            ])

            return result

        except Exception as e:
            logger.error("Volume at price analysis failed", error=str(e))
            raise

    def _validate_data(self, data: pl.DataFrame) -> None:
        """Validate input data format and content."""
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]

        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        if len(data) == 0:
            raise ValueError("Data is empty")

        # Check for null values
        for col in ["high", "low", "close", "volume"]:
            if data[col].null_count() > 0:
                raise ValueError(f"Data contains null values in '{col}' column")

    async def get_current_metrics(
        self,
        data: pl.DataFrame
    ) -> Optional[OrderFlowMetrics]:
        """
        Get current order flow metrics.

        Args:
            data: DataFrame with calculated metrics

        Returns:
            OrderFlowMetrics dataclass or None
        """
        try:
            if len(data) == 0:
                return None

            last_row = data.row(-1, named=True)

            required = [
                "volume_delta", "cumulative_delta", "buy_volume",
                "sell_volume", "delta_percentage", "imbalance_ratio", "pattern"
            ]

            if any(last_row.get(k) is None for k in required):
                return None

            return OrderFlowMetrics(
                volume_delta=Decimal(str(last_row["volume_delta"])),
                cumulative_delta=Decimal(str(last_row["cumulative_delta"])),
                buy_volume=Decimal(str(last_row["buy_volume"])),
                sell_volume=Decimal(str(last_row["sell_volume"])),
                delta_percentage=Decimal(str(last_row["delta_percentage"])),
                imbalance_ratio=Decimal(str(last_row["imbalance_ratio"])),
                pattern=OrderFlowPattern(last_row["pattern"])
            )

        except Exception as e:
            logger.error("Failed to get current metrics", error=str(e))
            return None


async def create_order_flow_indicator(
    config: Dict[str, Any]
) -> OrderFlowIndicator:
    """
    Factory function to create Order Flow indicator instance.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized Order Flow indicator
    """
    return OrderFlowIndicator(config)
