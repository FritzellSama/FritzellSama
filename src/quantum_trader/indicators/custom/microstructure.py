"""
Market Microstructure Indicator - Analyzes high-frequency market dynamics.

This indicator captures market microstructure signals including:
- Bid-ask spread dynamics
- Price impact
- Order arrival rates
- Tick direction
- Trade classification
"""

import asyncio
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any, Tuple
import polars as pl
from structlog import get_logger
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

logger = get_logger(__name__)

# Set high precision for Decimal calculations
getcontext().prec = 28


class TradeDirection(Enum):
    """Trade direction classification."""
    BUY = "buy"
    SELL = "sell"
    NEUTRAL = "neutral"


class MicrostructureSignal(Enum):
    """Microstructure trading signals."""
    STRONG_BUYING = "strong_buying"
    BUYING = "buying"
    NEUTRAL = "neutral"
    SELLING = "selling"
    STRONG_SELLING = "strong_selling"


@dataclass
class MicrostructureMetrics:
    """Market microstructure metrics."""
    effective_spread: Decimal
    realized_spread: Decimal
    price_impact: Decimal
    order_imbalance: Decimal
    tick_rule_direction: Decimal
    trade_intensity: Decimal
    volatility_adjusted_spread: Decimal


class MicrostructureIndicator:
    """
    Market microstructure analysis indicator.

    Analyzes tick-level data to identify:
    - Information asymmetry
    - Liquidity conditions
    - Order flow toxicity
    - Market maker costs

    Attributes:
        config: Configuration dictionary
        spread_window: Window for spread calculations
        intensity_window: Window for trade intensity
        impact_window: Window for price impact

    Example:
        >>> config = {
        ...     "spread_window": 100,
        ...     "intensity_window": 60,
        ...     "impact_window": 20
        ... }
        >>> micro = MicrostructureIndicator(config)
        >>> result = await micro.calculate(tick_data)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize microstructure indicator.

        Args:
            config: Configuration dictionary containing:
                - spread_window: Window for spread calculations
                - intensity_window: Window for intensity (seconds)
                - impact_window: Window for impact calculations
                - volatility_window: Window for volatility
                - min_tick_size: Minimum price tick size

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.spread_window: int = int(config.get("spread_window", 100))
        self.intensity_window: int = int(config.get("intensity_window", 60))
        self.impact_window: int = int(config.get("impact_window", 20))
        self.volatility_window: int = int(config.get("volatility_window", 100))
        self.min_tick_size: Decimal = Decimal(str(config.get("min_tick_size", "0.01")))

        logger.info(
            "Microstructure indicator initialized",
            spread_window=self.spread_window,
            intensity_window=self.intensity_window
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        for field in ["spread_window", "intensity_window", "impact_window"]:
            if field in self.config:
                if int(self.config[field]) < 1:
                    raise ValueError(f"{field} must be positive")

        if "min_tick_size" in self.config:
            if Decimal(str(self.config["min_tick_size"])) <= Decimal("0"):
                raise ValueError("min_tick_size must be positive")

        logger.debug("Microstructure configuration validated")

    async def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate market microstructure indicators.

        Args:
            data: Polars DataFrame with columns:
                - timestamp
                - price (or close)
                - volume
                - bid (optional)
                - ask (optional)

        Returns:
            DataFrame with microstructure metrics and signals

        Raises:
            ValueError: If data is invalid or insufficient
        """
        try:
            # Validate input data
            self._validate_data(data)

            # Classify trade direction using tick rule
            result = await self._classify_trade_direction(data)

            # Calculate effective spread
            result = await self._calculate_effective_spread(result)

            # Calculate order imbalance
            result = await self._calculate_order_imbalance(result)

            # Calculate trade intensity
            result = await self._calculate_trade_intensity(result)

            # Calculate price impact
            result = await self._calculate_price_impact(result)

            # Calculate realized spread
            result = await self._calculate_realized_spread(result)

            # Calculate volatility-adjusted spread
            result = await self._calculate_volatility_adjusted_spread(result)

            # Generate microstructure signals
            result = await self._generate_signals(result)

            logger.info(
                "Microstructure indicators calculated",
                rows=len(result)
            )

            return result

        except Exception as e:
            logger.error("Microstructure calculation failed", error=str(e))
            raise

    async def _classify_trade_direction(
        self,
        data: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Classify trade direction using tick rule.

        Tick rule: Compare current price to previous price
        - Uptick: Buy
        - Downtick: Sell
        - Zero tick: Use last non-zero tick

        Args:
            data: Input data

        Returns:
            DataFrame with trade_direction column
        """
        try:
            # Determine price column
            price_col = "price" if "price" in data.columns else "close"

            directions = []
            last_direction = TradeDirection.NEUTRAL

            prices = data[price_col].to_list()

            for i in range(len(prices)):
                if i == 0:
                    directions.append(TradeDirection.NEUTRAL.value)
                    continue

                curr_price = Decimal(str(prices[i]))
                prev_price = Decimal(str(prices[i-1]))

                if curr_price > prev_price:
                    last_direction = TradeDirection.BUY
                    directions.append(TradeDirection.BUY.value)
                elif curr_price < prev_price:
                    last_direction = TradeDirection.SELL
                    directions.append(TradeDirection.SELL.value)
                else:
                    # Zero tick: use last direction
                    directions.append(last_direction.value)

            result = data.with_columns([
                pl.Series("trade_direction", directions)
            ])

            # Add numeric direction (-1, 0, 1)
            direction_map = {
                TradeDirection.BUY.value: "1",
                TradeDirection.NEUTRAL.value: "0",
                TradeDirection.SELL.value: "-1"
            }

            result = result.with_columns([
                pl.col("trade_direction").map_elements(
                    lambda d: direction_map.get(d, "0"),
                    return_dtype=pl.Utf8
                ).alias("direction_numeric")
            ])

            return result

        except Exception as e:
            logger.error("Trade direction classification failed", error=str(e))
            raise

    async def _calculate_effective_spread(
        self,
        data: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Calculate effective spread.

        If bid/ask available: Effective Spread = ask - bid
        Otherwise: Estimate from price volatility

        Args:
            data: DataFrame with price data

        Returns:
            DataFrame with effective_spread column
        """
        try:
            has_bid_ask = "bid" in data.columns and "ask" in data.columns

            if has_bid_ask:
                # Use actual bid-ask spread
                spreads = []
                for row in data.iter_rows(named=True):
                    bid = Decimal(str(row["bid"]))
                    ask = Decimal(str(row["ask"]))
                    spread = ask - bid
                    spreads.append(str(spread))
            else:
                # Estimate spread from high-low range
                price_col = "price" if "price" in data.columns else "close"
                prices = [Decimal(str(p)) for p in data[price_col].to_list()]

                spreads = []
                for i in range(len(prices)):
                    if i < self.spread_window:
                        spreads.append(None)
                        continue

                    # Calculate volatility-based spread estimate
                    window_prices = prices[i-self.spread_window:i+1]
                    price_range = max(window_prices) - min(window_prices)

                    # Estimate spread as fraction of range
                    estimated_spread = price_range / Decimal(str(self.spread_window))
                    spreads.append(str(estimated_spread))

            result = data.with_columns([
                pl.Series("effective_spread", spreads)
            ])

            return result

        except Exception as e:
            logger.error("Effective spread calculation failed", error=str(e))
            raise

    async def _calculate_order_imbalance(
        self,
        data: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Calculate order imbalance.

        Order Imbalance = (Buy Volume - Sell Volume) / (Buy Volume + Sell Volume)

        Args:
            data: DataFrame with trade direction and volume

        Returns:
            DataFrame with order_imbalance column
        """
        try:
            imbalances = []

            volumes = [Decimal(str(v)) for v in data["volume"].to_list()]
            directions = data["direction_numeric"].to_list()

            for i in range(len(data)):
                if i < self.spread_window:
                    imbalances.append(None)
                    continue

                # Calculate buy and sell volumes over window
                buy_volume = Decimal("0")
                sell_volume = Decimal("0")

                for j in range(i - self.spread_window + 1, i + 1):
                    direction = Decimal(str(directions[j]))
                    volume = volumes[j]

                    if direction > Decimal("0"):
                        buy_volume += volume
                    elif direction < Decimal("0"):
                        sell_volume += volume

                total_volume = buy_volume + sell_volume

                if total_volume == Decimal("0"):
                    imbalance = Decimal("0")
                else:
                    imbalance = (buy_volume - sell_volume) / total_volume

                imbalances.append(str(imbalance))

            result = data.with_columns([
                pl.Series("order_imbalance", imbalances)
            ])

            return result

        except Exception as e:
            logger.error("Order imbalance calculation failed", error=str(e))
            raise

    async def _calculate_trade_intensity(
        self,
        data: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Calculate trade intensity (trades per second).

        Args:
            data: DataFrame with timestamps

        Returns:
            DataFrame with trade_intensity column
        """
        try:
            intensities = []

            timestamps = data["timestamp"].to_list()

            for i in range(len(data)):
                if i == 0:
                    intensities.append(None)
                    continue

                # Convert to datetime if needed
                curr_time = timestamps[i]
                if isinstance(curr_time, str):
                    curr_time = datetime.fromisoformat(curr_time.replace('Z', '+00:00'))

                # Count trades in window
                trade_count = 0
                window_start = curr_time - timedelta(seconds=self.intensity_window)

                for j in range(i, -1, -1):
                    trade_time = timestamps[j]
                    if isinstance(trade_time, str):
                        trade_time = datetime.fromisoformat(trade_time.replace('Z', '+00:00'))

                    if trade_time < window_start:
                        break

                    trade_count += 1

                # Calculate intensity (trades per second)
                intensity = Decimal(str(trade_count)) / Decimal(str(self.intensity_window))
                intensities.append(str(intensity))

            result = data.with_columns([
                pl.Series("trade_intensity", intensities)
            ])

            return result

        except Exception as e:
            logger.error("Trade intensity calculation failed", error=str(e))
            raise

    async def _calculate_price_impact(
        self,
        data: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Calculate price impact.

        Price Impact = |midpoint(t+n) - midpoint(t)| / volume(t)

        Args:
            data: DataFrame with price and volume

        Returns:
            DataFrame with price_impact column
        """
        try:
            price_col = "price" if "price" in data.columns else "close"
            prices = [Decimal(str(p)) for p in data[price_col].to_list()]
            volumes = [Decimal(str(v)) for v in data["volume"].to_list()]

            impacts = []

            for i in range(len(data)):
                if i + self.impact_window >= len(data):
                    impacts.append(None)
                    continue

                # Price change over impact window
                price_change = abs(prices[i + self.impact_window] - prices[i])

                # Normalize by volume
                if volumes[i] == Decimal("0"):
                    impact = Decimal("0")
                else:
                    impact = price_change / volumes[i]

                impacts.append(str(impact))

            result = data.with_columns([
                pl.Series("price_impact", impacts)
            ])

            return result

        except Exception as e:
            logger.error("Price impact calculation failed", error=str(e))
            raise

    async def _calculate_realized_spread(
        self,
        data: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Calculate realized spread.

        Realized Spread = 2 × direction × (price(t) - midpoint(t+n))

        Args:
            data: DataFrame with prices and directions

        Returns:
            DataFrame with realized_spread column
        """
        try:
            price_col = "price" if "price" in data.columns else "close"
            prices = [Decimal(str(p)) for p in data[price_col].to_list()]
            directions = [Decimal(str(d)) for d in data["direction_numeric"].to_list()]

            realized_spreads = []

            for i in range(len(data)):
                if i + self.impact_window >= len(data):
                    realized_spreads.append(None)
                    continue

                future_midpoint = prices[i + self.impact_window]
                current_price = prices[i]
                direction = directions[i]

                realized_spread = Decimal("2") * direction * (current_price - future_midpoint)
                realized_spreads.append(str(realized_spread))

            result = data.with_columns([
                pl.Series("realized_spread", realized_spreads)
            ])

            return result

        except Exception as e:
            logger.error("Realized spread calculation failed", error=str(e))
            raise

    async def _calculate_volatility_adjusted_spread(
        self,
        data: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Calculate volatility-adjusted spread.

        Adjusts spread for current volatility conditions.

        Args:
            data: DataFrame with spreads

        Returns:
            DataFrame with volatility_adjusted_spread column
        """
        try:
            price_col = "price" if "price" in data.columns else "close"
            prices = [Decimal(str(p)) for p in data[price_col].to_list()]

            spreads = data["effective_spread"].to_list()

            adjusted_spreads = []

            for i in range(len(data)):
                if i < self.volatility_window or spreads[i] is None:
                    adjusted_spreads.append(None)
                    continue

                # Calculate volatility over window
                window_prices = prices[i-self.volatility_window:i+1]

                # Standard deviation of returns
                returns = []
                for j in range(1, len(window_prices)):
                    ret = (window_prices[j] - window_prices[j-1]) / window_prices[j-1]
                    returns.append(ret)

                if len(returns) == 0:
                    adjusted_spreads.append(spreads[i])
                    continue

                mean_return = sum(returns) / Decimal(str(len(returns)))
                variance = sum((r - mean_return) ** 2 for r in returns) / Decimal(str(len(returns)))
                volatility = variance.sqrt() if variance > Decimal("0") else Decimal("0.00001")

                # Adjust spread by volatility
                spread = Decimal(str(spreads[i]))
                adjusted_spread = spread / volatility

                adjusted_spreads.append(str(adjusted_spread))

            result = data.with_columns([
                pl.Series("volatility_adjusted_spread", adjusted_spreads)
            ])

            return result

        except Exception as e:
            logger.error("Volatility-adjusted spread calculation failed", error=str(e))
            raise

    async def _generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Generate microstructure trading signals.

        Signal based on:
        - Order imbalance
        - Trade intensity
        - Price impact

        Args:
            data: DataFrame with microstructure metrics

        Returns:
            DataFrame with signal column
        """
        try:
            def determine_signal(row) -> str:
                try:
                    imbalance = row.get("order_imbalance")
                    intensity = row.get("trade_intensity")

                    if imbalance is None or intensity is None:
                        return MicrostructureSignal.NEUTRAL.value

                    imb = Decimal(str(imbalance))
                    ints = Decimal(str(intensity))

                    # High intensity + positive imbalance = buying pressure
                    # High intensity + negative imbalance = selling pressure

                    intensity_threshold = Decimal("1.0")  # From config ideally

                    if ints > intensity_threshold:
                        if imb > Decimal("0.3"):
                            return MicrostructureSignal.STRONG_BUYING.value
                        elif imb > Decimal("0.1"):
                            return MicrostructureSignal.BUYING.value
                        elif imb < Decimal("-0.3"):
                            return MicrostructureSignal.STRONG_SELLING.value
                        elif imb < Decimal("-0.1"):
                            return MicrostructureSignal.SELLING.value

                    # Normal intensity
                    if imb > Decimal("0.2"):
                        return MicrostructureSignal.BUYING.value
                    elif imb < Decimal("-0.2"):
                        return MicrostructureSignal.SELLING.value

                    return MicrostructureSignal.NEUTRAL.value

                except Exception:
                    return MicrostructureSignal.NEUTRAL.value

            signals = []
            for row in data.iter_rows(named=True):
                signals.append(determine_signal(row))

            result = data.with_columns([
                pl.Series("signal", signals)
            ])

            # Add signal strength
            signal_strength_map = {
                MicrostructureSignal.STRONG_BUYING.value: Decimal("2"),
                MicrostructureSignal.BUYING.value: Decimal("1"),
                MicrostructureSignal.NEUTRAL.value: Decimal("0"),
                MicrostructureSignal.SELLING.value: Decimal("-1"),
                MicrostructureSignal.STRONG_SELLING.value: Decimal("-2")
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

    def _validate_data(self, data: pl.DataFrame) -> None:
        """Validate input data format and content."""
        required_columns = ["timestamp", "volume"]

        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Need either 'price' or 'close'
        if "price" not in data.columns and "close" not in data.columns:
            raise ValueError("Data must contain 'price' or 'close' column")

        if len(data) == 0:
            raise ValueError("Data is empty")

    async def get_current_metrics(
        self,
        data: pl.DataFrame
    ) -> Optional[MicrostructureMetrics]:
        """
        Get current microstructure metrics.

        Args:
            data: DataFrame with calculated metrics

        Returns:
            MicrostructureMetrics dataclass or None
        """
        try:
            if len(data) == 0:
                return None

            last_row = data.row(-1, named=True)

            required = [
                "effective_spread", "realized_spread", "price_impact",
                "order_imbalance", "direction_numeric", "trade_intensity",
                "volatility_adjusted_spread"
            ]

            if any(last_row.get(k) is None for k in required):
                return None

            return MicrostructureMetrics(
                effective_spread=Decimal(str(last_row["effective_spread"])),
                realized_spread=Decimal(str(last_row["realized_spread"])),
                price_impact=Decimal(str(last_row["price_impact"])),
                order_imbalance=Decimal(str(last_row["order_imbalance"])),
                tick_rule_direction=Decimal(str(last_row["direction_numeric"])),
                trade_intensity=Decimal(str(last_row["trade_intensity"])),
                volatility_adjusted_spread=Decimal(str(last_row["volatility_adjusted_spread"]))
            )

        except Exception as e:
            logger.error("Failed to get current metrics", error=str(e))
            return None


async def create_microstructure_indicator(
    config: Dict[str, Any]
) -> MicrostructureIndicator:
    """
    Factory function to create Microstructure indicator instance.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized Microstructure indicator
    """
    return MicrostructureIndicator(config)
