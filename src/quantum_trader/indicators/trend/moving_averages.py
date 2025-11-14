"""
Moving Averages Indicator - Comprehensive trend-following indicators.

Implements multiple types of moving averages:
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Weighted Moving Average (WMA)
- Hull Moving Average (HMA)
- Volume Weighted Moving Average (VWMA)
"""

import asyncio
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any, Tuple
import polars as pl
from structlog import get_logger
from dataclasses import dataclass
from enum import Enum

logger = get_logger(__name__)

# Set high precision for Decimal calculations
getcontext().prec = 28


class MAType(Enum):
    """Moving average types."""
    SMA = "sma"  # Simple Moving Average
    EMA = "ema"  # Exponential Moving Average
    WMA = "wma"  # Weighted Moving Average
    HMA = "hma"  # Hull Moving Average
    VWMA = "vwma"  # Volume Weighted Moving Average


class TrendSignal(Enum):
    """Trend signals based on moving averages."""
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    NEUTRAL = "neutral"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


@dataclass
class MAValues:
    """Moving average values."""
    ma_value: Decimal
    ma_type: str
    period: int


class MovingAverageIndicator:
    """
    Comprehensive moving average indicator.

    Supports multiple MA types and can calculate multiple MAs simultaneously.

    Attributes:
        config: Configuration dictionary
        ma_type: Type of moving average
        periods: List of periods to calculate
        source: Price source column (close, high, low, etc.)

    Example:
        >>> config = {
        ...     "ma_type": "ema",
        ...     "periods": [20, 50, 200],
        ...     "source": "close"
        ... }
        >>> ma = MovingAverageIndicator(config)
        >>> result = await ma.calculate(price_data)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize moving average indicator.

        Args:
            config: Configuration dictionary containing:
                - ma_type: Type of MA (sma, ema, wma, hma, vwma)
                - periods: List of periods or single period
                - source: Source column (default: close)
                - min_periods: Minimum periods required

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.ma_type: MAType = MAType(config["ma_type"])
        self.source: str = config.get("source", "close")

        # Handle single period or list of periods
        periods_config = config["periods"]
        if isinstance(periods_config, (list, tuple)):
            self.periods: List[int] = [int(p) for p in periods_config]
        else:
            self.periods = [int(periods_config)]

        self.min_periods: int = int(config.get("min_periods", max(self.periods)))

        logger.info(
            "Moving Average indicator initialized",
            ma_type=self.ma_type.value,
            periods=self.periods,
            source=self.source
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_fields = ["ma_type", "periods"]

        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")

        # Validate MA type
        try:
            MAType(self.config["ma_type"])
        except ValueError:
            valid_types = [t.value for t in MAType]
            raise ValueError(f"Invalid ma_type. Must be one of: {valid_types}")

        # Validate periods
        periods_config = self.config["periods"]
        if isinstance(periods_config, (list, tuple)):
            for period in periods_config:
                if int(period) < 1:
                    raise ValueError("All periods must be positive")
        else:
            if int(periods_config) < 1:
                raise ValueError("Period must be positive")

        logger.debug("Moving Average configuration validated")

    async def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate moving averages.

        Args:
            data: Polars DataFrame with price data

        Returns:
            DataFrame with calculated moving averages

        Raises:
            ValueError: If data is invalid or insufficient
        """
        try:
            # Validate input data
            self._validate_data(data)

            # Calculate MAs based on type
            if self.ma_type == MAType.SMA:
                result = await self._calculate_sma(data)
            elif self.ma_type == MAType.EMA:
                result = await self._calculate_ema(data)
            elif self.ma_type == MAType.WMA:
                result = await self._calculate_wma(data)
            elif self.ma_type == MAType.HMA:
                result = await self._calculate_hma(data)
            elif self.ma_type == MAType.VWMA:
                result = await self._calculate_vwma(data)
            else:
                raise ValueError(f"Unsupported MA type: {self.ma_type}")

            # Generate trend signals if multiple MAs
            if len(self.periods) >= 2:
                result = await self._generate_trend_signals(result)

            # Detect crossovers
            result = await self._detect_crossovers(result)

            logger.info(
                "Moving averages calculated",
                ma_type=self.ma_type.value,
                periods=self.periods,
                rows=len(result)
            )

            return result

        except Exception as e:
            logger.error("Moving average calculation failed", error=str(e))
            raise

    async def _calculate_sma(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Simple Moving Average.

        SMA = Sum(prices, period) / period

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with SMA columns
        """
        try:
            result = data.clone()
            source_values = [Decimal(str(x)) for x in data[self.source].to_list()]

            for period in self.periods:
                sma_values = []

                for i in range(len(source_values)):
                    if i < period - 1:
                        sma_values.append(None)
                        continue

                    # Calculate SMA
                    window = source_values[i-period+1:i+1]
                    sma = sum(window) / Decimal(str(period))
                    sma_values.append(str(sma))

                col_name = f"sma_{period}"
                result = result.with_columns([
                    pl.Series(col_name, sma_values)
                ])

            return result

        except Exception as e:
            logger.error("SMA calculation failed", error=str(e))
            raise

    async def _calculate_ema(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Exponential Moving Average.

        EMA(t) = α × Price(t) + (1-α) × EMA(t-1)
        where α = 2 / (period + 1)

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with EMA columns
        """
        try:
            result = data.clone()
            source_values = [Decimal(str(x)) for x in data[self.source].to_list()]

            for period in self.periods:
                alpha = Decimal("2") / (Decimal(str(period)) + Decimal("1"))
                ema_values = []

                # Initialize with SMA
                if len(source_values) < period:
                    result = result.with_columns([
                        pl.Series(f"ema_{period}", [None] * len(source_values))
                    ])
                    continue

                # Calculate initial SMA
                initial_sma = sum(source_values[:period]) / Decimal(str(period))

                # Fill initial values with None
                ema_values = [None] * (period - 1)
                ema_values.append(str(initial_sma))

                # Calculate EMA for remaining values
                ema = initial_sma
                for i in range(period, len(source_values)):
                    ema = alpha * source_values[i] + (Decimal("1") - alpha) * ema
                    ema_values.append(str(ema))

                col_name = f"ema_{period}"
                result = result.with_columns([
                    pl.Series(col_name, ema_values)
                ])

            return result

        except Exception as e:
            logger.error("EMA calculation failed", error=str(e))
            raise

    async def _calculate_wma(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Weighted Moving Average.

        WMA gives more weight to recent prices.
        Weight = period, period-1, ..., 2, 1

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with WMA columns
        """
        try:
            result = data.clone()
            source_values = [Decimal(str(x)) for x in data[self.source].to_list()]

            for period in self.periods:
                wma_values = []

                # Calculate weights
                weights = [Decimal(str(i)) for i in range(1, period + 1)]
                weight_sum = sum(weights)

                for i in range(len(source_values)):
                    if i < period - 1:
                        wma_values.append(None)
                        continue

                    # Calculate WMA
                    window = source_values[i-period+1:i+1]
                    weighted_sum = sum(price * weight for price, weight in zip(window, weights))
                    wma = weighted_sum / weight_sum
                    wma_values.append(str(wma))

                col_name = f"wma_{period}"
                result = result.with_columns([
                    pl.Series(col_name, wma_values)
                ])

            return result

        except Exception as e:
            logger.error("WMA calculation failed", error=str(e))
            raise

    async def _calculate_hma(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Hull Moving Average.

        HMA aims to reduce lag while maintaining smoothness.
        HMA = WMA(2×WMA(n/2) - WMA(n), sqrt(n))

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with HMA columns
        """
        try:
            result = data.clone()
            source_values = [Decimal(str(x)) for x in data[self.source].to_list()]

            for period in self.periods:
                half_period = period // 2
                sqrt_period = int(Decimal(str(period)).sqrt())

                # Calculate WMA(n/2)
                wma_half = await self._calculate_wma_series(source_values, half_period)

                # Calculate WMA(n)
                wma_full = await self._calculate_wma_series(source_values, period)

                # Calculate 2×WMA(n/2) - WMA(n)
                diff_values = []
                for i in range(len(source_values)):
                    if wma_half[i] is None or wma_full[i] is None:
                        diff_values.append(None)
                    else:
                        diff = Decimal("2") * wma_half[i] - wma_full[i]
                        diff_values.append(diff)

                # Calculate HMA = WMA(diff, sqrt(n))
                hma_values = await self._calculate_wma_series(diff_values, sqrt_period)

                col_name = f"hma_{period}"
                result = result.with_columns([
                    pl.Series(col_name, [str(v) if v is not None else None for v in hma_values])
                ])

            return result

        except Exception as e:
            logger.error("HMA calculation failed", error=str(e))
            raise

    async def _calculate_wma_series(
        self,
        values: List[Optional[Decimal]],
        period: int
    ) -> List[Optional[Decimal]]:
        """Helper to calculate WMA for a series."""
        try:
            wma_values = []
            weights = [Decimal(str(i)) for i in range(1, period + 1)]
            weight_sum = sum(weights)

            for i in range(len(values)):
                if i < period - 1:
                    wma_values.append(None)
                    continue

                # Check for None values in window
                window = values[i-period+1:i+1]
                if any(v is None for v in window):
                    wma_values.append(None)
                    continue

                # Calculate WMA
                weighted_sum = sum(price * weight for price, weight in zip(window, weights))
                wma = weighted_sum / weight_sum
                wma_values.append(wma)

            return wma_values

        except Exception as e:
            logger.error("WMA series calculation failed", error=str(e))
            raise

    async def _calculate_vwma(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Volume Weighted Moving Average.

        VWMA = Sum(price × volume) / Sum(volume)

        Args:
            data: Input DataFrame (must have volume column)

        Returns:
            DataFrame with VWMA columns
        """
        try:
            if "volume" not in data.columns:
                raise ValueError("Volume column required for VWMA")

            result = data.clone()
            source_values = [Decimal(str(x)) for x in data[self.source].to_list()]
            volume_values = [Decimal(str(x)) for x in data["volume"].to_list()]

            for period in self.periods:
                vwma_values = []

                for i in range(len(source_values)):
                    if i < period - 1:
                        vwma_values.append(None)
                        continue

                    # Calculate VWMA
                    price_window = source_values[i-period+1:i+1]
                    volume_window = volume_values[i-period+1:i+1]

                    price_volume_sum = sum(p * v for p, v in zip(price_window, volume_window))
                    volume_sum = sum(volume_window)

                    if volume_sum == Decimal("0"):
                        vwma = sum(price_window) / Decimal(str(period))  # Fallback to SMA
                    else:
                        vwma = price_volume_sum / volume_sum

                    vwma_values.append(str(vwma))

                col_name = f"vwma_{period}"
                result = result.with_columns([
                    pl.Series(col_name, vwma_values)
                ])

            return result

        except Exception as e:
            logger.error("VWMA calculation failed", error=str(e))
            raise

    async def _generate_trend_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Generate trend signals based on multiple MA alignments.

        Uses the smallest and largest periods.

        Args:
            data: DataFrame with calculated MAs

        Returns:
            DataFrame with trend_signal column
        """
        try:
            if len(self.periods) < 2:
                return data

            # Get fast and slow MA column names
            fast_period = min(self.periods)
            slow_period = max(self.periods)

            fast_col = f"{self.ma_type.value}_{fast_period}"
            slow_col = f"{self.ma_type.value}_{slow_period}"

            def determine_signal(row) -> str:
                try:
                    price = row[self.source]
                    fast_ma = row.get(fast_col)
                    slow_ma = row.get(slow_col)

                    if any(v is None for v in [price, fast_ma, slow_ma]):
                        return TrendSignal.NEUTRAL.value

                    price_dec = Decimal(str(price))
                    fast_dec = Decimal(str(fast_ma))
                    slow_dec = Decimal(str(slow_ma))

                    # Strong uptrend: price > fast MA > slow MA
                    if price_dec > fast_dec > slow_dec:
                        return TrendSignal.STRONG_UPTREND.value
                    # Uptrend: price > fast MA or fast MA > slow MA
                    elif price_dec > fast_dec or fast_dec > slow_dec:
                        return TrendSignal.UPTREND.value
                    # Strong downtrend: price < fast MA < slow MA
                    elif price_dec < fast_dec < slow_dec:
                        return TrendSignal.STRONG_DOWNTREND.value
                    # Downtrend: price < fast MA or fast MA < slow MA
                    elif price_dec < fast_dec or fast_dec < slow_dec:
                        return TrendSignal.DOWNTREND.value
                    else:
                        return TrendSignal.NEUTRAL.value

                except Exception:
                    return TrendSignal.NEUTRAL.value

            signals = []
            for row in data.iter_rows(named=True):
                signals.append(determine_signal(row))

            result = data.with_columns([
                pl.Series("trend_signal", signals)
            ])

            # Add signal strength
            signal_strength_map = {
                TrendSignal.STRONG_UPTREND.value: Decimal("2"),
                TrendSignal.UPTREND.value: Decimal("1"),
                TrendSignal.NEUTRAL.value: Decimal("0"),
                TrendSignal.DOWNTREND.value: Decimal("-1"),
                TrendSignal.STRONG_DOWNTREND.value: Decimal("-2")
            }

            result = result.with_columns([
                pl.col("trend_signal").map_elements(
                    lambda s: str(signal_strength_map.get(s, Decimal("0"))),
                    return_dtype=pl.Utf8
                ).alias("signal_strength")
            ])

            return result

        except Exception as e:
            logger.error("Trend signal generation failed", error=str(e))
            raise

    async def _detect_crossovers(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Detect MA crossovers (golden cross, death cross).

        Args:
            data: DataFrame with MAs

        Returns:
            DataFrame with crossover columns
        """
        try:
            if len(self.periods) < 2:
                # No crossovers with single MA
                return data

            fast_period = min(self.periods)
            slow_period = max(self.periods)

            fast_col = f"{self.ma_type.value}_{fast_period}"
            slow_col = f"{self.ma_type.value}_{slow_period}"

            golden_crosses = []
            death_crosses = []

            fast_values = data[fast_col].to_list()
            slow_values = data[slow_col].to_list()

            for i in range(len(data)):
                if i == 0:
                    golden_crosses.append(False)
                    death_crosses.append(False)
                    continue

                curr_fast = fast_values[i]
                curr_slow = slow_values[i]
                prev_fast = fast_values[i-1]
                prev_slow = slow_values[i-1]

                if any(v is None for v in [curr_fast, curr_slow, prev_fast, prev_slow]):
                    golden_crosses.append(False)
                    death_crosses.append(False)
                    continue

                curr_fast_dec = Decimal(str(curr_fast))
                curr_slow_dec = Decimal(str(curr_slow))
                prev_fast_dec = Decimal(str(prev_fast))
                prev_slow_dec = Decimal(str(prev_slow))

                # Golden cross: fast crosses above slow
                is_golden = (
                    prev_fast_dec <= prev_slow_dec and
                    curr_fast_dec > curr_slow_dec
                )
                golden_crosses.append(is_golden)

                # Death cross: fast crosses below slow
                is_death = (
                    prev_fast_dec >= prev_slow_dec and
                    curr_fast_dec < curr_slow_dec
                )
                death_crosses.append(is_death)

            result = data.with_columns([
                pl.Series("golden_cross", golden_crosses),
                pl.Series("death_cross", death_crosses)
            ])

            return result

        except Exception as e:
            logger.error("Crossover detection failed", error=str(e))
            raise

    def _validate_data(self, data: pl.DataFrame) -> None:
        """Validate input data format and content."""
        required_columns = ["timestamp", self.source]

        # VWMA needs volume
        if self.ma_type == MAType.VWMA:
            required_columns.append("volume")

        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        if len(data) < self.min_periods:
            raise ValueError(
                f"Insufficient data points: {len(data)} < {self.min_periods}"
            )

        # Check for null values
        if data[self.source].null_count() > 0:
            raise ValueError(f"Data contains null values in '{self.source}' column")

    async def get_current_values(
        self,
        data: pl.DataFrame
    ) -> Dict[int, MAValues]:
        """
        Get current MA values for all periods.

        Args:
            data: DataFrame with calculated MAs

        Returns:
            Dictionary mapping periods to MAValues
        """
        try:
            if len(data) == 0:
                return {}

            last_row = data.row(-1, named=True)
            results = {}

            for period in self.periods:
                col_name = f"{self.ma_type.value}_{period}"
                if col_name in last_row and last_row[col_name] is not None:
                    results[period] = MAValues(
                        ma_value=Decimal(str(last_row[col_name])),
                        ma_type=self.ma_type.value,
                        period=period
                    )

            return results

        except Exception as e:
            logger.error("Failed to get current values", error=str(e))
            return {}


async def create_moving_average_indicator(
    config: Dict[str, Any]
) -> MovingAverageIndicator:
    """
    Factory function to create Moving Average indicator instance.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized Moving Average indicator
    """
    return MovingAverageIndicator(config)
