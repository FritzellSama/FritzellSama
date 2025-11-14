"""
MACD (Moving Average Convergence Divergence) Indicator.

MACD is a trend-following momentum indicator that shows the relationship
between two moving averages of prices.
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


class MACDSignal(Enum):
    """MACD trading signals."""
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


@dataclass
class MACDValues:
    """MACD indicator values."""
    macd_line: Decimal
    signal_line: Decimal
    histogram: Decimal
    trend: str


class MACDIndicator:
    """
    Moving Average Convergence Divergence (MACD) indicator.

    MACD consists of:
    - MACD Line: Fast EMA - Slow EMA
    - Signal Line: EMA of MACD Line
    - Histogram: MACD Line - Signal Line

    Attributes:
        config: Configuration dictionary
        fast_period: Fast EMA period (typically 12)
        slow_period: Slow EMA period (typically 26)
        signal_period: Signal line EMA period (typically 9)

    Example:
        >>> config = {
        ...     "fast_period": 12,
        ...     "slow_period": 26,
        ...     "signal_period": 9
        ... }
        >>> macd = MACDIndicator(config)
        >>> result = await macd.calculate(price_data)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize MACD indicator.

        Args:
            config: Configuration dictionary containing:
                - fast_period: Fast EMA period
                - slow_period: Slow EMA period
                - signal_period: Signal line period
                - min_periods: Minimum periods required

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.fast_period: int = int(config["fast_period"])
        self.slow_period: int = int(config["slow_period"])
        self.signal_period: int = int(config["signal_period"])
        self.min_periods: int = int(config.get(
            "min_periods",
            self.slow_period + self.signal_period
        ))

        logger.info(
            "MACD indicator initialized",
            fast_period=self.fast_period,
            slow_period=self.slow_period,
            signal_period=self.signal_period
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_fields = ["fast_period", "slow_period", "signal_period"]

        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")

        fast = int(self.config["fast_period"])
        slow = int(self.config["slow_period"])
        signal = int(self.config["signal_period"])

        if fast < 1:
            raise ValueError("fast_period must be positive")
        if slow < 1:
            raise ValueError("slow_period must be positive")
        if signal < 1:
            raise ValueError("signal_period must be positive")
        if fast >= slow:
            raise ValueError("fast_period must be less than slow_period")

        logger.debug("MACD configuration validated")

    async def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate MACD indicator.

        Args:
            data: Polars DataFrame with columns: timestamp, close

        Returns:
            DataFrame with MACD values and signals

        Raises:
            ValueError: If data is invalid or insufficient
        """
        try:
            # Validate input data
            self._validate_data(data)

            # Calculate fast and slow EMAs
            result = await self._calculate_emas(data)

            # Calculate MACD line
            result = await self._calculate_macd_line(result)

            # Calculate signal line
            result = await self._calculate_signal_line(result)

            # Calculate histogram
            result = await self._calculate_histogram(result)

            # Generate signals
            result = await self._generate_signals(result)

            logger.info(
                "MACD calculated",
                rows=len(result)
            )

            return result

        except Exception as e:
            logger.error("MACD calculation failed", error=str(e))
            raise

    async def _calculate_emas(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate fast and slow Exponential Moving Averages.

        Args:
            data: DataFrame with close prices

        Returns:
            DataFrame with fast_ema and slow_ema columns
        """
        try:
            close_values = [Decimal(str(x)) for x in data["close"].to_list()]

            # Calculate fast EMA
            fast_ema = await self._calculate_ema(close_values, self.fast_period)

            # Calculate slow EMA
            slow_ema = await self._calculate_ema(close_values, self.slow_period)

            # Add to DataFrame
            result = data.with_columns([
                pl.Series("fast_ema", [str(v) if v is not None else None for v in fast_ema]),
                pl.Series("slow_ema", [str(v) if v is not None else None for v in slow_ema])
            ])

            return result

        except Exception as e:
            logger.error("EMA calculation failed", error=str(e))
            raise

    async def _calculate_ema(
        self,
        values: List[Decimal],
        period: int
    ) -> List[Optional[Decimal]]:
        """
        Calculate Exponential Moving Average.

        EMA formula: EMA(t) = α × Value(t) + (1-α) × EMA(t-1)
        where α = 2 / (period + 1)

        Args:
            values: List of Decimal values
            period: EMA period

        Returns:
            List of EMA values (None for initial period)
        """
        try:
            alpha = Decimal("2") / (Decimal(str(period)) + Decimal("1"))
            ema_values: List[Optional[Decimal]] = []

            # Initial EMA is SMA of first period
            if len(values) < period:
                return [None] * len(values)

            # Calculate initial SMA
            initial_sma = sum(values[:period]) / Decimal(str(period))

            # Fill initial values with None
            ema_values = [None] * (period - 1)
            ema_values.append(initial_sma)

            # Calculate EMA for remaining values
            ema = initial_sma
            for i in range(period, len(values)):
                ema = alpha * values[i] + (Decimal("1") - alpha) * ema
                ema_values.append(ema)

            return ema_values

        except Exception as e:
            logger.error("EMA calculation failed", error=str(e))
            raise

    async def _calculate_macd_line(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate MACD line (fast EMA - slow EMA).

        Args:
            data: DataFrame with fast_ema and slow_ema

        Returns:
            DataFrame with macd_line column
        """
        try:
            def calc_macd(row):
                fast = row["fast_ema"]
                slow = row["slow_ema"]

                if fast is None or slow is None:
                    return None

                return str(Decimal(str(fast)) - Decimal(str(slow)))

            macd_values = []
            for row in data.iter_rows(named=True):
                macd_values.append(calc_macd(row))

            result = data.with_columns([
                pl.Series("macd_line", macd_values)
            ])

            return result

        except Exception as e:
            logger.error("MACD line calculation failed", error=str(e))
            raise

    async def _calculate_signal_line(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate signal line (EMA of MACD line).

        Args:
            data: DataFrame with macd_line

        Returns:
            DataFrame with signal_line column
        """
        try:
            # Get MACD values as Decimal list
            macd_values = []
            for val in data["macd_line"].to_list():
                if val is not None:
                    macd_values.append(Decimal(str(val)))
                else:
                    macd_values.append(None)

            # Calculate EMA of MACD line
            signal_values = await self._calculate_ema_with_nulls(
                macd_values,
                self.signal_period
            )

            result = data.with_columns([
                pl.Series("signal_line", [str(v) if v is not None else None for v in signal_values])
            ])

            return result

        except Exception as e:
            logger.error("Signal line calculation failed", error=str(e))
            raise

    async def _calculate_ema_with_nulls(
        self,
        values: List[Optional[Decimal]],
        period: int
    ) -> List[Optional[Decimal]]:
        """
        Calculate EMA handling None values.

        Args:
            values: List of optional Decimal values
            period: EMA period

        Returns:
            List of EMA values
        """
        try:
            # Filter out None values to find start
            non_null_values = [v for v in values if v is not None]
            if len(non_null_values) < period:
                return [None] * len(values)

            # Find index where we have enough non-null values
            non_null_count = 0
            start_idx = 0
            for i, v in enumerate(values):
                if v is not None:
                    non_null_count += 1
                    if non_null_count == period:
                        start_idx = i
                        break

            # Calculate initial SMA
            initial_values = [v for v in values[:start_idx+1] if v is not None][-period:]
            initial_sma = sum(initial_values) / Decimal(str(period))

            # Initialize result
            ema_values: List[Optional[Decimal]] = [None] * start_idx
            ema_values.append(initial_sma)

            # Calculate EMA for remaining values
            alpha = Decimal("2") / (Decimal(str(period)) + Decimal("1"))
            ema = initial_sma

            for i in range(start_idx + 1, len(values)):
                if values[i] is None:
                    ema_values.append(None)
                else:
                    ema = alpha * values[i] + (Decimal("1") - alpha) * ema
                    ema_values.append(ema)

            return ema_values

        except Exception as e:
            logger.error("EMA with nulls calculation failed", error=str(e))
            raise

    async def _calculate_histogram(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate MACD histogram (MACD line - signal line).

        Args:
            data: DataFrame with macd_line and signal_line

        Returns:
            DataFrame with histogram column
        """
        try:
            def calc_histogram(row):
                macd = row["macd_line"]
                signal = row["signal_line"]

                if macd is None or signal is None:
                    return None

                return str(Decimal(str(macd)) - Decimal(str(signal)))

            histogram_values = []
            for row in data.iter_rows(named=True):
                histogram_values.append(calc_histogram(row))

            result = data.with_columns([
                pl.Series("histogram", histogram_values)
            ])

            return result

        except Exception as e:
            logger.error("Histogram calculation failed", error=str(e))
            raise

    async def _generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Generate trading signals based on MACD.

        Signal logic:
        - STRONG_BULLISH: MACD > signal AND histogram increasing AND MACD > 0
        - BULLISH: MACD crosses above signal OR histogram > 0
        - NEUTRAL: No clear signal
        - BEARISH: MACD crosses below signal OR histogram < 0
        - STRONG_BEARISH: MACD < signal AND histogram decreasing AND MACD < 0

        Args:
            data: DataFrame with MACD values

        Returns:
            DataFrame with signal column
        """
        try:
            signals = []
            histogram_values = data["histogram"].to_list()
            macd_values = data["macd_line"].to_list()
            signal_values = data["signal_line"].to_list()

            for i in range(len(data)):
                try:
                    macd = macd_values[i]
                    signal = signal_values[i]
                    histogram = histogram_values[i]

                    if any(v is None for v in [macd, signal, histogram]):
                        signals.append(MACDSignal.NEUTRAL.value)
                        continue

                    macd_dec = Decimal(str(macd))
                    signal_dec = Decimal(str(signal))
                    histogram_dec = Decimal(str(histogram))

                    # Check histogram trend
                    histogram_increasing = False
                    histogram_decreasing = False

                    if i > 0 and histogram_values[i-1] is not None:
                        prev_histogram = Decimal(str(histogram_values[i-1]))
                        histogram_increasing = histogram_dec > prev_histogram
                        histogram_decreasing = histogram_dec < prev_histogram

                    # Determine signal
                    if macd_dec > signal_dec:
                        if histogram_increasing and macd_dec > Decimal("0"):
                            signals.append(MACDSignal.STRONG_BULLISH.value)
                        else:
                            signals.append(MACDSignal.BULLISH.value)
                    elif macd_dec < signal_dec:
                        if histogram_decreasing and macd_dec < Decimal("0"):
                            signals.append(MACDSignal.STRONG_BEARISH.value)
                        else:
                            signals.append(MACDSignal.BEARISH.value)
                    else:
                        signals.append(MACDSignal.NEUTRAL.value)

                except Exception:
                    signals.append(MACDSignal.NEUTRAL.value)

            result = data.with_columns([
                pl.Series("signal", signals)
            ])

            # Add signal strength
            signal_strength_map = {
                MACDSignal.STRONG_BULLISH.value: Decimal("2"),
                MACDSignal.BULLISH.value: Decimal("1"),
                MACDSignal.NEUTRAL.value: Decimal("0"),
                MACDSignal.BEARISH.value: Decimal("-1"),
                MACDSignal.STRONG_BEARISH.value: Decimal("-2")
            }

            result = result.with_columns([
                pl.col("signal").map_elements(
                    lambda s: str(signal_strength_map.get(s, Decimal("0"))),
                    return_dtype=pl.Utf8
                ).alias("signal_strength")
            ])

            # Detect crossovers
            result = await self._detect_crossovers(result)

            return result

        except Exception as e:
            logger.error("Signal generation failed", error=str(e))
            raise

    async def _detect_crossovers(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Detect MACD line and signal line crossovers.

        Args:
            data: DataFrame with MACD values

        Returns:
            DataFrame with crossover columns
        """
        try:
            bullish_cross = []
            bearish_cross = []

            macd_values = data["macd_line"].to_list()
            signal_values = data["signal_line"].to_list()

            for i in range(len(data)):
                if i == 0:
                    bullish_cross.append(False)
                    bearish_cross.append(False)
                    continue

                curr_macd = macd_values[i]
                curr_signal = signal_values[i]
                prev_macd = macd_values[i-1]
                prev_signal = signal_values[i-1]

                if any(v is None for v in [curr_macd, curr_signal, prev_macd, prev_signal]):
                    bullish_cross.append(False)
                    bearish_cross.append(False)
                    continue

                curr_macd_dec = Decimal(str(curr_macd))
                curr_signal_dec = Decimal(str(curr_signal))
                prev_macd_dec = Decimal(str(prev_macd))
                prev_signal_dec = Decimal(str(prev_signal))

                # Bullish crossover: MACD crosses above signal
                is_bullish = (
                    prev_macd_dec <= prev_signal_dec and
                    curr_macd_dec > curr_signal_dec
                )
                bullish_cross.append(is_bullish)

                # Bearish crossover: MACD crosses below signal
                is_bearish = (
                    prev_macd_dec >= prev_signal_dec and
                    curr_macd_dec < curr_signal_dec
                )
                bearish_cross.append(is_bearish)

            result = data.with_columns([
                pl.Series("bullish_crossover", bullish_cross),
                pl.Series("bearish_crossover", bearish_cross)
            ])

            return result

        except Exception as e:
            logger.error("Crossover detection failed", error=str(e))
            raise

    def _validate_data(self, data: pl.DataFrame) -> None:
        """Validate input data format and content."""
        required_columns = ["timestamp", "close"]

        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        if len(data) < self.min_periods:
            raise ValueError(
                f"Insufficient data points: {len(data)} < {self.min_periods}"
            )

        # Check for null values
        if data["close"].null_count() > 0:
            raise ValueError("Data contains null values in 'close' column")

    async def get_current_values(
        self,
        data: pl.DataFrame
    ) -> Optional[MACDValues]:
        """
        Get current MACD values.

        Args:
            data: DataFrame with calculated MACD values

        Returns:
            MACDValues dataclass or None
        """
        try:
            if len(data) == 0:
                return None

            last_row = data.row(-1, named=True)

            required = ["macd_line", "signal_line", "histogram", "signal"]
            if any(last_row.get(k) is None for k in required):
                return None

            return MACDValues(
                macd_line=Decimal(str(last_row["macd_line"])),
                signal_line=Decimal(str(last_row["signal_line"])),
                histogram=Decimal(str(last_row["histogram"])),
                trend=last_row["signal"]
            )

        except Exception as e:
            logger.error("Failed to get current values", error=str(e))
            return None


async def create_macd_indicator(config: Dict[str, Any]) -> MACDIndicator:
    """
    Factory function to create MACD indicator instance.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized MACD indicator
    """
    return MACDIndicator(config)
