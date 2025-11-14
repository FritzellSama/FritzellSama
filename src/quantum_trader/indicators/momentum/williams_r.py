"""Williams %R Momentum Indicator.

Production-ready implementation of Williams %R oscillator using Polars
for high-performance momentum analysis.
"""

import asyncio
from decimal import Decimal
from typing import Dict, Any, Optional, List
from enum import Enum

import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class WilliamsRSignal(Enum):
    """Williams %R signal types."""
    OVERSOLD = "oversold"
    OVERBOUGHT = "overbought"
    NEUTRAL = "neutral"
    OVERSOLD_EXIT = "oversold_exit"
    OVERBOUGHT_EXIT = "overbought_exit"


class WilliamsRCalculator:
    """Calculate Williams %R momentum indicator.

    Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

    The indicator oscillates between 0 and -100:
    - Values above -20 indicate overbought conditions
    - Values below -80 indicate oversold conditions

    Attributes:
        config: Configuration dictionary with calculation parameters
        period: Lookback period for high/low calculation
        overbought_level: Threshold for overbought signal
        oversold_level: Threshold for oversold signal

    Example:
        >>> config = {"period": 14, "overbought_level": -20, "oversold_level": -80}
        >>> calculator = WilliamsRCalculator(config)
        >>> df = await calculator.calculate(market_data)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Williams %R calculator.

        Args:
            config: Configuration dictionary containing:
                - period: Lookback period (default: 14)
                - overbought_level: Overbought threshold (default: -20)
                - oversold_level: Oversold threshold (default: -80)
                - signal_smoothing: Periods for signal smoothing (optional)

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.period = self.config.get("period", 14)
        self.overbought_level = Decimal(str(self.config.get("overbought_level", "-20")))
        self.oversold_level = Decimal(str(self.config.get("oversold_level", "-80")))
        self.signal_smoothing = self.config.get("signal_smoothing", None)

        logger.info(
            "williams_r_calculator_initialized",
            period=self.period,
            overbought_level=str(self.overbought_level),
            oversold_level=str(self.oversold_level),
            signal_smoothing=self.signal_smoothing
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        period = self.config.get("period", 14)
        if period < 1:
            raise ValueError(f"period must be positive, got {period}")

        overbought = self.config.get("overbought_level", -20)
        if overbought < -100 or overbought > 0:
            raise ValueError(f"overbought_level must be between -100 and 0, got {overbought}")

        oversold = self.config.get("oversold_level", -80)
        if oversold < -100 or oversold > 0:
            raise ValueError(f"oversold_level must be between -100 and 0, got {oversold}")

        if overbought <= oversold:
            raise ValueError("overbought_level must be greater than oversold_level")

        smoothing = self.config.get("signal_smoothing")
        if smoothing is not None and smoothing < 1:
            raise ValueError(f"signal_smoothing must be positive, got {smoothing}")

    async def calculate(
        self,
        data: pl.DataFrame,
        include_signals: bool = True
    ) -> pl.DataFrame:
        """Calculate Williams %R for market data.

        Args:
            data: Polars DataFrame with columns:
                - timestamp: UTC timestamp
                - high: High price
                - low: Low price
                - close: Close price
            include_signals: Whether to generate trading signals

        Returns:
            DataFrame with additional columns:
                - williams_r: Williams %R value (-100 to 0)
                - williams_r_signal: Signal type (if include_signals=True)
                - williams_r_smooth: Smoothed value (if smoothing enabled)

        Raises:
            ValueError: If data is invalid or missing required columns
        """
        try:
            self._validate_data(data)

            logger.debug(
                "calculating_williams_r",
                rows=len(data),
                period=self.period,
                include_signals=include_signals
            )

            # Calculate highest high and lowest low over the period
            result = data.with_columns([
                pl.col("high")
                .rolling_max(window_size=self.period)
                .alias("highest_high"),
                pl.col("low")
                .rolling_min(window_size=self.period)
                .alias("lowest_low")
            ])

            # Calculate Williams %R
            # Formula: (Highest High - Close) / (Highest High - Lowest Low) * -100
            result = result.with_columns([
                (
                    (pl.col("highest_high") - pl.col("close")) /
                    (pl.col("highest_high") - pl.col("lowest_low")) *
                    -100.0
                ).alias("williams_r")
            ])

            # Handle division by zero (when highest_high == lowest_low)
            result = result.with_columns([
                pl.when(pl.col("highest_high") == pl.col("lowest_low"))
                .then(pl.lit(-50.0))
                .otherwise(pl.col("williams_r"))
                .alias("williams_r")
            ])

            # Apply smoothing if configured
            if self.signal_smoothing is not None:
                result = result.with_columns([
                    pl.col("williams_r")
                    .rolling_mean(window_size=self.signal_smoothing)
                    .alias("williams_r_smooth")
                ])

            # Generate trading signals if requested
            if include_signals:
                result = await self._generate_signals(result)

            # Clean up intermediate columns
            result = result.drop(["highest_high", "lowest_low"])

            logger.debug("williams_r_calculated", rows=len(result))

            return result

        except Exception as e:
            logger.error("williams_r_calculation_failed", error=str(e))
            raise

    async def _generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """Generate trading signals from Williams %R values.

        Args:
            data: DataFrame with Williams %R calculated

        Returns:
            DataFrame with signal column added
        """
        try:
            overbought = float(self.overbought_level)
            oversold = float(self.oversold_level)

            # Determine which Williams %R to use for signals
            signal_col = "williams_r_smooth" if self.signal_smoothing else "williams_r"

            # Create signal based on thresholds
            result = data.with_columns([
                pl.when(pl.col(signal_col) >= overbought)
                .then(pl.lit(WilliamsRSignal.OVERBOUGHT.value))
                .when(pl.col(signal_col) <= oversold)
                .then(pl.lit(WilliamsRSignal.OVERSOLD.value))
                .otherwise(pl.lit(WilliamsRSignal.NEUTRAL.value))
                .alias("williams_r_signal")
            ])

            # Detect signal transitions (exits from overbought/oversold)
            result = result.with_columns([
                pl.col("williams_r_signal").shift(1).alias("prev_signal")
            ])

            result = result.with_columns([
                pl.when(
                    (pl.col("prev_signal") == WilliamsRSignal.OVERSOLD.value) &
                    (pl.col("williams_r_signal") == WilliamsRSignal.NEUTRAL.value)
                )
                .then(pl.lit(WilliamsRSignal.OVERSOLD_EXIT.value))
                .when(
                    (pl.col("prev_signal") == WilliamsRSignal.OVERBOUGHT.value) &
                    (pl.col("williams_r_signal") == WilliamsRSignal.NEUTRAL.value)
                )
                .then(pl.lit(WilliamsRSignal.OVERBOUGHT_EXIT.value))
                .otherwise(pl.col("williams_r_signal"))
                .alias("williams_r_signal")
            ])

            # Clean up temporary column
            result = result.drop(["prev_signal"])

            return result

        except Exception as e:
            logger.error("signal_generation_failed", error=str(e))
            raise

    def _validate_data(self, data: pl.DataFrame) -> None:
        """Validate input data.

        Args:
            data: Input DataFrame

        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(data, pl.DataFrame):
            raise ValueError("Data must be a Polars DataFrame")

        if len(data) == 0:
            raise ValueError("Data cannot be empty")

        if len(data) < self.period:
            raise ValueError(
                f"Data length ({len(data)}) must be >= period ({self.period})"
            )

        required_columns = ["timestamp", "high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Validate no null values in critical columns
        for col in required_columns:
            null_count = data[col].null_count()
            if null_count > 0:
                raise ValueError(f"Column '{col}' contains {null_count} null values")

    async def calculate_divergence(
        self,
        data: pl.DataFrame,
        lookback: Optional[int] = None
    ) -> pl.DataFrame:
        """Detect bullish/bearish divergence between price and Williams %R.

        Divergence occurs when price makes new highs/lows but Williams %R doesn't,
        or vice versa, signaling potential reversal.

        Args:
            data: DataFrame with Williams %R already calculated
            lookback: Periods to look back for divergence (default: 2 * period)

        Returns:
            DataFrame with divergence column added

        Raises:
            ValueError: If Williams %R not present
        """
        try:
            if "williams_r" not in data.columns:
                raise ValueError("Williams %R must be calculated first")

            if lookback is None:
                lookback = self.period * 2

            logger.debug("detecting_divergence", lookback=lookback)

            # Find local peaks and troughs in price
            result = data.with_columns([
                pl.col("close")
                .rolling_max(window_size=lookback)
                .alias("price_high"),
                pl.col("close")
                .rolling_min(window_size=lookback)
                .alias("price_low"),
                pl.col("williams_r")
                .rolling_max(window_size=lookback)
                .alias("wr_high"),
                pl.col("williams_r")
                .rolling_min(window_size=lookback)
                .alias("wr_low")
            ])

            # Detect bullish divergence: price makes lower low, WR makes higher low
            result = result.with_columns([
                pl.when(
                    (pl.col("close") == pl.col("price_low")) &
                    (pl.col("close") < pl.col("close").shift(lookback)) &
                    (pl.col("williams_r") > pl.col("williams_r").shift(lookback))
                )
                .then(pl.lit("bullish"))
                .when(
                    (pl.col("close") == pl.col("price_high")) &
                    (pl.col("close") > pl.col("close").shift(lookback)) &
                    (pl.col("williams_r") < pl.col("williams_r").shift(lookback))
                )
                .then(pl.lit("bearish"))
                .otherwise(pl.lit(None))
                .alias("williams_r_divergence")
            ])

            # Clean up intermediate columns
            result = result.drop(["price_high", "price_low", "wr_high", "wr_low"])

            return result

        except Exception as e:
            logger.error("divergence_detection_failed", error=str(e))
            raise

    def get_strength(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate momentum strength from Williams %R.

        Converts Williams %R to a 0-100 scale for easier interpretation.

        Args:
            data: DataFrame with Williams %R calculated

        Returns:
            DataFrame with williams_r_strength column (0-100)

        Raises:
            ValueError: If Williams %R not present
        """
        try:
            if "williams_r" not in data.columns:
                raise ValueError("Williams %R must be calculated first")

            # Convert -100 to 0 scale to 0 to 100 scale
            result = data.with_columns([
                (pl.col("williams_r") + 100.0).alias("williams_r_strength")
            ])

            return result

        except Exception as e:
            logger.error("strength_calculation_failed", error=str(e))
            raise

    async def calculate_multi_timeframe(
        self,
        data: pl.DataFrame,
        periods: List[int]
    ) -> pl.DataFrame:
        """Calculate Williams %R for multiple periods.

        Args:
            data: Input market data
            periods: List of periods to calculate

        Returns:
            DataFrame with Williams %R for each period

        Raises:
            ValueError: If periods list is empty or invalid
        """
        try:
            if not periods:
                raise ValueError("periods list cannot be empty")

            if any(p < 1 for p in periods):
                raise ValueError("All periods must be positive")

            logger.debug("calculating_multi_timeframe", periods=periods)

            result = data.clone()

            for period in periods:
                # Temporarily update period
                original_period = self.period
                self.period = period

                # Calculate Williams %R for this period
                temp = await self.calculate(data, include_signals=False)

                # Add to result with period suffix
                result = result.with_columns([
                    temp["williams_r"].alias(f"williams_r_{period}")
                ])

                # Restore original period
                self.period = original_period

            return result

        except Exception as e:
            logger.error("multi_timeframe_calculation_failed", error=str(e))
            raise
