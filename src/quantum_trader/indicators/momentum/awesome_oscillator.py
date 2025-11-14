"""Awesome Oscillator (AO) - Momentum Indicator.

The Awesome Oscillator is a momentum indicator that measures market momentum
by comparing recent market momentum with general momentum over a longer period.
Created by Bill Williams.
"""

import asyncio
from decimal import Decimal
from typing import Any, Dict, Optional

import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class AwesomeOscillator:
    """Awesome Oscillator momentum indicator.

    The AO is calculated as the difference between a 5-period and 34-period
    simple moving average of the median price (high + low) / 2. It shows
    what is happening to the market driving force at the present moment.

    Attributes:
        config: Configuration dictionary containing indicator parameters
        fast_period: Fast SMA period (typically 5)
        slow_period: Slow SMA period (typically 34)

    Example:
        >>> config = {"fast_period": 5, "slow_period": 34}
        >>> ao = AwesomeOscillator(config)
        >>> result = await ao.calculate(df)
        >>> print(result.select(["timestamp", "ao", "ao_signal"]))
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Awesome Oscillator.

        Args:
            config: Configuration dictionary with keys:
                - fast_period: int, fast SMA period (default from config)
                - slow_period: int, slow SMA period (default from config)
                - min_periods: int, minimum periods required

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()
        self.fast_period = int(self.config.get("fast_period", self.config.get("default_fast_period", 5)))
        self.slow_period = int(self.config.get("slow_period", self.config.get("default_slow_period", 34)))
        self.min_periods = int(self.config.get("min_periods", self.slow_period))

        logger.info(
            "awesome_oscillator_initialized",
            fast_period=self.fast_period,
            slow_period=self.slow_period,
            min_periods=self.min_periods
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        fast_period = self.config.get("fast_period", self.config.get("default_fast_period", 5))
        slow_period = self.config.get("slow_period", self.config.get("default_slow_period", 34))

        if not isinstance(fast_period, (int, str)) or int(fast_period) < 1:
            raise ValueError(f"Invalid fast_period: {fast_period}. Must be positive integer")

        if not isinstance(slow_period, (int, str)) or int(slow_period) < 1:
            raise ValueError(f"Invalid slow_period: {slow_period}. Must be positive integer")

        if int(fast_period) >= int(slow_period):
            raise ValueError(
                f"fast_period ({fast_period}) must be less than slow_period ({slow_period})"
            )

        logger.debug("awesome_oscillator_config_validated", config=self.config)

    async def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate Awesome Oscillator.

        Args:
            data: Polars DataFrame with columns:
                - high: High prices
                - low: Low prices
                - timestamp: Optional timestamp column

        Returns:
            DataFrame with additional columns:
                - median_price: (high + low) / 2
                - ao: Awesome Oscillator value
                - ao_histogram: Color indicator (1 for green, -1 for red, 0 for neutral)

        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        try:
            self._validate_data(data)

            logger.debug(
                "calculating_awesome_oscillator",
                rows=len(data),
                fast_period=self.fast_period,
                slow_period=self.slow_period
            )

            # Calculate median price
            result = data.with_columns([
                ((pl.col("high") + pl.col("low")) / pl.lit(2)).alias("median_price")
            ])

            # Calculate fast and slow SMAs
            result = result.with_columns([
                pl.col("median_price")
                  .rolling_mean(window_size=self.fast_period, min_periods=self.fast_period)
                  .alias("sma_fast"),
                pl.col("median_price")
                  .rolling_mean(window_size=self.slow_period, min_periods=self.slow_period)
                  .alias("sma_slow")
            ])

            # Calculate AO = fast SMA - slow SMA
            result = result.with_columns([
                (pl.col("sma_fast") - pl.col("sma_slow")).alias("ao")
            ])

            # Calculate histogram color (green if increasing, red if decreasing)
            result = result.with_columns([
                pl.col("ao").shift(1).alias("ao_prev")
            ])

            result = result.with_columns([
                pl.when(pl.col("ao") > pl.col("ao_prev"))
                  .then(pl.lit(1))  # Green - bullish
                  .when(pl.col("ao") < pl.col("ao_prev"))
                  .then(pl.lit(-1))  # Red - bearish
                  .otherwise(pl.lit(0))  # Neutral
                  .alias("ao_histogram")
            ])

            # Drop intermediate columns
            result = result.drop(["sma_fast", "sma_slow", "ao_prev"])

            logger.debug(
                "awesome_oscillator_calculated",
                rows=len(result),
                non_null_values=result.filter(pl.col("ao").is_not_null()).height,
                avg_ao=result["ao"].mean()
            )

            return result

        except Exception as e:
            logger.error(
                "awesome_oscillator_calculation_failed",
                error=str(e),
                error_type=type(e).__name__
            )
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

        required_columns = ["high", "low"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if len(data) < self.min_periods:
            raise ValueError(
                f"Insufficient data: {len(data)} rows, need at least {self.min_periods}"
            )

        # Check for null values in required columns
        for col in required_columns:
            null_count = data[col].null_count()
            if null_count > 0:
                logger.warning(
                    "null_values_detected",
                    column=col,
                    null_count=null_count
                )

    def detect_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """Detect trading signals from Awesome Oscillator.

        Classic AO signals:
        1. Zero-line crossover: AO crosses above/below zero
        2. Saucer: Three consecutive bars showing momentum shift
        3. Twin Peaks: Divergence pattern

        Args:
            data: DataFrame with AO values

        Returns:
            DataFrame with additional signal columns:
                - ao_zero_cross: "bullish", "bearish", or None
                - ao_saucer: "bullish", "bearish", or None
                - ao_momentum: "strengthening", "weakening", or "neutral"

        Raises:
            ValueError: If AO column is missing
        """
        try:
            if "ao" not in data.columns:
                raise ValueError("AO column not found. Calculate AO first.")

            result = data.with_columns([
                pl.col("ao").shift(1).alias("ao_prev"),
                pl.col("ao").shift(2).alias("ao_prev2")
            ])

            # Zero-line crossover
            result = result.with_columns([
                pl.when(
                    (pl.col("ao") > pl.lit(0)) & (pl.col("ao_prev") <= pl.lit(0))
                )
                  .then(pl.lit("bullish"))
                  .when(
                    (pl.col("ao") < pl.lit(0)) & (pl.col("ao_prev") >= pl.lit(0))
                  )
                  .then(pl.lit("bearish"))
                  .otherwise(pl.lit(None))
                  .alias("ao_zero_cross")
            ])

            # Saucer signal (3 consecutive bars showing momentum shift)
            # Bullish saucer: all bars below zero, middle bar is lowest, last bar is higher
            # Bearish saucer: all bars above zero, middle bar is highest, last bar is lower
            result = result.with_columns([
                pl.when(
                    (pl.col("ao") < pl.lit(0)) &
                    (pl.col("ao_prev") < pl.lit(0)) &
                    (pl.col("ao_prev2") < pl.lit(0)) &
                    (pl.col("ao_prev") < pl.col("ao_prev2")) &
                    (pl.col("ao") > pl.col("ao_prev"))
                )
                  .then(pl.lit("bullish"))
                  .when(
                    (pl.col("ao") > pl.lit(0)) &
                    (pl.col("ao_prev") > pl.lit(0)) &
                    (pl.col("ao_prev2") > pl.lit(0)) &
                    (pl.col("ao_prev") > pl.col("ao_prev2")) &
                    (pl.col("ao") < pl.col("ao_prev"))
                  )
                  .then(pl.lit("bearish"))
                  .otherwise(pl.lit(None))
                  .alias("ao_saucer")
            ])

            # Momentum assessment
            result = result.with_columns([
                pl.when(
                    (pl.col("ao") > pl.col("ao_prev")) &
                    (pl.col("ao_prev") > pl.col("ao_prev2"))
                )
                  .then(pl.lit("strengthening"))
                  .when(
                    (pl.col("ao") < pl.col("ao_prev")) &
                    (pl.col("ao_prev") < pl.col("ao_prev2"))
                  )
                  .then(pl.lit("weakening"))
                  .otherwise(pl.lit("neutral"))
                  .alias("ao_momentum")
            ])

            # Drop intermediate columns
            result = result.drop(["ao_prev", "ao_prev2"])

            logger.debug("awesome_oscillator_signals_detected", rows=len(result))

            return result

        except Exception as e:
            logger.error("awesome_oscillator_signal_detection_failed", error=str(e))
            raise

    def detect_divergence(
        self,
        data: pl.DataFrame,
        lookback_periods: Optional[int] = None
    ) -> pl.DataFrame:
        """Detect bullish and bearish divergences.

        Bullish divergence: Price makes lower lows, but AO makes higher lows
        Bearish divergence: Price makes higher highs, but AO makes lower highs

        Args:
            data: DataFrame with AO and price data
            lookback_periods: Periods to look back for divergence detection

        Returns:
            DataFrame with divergence signals

        Raises:
            ValueError: If required columns are missing
        """
        try:
            required_cols = ["ao", "close"]
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            if lookback_periods is None:
                lookback_periods = int(self.config.get("divergence_lookback", 20))

            # This is a simplified divergence detection
            # Full implementation would require peak/trough detection algorithms

            result = data.with_columns([
                pl.col("close").rolling_min(window_size=lookback_periods).alias("price_low"),
                pl.col("close").rolling_max(window_size=lookback_periods).alias("price_high"),
                pl.col("ao").rolling_min(window_size=lookback_periods).alias("ao_low"),
                pl.col("ao").rolling_max(window_size=lookback_periods).alias("ao_high")
            ])

            logger.debug(
                "divergence_detection_completed",
                lookback_periods=lookback_periods
            )

            return result

        except Exception as e:
            logger.error("divergence_detection_failed", error=str(e))
            raise


async def calculate_awesome_oscillator(
    data: pl.DataFrame,
    config: Dict[str, Any]
) -> pl.DataFrame:
    """Convenience function to calculate Awesome Oscillator.

    Args:
        data: Polars DataFrame with high and low prices
        config: Configuration dictionary

    Returns:
        DataFrame with Awesome Oscillator

    Example:
        >>> config = {"fast_period": 5, "slow_period": 34}
        >>> result = await calculate_awesome_oscillator(df, config)
    """
    indicator = AwesomeOscillator(config)
    return await indicator.calculate(data)
