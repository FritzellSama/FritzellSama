"""Aroon Indicator - Trend Strength and Direction.

The Aroon indicator identifies trend changes and measures trend strength.
It consists of two lines: Aroon Up and Aroon Down, oscillating between 0-100.
"""

import asyncio
from decimal import Decimal
from typing import Any, Dict, Optional

import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class AroonIndicator:
    """Aroon indicator for trend identification and strength measurement.

    The Aroon indicator measures the time elapsed since the highest high
    and lowest low over a given period. Values range from 0 to 100.

    Attributes:
        config: Configuration dictionary containing indicator parameters
        period: Lookback period for high/low calculation

    Example:
        >>> config = {"period": 25}
        >>> aroon = AroonIndicator(config)
        >>> result = await aroon.calculate(df)
        >>> print(result.select(["timestamp", "aroon_up", "aroon_down", "aroon_oscillator"]))
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Aroon indicator.

        Args:
            config: Configuration dictionary with keys:
                - period: int, lookback period (default from config)
                - min_periods: int, minimum periods required

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()
        self.period = int(self.config.get("period", self.config.get("default_period", 25)))
        self.min_periods = int(self.config.get("min_periods", self.period))

        logger.info(
            "aroon_initialized",
            period=self.period,
            min_periods=self.min_periods
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        period = self.config.get("period", self.config.get("default_period", 25))
        if not isinstance(period, (int, str)) or int(period) < 1:
            raise ValueError(f"Invalid period: {period}. Must be positive integer")

        logger.debug("aroon_config_validated", config=self.config)

    async def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate Aroon Up, Aroon Down, and Aroon Oscillator.

        Args:
            data: Polars DataFrame with columns:
                - high: High prices
                - low: Low prices
                - timestamp: Optional timestamp column

        Returns:
            DataFrame with additional columns:
                - aroon_up: Aroon Up line (0-100)
                - aroon_down: Aroon Down line (0-100)
                - aroon_oscillator: Difference between Up and Down (-100 to 100)

        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        try:
            self._validate_data(data)

            logger.debug(
                "calculating_aroon",
                rows=len(data),
                period=self.period
            )

            # Calculate periods since highest high and lowest low
            result = data.with_columns([
                # Get rolling max/min
                pl.col("high").rolling_max(window_size=self.period).alias("rolling_high"),
                pl.col("low").rolling_min(window_size=self.period).alias("rolling_low"),
            ])

            # Calculate Aroon Up and Down using a more efficient approach
            # We need to find the number of periods since the high/low occurred
            aroon_data = []

            highs = data["high"].to_list()
            lows = data["low"].to_list()

            for i in range(len(highs)):
                if i < self.min_periods - 1:
                    aroon_data.append({
                        "aroon_up": None,
                        "aroon_down": None,
                        "aroon_oscillator": None
                    })
                else:
                    # Get the window
                    start_idx = max(0, i - self.period + 1)
                    window_highs = highs[start_idx:i + 1]
                    window_lows = lows[start_idx:i + 1]

                    # Find periods since highest high (from end)
                    max_high = max(window_highs)
                    periods_since_high = len(window_highs) - 1 - window_highs[::-1].index(max_high)

                    # Find periods since lowest low (from end)
                    min_low = min(window_lows)
                    periods_since_low = len(window_lows) - 1 - window_lows[::-1].index(min_low)

                    # Calculate Aroon values (0-100)
                    actual_period = len(window_highs)
                    aroon_up = Decimal(((actual_period - periods_since_high) / actual_period) * 100)
                    aroon_down = Decimal(((actual_period - periods_since_low) / actual_period) * 100)
                    aroon_oscillator = aroon_up - aroon_down

                    aroon_data.append({
                        "aroon_up": float(aroon_up),
                        "aroon_down": float(aroon_down),
                        "aroon_oscillator": float(aroon_oscillator)
                    })

            # Create DataFrame from results and join with original
            aroon_df = pl.DataFrame(aroon_data)
            result = pl.concat([data, aroon_df], how="horizontal")

            logger.debug(
                "aroon_calculated",
                rows=len(result),
                non_null_values=result.filter(pl.col("aroon_up").is_not_null()).height
            )

            return result

        except Exception as e:
            logger.error(
                "aroon_calculation_failed",
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

    def interpret_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """Interpret Aroon signals for trading.

        Args:
            data: DataFrame with Aroon values

        Returns:
            DataFrame with additional signal columns:
                - aroon_signal: "bullish", "bearish", or "neutral"
                - aroon_strength: "strong", "moderate", or "weak"

        Raises:
            ValueError: If required Aroon columns are missing
        """
        try:
            required_cols = ["aroon_up", "aroon_down", "aroon_oscillator"]
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise ValueError(f"Missing Aroon columns: {missing}")

            # Thresholds from config
            strong_threshold = Decimal(str(self.config.get("strong_threshold", "70")))
            weak_threshold = Decimal(str(self.config.get("weak_threshold", "30")))

            result = data.with_columns([
                # Determine signal
                pl.when(pl.col("aroon_up") > strong_threshold)
                  .then(pl.lit("bullish"))
                  .when(pl.col("aroon_down") > strong_threshold)
                  .then(pl.lit("bearish"))
                  .otherwise(pl.lit("neutral"))
                  .alias("aroon_signal"),

                # Determine strength
                pl.when(
                    (pl.col("aroon_up") > strong_threshold) |
                    (pl.col("aroon_down") > strong_threshold)
                )
                  .then(pl.lit("strong"))
                  .when(
                    (pl.col("aroon_up") < weak_threshold) &
                    (pl.col("aroon_down") < weak_threshold)
                )
                  .then(pl.lit("weak"))
                  .otherwise(pl.lit("moderate"))
                  .alias("aroon_strength")
            ])

            logger.debug("aroon_signals_interpreted", rows=len(result))

            return result

        except Exception as e:
            logger.error("aroon_signal_interpretation_failed", error=str(e))
            raise


async def calculate_aroon(
    data: pl.DataFrame,
    config: Dict[str, Any]
) -> pl.DataFrame:
    """Convenience function to calculate Aroon indicator.

    Args:
        data: Polars DataFrame with high and low prices
        config: Configuration dictionary

    Returns:
        DataFrame with Aroon indicators

    Example:
        >>> config = {"period": 25}
        >>> result = await calculate_aroon(df, config)
    """
    indicator = AroonIndicator(config)
    return await indicator.calculate(data)
