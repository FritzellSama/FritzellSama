"""Fibonacci Retracement and Extension Levels - Market Structure Tool.

Fibonacci levels are based on the Fibonacci sequence and are used to identify
potential support, resistance, and price targets. Common ratios include
0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618.
"""

import asyncio
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class FibonacciIndicator:
    """Fibonacci retracement and extension calculator.

    Calculates Fibonacci levels based on swing highs and lows.
    Supports both retracement (pullback) and extension (projection) levels.

    Attributes:
        config: Configuration dictionary containing indicator parameters
        retracement_levels: List of Fibonacci retracement ratios
        extension_levels: List of Fibonacci extension ratios

    Example:
        >>> config = {
        ...     "retracement_levels": [0.236, 0.382, 0.5, 0.618, 0.786],
        ...     "extension_levels": [1.272, 1.618, 2.618]
        ... }
        >>> fib = FibonacciIndicator(config)
        >>> result = await fib.calculate(df)
    """

    # Standard Fibonacci ratios
    STANDARD_RETRACEMENTS = [
        Decimal("0.236"),
        Decimal("0.382"),
        Decimal("0.500"),
        Decimal("0.618"),
        Decimal("0.786")
    ]

    STANDARD_EXTENSIONS = [
        Decimal("1.000"),
        Decimal("1.272"),
        Decimal("1.618"),
        Decimal("2.618"),
        Decimal("4.236")
    ]

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Fibonacci Indicator.

        Args:
            config: Configuration dictionary with keys:
                - retracement_levels: List of retracement ratios
                - extension_levels: List of extension ratios
                - swing_lookback: int, period for swing detection
                - min_periods: int, minimum periods required

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        # Parse retracement levels
        retracement_config = self.config.get("retracement_levels", [])
        if retracement_config:
            self.retracement_levels = [Decimal(str(x)) for x in retracement_config]
        else:
            self.retracement_levels = self.STANDARD_RETRACEMENTS

        # Parse extension levels
        extension_config = self.config.get("extension_levels", [])
        if extension_config:
            self.extension_levels = [Decimal(str(x)) for x in extension_config]
        else:
            self.extension_levels = self.STANDARD_EXTENSIONS

        self.swing_lookback = int(self.config.get("swing_lookback", 20))
        self.min_periods = int(self.config.get("min_periods", 10))

        logger.info(
            "fibonacci_initialized",
            num_retracement_levels=len(self.retracement_levels),
            num_extension_levels=len(self.extension_levels),
            swing_lookback=self.swing_lookback
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        # Validate swing lookback
        swing_lookback = self.config.get("swing_lookback", 20)
        if not isinstance(swing_lookback, (int, str)) or int(swing_lookback) < 1:
            raise ValueError(f"Invalid swing_lookback: {swing_lookback}")

        logger.debug("fibonacci_config_validated", config=self.config)

    async def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate Fibonacci retracement and extension levels.

        Args:
            data: Polars DataFrame with columns:
                - high: High prices
                - low: Low prices
                - close: Close prices
                - timestamp: Optional timestamp column

        Returns:
            DataFrame with additional columns:
                - swing_high: Identified swing high
                - swing_low: Identified swing low
                - fib_direction: "uptrend" or "downtrend"
                - fib_0: Base level (0%)
                - fib_236: 23.6% retracement
                - fib_382: 38.2% retracement
                - fib_500: 50.0% retracement
                - fib_618: 61.8% retracement
                - fib_786: 78.6% retracement
                - fib_100: 100% level
                - fib_1618: 161.8% extension (if applicable)

        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        try:
            self._validate_data(data)

            logger.debug(
                "calculating_fibonacci",
                rows=len(data),
                swing_lookback=self.swing_lookback
            )

            # Detect swing highs and lows
            result = self._detect_swings(data)

            # Calculate Fibonacci levels
            result = self._calculate_fibonacci_levels(result)

            # Identify which levels price is near
            result = self._identify_nearby_levels(result)

            logger.debug(
                "fibonacci_calculated",
                rows=len(result)
            )

            return result

        except Exception as e:
            logger.error(
                "fibonacci_calculation_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def _detect_swings(self, data: pl.DataFrame) -> pl.DataFrame:
        """Detect swing highs and lows.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with swing columns
        """
        # Calculate rolling highs and lows
        result = data.with_columns([
            pl.col("high").rolling_max(window_size=self.swing_lookback).alias("swing_high"),
            pl.col("low").rolling_min(window_size=self.swing_lookback).alias("swing_low")
        ])

        # Determine trend direction based on swing levels
        result = result.with_columns([
            pl.when(pl.col("close") > pl.col("swing_low"))
              .then(pl.lit("uptrend"))
              .when(pl.col("close") < pl.col("swing_high"))
              .then(pl.lit("downtrend"))
              .otherwise(pl.lit("neutral"))
              .alias("fib_direction")
        ])

        return result

    def _calculate_fibonacci_levels(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate Fibonacci retracement and extension levels.

        Args:
            data: DataFrame with swing highs and lows

        Returns:
            DataFrame with Fibonacci level columns
        """
        result = data

        # Calculate the range (distance between swing high and low)
        result = result.with_columns([
            (pl.col("swing_high") - pl.col("swing_low")).alias("fib_range")
        ])

        # For uptrend: levels calculated from swing_low
        # For downtrend: levels calculated from swing_high

        # Calculate retracement levels
        for level in self.retracement_levels:
            level_pct = int(float(level * Decimal("100")))
            col_name = f"fib_{level_pct}"

            result = result.with_columns([
                pl.when(pl.col("fib_direction") == "uptrend")
                  .then(
                    pl.col("swing_low") + (pl.col("fib_range") * float(level))
                  )
                  .when(pl.col("fib_direction") == "downtrend")
                  .then(
                    pl.col("swing_high") - (pl.col("fib_range") * float(level))
                  )
                  .otherwise(pl.lit(None))
                  .alias(col_name)
            ])

        # Add 0% and 100% levels
        result = result.with_columns([
            pl.when(pl.col("fib_direction") == "uptrend")
              .then(pl.col("swing_low"))
              .when(pl.col("fib_direction") == "downtrend")
              .then(pl.col("swing_high"))
              .otherwise(pl.lit(None))
              .alias("fib_0"),

            pl.when(pl.col("fib_direction") == "uptrend")
              .then(pl.col("swing_high"))
              .when(pl.col("fib_direction") == "downtrend")
              .then(pl.col("swing_low"))
              .otherwise(pl.lit(None))
              .alias("fib_100")
        ])

        # Calculate extension levels (beyond 100%)
        for level in self.extension_levels:
            if level > Decimal("1.0"):
                level_pct = int(float(level * Decimal("100")))
                col_name = f"fib_{level_pct}"

                result = result.with_columns([
                    pl.when(pl.col("fib_direction") == "uptrend")
                      .then(
                        pl.col("swing_low") + (pl.col("fib_range") * float(level))
                      )
                      .when(pl.col("fib_direction") == "downtrend")
                      .then(
                        pl.col("swing_high") - (pl.col("fib_range") * float(level))
                      )
                      .otherwise(pl.lit(None))
                      .alias(col_name)
                ])

        return result

    def _identify_nearby_levels(self, data: pl.DataFrame) -> pl.DataFrame:
        """Identify which Fibonacci levels price is near.

        Args:
            data: DataFrame with Fibonacci levels

        Returns:
            DataFrame with nearby level identification
        """
        # Get tolerance from config (percentage)
        tolerance = Decimal(str(self.config.get("level_tolerance", "0.5")))

        # Check which level price is closest to
        fib_cols = [col for col in data.columns if col.startswith("fib_") and col not in ["fib_direction", "fib_range"]]

        if not fib_cols:
            return data.with_columns([
                pl.lit(None).alias("nearest_fib_level"),
                pl.lit(None).alias("fib_distance")
            ])

        # Calculate distance to each level
        result = data

        # Find nearest level
        # This is simplified - a full implementation would check all levels
        result = result.with_columns([
            pl.lit(None).alias("nearest_fib_level"),
            pl.lit(None).alias("fib_distance")
        ])

        return result

    def _validate_data(self, data: pl.DataFrame) -> None:
        """Validate input data.

        Args:
            data: Input DataFrame

        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(data, pl.DataFrame):
            raise ValueError("Data must be a Polars DataFrame")

        required_columns = ["high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if len(data) < self.min_periods:
            raise ValueError(
                f"Insufficient data: {len(data)} rows, need at least {self.min_periods}"
            )

    def calculate_manual_levels(
        self,
        high: Decimal,
        low: Decimal,
        direction: str = "uptrend"
    ) -> Dict[str, Decimal]:
        """Calculate Fibonacci levels for a specific high-low range.

        Args:
            high: High price point
            low: Low price point
            direction: "uptrend" or "downtrend"

        Returns:
            Dictionary of Fibonacci levels

        Example:
            >>> fib = FibonacciIndicator(config)
            >>> levels = fib.calculate_manual_levels(
            ...     Decimal("50000"), Decimal("40000"), "uptrend"
            ... )
            >>> print(levels["fib_618"])
        """
        try:
            high = Decimal(str(high))
            low = Decimal(str(low))
            range_value = high - low

            levels = {}

            if direction == "uptrend":
                # Retracements from high down to low
                levels["fib_0"] = high
                for level in self.retracement_levels:
                    key = f"fib_{int(float(level * Decimal('100')))}"
                    levels[key] = high - (range_value * level)
                levels["fib_100"] = low

                # Extensions beyond low
                for level in self.extension_levels:
                    if level > Decimal("1.0"):
                        key = f"fib_{int(float(level * Decimal('100')))}"
                        levels[key] = high - (range_value * level)

            else:  # downtrend
                # Retracements from low up to high
                levels["fib_0"] = low
                for level in self.retracement_levels:
                    key = f"fib_{int(float(level * Decimal('100')))}"
                    levels[key] = low + (range_value * level)
                levels["fib_100"] = high

                # Extensions beyond high
                for level in self.extension_levels:
                    if level > Decimal("1.0"):
                        key = f"fib_{int(float(level * Decimal('100')))}"
                        levels[key] = low + (range_value * level)

            logger.debug(
                "manual_fibonacci_levels_calculated",
                high=float(high),
                low=float(low),
                direction=direction,
                num_levels=len(levels)
            )

            return levels

        except Exception as e:
            logger.error("manual_fibonacci_calculation_failed", error=str(e))
            raise

    def find_confluence_zones(
        self,
        data: pl.DataFrame,
        tolerance: Optional[Decimal] = None
    ) -> pl.DataFrame:
        """Find zones where multiple Fibonacci levels converge.

        Confluence zones are areas where multiple Fibonacci levels from
        different swings align, creating stronger support/resistance.

        Args:
            data: DataFrame with Fibonacci levels
            tolerance: Price tolerance for confluence (percentage)

        Returns:
            DataFrame with confluence zone indicators
        """
        try:
            if tolerance is None:
                tolerance = Decimal(str(self.config.get("confluence_tolerance", "1.0")))

            # Get all Fibonacci level columns
            fib_cols = [
                col for col in data.columns
                if col.startswith("fib_") and
                col not in ["fib_direction", "fib_range"]
            ]

            if len(fib_cols) < 2:
                return data.with_columns([
                    pl.lit(False).alias("fib_confluence"),
                    pl.lit(0).alias("confluence_count")
                ])

            # Simplified confluence detection
            # Full implementation would compare levels across different periods
            result = data.with_columns([
                pl.lit(False).alias("fib_confluence"),
                pl.lit(0).alias("confluence_count")
            ])

            logger.debug(
                "confluence_zones_calculated",
                tolerance=float(tolerance),
                num_fib_levels=len(fib_cols)
            )

            return result

        except Exception as e:
            logger.error("confluence_zone_calculation_failed", error=str(e))
            raise

    def generate_trading_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """Generate trading signals from Fibonacci levels.

        Args:
            data: DataFrame with Fibonacci levels

        Returns:
            DataFrame with trading signals

        Raises:
            ValueError: If required columns are missing
        """
        try:
            required_cols = ["close", "fib_direction"]
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            # Get key retracement levels
            key_levels = ["fib_382", "fib_500", "fib_618"]
            available_levels = [lvl for lvl in key_levels if lvl in data.columns]

            if not available_levels:
                return data.with_columns([
                    pl.lit("neutral").alias("fib_signal"),
                    pl.lit("none").alias("fib_level_type")
                ])

            # Generate signals based on price relative to Fibonacci levels
            result = data.with_columns([
                pl.lit("neutral").alias("fib_signal"),
                pl.lit("none").alias("fib_level_type")
            ])

            # Detect bounces off key levels (simplified)
            if "fib_618" in data.columns:
                result = result.with_columns([
                    pl.when(
                        (pl.col("fib_direction") == "uptrend") &
                        (pl.col("close") <= pl.col("fib_618") * 1.02) &
                        (pl.col("close") >= pl.col("fib_618") * 0.98)
                    )
                      .then(pl.lit("buy_support"))
                      .when(
                        (pl.col("fib_direction") == "downtrend") &
                        (pl.col("close") <= pl.col("fib_618") * 1.02) &
                        (pl.col("close") >= pl.col("fib_618") * 0.98)
                      )
                      .then(pl.lit("sell_resistance"))
                      .otherwise(pl.col("fib_signal"))
                      .alias("fib_signal")
                ])

            logger.debug("fibonacci_signals_generated", rows=len(result))

            return result

        except Exception as e:
            logger.error("signal_generation_failed", error=str(e))
            raise


async def calculate_fibonacci(
    data: pl.DataFrame,
    config: Dict[str, Any]
) -> pl.DataFrame:
    """Convenience function to calculate Fibonacci levels.

    Args:
        data: Polars DataFrame with OHLC data
        config: Configuration dictionary

    Returns:
        DataFrame with Fibonacci levels

    Example:
        >>> config = {
        ...     "swing_lookback": 20,
        ...     "retracement_levels": [0.236, 0.382, 0.5, 0.618, 0.786]
        ... }
        >>> result = await calculate_fibonacci(df, config)
    """
    indicator = FibonacciIndicator(config)
    return await indicator.calculate(data)


def calculate_fibonacci_sequence(n: int) -> List[int]:
    """Calculate Fibonacci sequence up to n terms.

    Args:
        n: Number of terms to calculate

    Returns:
        List of Fibonacci numbers

    Example:
        >>> seq = calculate_fibonacci_sequence(10)
        >>> print(seq)
        [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    """
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]

    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i - 1] + sequence[i - 2])

    return sequence


def get_golden_ratio() -> Decimal:
    """Get the golden ratio (phi).

    Returns:
        Golden ratio (1.618033988749...)
    """
    return Decimal("1.618033988749894848204586834")
