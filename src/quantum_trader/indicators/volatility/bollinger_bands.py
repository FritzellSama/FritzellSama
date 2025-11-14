"""Bollinger Bands - Volatility and Price Channel Indicator.

Bollinger Bands consist of a middle band (SMA) and two outer bands
(standard deviations away from the middle band). They measure market
volatility and provide relative price levels.
"""

import asyncio
from decimal import Decimal
from typing import Any, Dict, Optional

import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class BollingerBands:
    """Bollinger Bands volatility indicator.

    Bollinger Bands use a moving average and standard deviation to create
    dynamic support and resistance levels. They expand during volatile
    periods and contract during quiet periods.

    Attributes:
        config: Configuration dictionary containing indicator parameters
        period: Lookback period for moving average
        std_dev: Number of standard deviations for bands

    Example:
        >>> config = {"period": 20, "std_dev": 2}
        >>> bb = BollingerBands(config)
        >>> result = await bb.calculate(df)
        >>> print(result.select(["timestamp", "bb_upper", "bb_middle", "bb_lower", "bb_width"]))
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Bollinger Bands.

        Args:
            config: Configuration dictionary with keys:
                - period: int, lookback period (default from config)
                - std_dev: float, standard deviation multiplier
                - ma_type: str, moving average type ('sma' or 'ema')
                - min_periods: int, minimum periods required

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()
        self.period = int(self.config.get("period", self.config.get("default_period", 20)))
        self.std_dev = Decimal(str(self.config.get("std_dev", self.config.get("default_std_dev", "2.0"))))
        self.ma_type = self.config.get("ma_type", "sma")
        self.min_periods = int(self.config.get("min_periods", self.period))

        logger.info(
            "bollinger_bands_initialized",
            period=self.period,
            std_dev=float(self.std_dev),
            ma_type=self.ma_type,
            min_periods=self.min_periods
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        period = self.config.get("period", self.config.get("default_period", 20))
        if not isinstance(period, (int, str)) or int(period) < 2:
            raise ValueError(f"Invalid period: {period}. Must be >= 2")

        std_dev = self.config.get("std_dev", self.config.get("default_std_dev", "2.0"))
        try:
            std_val = Decimal(str(std_dev))
            if std_val <= Decimal("0"):
                raise ValueError(f"Invalid std_dev: {std_dev}. Must be positive")
        except Exception:
            raise ValueError(f"Invalid std_dev: {std_dev}. Must be a number")

        ma_type = self.config.get("ma_type", "sma")
        if ma_type not in ["sma", "ema"]:
            raise ValueError(f"Invalid ma_type: {ma_type}. Must be 'sma' or 'ema'")

        logger.debug("bollinger_bands_config_validated", config=self.config)

    async def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate Bollinger Bands.

        Args:
            data: Polars DataFrame with columns:
                - close: Close prices
                - timestamp: Optional timestamp column

        Returns:
            DataFrame with additional columns:
                - bb_middle: Middle band (moving average)
                - bb_upper: Upper band (MA + std_dev × σ)
                - bb_lower: Lower band (MA - std_dev × σ)
                - bb_width: Band width (upper - lower)
                - bb_percent: %B indicator ((close - lower) / (upper - lower))

        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        try:
            self._validate_data(data)

            logger.debug(
                "calculating_bollinger_bands",
                rows=len(data),
                period=self.period,
                std_dev=float(self.std_dev),
                ma_type=self.ma_type
            )

            # Calculate middle band (moving average)
            if self.ma_type == "ema":
                result = data.with_columns([
                    pl.col("close")
                      .ewm_mean(span=self.period, min_periods=self.min_periods)
                      .alias("bb_middle")
                ])
            else:  # sma
                result = data.with_columns([
                    pl.col("close")
                      .rolling_mean(window_size=self.period, min_periods=self.min_periods)
                      .alias("bb_middle")
                ])

            # Calculate standard deviation
            result = result.with_columns([
                pl.col("close")
                  .rolling_std(window_size=self.period, min_periods=self.min_periods)
                  .alias("bb_std")
            ])

            # Calculate upper and lower bands
            std_multiplier = float(self.std_dev)
            result = result.with_columns([
                (pl.col("bb_middle") + (pl.col("bb_std") * std_multiplier))
                  .alias("bb_upper"),
                (pl.col("bb_middle") - (pl.col("bb_std") * std_multiplier))
                  .alias("bb_lower")
            ])

            # Calculate band width
            result = result.with_columns([
                (pl.col("bb_upper") - pl.col("bb_lower")).alias("bb_width")
            ])

            # Calculate %B (percent B) - position within bands
            # %B = (close - lower) / (upper - lower)
            result = result.with_columns([
                ((pl.col("close") - pl.col("bb_lower")) /
                 (pl.col("bb_upper") - pl.col("bb_lower")))
                  .alias("bb_percent")
            ])

            # Drop intermediate column
            result = result.drop(["bb_std"])

            logger.debug(
                "bollinger_bands_calculated",
                rows=len(result),
                non_null_values=result.filter(pl.col("bb_middle").is_not_null()).height,
                avg_width=result["bb_width"].mean()
            )

            return result

        except Exception as e:
            logger.error(
                "bollinger_bands_calculation_failed",
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

        if "close" not in data.columns:
            raise ValueError("Missing required column: close")

        if len(data) < self.min_periods:
            raise ValueError(
                f"Insufficient data: {len(data)} rows, need at least {self.min_periods}"
            )

        null_count = data["close"].null_count()
        if null_count > 0:
            logger.warning(
                "null_values_detected",
                column="close",
                null_count=null_count
            )

    def detect_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """Detect trading signals from Bollinger Bands.

        Common signals:
        1. Band squeeze: Low volatility, potential breakout
        2. Band touch/break: Overbought/oversold
        3. Walking the bands: Strong trend
        4. Centerline crossover: Trend change

        Args:
            data: DataFrame with Bollinger Bands values

        Returns:
            DataFrame with additional signal columns:
                - bb_squeeze: Boolean, True if bands are narrow
                - bb_position: "above_upper", "below_lower", "middle", or "neutral"
                - bb_trend: "strong_up", "strong_down", or "neutral"

        Raises:
            ValueError: If required BB columns are missing
        """
        try:
            required_cols = ["bb_upper", "bb_middle", "bb_lower", "bb_width", "close"]
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            # Calculate squeeze threshold from config
            squeeze_threshold = Decimal(str(self.config.get("squeeze_threshold", "0.05")))

            # Normalize width by price for squeeze detection
            result = data.with_columns([
                ((pl.col("bb_width") / pl.col("close")) < float(squeeze_threshold))
                  .alias("bb_squeeze")
            ])

            # Determine position relative to bands
            result = result.with_columns([
                pl.when(pl.col("close") > pl.col("bb_upper"))
                  .then(pl.lit("above_upper"))
                  .when(pl.col("close") < pl.col("bb_lower"))
                  .then(pl.lit("below_lower"))
                  .when(
                    (pl.col("close") >= pl.col("bb_middle") * 0.98) &
                    (pl.col("close") <= pl.col("bb_middle") * 1.02)
                  )
                  .then(pl.lit("middle"))
                  .otherwise(pl.lit("neutral"))
                  .alias("bb_position")
            ])

            # Detect "walking the bands" (strong trend)
            # If close stays near upper band for multiple periods = strong uptrend
            # If close stays near lower band for multiple periods = strong downtrend
            walk_periods = int(self.config.get("walk_periods", 3))

            result = result.with_columns([
                pl.col("bb_position").alias("bb_pos_current"),
                pl.col("bb_position").shift(1).alias("bb_pos_prev1"),
                pl.col("bb_position").shift(2).alias("bb_pos_prev2")
            ])

            if walk_periods >= 3:
                result = result.with_columns([
                    pl.when(
                        (pl.col("bb_pos_current") == "above_upper") &
                        (pl.col("bb_pos_prev1") == "above_upper") &
                        (pl.col("bb_pos_prev2") == "above_upper")
                    )
                      .then(pl.lit("strong_up"))
                      .when(
                        (pl.col("bb_pos_current") == "below_lower") &
                        (pl.col("bb_pos_prev1") == "below_lower") &
                        (pl.col("bb_pos_prev2") == "below_lower")
                      )
                      .then(pl.lit("strong_down"))
                      .otherwise(pl.lit("neutral"))
                      .alias("bb_trend")
                ])
            else:
                result = result.with_columns([
                    pl.lit("neutral").alias("bb_trend")
                ])

            # Drop intermediate columns
            result = result.drop(["bb_pos_current", "bb_pos_prev1", "bb_pos_prev2"])

            logger.debug("bollinger_bands_signals_detected", rows=len(result))

            return result

        except Exception as e:
            logger.error("bollinger_bands_signal_detection_failed", error=str(e))
            raise

    def calculate_bandwidth(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate Bollinger Band Width indicator.

        Bandwidth = (upper - lower) / middle × 100

        Args:
            data: DataFrame with Bollinger Bands

        Returns:
            DataFrame with bandwidth_percent column

        Raises:
            ValueError: If required columns are missing
        """
        try:
            required_cols = ["bb_upper", "bb_middle", "bb_lower"]
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            result = data.with_columns([
                (((pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("bb_middle")) * pl.lit(100))
                  .alias("bandwidth_percent")
            ])

            logger.debug("bandwidth_calculated", rows=len(result))

            return result

        except Exception as e:
            logger.error("bandwidth_calculation_failed", error=str(e))
            raise


async def calculate_bollinger_bands(
    data: pl.DataFrame,
    config: Dict[str, Any]
) -> pl.DataFrame:
    """Convenience function to calculate Bollinger Bands.

    Args:
        data: Polars DataFrame with close prices
        config: Configuration dictionary

    Returns:
        DataFrame with Bollinger Bands

    Example:
        >>> config = {"period": 20, "std_dev": 2}
        >>> result = await calculate_bollinger_bands(df, config)
    """
    indicator = BollingerBands(config)
    return await indicator.calculate(data)
