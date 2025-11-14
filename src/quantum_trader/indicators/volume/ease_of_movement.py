"""Ease of Movement (EMV) - Volume-Based Momentum Indicator.

The Ease of Movement indicator relates price change to volume and measures
how easily a price can move. It identifies periods when prices move easily
with little volume versus when significant volume is needed for movement.
"""

import asyncio
from decimal import Decimal
from typing import Any, Dict, Optional

import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class EaseOfMovement:
    """Ease of Movement (EMV) volume indicator.

    EMV combines price movement and volume to show how easily prices move.
    High positive values indicate prices moving up easily.
    High negative values indicate prices moving down easily.
    Values near zero indicate difficult price movement requiring high volume.

    Formula:
    Distance Moved = ((High + Low) / 2) - ((Prior High + Prior Low) / 2)
    Box Ratio = (Volume / Scale) / (High - Low)
    EMV = Distance Moved / Box Ratio

    Attributes:
        config: Configuration dictionary containing indicator parameters
        scale: Volume scaling factor (typically 10000 or 100000000)
        period: SMA period for smoothing (optional)

    Example:
        >>> config = {"scale": 10000, "sma_period": 14}
        >>> emv = EaseOfMovement(config)
        >>> result = await emv.calculate(df)
        >>> print(result.select(["timestamp", "emv", "emv_sma", "emv_signal"]))
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Ease of Movement indicator.

        Args:
            config: Configuration dictionary with keys:
                - scale: int, volume scaling factor
                - sma_period: int, smoothing period (optional)
                - min_periods: int, minimum periods required

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()
        self.scale = Decimal(str(self.config.get("scale", self.config.get("default_scale", "10000"))))
        self.sma_period = self.config.get("sma_period", self.config.get("default_sma_period", 14))
        if self.sma_period is not None:
            self.sma_period = int(self.sma_period)
        self.min_periods = int(self.config.get("min_periods", 2))

        logger.info(
            "ease_of_movement_initialized",
            scale=float(self.scale),
            sma_period=self.sma_period,
            min_periods=self.min_periods
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        scale = self.config.get("scale", self.config.get("default_scale", "10000"))
        try:
            scale_val = Decimal(str(scale))
            if scale_val <= Decimal("0"):
                raise ValueError(f"Invalid scale: {scale}. Must be positive")
        except Exception:
            raise ValueError(f"Invalid scale: {scale}. Must be a number")

        sma_period = self.config.get("sma_period", self.config.get("default_sma_period", 14))
        if sma_period is not None:
            if not isinstance(sma_period, (int, str)) or int(sma_period) < 1:
                raise ValueError(f"Invalid sma_period: {sma_period}. Must be positive integer")

        logger.debug("ease_of_movement_config_validated", config=self.config)

    async def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate Ease of Movement indicator.

        Args:
            data: Polars DataFrame with columns:
                - high: High prices
                - low: Low prices
                - volume: Trading volume
                - timestamp: Optional timestamp column

        Returns:
            DataFrame with additional columns:
                - distance_moved: Midpoint movement
                - box_ratio: Volume per price range
                - emv: Raw Ease of Movement
                - emv_sma: Smoothed EMV (if sma_period configured)

        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        try:
            self._validate_data(data)

            logger.debug(
                "calculating_ease_of_movement",
                rows=len(data),
                scale=float(self.scale),
                sma_period=self.sma_period
            )

            # Calculate midpoint and previous midpoint
            result = data.with_columns([
                ((pl.col("high") + pl.col("low")) / pl.lit(2)).alias("midpoint")
            ])

            result = result.with_columns([
                pl.col("midpoint").shift(1).alias("midpoint_prev")
            ])

            # Calculate Distance Moved
            result = result.with_columns([
                (pl.col("midpoint") - pl.col("midpoint_prev")).alias("distance_moved")
            ])

            # Calculate Box Ratio
            # Box Ratio = (Volume / Scale) / (High - Low)
            # Handle division by zero when high == low
            scale_float = float(self.scale)
            result = result.with_columns([
                pl.when(pl.col("high") == pl.col("low"))
                  .then(pl.lit(None))  # Set to None when range is zero
                  .otherwise(
                    (pl.col("volume") / scale_float) / (pl.col("high") - pl.col("low"))
                  )
                  .alias("box_ratio")
            ])

            # Calculate EMV
            # EMV = Distance Moved / Box Ratio
            result = result.with_columns([
                pl.when(
                    (pl.col("box_ratio").is_null()) |
                    (pl.col("box_ratio") == pl.lit(0))
                )
                  .then(pl.lit(None))
                  .otherwise(pl.col("distance_moved") / pl.col("box_ratio"))
                  .alias("emv")
            ])

            # Calculate smoothed EMV if sma_period is configured
            if self.sma_period is not None:
                result = result.with_columns([
                    pl.col("emv")
                      .rolling_mean(window_size=self.sma_period, min_periods=self.sma_period)
                      .alias("emv_sma")
                ])

            # Drop intermediate columns
            result = result.drop(["midpoint", "midpoint_prev"])

            logger.debug(
                "ease_of_movement_calculated",
                rows=len(result),
                non_null_values=result.filter(pl.col("emv").is_not_null()).height,
                avg_emv=result["emv"].mean()
            )

            return result

        except Exception as e:
            logger.error(
                "ease_of_movement_calculation_failed",
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

        required_columns = ["high", "low", "volume"]
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

        # Check for zero or negative volume
        invalid_volume = data.filter(pl.col("volume") <= 0).height
        if invalid_volume > 0:
            logger.warning(
                "invalid_volume_detected",
                zero_or_negative_count=invalid_volume
            )

    def detect_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """Detect trading signals from Ease of Movement.

        EMV signals:
        1. Positive EMV: Prices rising easily (bullish)
        2. Negative EMV: Prices falling easily (bearish)
        3. Zero-line crossover: Trend change
        4. Divergence with price

        Args:
            data: DataFrame with EMV values

        Returns:
            DataFrame with additional signal columns:
                - emv_signal: "bullish", "bearish", or "neutral"
                - emv_trend: "easy_up", "easy_down", "difficult", or "neutral"
                - emv_strength: "strong", "moderate", or "weak"

        Raises:
            ValueError: If EMV column is missing
        """
        try:
            # Use smoothed EMV if available, otherwise raw EMV
            emv_col = "emv_sma" if "emv_sma" in data.columns else "emv"

            if emv_col not in data.columns:
                raise ValueError(f"{emv_col} column not found. Calculate EMV first.")

            # Get thresholds from config
            strong_threshold = Decimal(str(self.config.get("strong_threshold", "1.0")))
            weak_threshold = Decimal(str(self.config.get("weak_threshold", "0.1")))

            result = data.with_columns([
                # Basic signal
                pl.when(pl.col(emv_col) > float(weak_threshold))
                  .then(pl.lit("bullish"))
                  .when(pl.col(emv_col) < -float(weak_threshold))
                  .then(pl.lit("bearish"))
                  .otherwise(pl.lit("neutral"))
                  .alias("emv_signal"),

                # Trend assessment
                pl.when(pl.col(emv_col) > float(strong_threshold))
                  .then(pl.lit("easy_up"))
                  .when(pl.col(emv_col) < -float(strong_threshold))
                  .then(pl.lit("easy_down"))
                  .when(
                    (pl.col(emv_col).abs() < float(weak_threshold)) &
                    (pl.col(emv_col).is_not_null())
                  )
                  .then(pl.lit("difficult"))
                  .otherwise(pl.lit("neutral"))
                  .alias("emv_trend"),

                # Strength
                pl.when(pl.col(emv_col).abs() > float(strong_threshold))
                  .then(pl.lit("strong"))
                  .when(pl.col(emv_col).abs() > float(weak_threshold))
                  .then(pl.lit("moderate"))
                  .otherwise(pl.lit("weak"))
                  .alias("emv_strength")
            ])

            logger.debug("emv_signals_detected", rows=len(result))

            return result

        except Exception as e:
            logger.error("emv_signal_detection_failed", error=str(e))
            raise

    def detect_divergence(
        self,
        data: pl.DataFrame,
        lookback_periods: Optional[int] = None
    ) -> pl.DataFrame:
        """Detect bullish and bearish divergences between price and EMV.

        Bullish divergence: Price makes lower lows, but EMV makes higher lows
        Bearish divergence: Price makes higher highs, but EMV makes lower highs

        Args:
            data: DataFrame with EMV and price data
            lookback_periods: Periods to look back for divergence detection

        Returns:
            DataFrame with divergence indicators

        Raises:
            ValueError: If required columns are missing
        """
        try:
            # Use smoothed EMV if available
            emv_col = "emv_sma" if "emv_sma" in data.columns else "emv"

            required_cols = [emv_col, "close"]
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            if lookback_periods is None:
                lookback_periods = int(self.config.get("divergence_lookback", 14))

            # Calculate rolling min/max for divergence detection
            result = data.with_columns([
                pl.col("close").rolling_min(window_size=lookback_periods).alias("price_low"),
                pl.col("close").rolling_max(window_size=lookback_periods).alias("price_high"),
                pl.col(emv_col).rolling_min(window_size=lookback_periods).alias("emv_low"),
                pl.col(emv_col).rolling_max(window_size=lookback_periods).alias("emv_high")
            ])

            # Detect potential divergence zones
            result = result.with_columns([
                pl.col("price_low").shift(1).alias("price_low_prev"),
                pl.col("emv_low").shift(1).alias("emv_low_prev"),
                pl.col("price_high").shift(1).alias("price_high_prev"),
                pl.col("emv_high").shift(1).alias("emv_high_prev")
            ])

            result = result.with_columns([
                # Bullish divergence: lower price low but higher EMV low
                pl.when(
                    (pl.col("price_low") < pl.col("price_low_prev")) &
                    (pl.col("emv_low") > pl.col("emv_low_prev"))
                )
                  .then(pl.lit(True))
                  .otherwise(pl.lit(False))
                  .alias("bullish_divergence"),

                # Bearish divergence: higher price high but lower EMV high
                pl.when(
                    (pl.col("price_high") > pl.col("price_high_prev")) &
                    (pl.col("emv_high") < pl.col("emv_high_prev"))
                )
                  .then(pl.lit(True))
                  .otherwise(pl.lit(False))
                  .alias("bearish_divergence")
            ])

            # Drop intermediate columns
            result = result.drop([
                "price_low", "price_high", "emv_low", "emv_high",
                "price_low_prev", "price_high_prev", "emv_low_prev", "emv_high_prev"
            ])

            logger.debug(
                "divergence_detection_completed",
                lookback_periods=lookback_periods
            )

            return result

        except Exception as e:
            logger.error("divergence_detection_failed", error=str(e))
            raise

    def compare_with_volume(self, data: pl.DataFrame) -> pl.DataFrame:
        """Compare EMV with raw volume for additional insights.

        Args:
            data: DataFrame with EMV and volume data

        Returns:
            DataFrame with volume comparison metrics

        Raises:
            ValueError: If required columns are missing
        """
        try:
            emv_col = "emv_sma" if "emv_sma" in data.columns else "emv"

            required_cols = [emv_col, "volume"]
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            # Calculate average volume for comparison
            avg_volume_period = int(self.config.get("avg_volume_period", 20))

            result = data.with_columns([
                pl.col("volume")
                  .rolling_mean(window_size=avg_volume_period)
                  .alias("avg_volume")
            ])

            # Categorize volume and EMV combinations
            result = result.with_columns([
                pl.when(
                    (pl.col(emv_col) > pl.lit(0)) &
                    (pl.col("volume") > pl.col("avg_volume"))
                )
                  .then(pl.lit("strong_bullish"))
                  .when(
                    (pl.col(emv_col) > pl.lit(0)) &
                    (pl.col("volume") <= pl.col("avg_volume"))
                  )
                  .then(pl.lit("easy_bullish"))
                  .when(
                    (pl.col(emv_col) < pl.lit(0)) &
                    (pl.col("volume") > pl.col("avg_volume"))
                  )
                  .then(pl.lit("strong_bearish"))
                  .when(
                    (pl.col(emv_col) < pl.lit(0)) &
                    (pl.col("volume") <= pl.col("avg_volume"))
                  )
                  .then(pl.lit("easy_bearish"))
                  .otherwise(pl.lit("neutral"))
                  .alias("volume_emv_combo")
            ])

            result = result.drop(["avg_volume"])

            logger.debug("volume_comparison_completed", rows=len(result))

            return result

        except Exception as e:
            logger.error("volume_comparison_failed", error=str(e))
            raise


async def calculate_ease_of_movement(
    data: pl.DataFrame,
    config: Dict[str, Any]
) -> pl.DataFrame:
    """Convenience function to calculate Ease of Movement.

    Args:
        data: Polars DataFrame with high, low, and volume
        config: Configuration dictionary

    Returns:
        DataFrame with Ease of Movement indicator

    Example:
        >>> config = {"scale": 10000, "sma_period": 14}
        >>> result = await calculate_ease_of_movement(df, config)
    """
    indicator = EaseOfMovement(config)
    return await indicator.calculate(data)
