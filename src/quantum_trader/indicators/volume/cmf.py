"""Chaikin Money Flow (CMF) - Volume-Weighted Indicator.

CMF measures the amount of Money Flow Volume over a specific period.
It oscillates between -1 and +1, indicating buying and selling pressure.
"""

import asyncio
from decimal import Decimal
from typing import Any, Dict, Optional

import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class ChaikinMoneyFlow:
    """Chaikin Money Flow (CMF) volume indicator.

    CMF is a volume-weighted average of accumulation and distribution over
    a specified period. It combines price and volume to measure buying and
    selling pressure.

    CMF = Sum(Money Flow Volume for n periods) / Sum(Volume for n periods)
    Money Flow Volume = Money Flow Multiplier × Volume
    Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)

    Attributes:
        config: Configuration dictionary containing indicator parameters
        period: Lookback period for CMF calculation

    Example:
        >>> config = {"period": 20}
        >>> cmf = ChaikinMoneyFlow(config)
        >>> result = await cmf.calculate(df)
        >>> print(result.select(["timestamp", "cmf", "cmf_signal"]))
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Chaikin Money Flow indicator.

        Args:
            config: Configuration dictionary with keys:
                - period: int, lookback period (default from config)
                - min_periods: int, minimum periods required

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()
        self.period = int(self.config.get("period", self.config.get("default_period", 20)))
        self.min_periods = int(self.config.get("min_periods", self.period))

        logger.info(
            "cmf_initialized",
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

        period = self.config.get("period", self.config.get("default_period", 20))
        if not isinstance(period, (int, str)) or int(period) < 1:
            raise ValueError(f"Invalid period: {period}. Must be positive integer")

        logger.debug("cmf_config_validated", config=self.config)

    async def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate Chaikin Money Flow.

        Args:
            data: Polars DataFrame with columns:
                - high: High prices
                - low: Low prices
                - close: Close prices
                - volume: Trading volume
                - timestamp: Optional timestamp column

        Returns:
            DataFrame with additional columns:
                - money_flow_multiplier: MF multiplier
                - money_flow_volume: MF volume
                - cmf: Chaikin Money Flow value

        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        try:
            self._validate_data(data)

            logger.debug(
                "calculating_cmf",
                rows=len(data),
                period=self.period
            )

            # Calculate Money Flow Multiplier
            # MFM = ((Close - Low) - (High - Close)) / (High - Low)
            # Simplified: MFM = (2 × Close - High - Low) / (High - Low)
            result = data.with_columns([
                # Handle division by zero when high == low
                pl.when(pl.col("high") == pl.col("low"))
                  .then(pl.lit(0.0))
                  .otherwise(
                    ((pl.lit(2) * pl.col("close")) - pl.col("high") - pl.col("low")) /
                    (pl.col("high") - pl.col("low"))
                  )
                  .alias("money_flow_multiplier")
            ])

            # Calculate Money Flow Volume
            # MFV = MFM × Volume
            result = result.with_columns([
                (pl.col("money_flow_multiplier") * pl.col("volume"))
                  .alias("money_flow_volume")
            ])

            # Calculate CMF
            # CMF = Sum(MFV for n periods) / Sum(Volume for n periods)
            result = result.with_columns([
                pl.col("money_flow_volume")
                  .rolling_sum(window_size=self.period, min_periods=self.min_periods)
                  .alias("sum_mfv"),
                pl.col("volume")
                  .rolling_sum(window_size=self.period, min_periods=self.min_periods)
                  .alias("sum_volume")
            ])

            # Calculate final CMF
            result = result.with_columns([
                pl.when(pl.col("sum_volume") == pl.lit(0))
                  .then(pl.lit(0.0))
                  .otherwise(pl.col("sum_mfv") / pl.col("sum_volume"))
                  .alias("cmf")
            ])

            # Drop intermediate columns
            result = result.drop(["sum_mfv", "sum_volume"])

            logger.debug(
                "cmf_calculated",
                rows=len(result),
                non_null_values=result.filter(pl.col("cmf").is_not_null()).height,
                avg_cmf=result["cmf"].mean()
            )

            return result

        except Exception as e:
            logger.error(
                "cmf_calculation_failed",
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

        required_columns = ["high", "low", "close", "volume"]
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

        # Check for negative or zero volume
        zero_volume = data.filter(pl.col("volume") <= 0).height
        if zero_volume > 0:
            logger.warning(
                "invalid_volume_detected",
                zero_or_negative_count=zero_volume
            )

    def detect_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """Detect trading signals from CMF.

        CMF signals:
        1. Positive CMF: Buying pressure (accumulation)
        2. Negative CMF: Selling pressure (distribution)
        3. Strong levels: > 0.25 or < -0.25
        4. Zero-line crossover
        5. Divergences with price

        Args:
            data: DataFrame with CMF values

        Returns:
            DataFrame with additional signal columns:
                - cmf_signal: "strong_buy", "buy", "neutral", "sell", "strong_sell"
                - cmf_trend: "accumulation", "distribution", or "neutral"
                - cmf_strength: "strong", "moderate", or "weak"

        Raises:
            ValueError: If CMF column is missing
        """
        try:
            if "cmf" not in data.columns:
                raise ValueError("CMF column not found. Calculate CMF first.")

            # Get thresholds from config
            strong_threshold = Decimal(str(self.config.get("strong_threshold", "0.25")))
            weak_threshold = Decimal(str(self.config.get("weak_threshold", "0.05")))

            result = data.with_columns([
                # Signal based on CMF value
                pl.when(pl.col("cmf") > float(strong_threshold))
                  .then(pl.lit("strong_buy"))
                  .when((pl.col("cmf") > float(weak_threshold)) & (pl.col("cmf") <= float(strong_threshold)))
                  .then(pl.lit("buy"))
                  .when((pl.col("cmf") >= -float(weak_threshold)) & (pl.col("cmf") <= float(weak_threshold)))
                  .then(pl.lit("neutral"))
                  .when((pl.col("cmf") < -float(weak_threshold)) & (pl.col("cmf") >= -float(strong_threshold)))
                  .then(pl.lit("sell"))
                  .when(pl.col("cmf") < -float(strong_threshold))
                  .then(pl.lit("strong_sell"))
                  .otherwise(pl.lit("neutral"))
                  .alias("cmf_signal"),

                # Trend (accumulation/distribution)
                pl.when(pl.col("cmf") > pl.lit(0))
                  .then(pl.lit("accumulation"))
                  .when(pl.col("cmf") < pl.lit(0))
                  .then(pl.lit("distribution"))
                  .otherwise(pl.lit("neutral"))
                  .alias("cmf_trend")
            ])

            # Strength assessment
            result = result.with_columns([
                pl.when(pl.col("cmf").abs() > float(strong_threshold))
                  .then(pl.lit("strong"))
                  .when(pl.col("cmf").abs() > float(weak_threshold))
                  .then(pl.lit("moderate"))
                  .otherwise(pl.lit("weak"))
                  .alias("cmf_strength")
            ])

            logger.debug("cmf_signals_detected", rows=len(result))

            return result

        except Exception as e:
            logger.error("cmf_signal_detection_failed", error=str(e))
            raise

    def detect_divergence(
        self,
        data: pl.DataFrame,
        lookback_periods: Optional[int] = None
    ) -> pl.DataFrame:
        """Detect bullish and bearish divergences between price and CMF.

        Bullish divergence: Price makes lower lows, but CMF makes higher lows
        Bearish divergence: Price makes higher highs, but CMF makes lower highs

        Args:
            data: DataFrame with CMF and price data
            lookback_periods: Periods to look back for divergence detection

        Returns:
            DataFrame with divergence indicators

        Raises:
            ValueError: If required columns are missing
        """
        try:
            required_cols = ["cmf", "close"]
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            if lookback_periods is None:
                lookback_periods = int(self.config.get("divergence_lookback", 14))

            # Calculate rolling min/max for divergence detection
            result = data.with_columns([
                pl.col("close").rolling_min(window_size=lookback_periods).alias("price_low"),
                pl.col("close").rolling_max(window_size=lookback_periods).alias("price_high"),
                pl.col("cmf").rolling_min(window_size=lookback_periods).alias("cmf_low"),
                pl.col("cmf").rolling_max(window_size=lookback_periods).alias("cmf_high")
            ])

            # Detect potential divergence zones
            result = result.with_columns([
                pl.col("price_low").shift(1).alias("price_low_prev"),
                pl.col("cmf_low").shift(1).alias("cmf_low_prev"),
                pl.col("price_high").shift(1).alias("price_high_prev"),
                pl.col("cmf_high").shift(1).alias("cmf_high_prev")
            ])

            result = result.with_columns([
                # Bullish divergence: lower price low but higher CMF low
                pl.when(
                    (pl.col("price_low") < pl.col("price_low_prev")) &
                    (pl.col("cmf_low") > pl.col("cmf_low_prev"))
                )
                  .then(pl.lit(True))
                  .otherwise(pl.lit(False))
                  .alias("bullish_divergence"),

                # Bearish divergence: higher price high but lower CMF high
                pl.when(
                    (pl.col("price_high") > pl.col("price_high_prev")) &
                    (pl.col("cmf_high") < pl.col("cmf_high_prev"))
                )
                  .then(pl.lit(True))
                  .otherwise(pl.lit(False))
                  .alias("bearish_divergence")
            ])

            # Drop intermediate columns
            result = result.drop([
                "price_low", "price_high", "cmf_low", "cmf_high",
                "price_low_prev", "price_high_prev", "cmf_low_prev", "cmf_high_prev"
            ])

            logger.debug(
                "divergence_detection_completed",
                lookback_periods=lookback_periods
            )

            return result

        except Exception as e:
            logger.error("divergence_detection_failed", error=str(e))
            raise

    def calculate_volume_strength(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate volume strength based on CMF and volume.

        Args:
            data: DataFrame with CMF and volume data

        Returns:
            DataFrame with volume_strength column

        Raises:
            ValueError: If required columns are missing
        """
        try:
            required_cols = ["cmf", "volume"]
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

            # Volume strength combines CMF direction with volume level
            result = result.with_columns([
                (pl.col("cmf") * (pl.col("volume") / pl.col("avg_volume")))
                  .alias("volume_strength")
            ])

            result = result.drop(["avg_volume"])

            logger.debug("volume_strength_calculated", rows=len(result))

            return result

        except Exception as e:
            logger.error("volume_strength_calculation_failed", error=str(e))
            raise


async def calculate_cmf(
    data: pl.DataFrame,
    config: Dict[str, Any]
) -> pl.DataFrame:
    """Convenience function to calculate Chaikin Money Flow.

    Args:
        data: Polars DataFrame with high, low, close, and volume
        config: Configuration dictionary

    Returns:
        DataFrame with CMF indicator

    Example:
        >>> config = {"period": 20}
        >>> result = await calculate_cmf(df, config)
    """
    indicator = ChaikinMoneyFlow(config)
    return await indicator.calculate(data)
