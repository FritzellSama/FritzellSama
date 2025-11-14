"""Commodity Channel Index (CCI) - Momentum Oscillator.

The CCI measures the current price level relative to an average price level
over a given period. It identifies cyclical turns in commodities but can
be applied to any market.
"""

import asyncio
from decimal import Decimal
from typing import Any, Dict, Optional

import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class CCIIndicator:
    """Commodity Channel Index (CCI) momentum indicator.

    CCI = (Typical Price - SMA of Typical Price) / (0.015 × Mean Deviation)

    The constant 0.015 is used to ensure approximately 70-80% of CCI values
    fall between -100 and +100.

    Attributes:
        config: Configuration dictionary containing indicator parameters
        period: Lookback period for CCI calculation
        constant: Scaling constant (typically 0.015)

    Example:
        >>> config = {"period": 20}
        >>> cci = CCIIndicator(config)
        >>> result = await cci.calculate(df)
        >>> print(result.select(["timestamp", "cci", "cci_signal"]))
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize CCI indicator.

        Args:
            config: Configuration dictionary with keys:
                - period: int, lookback period (default from config)
                - constant: float, scaling constant (default 0.015)
                - min_periods: int, minimum periods required

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()
        self.period = int(self.config.get("period", self.config.get("default_period", 20)))
        self.constant = Decimal(str(self.config.get("constant", "0.015")))
        self.min_periods = int(self.config.get("min_periods", self.period))

        logger.info(
            "cci_initialized",
            period=self.period,
            constant=float(self.constant),
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

        constant = self.config.get("constant", "0.015")
        try:
            const_val = Decimal(str(constant))
            if const_val <= Decimal("0"):
                raise ValueError(f"Invalid constant: {constant}. Must be positive")
        except Exception:
            raise ValueError(f"Invalid constant: {constant}. Must be a number")

        logger.debug("cci_config_validated", config=self.config)

    async def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate Commodity Channel Index.

        Args:
            data: Polars DataFrame with columns:
                - high: High prices
                - low: Low prices
                - close: Close prices
                - timestamp: Optional timestamp column

        Returns:
            DataFrame with additional columns:
                - typical_price: (high + low + close) / 3
                - cci: Commodity Channel Index value
                - cci_sma: SMA of typical price

        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        try:
            self._validate_data(data)

            logger.debug(
                "calculating_cci",
                rows=len(data),
                period=self.period,
                constant=float(self.constant)
            )

            # Calculate typical price (HLC/3)
            result = data.with_columns([
                ((pl.col("high") + pl.col("low") + pl.col("close")) / pl.lit(3))
                  .alias("typical_price")
            ])

            # Calculate SMA of typical price
            result = result.with_columns([
                pl.col("typical_price")
                  .rolling_mean(window_size=self.period, min_periods=self.min_periods)
                  .alias("cci_sma")
            ])

            # Calculate mean absolute deviation
            # This requires calculating the absolute difference from the mean for each value
            tp_values = result["typical_price"].to_list()
            sma_values = result["cci_sma"].to_list()

            mean_deviations = []
            for i in range(len(tp_values)):
                if i < self.min_periods - 1 or sma_values[i] is None:
                    mean_deviations.append(None)
                else:
                    # Get the window
                    start_idx = max(0, i - self.period + 1)
                    window_tp = tp_values[start_idx:i + 1]
                    sma = sma_values[i]

                    # Calculate mean absolute deviation
                    deviations = [abs(tp - sma) for tp in window_tp if tp is not None]
                    if deviations:
                        mad = sum(deviations) / len(deviations)
                        mean_deviations.append(mad)
                    else:
                        mean_deviations.append(None)

            result = result.with_columns([
                pl.Series("mean_deviation", mean_deviations)
            ])

            # Calculate CCI
            # CCI = (Typical Price - SMA) / (constant × Mean Deviation)
            constant_float = float(self.constant)
            result = result.with_columns([
                ((pl.col("typical_price") - pl.col("cci_sma")) /
                 (pl.col("mean_deviation") * constant_float))
                  .alias("cci")
            ])

            # Drop intermediate column
            result = result.drop(["mean_deviation"])

            logger.debug(
                "cci_calculated",
                rows=len(result),
                non_null_values=result.filter(pl.col("cci").is_not_null()).height,
                avg_cci=result["cci"].mean()
            )

            return result

        except Exception as e:
            logger.error(
                "cci_calculation_failed",
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

        required_columns = ["high", "low", "close"]
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
        """Detect trading signals from CCI.

        Classic CCI signals:
        1. Overbought/Oversold: > +100 / < -100
        2. Zero-line crossover
        3. Extreme levels: > +200 / < -200
        4. Divergences

        Args:
            data: DataFrame with CCI values

        Returns:
            DataFrame with additional signal columns:
                - cci_signal: "overbought", "oversold", or "neutral"
                - cci_extreme: "extreme_high", "extreme_low", or None
                - cci_trend: "bullish", "bearish", or "neutral"

        Raises:
            ValueError: If CCI column is missing
        """
        try:
            if "cci" not in data.columns:
                raise ValueError("CCI column not found. Calculate CCI first.")

            # Get thresholds from config
            overbought = Decimal(str(self.config.get("overbought_threshold", "100")))
            oversold = Decimal(str(self.config.get("oversold_threshold", "-100")))
            extreme_high = Decimal(str(self.config.get("extreme_high_threshold", "200")))
            extreme_low = Decimal(str(self.config.get("extreme_low_threshold", "-200")))

            result = data.with_columns([
                # Basic overbought/oversold signals
                pl.when(pl.col("cci") > float(overbought))
                  .then(pl.lit("overbought"))
                  .when(pl.col("cci") < float(oversold))
                  .then(pl.lit("oversold"))
                  .otherwise(pl.lit("neutral"))
                  .alias("cci_signal"),

                # Extreme levels
                pl.when(pl.col("cci") > float(extreme_high))
                  .then(pl.lit("extreme_high"))
                  .when(pl.col("cci") < float(extreme_low))
                  .then(pl.lit("extreme_low"))
                  .otherwise(pl.lit(None))
                  .alias("cci_extreme")
            ])

            # Zero-line crossover for trend
            result = result.with_columns([
                pl.col("cci").shift(1).alias("cci_prev")
            ])

            result = result.with_columns([
                pl.when(
                    (pl.col("cci") > pl.lit(0)) & (pl.col("cci_prev") <= pl.lit(0))
                )
                  .then(pl.lit("bullish"))
                  .when(
                    (pl.col("cci") < pl.lit(0)) & (pl.col("cci_prev") >= pl.lit(0))
                  )
                  .then(pl.lit("bearish"))
                  .when(pl.col("cci") > pl.lit(0))
                  .then(pl.lit("bullish"))
                  .when(pl.col("cci") < pl.lit(0))
                  .then(pl.lit("bearish"))
                  .otherwise(pl.lit("neutral"))
                  .alias("cci_trend")
            ])

            # Drop intermediate column
            result = result.drop(["cci_prev"])

            logger.debug("cci_signals_detected", rows=len(result))

            return result

        except Exception as e:
            logger.error("cci_signal_detection_failed", error=str(e))
            raise

    def detect_divergence(
        self,
        data: pl.DataFrame,
        lookback_periods: Optional[int] = None
    ) -> pl.DataFrame:
        """Detect bullish and bearish divergences.

        Bullish divergence: Price makes lower lows, but CCI makes higher lows
        Bearish divergence: Price makes higher highs, but CCI makes lower highs

        Args:
            data: DataFrame with CCI and price data
            lookback_periods: Periods to look back for divergence detection

        Returns:
            DataFrame with divergence indicators

        Raises:
            ValueError: If required columns are missing
        """
        try:
            required_cols = ["cci", "close"]
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            if lookback_periods is None:
                lookback_periods = int(self.config.get("divergence_lookback", 14))

            # Calculate rolling min/max for divergence detection
            result = data.with_columns([
                pl.col("close").rolling_min(window_size=lookback_periods).alias("price_low"),
                pl.col("close").rolling_max(window_size=lookback_periods).alias("price_high"),
                pl.col("cci").rolling_min(window_size=lookback_periods).alias("cci_low"),
                pl.col("cci").rolling_max(window_size=lookback_periods).alias("cci_high")
            ])

            # Detect potential divergence zones
            result = result.with_columns([
                pl.col("price_low").shift(1).alias("price_low_prev"),
                pl.col("cci_low").shift(1).alias("cci_low_prev"),
                pl.col("price_high").shift(1).alias("price_high_prev"),
                pl.col("cci_high").shift(1).alias("cci_high_prev")
            ])

            result = result.with_columns([
                # Bullish divergence: lower price low but higher CCI low
                pl.when(
                    (pl.col("price_low") < pl.col("price_low_prev")) &
                    (pl.col("cci_low") > pl.col("cci_low_prev"))
                )
                  .then(pl.lit(True))
                  .otherwise(pl.lit(False))
                  .alias("bullish_divergence"),

                # Bearish divergence: higher price high but lower CCI high
                pl.when(
                    (pl.col("price_high") > pl.col("price_high_prev")) &
                    (pl.col("cci_high") < pl.col("cci_high_prev"))
                )
                  .then(pl.lit(True))
                  .otherwise(pl.lit(False))
                  .alias("bearish_divergence")
            ])

            # Drop intermediate columns
            result = result.drop([
                "price_low", "price_high", "cci_low", "cci_high",
                "price_low_prev", "price_high_prev", "cci_low_prev", "cci_high_prev"
            ])

            logger.debug(
                "divergence_detection_completed",
                lookback_periods=lookback_periods
            )

            return result

        except Exception as e:
            logger.error("divergence_detection_failed", error=str(e))
            raise


async def calculate_cci(
    data: pl.DataFrame,
    config: Dict[str, Any]
) -> pl.DataFrame:
    """Convenience function to calculate CCI indicator.

    Args:
        data: Polars DataFrame with high, low, and close prices
        config: Configuration dictionary

    Returns:
        DataFrame with CCI indicator

    Example:
        >>> config = {"period": 20}
        >>> result = await calculate_cci(df, config)
    """
    indicator = CCIIndicator(config)
    return await indicator.calculate(data)
