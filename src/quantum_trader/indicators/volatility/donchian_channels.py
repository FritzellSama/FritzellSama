"""Donchian Channels - Trend and Volatility Indicator.

Donchian Channels plot the highest high and lowest low over a given period,
creating an upper and lower channel. Developed by Richard Donchian, they're
used for trend following and breakout trading.
"""

import asyncio
from decimal import Decimal
from typing import Any, Dict, Optional

import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class DonchianChannels:
    """Donchian Channels volatility and trend indicator.

    Donchian Channels consist of:
    - Upper band: Highest high over N periods
    - Lower band: Lowest low over N periods
    - Middle band: Average of upper and lower bands

    Attributes:
        config: Configuration dictionary containing indicator parameters
        period: Lookback period for high/low calculation

    Example:
        >>> config = {"period": 20}
        >>> dc = DonchianChannels(config)
        >>> result = await dc.calculate(df)
        >>> print(result.select(["timestamp", "dc_upper", "dc_middle", "dc_lower"]))
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Donchian Channels.

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
            "donchian_channels_initialized",
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

        logger.debug("donchian_channels_config_validated", config=self.config)

    async def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate Donchian Channels.

        Args:
            data: Polars DataFrame with columns:
                - high: High prices
                - low: Low prices
                - close: Close prices (optional, for position calculation)
                - timestamp: Optional timestamp column

        Returns:
            DataFrame with additional columns:
                - dc_upper: Upper channel (highest high)
                - dc_lower: Lower channel (lowest low)
                - dc_middle: Middle channel (average of upper and lower)
                - dc_width: Channel width (upper - lower)
                - dc_percent: Price position within channel (0 to 1)

        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        try:
            self._validate_data(data)

            logger.debug(
                "calculating_donchian_channels",
                rows=len(data),
                period=self.period
            )

            # Calculate upper and lower channels
            result = data.with_columns([
                pl.col("high")
                  .rolling_max(window_size=self.period, min_periods=self.min_periods)
                  .alias("dc_upper"),
                pl.col("low")
                  .rolling_min(window_size=self.period, min_periods=self.min_periods)
                  .alias("dc_lower")
            ])

            # Calculate middle channel
            result = result.with_columns([
                ((pl.col("dc_upper") + pl.col("dc_lower")) / pl.lit(2))
                  .alias("dc_middle")
            ])

            # Calculate channel width
            result = result.with_columns([
                (pl.col("dc_upper") - pl.col("dc_lower")).alias("dc_width")
            ])

            # Calculate price position within channel (0 to 1)
            # %D = (Close - Lower) / (Upper - Lower)
            if "close" in data.columns:
                result = result.with_columns([
                    pl.when(pl.col("dc_width") == pl.lit(0))
                      .then(pl.lit(0.5))
                      .otherwise(
                        (pl.col("close") - pl.col("dc_lower")) / pl.col("dc_width")
                      )
                      .alias("dc_percent")
                ])

            logger.debug(
                "donchian_channels_calculated",
                rows=len(result),
                non_null_values=result.filter(pl.col("dc_upper").is_not_null()).height,
                avg_width=result["dc_width"].mean()
            )

            return result

        except Exception as e:
            logger.error(
                "donchian_channels_calculation_failed",
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

    def detect_breakouts(self, data: pl.DataFrame) -> pl.DataFrame:
        """Detect breakouts from Donchian Channels.

        Breakout signals:
        1. Upper breakout: Close above upper channel
        2. Lower breakout: Close below lower channel
        3. Middle breakout: Close crosses middle channel

        Args:
            data: DataFrame with Donchian Channels values

        Returns:
            DataFrame with additional signal columns:
                - dc_breakout: "upper", "lower", or None
                - dc_position: "above", "inside", or "below"
                - dc_trend: "bullish", "bearish", or "neutral"

        Raises:
            ValueError: If required DC columns are missing
        """
        try:
            required_cols = ["dc_upper", "dc_lower", "dc_middle", "close"]
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            # Detect breakouts
            result = data.with_columns([
                pl.col("close").shift(1).alias("close_prev")
            ])

            result = result.with_columns([
                # Breakout detection
                pl.when(
                    (pl.col("close") > pl.col("dc_upper")) &
                    (pl.col("close_prev") <= pl.col("dc_upper"))
                )
                  .then(pl.lit("upper"))
                  .when(
                    (pl.col("close") < pl.col("dc_lower")) &
                    (pl.col("close_prev") >= pl.col("dc_lower"))
                  )
                  .then(pl.lit("lower"))
                  .otherwise(pl.lit(None))
                  .alias("dc_breakout"),

                # Position relative to channels
                pl.when(pl.col("close") > pl.col("dc_upper"))
                  .then(pl.lit("above"))
                  .when(pl.col("close") < pl.col("dc_lower"))
                  .then(pl.lit("below"))
                  .otherwise(pl.lit("inside"))
                  .alias("dc_position")
            ])

            # Trend determination
            result = result.with_columns([
                pl.when(
                    (pl.col("close") > pl.col("dc_middle")) &
                    (pl.col("dc_position") != "below")
                )
                  .then(pl.lit("bullish"))
                  .when(
                    (pl.col("close") < pl.col("dc_middle")) &
                    (pl.col("dc_position") != "above")
                  )
                  .then(pl.lit("bearish"))
                  .otherwise(pl.lit("neutral"))
                  .alias("dc_trend")
            ])

            # Drop intermediate column
            result = result.drop(["close_prev"])

            logger.debug("donchian_breakouts_detected", rows=len(result))

            return result

        except Exception as e:
            logger.error("donchian_breakout_detection_failed", error=str(e))
            raise

    def calculate_volatility_squeeze(
        self,
        data: pl.DataFrame,
        lookback_periods: Optional[int] = None
    ) -> pl.DataFrame:
        """Detect volatility squeeze using channel width.

        A squeeze occurs when the channel width narrows significantly,
        indicating low volatility and potential for a breakout.

        Args:
            data: DataFrame with Donchian Channels
            lookback_periods: Periods for comparing channel width

        Returns:
            DataFrame with squeeze indicators

        Raises:
            ValueError: If dc_width column is missing
        """
        try:
            if "dc_width" not in data.columns:
                raise ValueError("dc_width not found. Calculate Donchian Channels first.")

            if lookback_periods is None:
                lookback_periods = int(self.config.get("squeeze_lookback", 100))

            # Calculate percentile of current width relative to historical
            result = data.with_columns([
                pl.col("dc_width")
                  .rolling_min(window_size=lookback_periods)
                  .alias("width_min"),
                pl.col("dc_width")
                  .rolling_max(window_size=lookback_periods)
                  .alias("width_max")
            ])

            # Normalize width to 0-100 percentile
            result = result.with_columns([
                pl.when(
                    (pl.col("width_max") - pl.col("width_min")) == pl.lit(0)
                )
                  .then(pl.lit(50.0))
                  .otherwise(
                    ((pl.col("dc_width") - pl.col("width_min")) /
                     (pl.col("width_max") - pl.col("width_min"))) * pl.lit(100)
                  )
                  .alias("width_percentile")
            ])

            # Detect squeeze (width in lower percentile)
            squeeze_threshold = Decimal(str(self.config.get("squeeze_threshold", "20")))

            result = result.with_columns([
                pl.when(pl.col("width_percentile") < float(squeeze_threshold))
                  .then(pl.lit(True))
                  .otherwise(pl.lit(False))
                  .alias("dc_squeeze")
            ])

            # Drop intermediate columns
            result = result.drop(["width_min", "width_max"])

            logger.debug(
                "volatility_squeeze_calculated",
                lookback_periods=lookback_periods,
                squeeze_threshold=float(squeeze_threshold)
            )

            return result

        except Exception as e:
            logger.error("volatility_squeeze_calculation_failed", error=str(e))
            raise

    def calculate_stop_loss(
        self,
        data: pl.DataFrame,
        position_type: str = "long"
    ) -> pl.DataFrame:
        """Calculate stop-loss levels based on Donchian Channels.

        Args:
            data: DataFrame with Donchian Channels
            position_type: "long" or "short"

        Returns:
            DataFrame with stop_loss column

        Raises:
            ValueError: If required columns are missing or invalid position_type
        """
        try:
            required_cols = ["dc_upper", "dc_lower", "dc_middle"]
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            if position_type not in ["long", "short"]:
                raise ValueError(f"Invalid position_type: {position_type}. Must be 'long' or 'short'")

            # Get buffer from config (percentage below/above channel)
            buffer_percent = Decimal(str(self.config.get("stop_loss_buffer_percent", "1.0")))

            if position_type == "long":
                # For long positions, stop loss below lower channel
                result = data.with_columns([
                    (pl.col("dc_lower") * (pl.lit(1) - (float(buffer_percent) / 100)))
                      .alias("stop_loss")
                ])
            else:  # short
                # For short positions, stop loss above upper channel
                result = data.with_columns([
                    (pl.col("dc_upper") * (pl.lit(1) + (float(buffer_percent) / 100)))
                      .alias("stop_loss")
                ])

            logger.debug(
                "stop_loss_calculated",
                position_type=position_type,
                buffer_percent=float(buffer_percent)
            )

            return result

        except Exception as e:
            logger.error("stop_loss_calculation_failed", error=str(e))
            raise

    def calculate_take_profit(
        self,
        data: pl.DataFrame,
        position_type: str = "long",
        risk_reward_ratio: Optional[Decimal] = None
    ) -> pl.DataFrame:
        """Calculate take-profit levels based on Donchian Channels.

        Args:
            data: DataFrame with Donchian Channels and entry price
            position_type: "long" or "short"
            risk_reward_ratio: Risk/reward ratio for take profit

        Returns:
            DataFrame with take_profit column

        Raises:
            ValueError: If required columns are missing
        """
        try:
            required_cols = ["dc_upper", "dc_lower", "close"]
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            if risk_reward_ratio is None:
                risk_reward_ratio = Decimal(str(self.config.get("risk_reward_ratio", "2.0")))
            else:
                risk_reward_ratio = Decimal(str(risk_reward_ratio))

            if position_type == "long":
                # For long positions, take profit at upper channel or based on R:R
                # Simple approach: use upper channel
                result = data.with_columns([
                    pl.col("dc_upper").alias("take_profit")
                ])
            else:  # short
                # For short positions, take profit at lower channel
                result = data.with_columns([
                    pl.col("dc_lower").alias("take_profit")
                ])

            logger.debug(
                "take_profit_calculated",
                position_type=position_type,
                risk_reward_ratio=float(risk_reward_ratio)
            )

            return result

        except Exception as e:
            logger.error("take_profit_calculation_failed", error=str(e))
            raise


async def calculate_donchian_channels(
    data: pl.DataFrame,
    config: Dict[str, Any]
) -> pl.DataFrame:
    """Convenience function to calculate Donchian Channels.

    Args:
        data: Polars DataFrame with high and low prices
        config: Configuration dictionary

    Returns:
        DataFrame with Donchian Channels

    Example:
        >>> config = {"period": 20}
        >>> result = await calculate_donchian_channels(df, config)
    """
    indicator = DonchianChannels(config)
    return await indicator.calculate(data)
