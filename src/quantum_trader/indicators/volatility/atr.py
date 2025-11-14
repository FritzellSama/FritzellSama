"""Average True Range (ATR) - Volatility Indicator.

ATR measures market volatility by decomposing the entire range of an asset
for a given period. It's particularly useful for stop-loss placement and
position sizing.
"""

import asyncio
from decimal import Decimal
from typing import Any, Dict, Optional

import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class ATRIndicator:
    """Average True Range (ATR) volatility indicator.

    ATR is a technical analysis volatility indicator that measures the degree
    of price volatility. It does not indicate price direction but rather the
    degree of price movement or volatility.

    Attributes:
        config: Configuration dictionary containing indicator parameters
        period: Lookback period for ATR calculation
        smoothing_method: Method for smoothing ('ema', 'sma', 'rma')

    Example:
        >>> config = {"period": 14, "smoothing_method": "rma"}
        >>> atr = ATRIndicator(config)
        >>> result = await atr.calculate(df)
        >>> print(result.select(["timestamp", "true_range", "atr"]))
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize ATR indicator.

        Args:
            config: Configuration dictionary with keys:
                - period: int, lookback period (default from config)
                - smoothing_method: str, smoothing method ('ema', 'sma', 'rma')
                - min_periods: int, minimum periods required

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()
        self.period = int(self.config.get("period", self.config.get("default_period", 14)))
        self.smoothing_method = self.config.get("smoothing_method", "rma")
        self.min_periods = int(self.config.get("min_periods", self.period))

        logger.info(
            "atr_initialized",
            period=self.period,
            smoothing_method=self.smoothing_method,
            min_periods=self.min_periods
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        period = self.config.get("period", self.config.get("default_period", 14))
        if not isinstance(period, (int, str)) or int(period) < 1:
            raise ValueError(f"Invalid period: {period}. Must be positive integer")

        smoothing_method = self.config.get("smoothing_method", "rma")
        valid_methods = ["ema", "sma", "rma", "wilder"]
        if smoothing_method not in valid_methods:
            raise ValueError(
                f"Invalid smoothing_method: {smoothing_method}. "
                f"Must be one of {valid_methods}"
            )

        logger.debug("atr_config_validated", config=self.config)

    async def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate Average True Range.

        Args:
            data: Polars DataFrame with columns:
                - high: High prices
                - low: Low prices
                - close: Close prices
                - timestamp: Optional timestamp column

        Returns:
            DataFrame with additional columns:
                - true_range: True Range values
                - atr: Average True Range
                - atr_percent: ATR as percentage of close price

        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        try:
            self._validate_data(data)

            logger.debug(
                "calculating_atr",
                rows=len(data),
                period=self.period,
                smoothing_method=self.smoothing_method
            )

            # Calculate True Range
            # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
            result = data.with_columns([
                pl.col("close").shift(1).alias("prev_close")
            ])

            # Calculate the three components of True Range
            result = result.with_columns([
                (pl.col("high") - pl.col("low")).alias("high_low"),
                (pl.col("high") - pl.col("prev_close")).abs().alias("high_prev_close"),
                (pl.col("low") - pl.col("prev_close")).abs().alias("low_prev_close")
            ])

            # True Range is the maximum of the three
            result = result.with_columns([
                pl.max_horizontal(["high_low", "high_prev_close", "low_prev_close"])
                  .alias("true_range")
            ])

            # Calculate ATR based on smoothing method
            if self.smoothing_method in ["rma", "wilder"]:
                # Wilder's smoothing (RMA) - this is the traditional ATR calculation
                result = self._calculate_wilder_atr(result)
            elif self.smoothing_method == "ema":
                result = result.with_columns([
                    pl.col("true_range")
                      .ewm_mean(span=self.period, min_periods=self.min_periods)
                      .alias("atr")
                ])
            else:  # sma
                result = result.with_columns([
                    pl.col("true_range")
                      .rolling_mean(window_size=self.period, min_periods=self.min_periods)
                      .alias("atr")
                ])

            # Calculate ATR as percentage of price
            result = result.with_columns([
                ((pl.col("atr") / pl.col("close")) * pl.lit(100)).alias("atr_percent")
            ])

            # Drop intermediate columns
            result = result.drop([
                "prev_close", "high_low", "high_prev_close", "low_prev_close"
            ])

            logger.debug(
                "atr_calculated",
                rows=len(result),
                non_null_values=result.filter(pl.col("atr").is_not_null()).height,
                avg_atr=result["atr"].mean()
            )

            return result

        except Exception as e:
            logger.error(
                "atr_calculation_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def _calculate_wilder_atr(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate ATR using Wilder's smoothing method.

        Wilder's smoothing: ATR = ((prior ATR × (n-1)) + current TR) / n

        Args:
            data: DataFrame with true_range column

        Returns:
            DataFrame with atr column added
        """
        tr_values = data["true_range"].to_list()
        atr_values = []

        for i in range(len(tr_values)):
            if i < self.min_periods - 1:
                atr_values.append(None)
            elif i == self.min_periods - 1:
                # First ATR is simple average
                first_atr = sum(tr_values[:i + 1]) / (i + 1)
                atr_values.append(first_atr)
            else:
                # Subsequent ATRs use Wilder's smoothing
                prev_atr = atr_values[i - 1]
                current_tr = tr_values[i]
                if prev_atr is not None and current_tr is not None:
                    new_atr = ((prev_atr * (self.period - 1)) + current_tr) / self.period
                    atr_values.append(new_atr)
                else:
                    atr_values.append(None)

        return data.with_columns([
            pl.Series("atr", atr_values)
        ])

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

    def calculate_stop_loss(
        self,
        data: pl.DataFrame,
        multiplier: Optional[Decimal] = None
    ) -> pl.DataFrame:
        """Calculate ATR-based stop-loss levels.

        Args:
            data: DataFrame with ATR values
            multiplier: ATR multiplier for stop-loss (default from config)

        Returns:
            DataFrame with additional columns:
                - stop_loss_long: Stop loss for long positions
                - stop_loss_short: Stop loss for short positions

        Raises:
            ValueError: If ATR column is missing
        """
        try:
            if "atr" not in data.columns:
                raise ValueError("ATR column not found. Calculate ATR first.")

            if multiplier is None:
                multiplier = Decimal(str(self.config.get("stop_loss_multiplier", "2.0")))
            else:
                multiplier = Decimal(str(multiplier))

            result = data.with_columns([
                # Long stop loss: close - (ATR × multiplier)
                (pl.col("close") - (pl.col("atr") * float(multiplier)))
                  .alias("stop_loss_long"),

                # Short stop loss: close + (ATR × multiplier)
                (pl.col("close") + (pl.col("atr") * float(multiplier)))
                  .alias("stop_loss_short")
            ])

            logger.debug(
                "atr_stop_loss_calculated",
                multiplier=float(multiplier),
                rows=len(result)
            )

            return result

        except Exception as e:
            logger.error("atr_stop_loss_calculation_failed", error=str(e))
            raise

    def calculate_position_size(
        self,
        data: pl.DataFrame,
        account_balance: Decimal,
        risk_percent: Optional[Decimal] = None
    ) -> pl.DataFrame:
        """Calculate position size based on ATR and risk management.

        Args:
            data: DataFrame with ATR values
            account_balance: Account balance in quote currency
            risk_percent: Risk percentage per trade (default from config)

        Returns:
            DataFrame with position_size column

        Raises:
            ValueError: If ATR column is missing or parameters invalid
        """
        try:
            if "atr" not in data.columns:
                raise ValueError("ATR column not found. Calculate ATR first.")

            if not isinstance(account_balance, Decimal):
                account_balance = Decimal(str(account_balance))

            if risk_percent is None:
                risk_percent = Decimal(str(self.config.get("risk_percent", "1.0")))
            else:
                risk_percent = Decimal(str(risk_percent))

            # Risk amount = account balance × risk %
            risk_amount = account_balance * (risk_percent / Decimal("100"))

            # Position size = risk amount / (ATR × multiplier)
            atr_multiplier = Decimal(str(self.config.get("position_size_atr_multiplier", "2.0")))

            result = data.with_columns([
                (pl.lit(float(risk_amount)) / (pl.col("atr") * float(atr_multiplier)))
                  .alias("position_size")
            ])

            logger.debug(
                "atr_position_size_calculated",
                account_balance=float(account_balance),
                risk_percent=float(risk_percent),
                atr_multiplier=float(atr_multiplier)
            )

            return result

        except Exception as e:
            logger.error("atr_position_size_calculation_failed", error=str(e))
            raise


async def calculate_atr(
    data: pl.DataFrame,
    config: Dict[str, Any]
) -> pl.DataFrame:
    """Convenience function to calculate ATR indicator.

    Args:
        data: Polars DataFrame with high, low, and close prices
        config: Configuration dictionary

    Returns:
        DataFrame with ATR indicators

    Example:
        >>> config = {"period": 14, "smoothing_method": "rma"}
        >>> result = await calculate_atr(df, config)
    """
    indicator = ATRIndicator(config)
    return await indicator.calculate(data)
