"""
Standard Deviation Volatility Indicator.

This module implements standard deviation calculation for measuring
price dispersion and volatility in institutional trading systems.
"""

import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Optional
import polars as pl
from structlog import get_logger
import math

logger = get_logger(__name__)


class StandardDeviationError(Exception):
    """Base exception for Standard Deviation indicator errors."""
    pass


class StandardDeviationValidator:
    """Validates Standard Deviation configuration and input data."""

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate Standard Deviation configuration.

        Args:
            config: Configuration dictionary

        Raises:
            StandardDeviationError: If configuration is invalid
        """
        if 'period' not in config:
            raise StandardDeviationError("Missing required config key: period")

        try:
            period = int(config['period'])
            if period < 2:
                raise StandardDeviationError("period must be at least 2")

            if 'num_std_dev' in config:
                num_std = Decimal(str(config['num_std_dev']))
                if num_std <= Decimal('0'):
                    raise StandardDeviationError("num_std_dev must be positive")

        except (ValueError, TypeError) as e:
            raise StandardDeviationError(f"Invalid configuration values: {e}")

    @staticmethod
    def validate_dataframe(df: pl.DataFrame) -> None:
        """Validate input dataframe structure.

        Args:
            df: Input dataframe

        Raises:
            StandardDeviationError: If dataframe is invalid
        """
        if 'close' not in df.columns:
            raise StandardDeviationError("Missing required column: close")

        if len(df) < 2:
            raise StandardDeviationError("Dataframe must have at least 2 rows")


class StandardDeviation:
    """Standard Deviation volatility indicator.

    Calculates rolling standard deviation of price to measure volatility
    and identify potential breakout/breakdown points.

    Attributes:
        config: Configuration dictionary
        period: Lookback period for calculation
        num_std_dev: Number of standard deviations for bands

    Example:
        >>> config = {
        ...     'period': 20,
        ...     'num_std_dev': '2.0'
        ... }
        >>> std_dev = StandardDeviation(config)
        >>> result = await std_dev.calculate(df)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Standard Deviation indicator.

        Args:
            config: Configuration dictionary

        Raises:
            StandardDeviationError: If configuration is invalid
        """
        StandardDeviationValidator.validate_config(config)

        self.config = config
        self.period = int(config['period'])
        self.num_std_dev = Decimal(str(config.get('num_std_dev', '2.0')))

        logger.info(
            "StandardDeviation initialized",
            period=self.period,
            num_std_dev=str(self.num_std_dev)
        )

    async def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate standard deviation and bands.

        Args:
            df: Input dataframe with column: close

        Returns:
            DataFrame with additional columns:
                - sma: Simple moving average
                - std_dev: Standard deviation
                - upper_band: SMA + (num_std_dev * std_dev)
                - lower_band: SMA - (num_std_dev * std_dev)
                - bandwidth: Measure of band width

        Raises:
            StandardDeviationError: If calculation fails

        Example:
            >>> df = pl.DataFrame({
            ...     'close': [100.0, 102.0, 101.0, 105.0, 103.0]
            ... })
            >>> result = await std_dev.calculate(df)
        """
        try:
            StandardDeviationValidator.validate_dataframe(df)

            logger.debug("Calculating standard deviation", rows=len(df))

            closes = [Decimal(str(x)) for x in df['close'].to_list()]

            sma_values = []
            std_dev_values = []
            upper_band_values = []
            lower_band_values = []
            bandwidth_values = []

            for i in range(len(closes)):
                if i < self.period - 1:
                    # Not enough data yet
                    sma_values.append(None)
                    std_dev_values.append(None)
                    upper_band_values.append(None)
                    lower_band_values.append(None)
                    bandwidth_values.append(None)
                else:
                    # Get window of prices
                    window = closes[i - self.period + 1:i + 1]

                    # Calculate simple moving average
                    sma = sum(window) / Decimal(str(len(window)))

                    # Calculate variance
                    variance = sum((price - sma) ** 2 for price in window) / Decimal(str(len(window)))

                    # Calculate standard deviation
                    std_dev = Decimal(str(math.sqrt(float(variance))))

                    # Calculate bands
                    upper_band = sma + (self.num_std_dev * std_dev)
                    lower_band = sma - (self.num_std_dev * std_dev)

                    # Calculate bandwidth (normalized)
                    if sma > Decimal('0'):
                        bandwidth = ((upper_band - lower_band) / sma) * Decimal('100')
                    else:
                        bandwidth = Decimal('0')

                    sma_values.append(float(sma))
                    std_dev_values.append(float(std_dev))
                    upper_band_values.append(float(upper_band))
                    lower_band_values.append(float(lower_band))
                    bandwidth_values.append(float(bandwidth))

            # Add columns to dataframe
            result = df.with_columns([
                pl.Series('sma', sma_values),
                pl.Series('std_dev', std_dev_values),
                pl.Series('upper_band', upper_band_values),
                pl.Series('lower_band', lower_band_values),
                pl.Series('bandwidth', bandwidth_values)
            ])

            logger.info("Standard deviation calculated successfully", rows=len(result))

            return result

        except StandardDeviationError:
            raise
        except Exception as e:
            logger.error("Standard deviation calculation failed", error=str(e))
            raise StandardDeviationError(f"Calculation failed: {e}")

    async def calculate_bollinger_bands(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Bollinger Bands (alias for standard calculation).

        Bollinger Bands are simply SMA +/- N standard deviations.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with Bollinger Bands
        """
        result = await self.calculate(df)

        # Add %B indicator (position within bands)
        result = self._calculate_percent_b(result)

        return result

    def _calculate_percent_b(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate %B indicator.

        %B shows where price is relative to the bands.
        %B = (Close - Lower Band) / (Upper Band - Lower Band)

        Args:
            df: DataFrame with band calculations

        Returns:
            DataFrame with %B column
        """
        required = ['close', 'upper_band', 'lower_band']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise StandardDeviationError(f"Missing required columns: {missing}")

        closes = df['close'].to_list()
        upper_bands = df['upper_band'].to_list()
        lower_bands = df['lower_band'].to_list()

        percent_b_values = []

        for close, upper, lower in zip(closes, upper_bands, lower_bands):
            if upper is None or lower is None:
                percent_b_values.append(None)
            else:
                close_dec = Decimal(str(close))
                upper_dec = Decimal(str(upper))
                lower_dec = Decimal(str(lower))

                band_width = upper_dec - lower_dec

                if band_width > Decimal('0'):
                    percent_b = (close_dec - lower_dec) / band_width
                    percent_b_values.append(float(percent_b))
                else:
                    percent_b_values.append(None)

        return df.with_columns([
            pl.Series('percent_b', percent_b_values)
        ])

    def get_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate trading signals from standard deviation bands.

        Args:
            df: DataFrame with band calculations

        Returns:
            DataFrame with signal columns:
                - band_signal: 1 when below lower band, -1 when above upper band
                - squeeze: True when bandwidth is very low
                - breakout: True when price breaks out of bands

        Raises:
            StandardDeviationError: If required columns missing
        """
        try:
            required = ['close', 'upper_band', 'lower_band', 'bandwidth']
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise StandardDeviationError(f"Missing required columns: {missing}")

            closes = df['close'].to_list()
            upper_bands = df['upper_band'].to_list()
            lower_bands = df['lower_band'].to_list()
            bandwidths = df['bandwidth'].to_list()

            signals = []
            squeezes = []
            breakouts = []

            # Calculate bandwidth percentile for squeeze detection
            valid_bandwidths = [b for b in bandwidths if b is not None]
            if len(valid_bandwidths) > 0:
                sorted_bw = sorted(valid_bandwidths)
                squeeze_threshold = sorted_bw[len(sorted_bw) // 5]  # 20th percentile
            else:
                squeeze_threshold = 0

            for i in range(len(closes)):
                if upper_bands[i] is None or lower_bands[i] is None:
                    signals.append(0)
                    squeezes.append(False)
                    breakouts.append(False)
                else:
                    close = closes[i]
                    upper = upper_bands[i]
                    lower = lower_bands[i]
                    bandwidth = bandwidths[i] if bandwidths[i] is not None else 0

                    # Signal: buy when below lower band, sell when above upper band
                    if close < lower:
                        signals.append(1)
                    elif close > upper:
                        signals.append(-1)
                    else:
                        signals.append(0)

                    # Squeeze: bandwidth is very low
                    squeezes.append(bandwidth < squeeze_threshold)

                    # Breakout: price crosses band
                    if i > 0 and upper_bands[i-1] is not None:
                        prev_close = closes[i-1]
                        prev_upper = upper_bands[i-1]
                        prev_lower = lower_bands[i-1]

                        crossed_upper = (prev_close <= prev_upper and close > upper)
                        crossed_lower = (prev_close >= prev_lower and close < lower)
                        breakouts.append(crossed_upper or crossed_lower)
                    else:
                        breakouts.append(False)

            result = df.with_columns([
                pl.Series('band_signal', signals),
                pl.Series('squeeze', squeezes),
                pl.Series('breakout', breakouts)
            ])

            logger.debug(
                "Standard deviation signals generated",
                buy_signals=sum(1 for s in signals if s == 1),
                sell_signals=sum(1 for s in signals if s == -1),
                squeezes=sum(squeezes),
                breakouts=sum(breakouts)
            )

            return result

        except Exception as e:
            logger.error("Signal generation failed", error=str(e))
            raise StandardDeviationError(f"Signal generation failed: {e}")

    async def calculate_historical_volatility(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate annualized historical volatility.

        Args:
            df: Input dataframe with close prices

        Returns:
            DataFrame with historical_volatility column

        Raises:
            StandardDeviationError: If calculation fails
        """
        try:
            StandardDeviationValidator.validate_dataframe(df)

            closes = [Decimal(str(x)) for x in df['close'].to_list()]

            # Calculate log returns
            log_returns = []
            for i in range(len(closes)):
                if i == 0:
                    log_returns.append(Decimal('0'))
                elif closes[i-1] > Decimal('0'):
                    ratio = closes[i] / closes[i-1]
                    log_ret = Decimal(str(math.log(float(ratio))))
                    log_returns.append(log_ret)
                else:
                    log_returns.append(Decimal('0'))

            # Calculate rolling standard deviation of returns
            hv_values = []
            annualization_factor = Decimal(str(math.sqrt(252)))  # For daily data

            for i in range(len(log_returns)):
                if i < self.period - 1:
                    hv_values.append(None)
                else:
                    window = log_returns[i - self.period + 1:i + 1]

                    # Calculate standard deviation
                    mean_return = sum(window) / Decimal(str(len(window)))
                    variance = sum((r - mean_return) ** 2 for r in window) / Decimal(str(len(window)))
                    std_dev = Decimal(str(math.sqrt(float(variance))))

                    # Annualize
                    hv = std_dev * annualization_factor * Decimal('100')  # Convert to percentage
                    hv_values.append(float(hv))

            result = df.with_columns([
                pl.Series('historical_volatility', hv_values)
            ])

            logger.info("Historical volatility calculated successfully")

            return result

        except Exception as e:
            logger.error("Historical volatility calculation failed", error=str(e))
            raise StandardDeviationError(f"Historical volatility calculation failed: {e}")


async def calculate_standard_deviation(
    df: pl.DataFrame,
    config: Dict[str, Any]
) -> pl.DataFrame:
    """Convenience function to calculate standard deviation.

    Args:
        df: Input dataframe with price data
        config: Configuration dictionary

    Returns:
        DataFrame with standard deviation calculations

    Example:
        >>> config = {
        ...     'period': 20,
        ...     'num_std_dev': '2.0'
        ... }
        >>> result = await calculate_standard_deviation(df, config)
    """
    std_dev = StandardDeviation(config)
    return await std_dev.calculate(df)
