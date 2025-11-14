"""
SuperTrend Indicator.

This module implements the SuperTrend indicator for trend identification
and entry/exit signals in institutional trading systems.
"""

import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Optional
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class SuperTrendError(Exception):
    """Base exception for SuperTrend indicator errors."""
    pass


class SuperTrendValidator:
    """Validates SuperTrend configuration and input data."""

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate SuperTrend configuration.

        Args:
            config: Configuration dictionary

        Raises:
            SuperTrendError: If configuration is invalid
        """
        required_keys = ['atr_period', 'multiplier']
        for key in required_keys:
            if key not in config:
                raise SuperTrendError(f"Missing required config key: {key}")

        try:
            atr_period = int(config['atr_period'])
            multiplier = Decimal(str(config['multiplier']))

            if atr_period < 1:
                raise SuperTrendError("atr_period must be at least 1")
            if multiplier <= Decimal('0'):
                raise SuperTrendError("multiplier must be positive")

        except (ValueError, TypeError) as e:
            raise SuperTrendError(f"Invalid configuration values: {e}")

    @staticmethod
    def validate_dataframe(df: pl.DataFrame) -> None:
        """Validate input dataframe structure.

        Args:
            df: Input dataframe

        Raises:
            SuperTrendError: If dataframe is invalid
        """
        required_columns = ['high', 'low', 'close']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise SuperTrendError(f"Missing required columns: {missing}")

        if len(df) < 2:
            raise SuperTrendError("Dataframe must have at least 2 rows")


class SuperTrend:
    """SuperTrend indicator implementation.

    SuperTrend uses Average True Range (ATR) to identify trend direction
    and potential reversal points.

    Formula:
        Basic Upper Band = (High + Low) / 2 + Multiplier * ATR
        Basic Lower Band = (High + Low) / 2 - Multiplier * ATR

    The final bands are adjusted based on previous values and trend direction.

    Attributes:
        config: Configuration dictionary
        atr_period: Period for ATR calculation (typically 10)
        multiplier: ATR multiplier for band calculation (typically 3.0)

    Example:
        >>> config = {
        ...     'atr_period': 10,
        ...     'multiplier': '3.0'
        ... }
        >>> supertrend = SuperTrend(config)
        >>> result = await supertrend.calculate(df)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize SuperTrend indicator.

        Args:
            config: Configuration dictionary

        Raises:
            SuperTrendError: If configuration is invalid
        """
        SuperTrendValidator.validate_config(config)

        self.config = config
        self.atr_period = int(config['atr_period'])
        self.multiplier = Decimal(str(config['multiplier']))

        logger.info(
            "SuperTrend initialized",
            atr_period=self.atr_period,
            multiplier=str(self.multiplier)
        )

    async def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate SuperTrend values.

        Args:
            df: Input dataframe with columns: high, low, close

        Returns:
            DataFrame with additional columns:
                - atr: Average True Range
                - supertrend: SuperTrend line value
                - supertrend_direction: 1 for uptrend, -1 for downtrend
                - supertrend_signal: True when trend changes

        Raises:
            SuperTrendError: If calculation fails

        Example:
            >>> df = pl.DataFrame({
            ...     'high': [102.0, 103.0, 104.0],
            ...     'low': [99.0, 100.0, 101.0],
            ...     'close': [101.0, 102.0, 103.0]
            ... })
            >>> result = await supertrend.calculate(df)
        """
        try:
            SuperTrendValidator.validate_dataframe(df)

            logger.debug("Calculating SuperTrend", rows=len(df))

            highs = [Decimal(str(x)) for x in df['high'].to_list()]
            lows = [Decimal(str(x)) for x in df['low'].to_list()]
            closes = [Decimal(str(x)) for x in df['close'].to_list()]

            # Calculate Average True Range (ATR)
            atr_values = self._calculate_atr(highs, lows, closes)

            # Calculate basic upper and lower bands
            basic_upper_bands = []
            basic_lower_bands = []

            for i in range(len(closes)):
                if atr_values[i] is None:
                    basic_upper_bands.append(None)
                    basic_lower_bands.append(None)
                else:
                    hl_avg = (highs[i] + lows[i]) / Decimal('2')
                    atr_mult = atr_values[i] * self.multiplier

                    basic_upper = hl_avg + atr_mult
                    basic_lower = hl_avg - atr_mult

                    basic_upper_bands.append(basic_upper)
                    basic_lower_bands.append(basic_lower)

            # Calculate final bands with trend logic
            final_upper_bands = []
            final_lower_bands = []

            for i in range(len(closes)):
                if basic_upper_bands[i] is None:
                    final_upper_bands.append(None)
                    final_lower_bands.append(None)
                else:
                    # Upper band
                    if i == 0 or final_upper_bands[i-1] is None:
                        final_upper = basic_upper_bands[i]
                    else:
                        if basic_upper_bands[i] < final_upper_bands[i-1] or closes[i-1] > final_upper_bands[i-1]:
                            final_upper = basic_upper_bands[i]
                        else:
                            final_upper = final_upper_bands[i-1]

                    # Lower band
                    if i == 0 or final_lower_bands[i-1] is None:
                        final_lower = basic_lower_bands[i]
                    else:
                        if basic_lower_bands[i] > final_lower_bands[i-1] or closes[i-1] < final_lower_bands[i-1]:
                            final_lower = basic_lower_bands[i]
                        else:
                            final_lower = final_lower_bands[i-1]

                    final_upper_bands.append(final_upper)
                    final_lower_bands.append(final_lower)

            # Determine SuperTrend line and direction
            supertrend_values = []
            directions = []
            signals = []

            for i in range(len(closes)):
                if final_upper_bands[i] is None or final_lower_bands[i] is None:
                    supertrend_values.append(None)
                    directions.append(0)
                    signals.append(False)
                else:
                    # Determine direction
                    if i == 0:
                        # Initialize with uptrend if close > lower band
                        if closes[i] > final_lower_bands[i]:
                            direction = 1
                            supertrend = final_lower_bands[i]
                        else:
                            direction = -1
                            supertrend = final_upper_bands[i]
                    else:
                        prev_direction = directions[i-1] if directions[i-1] != 0 else -1

                        if prev_direction == 1:
                            # In uptrend
                            if closes[i] <= final_lower_bands[i]:
                                # Switch to downtrend
                                direction = -1
                                supertrend = final_upper_bands[i]
                            else:
                                # Continue uptrend
                                direction = 1
                                supertrend = final_lower_bands[i]
                        else:
                            # In downtrend
                            if closes[i] >= final_upper_bands[i]:
                                # Switch to uptrend
                                direction = 1
                                supertrend = final_lower_bands[i]
                            else:
                                # Continue downtrend
                                direction = -1
                                supertrend = final_upper_bands[i]

                    supertrend_values.append(float(supertrend))
                    directions.append(direction)

                    # Signal when direction changes
                    if i > 0 and directions[i-1] != 0:
                        signals.append(direction != directions[i-1])
                    else:
                        signals.append(False)

            # Add columns to dataframe
            result = df.with_columns([
                pl.Series('atr', [float(x) if x is not None else None for x in atr_values]),
                pl.Series('supertrend', supertrend_values),
                pl.Series('supertrend_direction', directions),
                pl.Series('supertrend_signal', signals)
            ])

            logger.info(
                "SuperTrend calculated successfully",
                rows=len(result),
                signals=sum(signals)
            )

            return result

        except SuperTrendError:
            raise
        except Exception as e:
            logger.error("SuperTrend calculation failed", error=str(e))
            raise SuperTrendError(f"Calculation failed: {e}")

    def _calculate_atr(
        self,
        highs: list,
        lows: list,
        closes: list
    ) -> list:
        """Calculate Average True Range.

        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of close prices

        Returns:
            List of ATR values
        """
        # Calculate True Range
        tr_values = []

        for i in range(len(highs)):
            high_low = highs[i] - lows[i]

            if i == 0:
                tr = high_low
            else:
                high_close = abs(highs[i] - closes[i-1])
                low_close = abs(lows[i] - closes[i-1])
                tr = max(high_low, high_close, low_close)

            tr_values.append(tr)

        # Calculate ATR using Wilder's smoothing
        atr_values = []
        atr = None

        for i in range(len(tr_values)):
            if i < self.atr_period - 1:
                atr_values.append(None)
            elif i == self.atr_period - 1:
                # First ATR is simple average
                atr = sum(tr_values[:i+1]) / Decimal(str(self.atr_period))
                atr_values.append(atr)
            else:
                # Subsequent ATR uses Wilder's smoothing
                # ATR = ((ATR_prev * (period - 1)) + TR) / period
                atr = ((atr * Decimal(str(self.atr_period - 1))) + tr_values[i]) / Decimal(str(self.atr_period))
                atr_values.append(atr)

        return atr_values

    def get_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract trading signals from SuperTrend.

        Args:
            df: DataFrame with SuperTrend calculations

        Returns:
            DataFrame with enhanced signal columns:
                - supertrend_trade_signal: 1 for long, -1 for short, 0 for no change
                - supertrend_long: True when in uptrend
                - supertrend_short: True when in downtrend

        Raises:
            SuperTrendError: If required columns missing
        """
        try:
            required = ['supertrend_direction', 'supertrend_signal']
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise SuperTrendError(f"Missing required columns: {missing}")

            directions = df['supertrend_direction'].to_list()
            signals = df['supertrend_signal'].to_list()

            trade_signals = []
            long_flags = []
            short_flags = []

            for i in range(len(directions)):
                direction = directions[i]
                signal = signals[i]

                # Trade signal only when trend changes
                if signal:
                    trade_signals.append(direction)
                else:
                    trade_signals.append(0)

                # Position flags
                long_flags.append(direction == 1)
                short_flags.append(direction == -1)

            result = df.with_columns([
                pl.Series('supertrend_trade_signal', trade_signals),
                pl.Series('supertrend_long', long_flags),
                pl.Series('supertrend_short', short_flags)
            ])

            logger.debug(
                "SuperTrend signals extracted",
                long_signals=sum(1 for s in trade_signals if s == 1),
                short_signals=sum(1 for s in trade_signals if s == -1)
            )

            return result

        except Exception as e:
            logger.error("Signal extraction failed", error=str(e))
            raise SuperTrendError(f"Signal extraction failed: {e}")


async def calculate_supertrend(
    df: pl.DataFrame,
    config: Dict[str, Any]
) -> pl.DataFrame:
    """Convenience function to calculate SuperTrend.

    Args:
        df: Input dataframe with OHLC data
        config: Configuration dictionary

    Returns:
        DataFrame with SuperTrend calculations

    Example:
        >>> config = {
        ...     'atr_period': 10,
        ...     'multiplier': '3.0'
        ... }
        >>> result = await calculate_supertrend(df, config)
    """
    supertrend = SuperTrend(config)
    return await supertrend.calculate(df)
