"""
Parabolic SAR (Stop and Reverse) Trend Indicator.

This module implements the Parabolic SAR indicator for trend identification
and reversal detection in institutional trading systems.
"""

import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Optional
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class ParabolicSARError(Exception):
    """Base exception for Parabolic SAR indicator errors."""
    pass


class ParabolicSARValidator:
    """Validates Parabolic SAR configuration and input data."""

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate Parabolic SAR configuration.

        Args:
            config: Configuration dictionary

        Raises:
            ParabolicSARError: If configuration is invalid
        """
        required_keys = ['acceleration_start', 'acceleration_increment', 'acceleration_max']
        for key in required_keys:
            if key not in config:
                raise ParabolicSARError(f"Missing required config key: {key}")

        try:
            start = Decimal(str(config['acceleration_start']))
            increment = Decimal(str(config['acceleration_increment']))
            max_accel = Decimal(str(config['acceleration_max']))

            if start <= Decimal('0'):
                raise ParabolicSARError("acceleration_start must be positive")
            if increment <= Decimal('0'):
                raise ParabolicSARError("acceleration_increment must be positive")
            if max_accel <= start:
                raise ParabolicSARError("acceleration_max must be greater than acceleration_start")

        except (ValueError, TypeError) as e:
            raise ParabolicSARError(f"Invalid configuration values: {e}")

    @staticmethod
    def validate_dataframe(df: pl.DataFrame) -> None:
        """Validate input dataframe structure.

        Args:
            df: Input dataframe

        Raises:
            ParabolicSARError: If dataframe is invalid
        """
        required_columns = ['high', 'low', 'close']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ParabolicSARError(f"Missing required columns: {missing}")

        if len(df) < 2:
            raise ParabolicSARError("Dataframe must have at least 2 rows")


class ParabolicSAR:
    """Parabolic SAR (Stop and Reverse) indicator implementation.

    The Parabolic SAR is a trend-following indicator that provides potential
    reversal points. It appears as dots above or below price, indicating
    bearish or bullish trends respectively.

    Attributes:
        config: Configuration dictionary containing acceleration parameters
        af_start: Initial acceleration factor
        af_increment: Acceleration factor increment
        af_max: Maximum acceleration factor

    Example:
        >>> config = {
        ...     'acceleration_start': '0.02',
        ...     'acceleration_increment': '0.02',
        ...     'acceleration_max': '0.20'
        ... }
        >>> psar = ParabolicSAR(config)
        >>> result = await psar.calculate(df)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Parabolic SAR indicator.

        Args:
            config: Configuration dictionary

        Raises:
            ParabolicSARError: If configuration is invalid
        """
        ParabolicSARValidator.validate_config(config)

        self.config = config
        self.af_start = Decimal(str(config['acceleration_start']))
        self.af_increment = Decimal(str(config['acceleration_increment']))
        self.af_max = Decimal(str(config['acceleration_max']))

        logger.info(
            "ParabolicSAR initialized",
            af_start=str(self.af_start),
            af_increment=str(self.af_increment),
            af_max=str(self.af_max)
        )

    async def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Parabolic SAR values.

        Args:
            df: Input dataframe with columns: high, low, close

        Returns:
            DataFrame with additional columns:
                - psar: Parabolic SAR values
                - psar_trend: 1 for bullish, -1 for bearish
                - psar_reversal: True when trend reversal occurs

        Raises:
            ParabolicSARError: If calculation fails

        Example:
            >>> df = pl.DataFrame({
            ...     'high': [100.5, 101.2, 102.0],
            ...     'low': [99.0, 100.0, 100.5],
            ...     'close': [100.0, 101.0, 101.5]
            ... })
            >>> result = await psar.calculate(df)
        """
        try:
            ParabolicSARValidator.validate_dataframe(df)

            logger.debug("Calculating Parabolic SAR", rows=len(df))

            # Convert to list for efficient iteration
            highs = df['high'].to_list()
            lows = df['low'].to_list()

            # Initialize arrays
            psar_values = [Decimal('0')] * len(df)
            trends = [0] * len(df)
            reversals = [False] * len(df)

            # Initialize first position
            # Start with bullish trend assumption
            is_bullish = True
            sar = Decimal(str(lows[0]))
            ep = Decimal(str(highs[0]))  # Extreme point
            af = self.af_start

            psar_values[0] = sar
            trends[0] = 1 if is_bullish else -1

            # Calculate SAR for each period
            for i in range(1, len(df)):
                high = Decimal(str(highs[i]))
                low = Decimal(str(lows[i]))

                # Calculate new SAR
                sar = sar + af * (ep - sar)

                # Check for reversal
                reversal_occurred = False

                if is_bullish:
                    # In uptrend, SAR should be below price
                    if low < sar:
                        # Reversal to downtrend
                        is_bullish = False
                        reversal_occurred = True
                        sar = ep  # SAR becomes the extreme point
                        ep = low  # New extreme point
                        af = self.af_start  # Reset acceleration
                    else:
                        # Continue uptrend
                        # Ensure SAR doesn't go above previous two lows
                        if i >= 2:
                            sar = min(sar, Decimal(str(lows[i-1])), Decimal(str(lows[i-2])))
                        elif i >= 1:
                            sar = min(sar, Decimal(str(lows[i-1])))

                        # Update extreme point and acceleration
                        if high > ep:
                            ep = high
                            af = min(af + self.af_increment, self.af_max)
                else:
                    # In downtrend, SAR should be above price
                    if high > sar:
                        # Reversal to uptrend
                        is_bullish = True
                        reversal_occurred = True
                        sar = ep  # SAR becomes the extreme point
                        ep = high  # New extreme point
                        af = self.af_start  # Reset acceleration
                    else:
                        # Continue downtrend
                        # Ensure SAR doesn't go below previous two highs
                        if i >= 2:
                            sar = max(sar, Decimal(str(highs[i-1])), Decimal(str(highs[i-2])))
                        elif i >= 1:
                            sar = max(sar, Decimal(str(highs[i-1])))

                        # Update extreme point and acceleration
                        if low < ep:
                            ep = low
                            af = min(af + self.af_increment, self.af_max)

                psar_values[i] = sar
                trends[i] = 1 if is_bullish else -1
                reversals[i] = reversal_occurred

            # Add columns to dataframe
            result = df.with_columns([
                pl.Series('psar', [float(x) for x in psar_values]),
                pl.Series('psar_trend', trends),
                pl.Series('psar_reversal', reversals)
            ])

            logger.info(
                "Parabolic SAR calculated successfully",
                rows=len(result),
                reversals=sum(reversals)
            )

            return result

        except ParabolicSARError:
            raise
        except Exception as e:
            logger.error("Parabolic SAR calculation failed", error=str(e))
            raise ParabolicSARError(f"Calculation failed: {e}")

    def get_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract trading signals from Parabolic SAR.

        Args:
            df: DataFrame with PSAR calculations

        Returns:
            DataFrame with signal column:
                1 for long entry (reversal to uptrend)
                -1 for short entry (reversal to downtrend)
                0 for no signal

        Raises:
            ParabolicSARError: If required columns missing
        """
        try:
            if 'psar_reversal' not in df.columns or 'psar_trend' not in df.columns:
                raise ParabolicSARError("DataFrame must contain PSAR calculations")

            # Signal occurs on reversal
            signals = []
            for i in range(len(df)):
                if df['psar_reversal'][i]:
                    signals.append(df['psar_trend'][i])
                else:
                    signals.append(0)

            result = df.with_columns([
                pl.Series('psar_signal', signals)
            ])

            logger.debug("PSAR signals extracted", signals_count=sum(1 for s in signals if s != 0))

            return result

        except Exception as e:
            logger.error("Signal extraction failed", error=str(e))
            raise ParabolicSARError(f"Signal extraction failed: {e}")


async def calculate_parabolic_sar(
    df: pl.DataFrame,
    config: Dict[str, Any]
) -> pl.DataFrame:
    """Convenience function to calculate Parabolic SAR.

    Args:
        df: Input dataframe with OHLC data
        config: Configuration dictionary

    Returns:
        DataFrame with PSAR calculations

    Example:
        >>> config = {
        ...     'acceleration_start': '0.02',
        ...     'acceleration_increment': '0.02',
        ...     'acceleration_max': '0.20'
        ... }
        >>> result = await calculate_parabolic_sar(df, config)
    """
    psar = ParabolicSAR(config)
    return await psar.calculate(df)
