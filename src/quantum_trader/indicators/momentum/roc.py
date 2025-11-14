"""
Rate of Change (ROC) Momentum Indicator.

This module implements the ROC indicator for measuring price momentum
in institutional trading systems.
"""

import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Optional
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class ROCError(Exception):
    """Base exception for ROC indicator errors."""
    pass


class ROCValidator:
    """Validates ROC configuration and input data."""

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate ROC configuration.

        Args:
            config: Configuration dictionary

        Raises:
            ROCError: If configuration is invalid
        """
        if 'period' not in config:
            raise ROCError("Missing required config key: period")

        try:
            period = int(config['period'])
            if period < 1:
                raise ROCError("period must be at least 1")

        except (ValueError, TypeError) as e:
            raise ROCError(f"Invalid configuration values: {e}")

    @staticmethod
    def validate_dataframe(df: pl.DataFrame) -> None:
        """Validate input dataframe structure.

        Args:
            df: Input dataframe

        Raises:
            ROCError: If dataframe is invalid
        """
        if 'close' not in df.columns:
            raise ROCError("Missing required column: close")

        if len(df) < 2:
            raise ROCError("Dataframe must have at least 2 rows")


class ROC:
    """Rate of Change (ROC) momentum indicator.

    ROC measures the percentage change in price over a specified period.
    It's useful for identifying overbought/oversold conditions and divergences.

    Formula:
        ROC = ((Close - Close_n) / Close_n) * 100

    Where n is the lookback period.

    Attributes:
        config: Configuration dictionary
        period: Lookback period for ROC calculation

    Example:
        >>> config = {'period': 12}
        >>> roc = ROC(config)
        >>> result = await roc.calculate(df)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize ROC indicator.

        Args:
            config: Configuration dictionary with 'period' key

        Raises:
            ROCError: If configuration is invalid
        """
        ROCValidator.validate_config(config)

        self.config = config
        self.period = int(config['period'])

        logger.info("ROC initialized", period=self.period)

    async def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Rate of Change.

        Args:
            df: Input dataframe with column: close

        Returns:
            DataFrame with additional columns:
                - roc: Rate of change percentage
                - roc_ema: Exponential moving average of ROC (if smoothing enabled)

        Raises:
            ROCError: If calculation fails

        Example:
            >>> df = pl.DataFrame({
            ...     'close': [100.0, 102.0, 101.0, 105.0, 103.0]
            ... })
            >>> result = await roc.calculate(df)
        """
        try:
            ROCValidator.validate_dataframe(df)

            logger.debug("Calculating ROC", rows=len(df), period=self.period)

            closes = [Decimal(str(x)) for x in df['close'].to_list()]

            # Calculate ROC
            roc_values = []

            for i in range(len(closes)):
                if i < self.period:
                    # Not enough data yet
                    roc_values.append(None)
                else:
                    close_current = closes[i]
                    close_previous = closes[i - self.period]

                    if close_previous != Decimal('0'):
                        # ROC = ((Close - Close_n) / Close_n) * 100
                        roc = ((close_current - close_previous) / close_previous) * Decimal('100')
                        roc_values.append(float(roc))
                    else:
                        roc_values.append(None)

            # Add ROC column
            result = df.with_columns([
                pl.Series('roc', roc_values)
            ])

            # Calculate smoothed ROC if configured
            if 'smoothing_period' in self.config:
                smoothing_period = int(self.config['smoothing_period'])
                result = await self._calculate_smoothed_roc(result, smoothing_period)

            logger.info("ROC calculated successfully", rows=len(result))

            return result

        except ROCError:
            raise
        except Exception as e:
            logger.error("ROC calculation failed", error=str(e))
            raise ROCError(f"Calculation failed: {e}")

    async def _calculate_smoothed_roc(
        self,
        df: pl.DataFrame,
        smoothing_period: int
    ) -> pl.DataFrame:
        """Calculate exponential moving average of ROC.

        Args:
            df: DataFrame with ROC values
            smoothing_period: EMA period

        Returns:
            DataFrame with roc_ema column
        """
        if 'roc' not in df.columns:
            raise ROCError("DataFrame must contain ROC values")

        roc_values = df['roc'].to_list()

        # Calculate EMA multiplier
        multiplier = Decimal('2') / Decimal(str(smoothing_period + 1))

        ema_values = []
        ema = None

        for roc_val in roc_values:
            if roc_val is None:
                ema_values.append(None)
            else:
                roc_decimal = Decimal(str(roc_val))

                if ema is None:
                    # First value
                    ema = roc_decimal
                else:
                    # EMA = (Close - EMA_prev) * multiplier + EMA_prev
                    ema = (roc_decimal - ema) * multiplier + ema

                ema_values.append(float(ema))

        return df.with_columns([
            pl.Series('roc_ema', ema_values)
        ])

    def get_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate trading signals from ROC.

        Args:
            df: DataFrame with ROC calculations

        Returns:
            DataFrame with signal columns:
                - roc_signal: 1 for bullish, -1 for bearish, 0 for neutral
                - roc_crossover: True when ROC crosses zero line

        Raises:
            ROCError: If required columns missing
        """
        try:
            if 'roc' not in df.columns:
                raise ROCError("DataFrame must contain ROC values")

            roc_values = df['roc'].to_list()

            signals = []
            crossovers = []

            for i in range(len(roc_values)):
                if roc_values[i] is None:
                    signals.append(0)
                    crossovers.append(False)
                else:
                    # Generate signal based on ROC value
                    if roc_values[i] > 0:
                        signals.append(1)
                    elif roc_values[i] < 0:
                        signals.append(-1)
                    else:
                        signals.append(0)

                    # Detect zero-line crossover
                    if i > 0 and roc_values[i-1] is not None:
                        if (roc_values[i-1] < 0 and roc_values[i] > 0) or \
                           (roc_values[i-1] > 0 and roc_values[i] < 0):
                            crossovers.append(True)
                        else:
                            crossovers.append(False)
                    else:
                        crossovers.append(False)

            result = df.with_columns([
                pl.Series('roc_signal', signals),
                pl.Series('roc_crossover', crossovers)
            ])

            logger.debug(
                "ROC signals generated",
                crossovers=sum(crossovers)
            )

            return result

        except Exception as e:
            logger.error("Signal generation failed", error=str(e))
            raise ROCError(f"Signal generation failed: {e}")

    def detect_divergence(
        self,
        df: pl.DataFrame,
        lookback: Optional[int] = None
    ) -> pl.DataFrame:
        """Detect bullish and bearish divergences.

        Args:
            df: DataFrame with price and ROC data
            lookback: Number of periods to look back for divergence

        Returns:
            DataFrame with divergence columns:
                - bullish_divergence: True when detected
                - bearish_divergence: True when detected

        Raises:
            ROCError: If required columns missing
        """
        try:
            required = ['close', 'roc']
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise ROCError(f"Missing required columns: {missing}")

            if lookback is None:
                lookback = self.period * 2

            closes = df['close'].to_list()
            roc_values = df['roc'].to_list()

            bullish_div = []
            bearish_div = []

            for i in range(len(closes)):
                if i < lookback or roc_values[i] is None:
                    bullish_div.append(False)
                    bearish_div.append(False)
                    continue

                # Look for divergences in the lookback window
                window_closes = closes[i - lookback:i + 1]
                window_roc = roc_values[i - lookback:i + 1]

                # Bullish divergence: price makes lower low, ROC makes higher low
                price_min_idx = window_closes.index(min(window_closes))
                roc_valid = [r for r in window_roc if r is not None]

                if len(roc_valid) > 0:
                    roc_min_val = min(roc_valid)
                    roc_min_idx = window_roc.index(roc_min_val)

                    # Simple divergence detection
                    if price_min_idx < len(window_closes) - 1 and roc_min_idx < price_min_idx:
                        bullish_div.append(True)
                    else:
                        bullish_div.append(False)

                    # Bearish divergence: price makes higher high, ROC makes lower high
                    price_max_idx = window_closes.index(max(window_closes))
                    roc_max_val = max(roc_valid)
                    roc_max_idx = window_roc.index(roc_max_val)

                    if price_max_idx < len(window_closes) - 1 and roc_max_idx < price_max_idx:
                        bearish_div.append(True)
                    else:
                        bearish_div.append(False)
                else:
                    bullish_div.append(False)
                    bearish_div.append(False)

            result = df.with_columns([
                pl.Series('bullish_divergence', bullish_div),
                pl.Series('bearish_divergence', bearish_div)
            ])

            logger.debug(
                "Divergences detected",
                bullish=sum(bullish_div),
                bearish=sum(bearish_div)
            )

            return result

        except Exception as e:
            logger.error("Divergence detection failed", error=str(e))
            raise ROCError(f"Divergence detection failed: {e}")


async def calculate_roc(
    df: pl.DataFrame,
    config: Dict[str, Any]
) -> pl.DataFrame:
    """Convenience function to calculate ROC.

    Args:
        df: Input dataframe with price data
        config: Configuration dictionary with 'period' key

    Returns:
        DataFrame with ROC calculations

    Example:
        >>> config = {'period': 12}
        >>> result = await calculate_roc(df, config)
    """
    roc = ROC(config)
    return await roc.calculate(df)
