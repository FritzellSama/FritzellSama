"""
Relative Strength Index (RSI) Momentum Indicator.

This module implements the RSI indicator for measuring momentum and
identifying overbought/oversold conditions in institutional trading systems.
"""

import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Optional
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class RSIError(Exception):
    """Base exception for RSI indicator errors."""
    pass


class RSIValidator:
    """Validates RSI configuration and input data."""

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate RSI configuration.

        Args:
            config: Configuration dictionary

        Raises:
            RSIError: If configuration is invalid
        """
        required_keys = ['period', 'overbought', 'oversold']
        for key in required_keys:
            if key not in config:
                raise RSIError(f"Missing required config key: {key}")

        try:
            period = int(config['period'])
            overbought = Decimal(str(config['overbought']))
            oversold = Decimal(str(config['oversold']))

            if period < 2:
                raise RSIError("period must be at least 2")
            if not (Decimal('0') <= oversold < overbought <= Decimal('100')):
                raise RSIError("Must have: 0 <= oversold < overbought <= 100")

        except (ValueError, TypeError) as e:
            raise RSIError(f"Invalid configuration values: {e}")

    @staticmethod
    def validate_dataframe(df: pl.DataFrame) -> None:
        """Validate input dataframe structure.

        Args:
            df: Input dataframe

        Raises:
            RSIError: If dataframe is invalid
        """
        if 'close' not in df.columns:
            raise RSIError("Missing required column: close")

        if len(df) < 2:
            raise RSIError("Dataframe must have at least 2 rows")


class RSI:
    """Relative Strength Index (RSI) momentum indicator.

    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions.

    Formula:
        RS = Average Gain / Average Loss
        RSI = 100 - (100 / (1 + RS))

    Attributes:
        config: Configuration dictionary
        period: Lookback period for RSI calculation (typically 14)
        overbought: Overbought threshold (typically 70)
        oversold: Oversold threshold (typically 30)

    Example:
        >>> config = {
        ...     'period': 14,
        ...     'overbought': '70',
        ...     'oversold': '30'
        ... }
        >>> rsi = RSI(config)
        >>> result = await rsi.calculate(df)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize RSI indicator.

        Args:
            config: Configuration dictionary

        Raises:
            RSIError: If configuration is invalid
        """
        RSIValidator.validate_config(config)

        self.config = config
        self.period = int(config['period'])
        self.overbought = Decimal(str(config['overbought']))
        self.oversold = Decimal(str(config['oversold']))

        logger.info(
            "RSI initialized",
            period=self.period,
            overbought=str(self.overbought),
            oversold=str(self.oversold)
        )

    async def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate RSI values.

        Args:
            df: Input dataframe with column: close

        Returns:
            DataFrame with additional columns:
                - rsi: RSI values (0-100)
                - rsi_overbought: True when RSI > overbought threshold
                - rsi_oversold: True when RSI < oversold threshold

        Raises:
            RSIError: If calculation fails

        Example:
            >>> df = pl.DataFrame({
            ...     'close': [100.0, 102.0, 101.0, 105.0, 103.0, 107.0]
            ... })
            >>> result = await rsi.calculate(df)
        """
        try:
            RSIValidator.validate_dataframe(df)

            logger.debug("Calculating RSI", rows=len(df), period=self.period)

            closes = [Decimal(str(x)) for x in df['close'].to_list()]

            # Calculate price changes
            changes = [Decimal('0')]  # First value has no change
            for i in range(1, len(closes)):
                change = closes[i] - closes[i-1]
                changes.append(change)

            # Separate gains and losses
            gains = []
            losses = []

            for change in changes:
                if change > Decimal('0'):
                    gains.append(change)
                    losses.append(Decimal('0'))
                else:
                    gains.append(Decimal('0'))
                    losses.append(abs(change))

            # Calculate RSI using Wilder's smoothing method
            rsi_values = []
            avg_gain = None
            avg_loss = None

            for i in range(len(gains)):
                if i < self.period:
                    # Not enough data yet
                    rsi_values.append(None)
                elif i == self.period:
                    # First RSI calculation - simple average
                    avg_gain = sum(gains[1:i+1]) / Decimal(str(self.period))
                    avg_loss = sum(losses[1:i+1]) / Decimal(str(self.period))

                    if avg_loss == Decimal('0'):
                        rsi = Decimal('100')
                    else:
                        rs = avg_gain / avg_loss
                        rsi = Decimal('100') - (Decimal('100') / (Decimal('1') + rs))

                    rsi_values.append(float(rsi))
                else:
                    # Subsequent RSI calculations - Wilder's smoothing
                    # Average Gain = [(Previous Average Gain) × 13 + Current Gain] / 14
                    avg_gain = ((avg_gain * Decimal(str(self.period - 1))) + gains[i]) / Decimal(str(self.period))
                    avg_loss = ((avg_loss * Decimal(str(self.period - 1))) + losses[i]) / Decimal(str(self.period))

                    if avg_loss == Decimal('0'):
                        rsi = Decimal('100')
                    else:
                        rs = avg_gain / avg_loss
                        rsi = Decimal('100') - (Decimal('100') / (Decimal('1') + rs))

                    rsi_values.append(float(rsi))

            # Identify overbought and oversold conditions
            overbought_flags = []
            oversold_flags = []

            for rsi_val in rsi_values:
                if rsi_val is None:
                    overbought_flags.append(False)
                    oversold_flags.append(False)
                else:
                    rsi_decimal = Decimal(str(rsi_val))
                    overbought_flags.append(rsi_decimal > self.overbought)
                    oversold_flags.append(rsi_decimal < self.oversold)

            # Add columns to dataframe
            result = df.with_columns([
                pl.Series('rsi', rsi_values),
                pl.Series('rsi_overbought', overbought_flags),
                pl.Series('rsi_oversold', oversold_flags)
            ])

            logger.info(
                "RSI calculated successfully",
                rows=len(result),
                overbought_count=sum(overbought_flags),
                oversold_count=sum(oversold_flags)
            )

            return result

        except RSIError:
            raise
        except Exception as e:
            logger.error("RSI calculation failed", error=str(e))
            raise RSIError(f"Calculation failed: {e}")

    def get_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate trading signals from RSI.

        Args:
            df: DataFrame with RSI calculations

        Returns:
            DataFrame with signal columns:
                - rsi_signal: 1 for buy, -1 for sell, 0 for neutral
                - rsi_extreme: True when in extreme territory

        Raises:
            RSIError: If required columns missing
        """
        try:
            if 'rsi' not in df.columns:
                raise RSIError("DataFrame must contain RSI values")

            rsi_values = df['rsi'].to_list()

            signals = []
            extremes = []

            for i in range(len(rsi_values)):
                if rsi_values[i] is None:
                    signals.append(0)
                    extremes.append(False)
                else:
                    rsi_decimal = Decimal(str(rsi_values[i]))

                    # Generate signal
                    if rsi_decimal < self.oversold:
                        # Oversold - potential buy signal
                        signals.append(1)
                        extremes.append(True)
                    elif rsi_decimal > self.overbought:
                        # Overbought - potential sell signal
                        signals.append(-1)
                        extremes.append(True)
                    else:
                        signals.append(0)
                        extremes.append(False)

            result = df.with_columns([
                pl.Series('rsi_signal', signals),
                pl.Series('rsi_extreme', extremes)
            ])

            logger.debug(
                "RSI signals generated",
                buy_signals=sum(1 for s in signals if s == 1),
                sell_signals=sum(1 for s in signals if s == -1)
            )

            return result

        except Exception as e:
            logger.error("Signal generation failed", error=str(e))
            raise RSIError(f"Signal generation failed: {e}")

    def detect_divergence(
        self,
        df: pl.DataFrame,
        lookback: Optional[int] = None
    ) -> pl.DataFrame:
        """Detect bullish and bearish RSI divergences.

        Args:
            df: DataFrame with price and RSI data
            lookback: Number of periods to look back for divergence

        Returns:
            DataFrame with divergence columns:
                - rsi_bullish_divergence: Price lower low, RSI higher low
                - rsi_bearish_divergence: Price higher high, RSI lower high

        Raises:
            RSIError: If required columns missing
        """
        try:
            required = ['close', 'rsi']
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise RSIError(f"Missing required columns: {missing}")

            if lookback is None:
                lookback = self.period * 3

            closes = df['close'].to_list()
            rsi_values = df['rsi'].to_list()

            bullish_div = []
            bearish_div = []

            for i in range(len(closes)):
                if i < lookback or rsi_values[i] is None:
                    bullish_div.append(False)
                    bearish_div.append(False)
                    continue

                # Get window data
                window_closes = closes[i - lookback:i + 1]
                window_rsi = rsi_values[i - lookback:i + 1]

                # Find local extremes
                current_close = window_closes[-1]
                current_rsi = window_rsi[-1]

                # Bullish divergence: price makes lower low, RSI makes higher low
                price_lows = []
                rsi_lows = []

                for j in range(1, len(window_closes) - 1):
                    if window_closes[j] is not None and window_rsi[j] is not None:
                        if window_closes[j] < window_closes[j-1] and window_closes[j] < window_closes[j+1]:
                            price_lows.append((j, window_closes[j], window_rsi[j]))

                bullish_detected = False
                if len(price_lows) >= 2:
                    # Compare last two lows
                    if price_lows[-1][1] < price_lows[-2][1] and price_lows[-1][2] > price_lows[-2][2]:
                        bullish_detected = True

                bullish_div.append(bullish_detected)

                # Bearish divergence: price makes higher high, RSI makes lower high
                price_highs = []
                rsi_highs = []

                for j in range(1, len(window_closes) - 1):
                    if window_closes[j] is not None and window_rsi[j] is not None:
                        if window_closes[j] > window_closes[j-1] and window_closes[j] > window_closes[j+1]:
                            price_highs.append((j, window_closes[j], window_rsi[j]))

                bearish_detected = False
                if len(price_highs) >= 2:
                    # Compare last two highs
                    if price_highs[-1][1] > price_highs[-2][1] and price_highs[-1][2] < price_highs[-2][2]:
                        bearish_detected = True

                bearish_div.append(bearish_detected)

            result = df.with_columns([
                pl.Series('rsi_bullish_divergence', bullish_div),
                pl.Series('rsi_bearish_divergence', bearish_div)
            ])

            logger.debug(
                "RSI divergences detected",
                bullish=sum(bullish_div),
                bearish=sum(bearish_div)
            )

            return result

        except Exception as e:
            logger.error("Divergence detection failed", error=str(e))
            raise RSIError(f"Divergence detection failed: {e}")


async def calculate_rsi(
    df: pl.DataFrame,
    config: Dict[str, Any]
) -> pl.DataFrame:
    """Convenience function to calculate RSI.

    Args:
        df: Input dataframe with price data
        config: Configuration dictionary

    Returns:
        DataFrame with RSI calculations

    Example:
        >>> config = {
        ...     'period': 14,
        ...     'overbought': '70',
        ...     'oversold': '30'
        ... }
        >>> result = await calculate_rsi(df, config)
    """
    rsi = RSI(config)
    return await rsi.calculate(df)
