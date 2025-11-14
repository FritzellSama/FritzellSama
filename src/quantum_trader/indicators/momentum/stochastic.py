"""
Stochastic Oscillator Momentum Indicator.

This module implements the Stochastic Oscillator for identifying
overbought/oversold conditions and momentum in institutional trading systems.
"""

import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Optional
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class StochasticError(Exception):
    """Base exception for Stochastic indicator errors."""
    pass


class StochasticValidator:
    """Validates Stochastic configuration and input data."""

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate Stochastic configuration.

        Args:
            config: Configuration dictionary

        Raises:
            StochasticError: If configuration is invalid
        """
        required_keys = ['k_period', 'd_period', 'overbought', 'oversold']
        for key in required_keys:
            if key not in config:
                raise StochasticError(f"Missing required config key: {key}")

        try:
            k_period = int(config['k_period'])
            d_period = int(config['d_period'])
            overbought = Decimal(str(config['overbought']))
            oversold = Decimal(str(config['oversold']))

            if k_period < 1:
                raise StochasticError("k_period must be at least 1")
            if d_period < 1:
                raise StochasticError("d_period must be at least 1")
            if not (Decimal('0') <= oversold < overbought <= Decimal('100')):
                raise StochasticError("Must have: 0 <= oversold < overbought <= 100")

        except (ValueError, TypeError) as e:
            raise StochasticError(f"Invalid configuration values: {e}")

    @staticmethod
    def validate_dataframe(df: pl.DataFrame) -> None:
        """Validate input dataframe structure.

        Args:
            df: Input dataframe

        Raises:
            StochasticError: If dataframe is invalid
        """
        required_columns = ['high', 'low', 'close']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise StochasticError(f"Missing required columns: {missing}")

        if len(df) < 2:
            raise StochasticError("Dataframe must have at least 2 rows")


class Stochastic:
    """Stochastic Oscillator momentum indicator.

    The Stochastic Oscillator shows where the current close is relative
    to the high-low range over a specified period.

    Formula:
        %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
        %D = SMA of %K

    Attributes:
        config: Configuration dictionary
        k_period: Period for %K calculation (typically 14)
        d_period: Period for %D smoothing (typically 3)
        smooth_k: Optional smoothing for %K (typically 3 for slow stochastic)
        overbought: Overbought threshold (typically 80)
        oversold: Oversold threshold (typically 20)

    Example:
        >>> config = {
        ...     'k_period': 14,
        ...     'd_period': 3,
        ...     'overbought': '80',
        ...     'oversold': '20'
        ... }
        >>> stoch = Stochastic(config)
        >>> result = await stoch.calculate(df)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Stochastic indicator.

        Args:
            config: Configuration dictionary

        Raises:
            StochasticError: If configuration is invalid
        """
        StochasticValidator.validate_config(config)

        self.config = config
        self.k_period = int(config['k_period'])
        self.d_period = int(config['d_period'])
        self.smooth_k = int(config.get('smooth_k', 1))  # 1 = fast, 3 = slow
        self.overbought = Decimal(str(config['overbought']))
        self.oversold = Decimal(str(config['oversold']))

        logger.info(
            "Stochastic initialized",
            k_period=self.k_period,
            d_period=self.d_period,
            smooth_k=self.smooth_k,
            overbought=str(self.overbought),
            oversold=str(self.oversold)
        )

    async def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Stochastic Oscillator values.

        Args:
            df: Input dataframe with columns: high, low, close

        Returns:
            DataFrame with additional columns:
                - stoch_k: %K line (fast stochastic)
                - stoch_d: %D line (signal line)
                - stoch_overbought: True when in overbought territory
                - stoch_oversold: True when in oversold territory

        Raises:
            StochasticError: If calculation fails

        Example:
            >>> df = pl.DataFrame({
            ...     'high': [102.0, 103.0, 104.0],
            ...     'low': [99.0, 100.0, 101.0],
            ...     'close': [101.0, 102.0, 103.0]
            ... })
            >>> result = await stoch.calculate(df)
        """
        try:
            StochasticValidator.validate_dataframe(df)

            logger.debug("Calculating Stochastic", rows=len(df))

            highs = [Decimal(str(x)) for x in df['high'].to_list()]
            lows = [Decimal(str(x)) for x in df['low'].to_list()]
            closes = [Decimal(str(x)) for x in df['close'].to_list()]

            # Calculate raw %K
            raw_k_values = []

            for i in range(len(closes)):
                if i < self.k_period - 1:
                    raw_k_values.append(None)
                else:
                    # Get highest high and lowest low over k_period
                    period_highs = highs[i - self.k_period + 1:i + 1]
                    period_lows = lows[i - self.k_period + 1:i + 1]

                    highest_high = max(period_highs)
                    lowest_low = min(period_lows)

                    # Calculate %K
                    range_hl = highest_high - lowest_low
                    if range_hl > Decimal('0'):
                        k = ((closes[i] - lowest_low) / range_hl) * Decimal('100')
                        raw_k_values.append(k)
                    else:
                        raw_k_values.append(Decimal('50'))  # Middle value if no range

            # Apply smoothing to %K if configured (slow stochastic)
            if self.smooth_k > 1:
                k_values = self._smooth_values(raw_k_values, self.smooth_k)
            else:
                k_values = raw_k_values

            # Calculate %D (SMA of %K)
            d_values = self._smooth_values(k_values, self.d_period)

            # Convert to float and identify conditions
            k_floats = []
            d_floats = []
            overbought_flags = []
            oversold_flags = []

            for k, d in zip(k_values, d_values):
                if k is not None:
                    k_floats.append(float(k))
                    overbought_flags.append(k > self.overbought)
                    oversold_flags.append(k < self.oversold)
                else:
                    k_floats.append(None)
                    overbought_flags.append(False)
                    oversold_flags.append(False)

                if d is not None:
                    d_floats.append(float(d))
                else:
                    d_floats.append(None)

            # Add columns to dataframe
            result = df.with_columns([
                pl.Series('stoch_k', k_floats),
                pl.Series('stoch_d', d_floats),
                pl.Series('stoch_overbought', overbought_flags),
                pl.Series('stoch_oversold', oversold_flags)
            ])

            logger.info(
                "Stochastic calculated successfully",
                rows=len(result),
                overbought_count=sum(overbought_flags),
                oversold_count=sum(oversold_flags)
            )

            return result

        except StochasticError:
            raise
        except Exception as e:
            logger.error("Stochastic calculation failed", error=str(e))
            raise StochasticError(f"Calculation failed: {e}")

    def _smooth_values(self, values: list, period: int) -> list:
        """Apply simple moving average smoothing.

        Args:
            values: List of values to smooth
            period: Smoothing period

        Returns:
            List of smoothed values
        """
        smoothed = []

        for i in range(len(values)):
            if i < period - 1:
                smoothed.append(None)
            else:
                # Get valid values in window
                window = [v for v in values[i - period + 1:i + 1] if v is not None]

                if len(window) == period:
                    avg = sum(window) / Decimal(str(len(window)))
                    smoothed.append(avg)
                else:
                    smoothed.append(None)

        return smoothed

    def get_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate trading signals from Stochastic.

        Args:
            df: DataFrame with Stochastic calculations

        Returns:
            DataFrame with signal columns:
                - stoch_signal: 1 for buy, -1 for sell, 0 for neutral
                - stoch_crossover: True when %K crosses %D
                - stoch_extreme_reversal: True when reversing from extreme

        Raises:
            StochasticError: If required columns missing
        """
        try:
            required = ['stoch_k', 'stoch_d']
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise StochasticError(f"Missing required columns: {missing}")

            k_values = df['stoch_k'].to_list()
            d_values = df['stoch_d'].to_list()

            signals = []
            crossovers = []
            extreme_reversals = []

            for i in range(len(k_values)):
                if k_values[i] is None or d_values[i] is None:
                    signals.append(0)
                    crossovers.append(False)
                    extreme_reversals.append(False)
                else:
                    k = Decimal(str(k_values[i]))
                    d = Decimal(str(d_values[i]))

                    # Basic signal: oversold = buy, overbought = sell
                    if k < self.oversold and d < self.oversold:
                        signals.append(1)
                    elif k > self.overbought and d > self.overbought:
                        signals.append(-1)
                    else:
                        signals.append(0)

                    # Crossover detection
                    if i > 0 and k_values[i-1] is not None and d_values[i-1] is not None:
                        k_prev = Decimal(str(k_values[i-1]))
                        d_prev = Decimal(str(d_values[i-1]))

                        # Bullish crossover: %K crosses above %D
                        bullish_cross = (k_prev <= d_prev and k > d)
                        # Bearish crossover: %K crosses below %D
                        bearish_cross = (k_prev >= d_prev and k < d)

                        crossovers.append(bullish_cross or bearish_cross)

                        # Extreme reversal: crossover in extreme territory
                        if bullish_cross and k < self.oversold:
                            extreme_reversals.append(True)
                        elif bearish_cross and k > self.overbought:
                            extreme_reversals.append(True)
                        else:
                            extreme_reversals.append(False)
                    else:
                        crossovers.append(False)
                        extreme_reversals.append(False)

            result = df.with_columns([
                pl.Series('stoch_signal', signals),
                pl.Series('stoch_crossover', crossovers),
                pl.Series('stoch_extreme_reversal', extreme_reversals)
            ])

            logger.debug(
                "Stochastic signals generated",
                buy_signals=sum(1 for s in signals if s == 1),
                sell_signals=sum(1 for s in signals if s == -1),
                crossovers=sum(crossovers),
                extreme_reversals=sum(extreme_reversals)
            )

            return result

        except Exception as e:
            logger.error("Signal generation failed", error=str(e))
            raise StochasticError(f"Signal generation failed: {e}")

    def detect_divergence(
        self,
        df: pl.DataFrame,
        lookback: Optional[int] = None
    ) -> pl.DataFrame:
        """Detect bullish and bearish Stochastic divergences.

        Args:
            df: DataFrame with price and Stochastic data
            lookback: Number of periods to look back for divergence

        Returns:
            DataFrame with divergence columns:
                - stoch_bullish_divergence: Price lower low, Stochastic higher low
                - stoch_bearish_divergence: Price higher high, Stochastic lower high

        Raises:
            StochasticError: If required columns missing
        """
        try:
            required = ['close', 'stoch_k']
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise StochasticError(f"Missing required columns: {missing}")

            if lookback is None:
                lookback = self.k_period * 3

            closes = df['close'].to_list()
            k_values = df['stoch_k'].to_list()

            bullish_div = []
            bearish_div = []

            for i in range(len(closes)):
                if i < lookback or k_values[i] is None:
                    bullish_div.append(False)
                    bearish_div.append(False)
                    continue

                # Get window data
                window_closes = closes[i - lookback:i + 1]
                window_k = k_values[i - lookback:i + 1]

                # Find local lows and highs
                price_lows = []
                k_lows = []

                for j in range(1, len(window_closes) - 1):
                    if window_closes[j] is not None and window_k[j] is not None:
                        if window_closes[j] < window_closes[j-1] and window_closes[j] < window_closes[j+1]:
                            price_lows.append((j, window_closes[j], window_k[j]))

                # Bullish divergence
                bullish_detected = False
                if len(price_lows) >= 2:
                    if price_lows[-1][1] < price_lows[-2][1] and price_lows[-1][2] > price_lows[-2][2]:
                        bullish_detected = True

                bullish_div.append(bullish_detected)

                # Find local highs
                price_highs = []

                for j in range(1, len(window_closes) - 1):
                    if window_closes[j] is not None and window_k[j] is not None:
                        if window_closes[j] > window_closes[j-1] and window_closes[j] > window_closes[j+1]:
                            price_highs.append((j, window_closes[j], window_k[j]))

                # Bearish divergence
                bearish_detected = False
                if len(price_highs) >= 2:
                    if price_highs[-1][1] > price_highs[-2][1] and price_highs[-1][2] < price_highs[-2][2]:
                        bearish_detected = True

                bearish_div.append(bearish_detected)

            result = df.with_columns([
                pl.Series('stoch_bullish_divergence', bullish_div),
                pl.Series('stoch_bearish_divergence', bearish_div)
            ])

            logger.debug(
                "Stochastic divergences detected",
                bullish=sum(bullish_div),
                bearish=sum(bearish_div)
            )

            return result

        except Exception as e:
            logger.error("Divergence detection failed", error=str(e))
            raise StochasticError(f"Divergence detection failed: {e}")


async def calculate_stochastic(
    df: pl.DataFrame,
    config: Dict[str, Any]
) -> pl.DataFrame:
    """Convenience function to calculate Stochastic Oscillator.

    Args:
        df: Input dataframe with OHLC data
        config: Configuration dictionary

    Returns:
        DataFrame with Stochastic calculations

    Example:
        >>> config = {
        ...     'k_period': 14,
        ...     'd_period': 3,
        ...     'overbought': '80',
        ...     'oversold': '20'
        ... }
        >>> result = await calculate_stochastic(df, config)
    """
    stoch = Stochastic(config)
    return await stoch.calculate(df)
