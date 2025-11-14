"""
True Strength Index (TSI) Momentum Indicator.

This module implements the TSI indicator for measuring momentum with
double smoothing in institutional trading systems.
"""

import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Optional
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class TSIError(Exception):
    """Base exception for TSI indicator errors."""
    pass


class TSIValidator:
    """Validates TSI configuration and input data."""

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate TSI configuration.

        Args:
            config: Configuration dictionary

        Raises:
            TSIError: If configuration is invalid
        """
        required_keys = ['long_period', 'short_period', 'signal_period']
        for key in required_keys:
            if key not in config:
                raise TSIError(f"Missing required config key: {key}")

        try:
            long_period = int(config['long_period'])
            short_period = int(config['short_period'])
            signal_period = int(config['signal_period'])

            if long_period < 1:
                raise TSIError("long_period must be at least 1")
            if short_period < 1:
                raise TSIError("short_period must be at least 1")
            if signal_period < 1:
                raise TSIError("signal_period must be at least 1")
            if short_period > long_period:
                raise TSIError("short_period should not be greater than long_period")

        except (ValueError, TypeError) as e:
            raise TSIError(f"Invalid configuration values: {e}")

    @staticmethod
    def validate_dataframe(df: pl.DataFrame) -> None:
        """Validate input dataframe structure.

        Args:
            df: Input dataframe

        Raises:
            TSIError: If dataframe is invalid
        """
        if 'close' not in df.columns:
            raise TSIError("Missing required column: close")

        if len(df) < 2:
            raise TSIError("Dataframe must have at least 2 rows")


class TSI:
    """True Strength Index (TSI) momentum indicator.

    TSI uses double exponential smoothing of momentum to generate
    trading signals with reduced noise.

    Formula:
        Momentum = Close - Close[1]
        Double Smoothed Momentum = EMA(EMA(Momentum, long), short)
        Double Smoothed Absolute Momentum = EMA(EMA(|Momentum|, long), short)
        TSI = 100 * (Double Smoothed Momentum / Double Smoothed Absolute Momentum)

    Attributes:
        config: Configuration dictionary
        long_period: First (longer) EMA period (typically 25)
        short_period: Second (shorter) EMA period (typically 13)
        signal_period: Signal line EMA period (typically 7)

    Example:
        >>> config = {
        ...     'long_period': 25,
        ...     'short_period': 13,
        ...     'signal_period': 7
        ... }
        >>> tsi = TSI(config)
        >>> result = await tsi.calculate(df)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize TSI indicator.

        Args:
            config: Configuration dictionary

        Raises:
            TSIError: If configuration is invalid
        """
        TSIValidator.validate_config(config)

        self.config = config
        self.long_period = int(config['long_period'])
        self.short_period = int(config['short_period'])
        self.signal_period = int(config['signal_period'])

        logger.info(
            "TSI initialized",
            long_period=self.long_period,
            short_period=self.short_period,
            signal_period=self.signal_period
        )

    async def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate TSI values.

        Args:
            df: Input dataframe with column: close

        Returns:
            DataFrame with additional columns:
                - tsi: TSI values (-100 to 100)
                - tsi_signal: Signal line (EMA of TSI)
                - tsi_histogram: TSI - Signal line

        Raises:
            TSIError: If calculation fails

        Example:
            >>> df = pl.DataFrame({
            ...     'close': [100.0, 102.0, 101.0, 105.0, 103.0]
            ... })
            >>> result = await tsi.calculate(df)
        """
        try:
            TSIValidator.validate_dataframe(df)

            logger.debug("Calculating TSI", rows=len(df))

            closes = [Decimal(str(x)) for x in df['close'].to_list()]

            # Calculate price momentum (change)
            momentum = [Decimal('0')]  # First value has no change
            for i in range(1, len(closes)):
                change = closes[i] - closes[i-1]
                momentum.append(change)

            # Calculate absolute momentum
            abs_momentum = [abs(m) for m in momentum]

            # First smoothing (long period EMA)
            momentum_ema1 = self._calculate_ema(momentum, self.long_period)
            abs_momentum_ema1 = self._calculate_ema(abs_momentum, self.long_period)

            # Second smoothing (short period EMA)
            momentum_ema2 = self._calculate_ema(momentum_ema1, self.short_period)
            abs_momentum_ema2 = self._calculate_ema(abs_momentum_ema1, self.short_period)

            # Calculate TSI
            tsi_values = []

            for i in range(len(momentum_ema2)):
                if momentum_ema2[i] is None or abs_momentum_ema2[i] is None:
                    tsi_values.append(None)
                else:
                    if abs_momentum_ema2[i] != Decimal('0'):
                        tsi = (momentum_ema2[i] / abs_momentum_ema2[i]) * Decimal('100')
                        tsi_values.append(tsi)
                    else:
                        tsi_values.append(Decimal('0'))

            # Calculate signal line (EMA of TSI)
            signal_values = self._calculate_ema(tsi_values, self.signal_period)

            # Calculate histogram
            histogram_values = []
            for tsi, signal in zip(tsi_values, signal_values):
                if tsi is not None and signal is not None:
                    histogram_values.append(tsi - signal)
                else:
                    histogram_values.append(None)

            # Convert to float
            tsi_floats = [float(x) if x is not None else None for x in tsi_values]
            signal_floats = [float(x) if x is not None else None for x in signal_values]
            histogram_floats = [float(x) if x is not None else None for x in histogram_values]

            # Add columns to dataframe
            result = df.with_columns([
                pl.Series('tsi', tsi_floats),
                pl.Series('tsi_signal', signal_floats),
                pl.Series('tsi_histogram', histogram_floats)
            ])

            logger.info("TSI calculated successfully", rows=len(result))

            return result

        except TSIError:
            raise
        except Exception as e:
            logger.error("TSI calculation failed", error=str(e))
            raise TSIError(f"Calculation failed: {e}")

    def _calculate_ema(self, values: list, period: int) -> list:
        """Calculate Exponential Moving Average.

        Args:
            values: List of values to smooth
            period: EMA period

        Returns:
            List of EMA values
        """
        multiplier = Decimal('2') / Decimal(str(period + 1))
        ema_values = []
        ema = None

        for val in values:
            if val is None:
                ema_values.append(None)
                continue

            if ema is None:
                # First value
                ema = val
            else:
                # EMA = (Value - EMA_prev) * multiplier + EMA_prev
                ema = (val - ema) * multiplier + ema

            ema_values.append(ema)

        return ema_values

    def get_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate trading signals from TSI.

        Args:
            df: DataFrame with TSI calculations

        Returns:
            DataFrame with signal columns:
                - tsi_trade_signal: 1 for buy, -1 for sell, 0 for neutral
                - tsi_crossover: True when TSI crosses signal line
                - tsi_zero_cross: True when TSI crosses zero line

        Raises:
            TSIError: If required columns missing
        """
        try:
            required = ['tsi', 'tsi_signal']
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise TSIError(f"Missing required columns: {missing}")

            tsi_values = df['tsi'].to_list()
            signal_values = df['tsi_signal'].to_list()

            trade_signals = []
            crossovers = []
            zero_crosses = []

            for i in range(len(tsi_values)):
                if tsi_values[i] is None or signal_values[i] is None:
                    trade_signals.append(0)
                    crossovers.append(False)
                    zero_crosses.append(False)
                else:
                    tsi = Decimal(str(tsi_values[i]))
                    signal = Decimal(str(signal_values[i]))

                    # Detect crossovers
                    if i > 0 and tsi_values[i-1] is not None and signal_values[i-1] is not None:
                        tsi_prev = Decimal(str(tsi_values[i-1]))
                        signal_prev = Decimal(str(signal_values[i-1]))

                        # Bullish crossover: TSI crosses above signal
                        bullish_cross = (tsi_prev <= signal_prev and tsi > signal)
                        # Bearish crossover: TSI crosses below signal
                        bearish_cross = (tsi_prev >= signal_prev and tsi < signal)

                        if bullish_cross:
                            trade_signals.append(1)
                            crossovers.append(True)
                        elif bearish_cross:
                            trade_signals.append(-1)
                            crossovers.append(True)
                        else:
                            trade_signals.append(0)
                            crossovers.append(False)

                        # Zero line cross
                        zero_cross_up = (tsi_prev <= Decimal('0') and tsi > Decimal('0'))
                        zero_cross_down = (tsi_prev >= Decimal('0') and tsi < Decimal('0'))
                        zero_crosses.append(zero_cross_up or zero_cross_down)
                    else:
                        trade_signals.append(0)
                        crossovers.append(False)
                        zero_crosses.append(False)

            result = df.with_columns([
                pl.Series('tsi_trade_signal', trade_signals),
                pl.Series('tsi_crossover', crossovers),
                pl.Series('tsi_zero_cross', zero_crosses)
            ])

            logger.debug(
                "TSI signals generated",
                buy_signals=sum(1 for s in trade_signals if s == 1),
                sell_signals=sum(1 for s in trade_signals if s == -1),
                crossovers=sum(crossovers),
                zero_crosses=sum(zero_crosses)
            )

            return result

        except Exception as e:
            logger.error("Signal generation failed", error=str(e))
            raise TSIError(f"Signal generation failed: {e}")

    def detect_divergence(
        self,
        df: pl.DataFrame,
        lookback: Optional[int] = None
    ) -> pl.DataFrame:
        """Detect bullish and bearish TSI divergences.

        Args:
            df: DataFrame with price and TSI data
            lookback: Number of periods to look back for divergence

        Returns:
            DataFrame with divergence columns:
                - tsi_bullish_divergence: Price lower low, TSI higher low
                - tsi_bearish_divergence: Price higher high, TSI lower high

        Raises:
            TSIError: If required columns missing
        """
        try:
            required = ['close', 'tsi']
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise TSIError(f"Missing required columns: {missing}")

            if lookback is None:
                lookback = self.long_period * 2

            closes = df['close'].to_list()
            tsi_values = df['tsi'].to_list()

            bullish_div = []
            bearish_div = []

            for i in range(len(closes)):
                if i < lookback or tsi_values[i] is None:
                    bullish_div.append(False)
                    bearish_div.append(False)
                    continue

                # Get window data
                window_closes = closes[i - lookback:i + 1]
                window_tsi = tsi_values[i - lookback:i + 1]

                # Find local lows
                price_lows = []
                for j in range(1, len(window_closes) - 1):
                    if window_closes[j] is not None and window_tsi[j] is not None:
                        if window_closes[j] < window_closes[j-1] and window_closes[j] < window_closes[j+1]:
                            price_lows.append((j, window_closes[j], window_tsi[j]))

                # Bullish divergence
                bullish_detected = False
                if len(price_lows) >= 2:
                    if price_lows[-1][1] < price_lows[-2][1] and price_lows[-1][2] > price_lows[-2][2]:
                        bullish_detected = True

                bullish_div.append(bullish_detected)

                # Find local highs
                price_highs = []
                for j in range(1, len(window_closes) - 1):
                    if window_closes[j] is not None and window_tsi[j] is not None:
                        if window_closes[j] > window_closes[j-1] and window_closes[j] > window_closes[j+1]:
                            price_highs.append((j, window_closes[j], window_tsi[j]))

                # Bearish divergence
                bearish_detected = False
                if len(price_highs) >= 2:
                    if price_highs[-1][1] > price_highs[-2][1] and price_highs[-1][2] < price_highs[-2][2]:
                        bearish_detected = True

                bearish_div.append(bearish_detected)

            result = df.with_columns([
                pl.Series('tsi_bullish_divergence', bullish_div),
                pl.Series('tsi_bearish_divergence', bearish_div)
            ])

            logger.debug(
                "TSI divergences detected",
                bullish=sum(bullish_div),
                bearish=sum(bearish_div)
            )

            return result

        except Exception as e:
            logger.error("Divergence detection failed", error=str(e))
            raise TSIError(f"Divergence detection failed: {e}")


async def calculate_tsi(
    df: pl.DataFrame,
    config: Dict[str, Any]
) -> pl.DataFrame:
    """Convenience function to calculate TSI.

    Args:
        df: Input dataframe with price data
        config: Configuration dictionary

    Returns:
        DataFrame with TSI calculations

    Example:
        >>> config = {
        ...     'long_period': 25,
        ...     'short_period': 13,
        ...     'signal_period': 7
        ... }
        >>> result = await calculate_tsi(df, config)
    """
    tsi = TSI(config)
    return await tsi.calculate(df)
