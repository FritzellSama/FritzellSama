"""
Vortex Indicator (VI) Trend Indicator.

This module implements the Vortex Indicator for identifying trend direction
and reversals in institutional trading systems.
"""

import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Optional
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class VortexError(Exception):
    """Base exception for Vortex indicator errors."""
    pass


class VortexValidator:
    """Validates Vortex configuration and input data."""

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate Vortex configuration.

        Args:
            config: Configuration dictionary

        Raises:
            VortexError: If configuration is invalid
        """
        if 'period' not in config:
            raise VortexError("Missing required config key: period")

        try:
            period = int(config['period'])
            if period < 1:
                raise VortexError("period must be at least 1")

        except (ValueError, TypeError) as e:
            raise VortexError(f"Invalid configuration values: {e}")

    @staticmethod
    def validate_dataframe(df: pl.DataFrame) -> None:
        """Validate input dataframe structure.

        Args:
            df: Input dataframe

        Raises:
            VortexError: If dataframe is invalid
        """
        required_columns = ['high', 'low', 'close']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise VortexError(f"Missing required columns: {missing}")

        if len(df) < 2:
            raise VortexError("Dataframe must have at least 2 rows")


class Vortex:
    """Vortex Indicator (VI) implementation.

    The Vortex Indicator consists of two oscillators (VI+ and VI-) that
    capture positive and negative trend movement.

    Formula:
        Vortex Movement (VM):
            VM+ = |High[i] - Low[i-1]|
            VM- = |Low[i] - High[i-1]|

        True Range (TR):
            TR = max(High - Low, |High - Close[i-1]|, |Low - Close[i-1]|)

        Vortex Indicator:
            VI+ = Sum(VM+, period) / Sum(TR, period)
            VI- = Sum(VM-, period) / Sum(TR, period)

    Attributes:
        config: Configuration dictionary
        period: Lookback period for calculation (typically 14)

    Example:
        >>> config = {'period': 14}
        >>> vortex = Vortex(config)
        >>> result = await vortex.calculate(df)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Vortex indicator.

        Args:
            config: Configuration dictionary

        Raises:
            VortexError: If configuration is invalid
        """
        VortexValidator.validate_config(config)

        self.config = config
        self.period = int(config['period'])

        logger.info("Vortex initialized", period=self.period)

    async def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Vortex Indicator values.

        Args:
            df: Input dataframe with columns: high, low, close

        Returns:
            DataFrame with additional columns:
                - vi_plus: Positive vortex indicator
                - vi_minus: Negative vortex indicator
                - vi_diff: VI+ - VI- (trend strength)
                - vi_trend: 1 for uptrend (VI+ > VI-), -1 for downtrend

        Raises:
            VortexError: If calculation fails

        Example:
            >>> df = pl.DataFrame({
            ...     'high': [102.0, 103.0, 104.0],
            ...     'low': [99.0, 100.0, 101.0],
            ...     'close': [101.0, 102.0, 103.0]
            ... })
            >>> result = await vortex.calculate(df)
        """
        try:
            VortexValidator.validate_dataframe(df)

            logger.debug("Calculating Vortex Indicator", rows=len(df))

            highs = [Decimal(str(x)) for x in df['high'].to_list()]
            lows = [Decimal(str(x)) for x in df['low'].to_list()]
            closes = [Decimal(str(x)) for x in df['close'].to_list()]

            # Calculate Vortex Movement (VM+ and VM-)
            vm_plus = []
            vm_minus = []

            for i in range(len(highs)):
                if i == 0:
                    # First bar has no previous bar
                    vm_plus.append(Decimal('0'))
                    vm_minus.append(Decimal('0'))
                else:
                    # VM+ = |High[i] - Low[i-1]|
                    vmp = abs(highs[i] - lows[i-1])
                    # VM- = |Low[i] - High[i-1]|
                    vmm = abs(lows[i] - highs[i-1])

                    vm_plus.append(vmp)
                    vm_minus.append(vmm)

            # Calculate True Range (TR)
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

            # Calculate VI+ and VI-
            vi_plus_values = []
            vi_minus_values = []
            vi_diff_values = []
            vi_trend_values = []

            for i in range(len(highs)):
                if i < self.period:
                    # Not enough data yet
                    vi_plus_values.append(None)
                    vi_minus_values.append(None)
                    vi_diff_values.append(None)
                    vi_trend_values.append(0)
                else:
                    # Sum VM+ and VM- over period
                    sum_vmp = sum(vm_plus[i - self.period + 1:i + 1])
                    sum_vmm = sum(vm_minus[i - self.period + 1:i + 1])

                    # Sum TR over period
                    sum_tr = sum(tr_values[i - self.period + 1:i + 1])

                    if sum_tr > Decimal('0'):
                        # VI+ = Sum(VM+) / Sum(TR)
                        vi_plus = sum_vmp / sum_tr
                        # VI- = Sum(VM-) / Sum(TR)
                        vi_minus = sum_vmm / sum_tr

                        vi_plus_values.append(float(vi_plus))
                        vi_minus_values.append(float(vi_minus))

                        # Calculate difference
                        vi_diff = vi_plus - vi_minus
                        vi_diff_values.append(float(vi_diff))

                        # Determine trend
                        if vi_plus > vi_minus:
                            vi_trend_values.append(1)
                        elif vi_minus > vi_plus:
                            vi_trend_values.append(-1)
                        else:
                            vi_trend_values.append(0)
                    else:
                        vi_plus_values.append(None)
                        vi_minus_values.append(None)
                        vi_diff_values.append(None)
                        vi_trend_values.append(0)

            # Add columns to dataframe
            result = df.with_columns([
                pl.Series('vi_plus', vi_plus_values),
                pl.Series('vi_minus', vi_minus_values),
                pl.Series('vi_diff', vi_diff_values),
                pl.Series('vi_trend', vi_trend_values)
            ])

            logger.info("Vortex Indicator calculated successfully", rows=len(result))

            return result

        except VortexError:
            raise
        except Exception as e:
            logger.error("Vortex Indicator calculation failed", error=str(e))
            raise VortexError(f"Calculation failed: {e}")

    def get_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate trading signals from Vortex Indicator.

        Args:
            df: DataFrame with Vortex calculations

        Returns:
            DataFrame with signal columns:
                - vi_signal: 1 for buy, -1 for sell, 0 for neutral
                - vi_crossover: True when VI+ and VI- cross
                - vi_trend_strength: Absolute difference between VI+ and VI-

        Raises:
            VortexError: If required columns missing
        """
        try:
            required = ['vi_plus', 'vi_minus', 'vi_trend']
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise VortexError(f"Missing required columns: {missing}")

            vi_plus_values = df['vi_plus'].to_list()
            vi_minus_values = df['vi_minus'].to_list()
            vi_trend_values = df['vi_trend'].to_list()

            signals = []
            crossovers = []
            trend_strength = []

            for i in range(len(vi_plus_values)):
                if vi_plus_values[i] is None or vi_minus_values[i] is None:
                    signals.append(0)
                    crossovers.append(False)
                    trend_strength.append(None)
                else:
                    vi_plus = Decimal(str(vi_plus_values[i]))
                    vi_minus = Decimal(str(vi_minus_values[i]))

                    # Trend strength (absolute difference)
                    strength = abs(vi_plus - vi_minus)
                    trend_strength.append(float(strength))

                    # Detect crossovers
                    if i > 0 and vi_plus_values[i-1] is not None and vi_minus_values[i-1] is not None:
                        vi_plus_prev = Decimal(str(vi_plus_values[i-1]))
                        vi_minus_prev = Decimal(str(vi_minus_values[i-1]))

                        # Bullish crossover: VI+ crosses above VI-
                        bullish_cross = (vi_plus_prev <= vi_minus_prev and vi_plus > vi_minus)
                        # Bearish crossover: VI- crosses above VI+
                        bearish_cross = (vi_minus_prev <= vi_plus_prev and vi_minus > vi_plus)

                        if bullish_cross:
                            signals.append(1)
                            crossovers.append(True)
                        elif bearish_cross:
                            signals.append(-1)
                            crossovers.append(True)
                        else:
                            signals.append(0)
                            crossovers.append(False)
                    else:
                        signals.append(0)
                        crossovers.append(False)

            result = df.with_columns([
                pl.Series('vi_signal', signals),
                pl.Series('vi_crossover', crossovers),
                pl.Series('vi_trend_strength', trend_strength)
            ])

            logger.debug(
                "Vortex signals generated",
                buy_signals=sum(1 for s in signals if s == 1),
                sell_signals=sum(1 for s in signals if s == -1),
                crossovers=sum(crossovers)
            )

            return result

        except Exception as e:
            logger.error("Signal generation failed", error=str(e))
            raise VortexError(f"Signal generation failed: {e}")

    def detect_trend_reversal(
        self,
        df: pl.DataFrame,
        threshold: Optional[Decimal] = None
    ) -> pl.DataFrame:
        """Detect potential trend reversals.

        Args:
            df: DataFrame with Vortex calculations
            threshold: Minimum VI difference to confirm reversal

        Returns:
            DataFrame with reversal detection columns:
                - vi_reversal_bullish: Strong bullish reversal signal
                - vi_reversal_bearish: Strong bearish reversal signal

        Raises:
            VortexError: If required columns missing
        """
        try:
            required = ['vi_plus', 'vi_minus', 'vi_trend']
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise VortexError(f"Missing required columns: {missing}")

            if threshold is None:
                threshold = Decimal('0.05')  # 5% difference

            vi_plus_values = df['vi_plus'].to_list()
            vi_minus_values = df['vi_minus'].to_list()
            vi_trend_values = df['vi_trend'].to_list()

            bullish_reversals = []
            bearish_reversals = []

            for i in range(len(vi_plus_values)):
                if i < 2 or vi_plus_values[i] is None or vi_minus_values[i] is None:
                    bullish_reversals.append(False)
                    bearish_reversals.append(False)
                else:
                    vi_plus = Decimal(str(vi_plus_values[i]))
                    vi_minus = Decimal(str(vi_minus_values[i]))
                    prev_trend = vi_trend_values[i-1]
                    current_trend = vi_trend_values[i]

                    # Bullish reversal: was bearish, now bullish with strong VI+
                    if prev_trend == -1 and current_trend == 1:
                        if (vi_plus - vi_minus) > threshold:
                            bullish_reversals.append(True)
                        else:
                            bullish_reversals.append(False)
                    else:
                        bullish_reversals.append(False)

                    # Bearish reversal: was bullish, now bearish with strong VI-
                    if prev_trend == 1 and current_trend == -1:
                        if (vi_minus - vi_plus) > threshold:
                            bearish_reversals.append(True)
                        else:
                            bearish_reversals.append(False)
                    else:
                        bearish_reversals.append(False)

            result = df.with_columns([
                pl.Series('vi_reversal_bullish', bullish_reversals),
                pl.Series('vi_reversal_bearish', bearish_reversals)
            ])

            logger.debug(
                "Vortex reversals detected",
                bullish=sum(bullish_reversals),
                bearish=sum(bearish_reversals)
            )

            return result

        except Exception as e:
            logger.error("Reversal detection failed", error=str(e))
            raise VortexError(f"Reversal detection failed: {e}")


async def calculate_vortex(
    df: pl.DataFrame,
    config: Dict[str, Any]
) -> pl.DataFrame:
    """Convenience function to calculate Vortex Indicator.

    Args:
        df: Input dataframe with OHLC data
        config: Configuration dictionary

    Returns:
        DataFrame with Vortex calculations

    Example:
        >>> config = {'period': 14}
        >>> result = await calculate_vortex(df, config)
    """
    vortex = Vortex(config)
    return await vortex.calculate(df)
