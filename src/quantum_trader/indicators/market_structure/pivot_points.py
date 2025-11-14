"""
Pivot Points Market Structure Indicator.

This module implements various pivot point calculation methods for identifying
key support and resistance levels in institutional trading systems.
"""

import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional
from enum import Enum
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class PivotMethod(Enum):
    """Pivot point calculation methods."""
    STANDARD = "standard"
    FIBONACCI = "fibonacci"
    WOODIE = "woodie"
    CAMARILLA = "camarilla"
    DEMARK = "demark"


class PivotPointsError(Exception):
    """Base exception for Pivot Points indicator errors."""
    pass


class PivotPointsValidator:
    """Validates Pivot Points configuration and input data."""

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate Pivot Points configuration.

        Args:
            config: Configuration dictionary

        Raises:
            PivotPointsError: If configuration is invalid
        """
        if 'method' not in config:
            raise PivotPointsError("Missing required config key: method")

        method = config['method'].lower()
        valid_methods = [m.value for m in PivotMethod]
        if method not in valid_methods:
            raise PivotPointsError(
                f"Invalid method '{method}'. Must be one of: {valid_methods}"
            )

    @staticmethod
    def validate_dataframe(df: pl.DataFrame) -> None:
        """Validate input dataframe structure.

        Args:
            df: Input dataframe

        Raises:
            PivotPointsError: If dataframe is invalid
        """
        required_columns = ['high', 'low', 'close']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise PivotPointsError(f"Missing required columns: {missing}")

        if len(df) < 1:
            raise PivotPointsError("Dataframe must have at least 1 row")


class PivotPoints:
    """Pivot Points calculation for multiple methods.

    Pivot points are used to identify potential support and resistance levels
    based on previous period's high, low, and close prices.

    Attributes:
        config: Configuration dictionary
        method: Calculation method (standard, fibonacci, woodie, camarilla, demark)

    Example:
        >>> config = {'method': 'standard'}
        >>> pivot = PivotPoints(config)
        >>> result = await pivot.calculate(df)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Pivot Points indicator.

        Args:
            config: Configuration dictionary with 'method' key

        Raises:
            PivotPointsError: If configuration is invalid
        """
        PivotPointsValidator.validate_config(config)

        self.config = config
        self.method = PivotMethod(config['method'].lower())

        logger.info("PivotPoints initialized", method=self.method.value)

    async def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate pivot points and support/resistance levels.

        Args:
            df: Input dataframe with columns: high, low, close

        Returns:
            DataFrame with additional columns:
                - pivot: Pivot point
                - r1, r2, r3: Resistance levels
                - s1, s2, s3: Support levels
                - (additional levels depending on method)

        Raises:
            PivotPointsError: If calculation fails

        Example:
            >>> df = pl.DataFrame({
            ...     'high': [102.0, 103.0],
            ...     'low': [99.0, 100.0],
            ...     'close': [101.0, 102.0]
            ... })
            >>> result = await pivot.calculate(df)
        """
        try:
            PivotPointsValidator.validate_dataframe(df)

            logger.debug("Calculating pivot points", rows=len(df), method=self.method.value)

            # Route to appropriate calculation method
            if self.method == PivotMethod.STANDARD:
                result = await self._calculate_standard(df)
            elif self.method == PivotMethod.FIBONACCI:
                result = await self._calculate_fibonacci(df)
            elif self.method == PivotMethod.WOODIE:
                result = await self._calculate_woodie(df)
            elif self.method == PivotMethod.CAMARILLA:
                result = await self._calculate_camarilla(df)
            elif self.method == PivotMethod.DEMARK:
                result = await self._calculate_demark(df)
            else:
                raise PivotPointsError(f"Unsupported method: {self.method}")

            logger.info("Pivot points calculated successfully", rows=len(result))

            return result

        except PivotPointsError:
            raise
        except Exception as e:
            logger.error("Pivot points calculation failed", error=str(e))
            raise PivotPointsError(f"Calculation failed: {e}")

    async def _calculate_standard(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate standard pivot points.

        Formula:
            P = (H + L + C) / 3
            R1 = 2P - L
            R2 = P + (H - L)
            R3 = H + 2(P - L)
            S1 = 2P - H
            S2 = P - (H - L)
            S3 = L - 2(H - P)
        """
        highs = [Decimal(str(x)) for x in df['high'].to_list()]
        lows = [Decimal(str(x)) for x in df['low'].to_list()]
        closes = [Decimal(str(x)) for x in df['close'].to_list()]

        pivots = []
        r1_list, r2_list, r3_list = [], [], []
        s1_list, s2_list, s3_list = [], [], []

        for h, l, c in zip(highs, lows, closes):
            # Calculate pivot point
            p = (h + l + c) / Decimal('3')

            # Calculate resistance levels
            r1 = Decimal('2') * p - l
            r2 = p + (h - l)
            r3 = h + Decimal('2') * (p - l)

            # Calculate support levels
            s1 = Decimal('2') * p - h
            s2 = p - (h - l)
            s3 = l - Decimal('2') * (h - p)

            pivots.append(float(p))
            r1_list.append(float(r1))
            r2_list.append(float(r2))
            r3_list.append(float(r3))
            s1_list.append(float(s1))
            s2_list.append(float(s2))
            s3_list.append(float(s3))

        return df.with_columns([
            pl.Series('pivot', pivots),
            pl.Series('r1', r1_list),
            pl.Series('r2', r2_list),
            pl.Series('r3', r3_list),
            pl.Series('s1', s1_list),
            pl.Series('s2', s2_list),
            pl.Series('s3', s3_list)
        ])

    async def _calculate_fibonacci(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Fibonacci pivot points.

        Uses Fibonacci ratios (0.382, 0.618) for support/resistance.
        """
        highs = [Decimal(str(x)) for x in df['high'].to_list()]
        lows = [Decimal(str(x)) for x in df['low'].to_list()]
        closes = [Decimal(str(x)) for x in df['close'].to_list()]

        fib_382 = Decimal('0.382')
        fib_618 = Decimal('0.618')
        fib_1000 = Decimal('1.000')

        pivots = []
        r1_list, r2_list, r3_list = [], [], []
        s1_list, s2_list, s3_list = [], [], []

        for h, l, c in zip(highs, lows, closes):
            p = (h + l + c) / Decimal('3')
            range_hl = h - l

            r1 = p + (fib_382 * range_hl)
            r2 = p + (fib_618 * range_hl)
            r3 = p + (fib_1000 * range_hl)

            s1 = p - (fib_382 * range_hl)
            s2 = p - (fib_618 * range_hl)
            s3 = p - (fib_1000 * range_hl)

            pivots.append(float(p))
            r1_list.append(float(r1))
            r2_list.append(float(r2))
            r3_list.append(float(r3))
            s1_list.append(float(s1))
            s2_list.append(float(s2))
            s3_list.append(float(s3))

        return df.with_columns([
            pl.Series('pivot', pivots),
            pl.Series('r1', r1_list),
            pl.Series('r2', r2_list),
            pl.Series('r3', r3_list),
            pl.Series('s1', s1_list),
            pl.Series('s2', s2_list),
            pl.Series('s3', s3_list)
        ])

    async def _calculate_woodie(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Woodie's pivot points.

        Gives more weight to closing price.
        """
        highs = [Decimal(str(x)) for x in df['high'].to_list()]
        lows = [Decimal(str(x)) for x in df['low'].to_list()]
        closes = [Decimal(str(x)) for x in df['close'].to_list()]

        pivots = []
        r1_list, r2_list, r3_list = [], [], []
        s1_list, s2_list, s3_list = [], [], []

        for h, l, c in zip(highs, lows, closes):
            # Woodie's pivot uses (H + L + 2C) / 4
            p = (h + l + Decimal('2') * c) / Decimal('4')

            r1 = Decimal('2') * p - l
            r2 = p + (h - l)
            r3 = h + Decimal('2') * (p - l)

            s1 = Decimal('2') * p - h
            s2 = p - (h - l)
            s3 = l - Decimal('2') * (h - p)

            pivots.append(float(p))
            r1_list.append(float(r1))
            r2_list.append(float(r2))
            r3_list.append(float(r3))
            s1_list.append(float(s1))
            s2_list.append(float(s2))
            s3_list.append(float(s3))

        return df.with_columns([
            pl.Series('pivot', pivots),
            pl.Series('r1', r1_list),
            pl.Series('r2', r2_list),
            pl.Series('r3', r3_list),
            pl.Series('s1', s1_list),
            pl.Series('s2', s2_list),
            pl.Series('s3', s3_list)
        ])

    async def _calculate_camarilla(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Camarilla pivot points.

        Provides 4 levels of support and resistance.
        """
        highs = [Decimal(str(x)) for x in df['high'].to_list()]
        lows = [Decimal(str(x)) for x in df['low'].to_list()]
        closes = [Decimal(str(x)) for x in df['close'].to_list()]

        pivots = []
        r1_list, r2_list, r3_list, r4_list = [], [], [], []
        s1_list, s2_list, s3_list, s4_list = [], [], [], []

        for h, l, c in zip(highs, lows, closes):
            p = (h + l + c) / Decimal('3')
            range_hl = h - l

            # Camarilla uses multipliers: 1.1/12, 1.1/6, 1.1/4, 1.1/2
            r1 = c + (range_hl * Decimal('1.1') / Decimal('12'))
            r2 = c + (range_hl * Decimal('1.1') / Decimal('6'))
            r3 = c + (range_hl * Decimal('1.1') / Decimal('4'))
            r4 = c + (range_hl * Decimal('1.1') / Decimal('2'))

            s1 = c - (range_hl * Decimal('1.1') / Decimal('12'))
            s2 = c - (range_hl * Decimal('1.1') / Decimal('6'))
            s3 = c - (range_hl * Decimal('1.1') / Decimal('4'))
            s4 = c - (range_hl * Decimal('1.1') / Decimal('2'))

            pivots.append(float(p))
            r1_list.append(float(r1))
            r2_list.append(float(r2))
            r3_list.append(float(r3))
            r4_list.append(float(r4))
            s1_list.append(float(s1))
            s2_list.append(float(s2))
            s3_list.append(float(s3))
            s4_list.append(float(s4))

        return df.with_columns([
            pl.Series('pivot', pivots),
            pl.Series('r1', r1_list),
            pl.Series('r2', r2_list),
            pl.Series('r3', r3_list),
            pl.Series('r4', r4_list),
            pl.Series('s1', s1_list),
            pl.Series('s2', s2_list),
            pl.Series('s3', s3_list),
            pl.Series('s4', s4_list)
        ])

    async def _calculate_demark(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate DeMark pivot points.

        Uses different formulas based on close vs open relationship.
        """
        highs = [Decimal(str(x)) for x in df['high'].to_list()]
        lows = [Decimal(str(x)) for x in df['low'].to_list()]
        closes = [Decimal(str(x)) for x in df['close'].to_list()]

        # DeMark needs open prices, use close if not available
        if 'open' in df.columns:
            opens = [Decimal(str(x)) for x in df['open'].to_list()]
        else:
            opens = closes

        pivots = []
        r1_list, s1_list = [], []

        for h, l, c, o in zip(highs, lows, closes, opens):
            # Calculate X based on close vs open
            if c < o:
                x = h + Decimal('2') * l + c
            elif c > o:
                x = Decimal('2') * h + l + c
            else:
                x = h + l + Decimal('2') * c

            p = x / Decimal('4')
            r1 = x / Decimal('2') - l
            s1 = x / Decimal('2') - h

            pivots.append(float(p))
            r1_list.append(float(r1))
            s1_list.append(float(s1))

        return df.with_columns([
            pl.Series('pivot', pivots),
            pl.Series('r1', r1_list),
            pl.Series('s1', s1_list)
        ])


async def calculate_pivot_points(
    df: pl.DataFrame,
    config: Dict[str, Any]
) -> pl.DataFrame:
    """Convenience function to calculate pivot points.

    Args:
        df: Input dataframe with OHLC data
        config: Configuration dictionary with 'method' key

    Returns:
        DataFrame with pivot point calculations

    Example:
        >>> config = {'method': 'standard'}
        >>> result = await calculate_pivot_points(df, config)
    """
    pivot = PivotPoints(config)
    return await pivot.calculate(df)
