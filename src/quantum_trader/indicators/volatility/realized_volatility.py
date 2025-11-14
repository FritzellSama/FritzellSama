"""
Realized Volatility Indicator.

This module implements realized volatility calculation for measuring actual
historical price volatility in institutional trading systems.
"""

import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Optional
import polars as pl
from structlog import get_logger
import math

logger = get_logger(__name__)


class RealizedVolatilityError(Exception):
    """Base exception for Realized Volatility indicator errors."""
    pass


class RealizedVolatilityValidator:
    """Validates Realized Volatility configuration and input data."""

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate Realized Volatility configuration.

        Args:
            config: Configuration dictionary

        Raises:
            RealizedVolatilityError: If configuration is invalid
        """
        required_keys = ['window', 'annualization_factor']
        for key in required_keys:
            if key not in config:
                raise RealizedVolatilityError(f"Missing required config key: {key}")

        try:
            window = int(config['window'])
            ann_factor = Decimal(str(config['annualization_factor']))

            if window < 2:
                raise RealizedVolatilityError("window must be at least 2")
            if ann_factor <= Decimal('0'):
                raise RealizedVolatilityError("annualization_factor must be positive")

        except (ValueError, TypeError) as e:
            raise RealizedVolatilityError(f"Invalid configuration values: {e}")

    @staticmethod
    def validate_dataframe(df: pl.DataFrame) -> None:
        """Validate input dataframe structure.

        Args:
            df: Input dataframe

        Raises:
            RealizedVolatilityError: If dataframe is invalid
        """
        if 'close' not in df.columns:
            raise RealizedVolatilityError("Missing required column: close")

        if len(df) < 2:
            raise RealizedVolatilityError("Dataframe must have at least 2 rows")


class RealizedVolatility:
    """Realized Volatility calculation for price series.

    Realized volatility measures actual historical volatility based on
    logarithmic returns. It's commonly used in options pricing and risk management.

    Attributes:
        config: Configuration dictionary
        window: Rolling window size for volatility calculation
        annualization_factor: Factor to annualize volatility (e.g., sqrt(252) for daily data)

    Example:
        >>> config = {
        ...     'window': 20,
        ...     'annualization_factor': '15.8745'  # sqrt(252) for daily data
        ... }
        >>> rv = RealizedVolatility(config)
        >>> result = await rv.calculate(df)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Realized Volatility indicator.

        Args:
            config: Configuration dictionary

        Raises:
            RealizedVolatilityError: If configuration is invalid
        """
        RealizedVolatilityValidator.validate_config(config)

        self.config = config
        self.window = int(config['window'])
        self.annualization_factor = Decimal(str(config['annualization_factor']))

        logger.info(
            "RealizedVolatility initialized",
            window=self.window,
            annualization_factor=str(self.annualization_factor)
        )

    async def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate realized volatility.

        Args:
            df: Input dataframe with column: close

        Returns:
            DataFrame with additional columns:
                - log_return: Logarithmic returns
                - rv: Realized volatility (rolling standard deviation)
                - rv_annualized: Annualized realized volatility

        Raises:
            RealizedVolatilityError: If calculation fails

        Example:
            >>> df = pl.DataFrame({
            ...     'close': [100.0, 101.0, 102.0, 101.5, 103.0]
            ... })
            >>> result = await rv.calculate(df)
        """
        try:
            RealizedVolatilityValidator.validate_dataframe(df)

            logger.debug("Calculating realized volatility", rows=len(df))

            closes = [Decimal(str(x)) for x in df['close'].to_list()]

            # Calculate logarithmic returns
            log_returns = [Decimal('0')]  # First value has no return
            for i in range(1, len(closes)):
                if closes[i-1] > Decimal('0'):
                    # ln(P_t / P_t-1)
                    ratio = closes[i] / closes[i-1]
                    log_ret = Decimal(str(math.log(float(ratio))))
                    log_returns.append(log_ret)
                else:
                    log_returns.append(Decimal('0'))

            # Calculate rolling volatility (standard deviation of returns)
            rv_values = []
            rv_annualized_values = []

            for i in range(len(log_returns)):
                if i < self.window - 1:
                    # Not enough data yet
                    rv_values.append(None)
                    rv_annualized_values.append(None)
                else:
                    # Get window of returns
                    window_returns = log_returns[i - self.window + 1:i + 1]

                    # Calculate mean
                    mean_return = sum(window_returns) / Decimal(str(len(window_returns)))

                    # Calculate variance
                    variance = sum((r - mean_return) ** 2 for r in window_returns) / Decimal(str(len(window_returns)))

                    # Standard deviation (realized volatility)
                    if variance > Decimal('0'):
                        std_dev = Decimal(str(math.sqrt(float(variance))))
                        rv = std_dev
                        rv_ann = std_dev * self.annualization_factor
                    else:
                        rv = Decimal('0')
                        rv_ann = Decimal('0')

                    rv_values.append(float(rv))
                    rv_annualized_values.append(float(rv_ann))

            # Add columns to dataframe
            result = df.with_columns([
                pl.Series('log_return', [float(x) if x is not None else None for x in log_returns]),
                pl.Series('rv', rv_values),
                pl.Series('rv_annualized', rv_annualized_values)
            ])

            logger.info("Realized volatility calculated successfully", rows=len(result))

            return result

        except RealizedVolatilityError:
            raise
        except Exception as e:
            logger.error("Realized volatility calculation failed", error=str(e))
            raise RealizedVolatilityError(f"Calculation failed: {e}")

    async def calculate_parkinson(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Parkinson's volatility estimator.

        Uses high and low prices for more efficient volatility estimation.

        Args:
            df: Input dataframe with columns: high, low

        Returns:
            DataFrame with parkinson_volatility column

        Raises:
            RealizedVolatilityError: If calculation fails
        """
        try:
            required = ['high', 'low']
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise RealizedVolatilityError(f"Missing required columns: {missing}")

            logger.debug("Calculating Parkinson volatility", rows=len(df))

            highs = [Decimal(str(x)) for x in df['high'].to_list()]
            lows = [Decimal(str(x)) for x in df['low'].to_list()]

            # Parkinson's volatility: sqrt(1/(4*ln(2)) * sum(ln(H/L)^2))
            factor = Decimal('1') / (Decimal('4') * Decimal(str(math.log(2))))

            park_vol_values = []

            for i in range(len(highs)):
                if i < self.window - 1:
                    park_vol_values.append(None)
                else:
                    sum_squares = Decimal('0')
                    for j in range(i - self.window + 1, i + 1):
                        if lows[j] > Decimal('0') and highs[j] > lows[j]:
                            ratio = highs[j] / lows[j]
                            log_ratio = Decimal(str(math.log(float(ratio))))
                            sum_squares += log_ratio ** 2

                    park_vol = Decimal(str(math.sqrt(float(factor * sum_squares))))
                    park_vol_ann = park_vol * self.annualization_factor

                    park_vol_values.append(float(park_vol_ann))

            result = df.with_columns([
                pl.Series('parkinson_volatility', park_vol_values)
            ])

            logger.info("Parkinson volatility calculated successfully")

            return result

        except Exception as e:
            logger.error("Parkinson volatility calculation failed", error=str(e))
            raise RealizedVolatilityError(f"Parkinson calculation failed: {e}")

    async def calculate_garman_klass(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Garman-Klass volatility estimator.

        Uses open, high, low, and close for improved estimation.

        Args:
            df: Input dataframe with columns: open, high, low, close

        Returns:
            DataFrame with garman_klass_volatility column

        Raises:
            RealizedVolatilityError: If calculation fails
        """
        try:
            required = ['open', 'high', 'low', 'close']
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise RealizedVolatilityError(f"Missing required columns: {missing}")

            logger.debug("Calculating Garman-Klass volatility", rows=len(df))

            opens = [Decimal(str(x)) for x in df['open'].to_list()]
            highs = [Decimal(str(x)) for x in df['high'].to_list()]
            lows = [Decimal(str(x)) for x in df['low'].to_list()]
            closes = [Decimal(str(x)) for x in df['close'].to_list()]

            gk_vol_values = []

            for i in range(len(opens)):
                if i < self.window - 1:
                    gk_vol_values.append(None)
                else:
                    sum_gk = Decimal('0')
                    for j in range(i - self.window + 1, i + 1):
                        if opens[j] > Decimal('0') and lows[j] > Decimal('0'):
                            hl_ratio = highs[j] / lows[j]
                            co_ratio = closes[j] / opens[j]

                            log_hl = Decimal(str(math.log(float(hl_ratio))))
                            log_co = Decimal(str(math.log(float(co_ratio))))

                            # GK formula: 0.5 * (ln(H/L))^2 - (2*ln(2)-1) * (ln(C/O))^2
                            term1 = Decimal('0.5') * (log_hl ** 2)
                            term2 = (Decimal('2') * Decimal(str(math.log(2))) - Decimal('1')) * (log_co ** 2)
                            sum_gk += term1 - term2

                    gk_vol = Decimal(str(math.sqrt(float(sum_gk / Decimal(str(self.window))))))
                    gk_vol_ann = gk_vol * self.annualization_factor

                    gk_vol_values.append(float(gk_vol_ann))

            result = df.with_columns([
                pl.Series('garman_klass_volatility', gk_vol_values)
            ])

            logger.info("Garman-Klass volatility calculated successfully")

            return result

        except Exception as e:
            logger.error("Garman-Klass volatility calculation failed", error=str(e))
            raise RealizedVolatilityError(f"Garman-Klass calculation failed: {e}")


async def calculate_realized_volatility(
    df: pl.DataFrame,
    config: Dict[str, Any]
) -> pl.DataFrame:
    """Convenience function to calculate realized volatility.

    Args:
        df: Input dataframe with price data
        config: Configuration dictionary

    Returns:
        DataFrame with realized volatility calculations

    Example:
        >>> config = {
        ...     'window': 20,
        ...     'annualization_factor': '15.8745'
        ... }
        >>> result = await calculate_realized_volatility(df, config)
    """
    rv = RealizedVolatility(config)
    return await rv.calculate(df)
