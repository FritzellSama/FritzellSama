"""
Ichimoku Cloud Indicator - Advanced trend analysis system.

This module implements the complete Ichimoku Kinko Hyo indicator including:
- Tenkan-sen (Conversion Line)
- Kijun-sen (Base Line)
- Senkou Span A (Leading Span A)
- Senkou Span B (Leading Span B)
- Chikou Span (Lagging Span)
"""

import asyncio
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any, Tuple
import polars as pl
from structlog import get_logger
from dataclasses import dataclass
from enum import Enum

logger = get_logger(__name__)

# Set high precision for Decimal calculations
getcontext().prec = 28


class IchimokuSignal(Enum):
    """Ichimoku trading signals."""
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


@dataclass
class IchimokuComponents:
    """Ichimoku Cloud components."""
    tenkan_sen: Decimal
    kijun_sen: Decimal
    senkou_span_a: Decimal
    senkou_span_b: Decimal
    chikou_span: Decimal
    cloud_top: Decimal
    cloud_bottom: Decimal
    cloud_thickness: Decimal


class IchimokuIndicator:
    """
    Ichimoku Kinko Hyo (Equilibrium Cloud) indicator.

    A comprehensive trend-following system that identifies support/resistance,
    trend direction, and momentum.

    Attributes:
        config: Configuration dictionary with period parameters
        tenkan_period: Conversion line period
        kijun_period: Base line period
        senkou_b_period: Leading Span B period
        displacement: Cloud displacement (typically 26)

    Example:
        >>> config = {
        ...     "tenkan_period": 9,
        ...     "kijun_period": 26,
        ...     "senkou_b_period": 52,
        ...     "displacement": 26
        ... }
        >>> ichimoku = IchimokuIndicator(config)
        >>> result = await ichimoku.calculate(ohlc_data)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize Ichimoku indicator.

        Args:
            config: Configuration dictionary containing:
                - tenkan_period: Tenkan-sen period (default 9)
                - kijun_period: Kijun-sen period (default 26)
                - senkou_b_period: Senkou Span B period (default 52)
                - displacement: Cloud displacement (default 26)
                - min_periods: Minimum periods required

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.tenkan_period: int = int(config.get("tenkan_period", 9))
        self.kijun_period: int = int(config.get("kijun_period", 26))
        self.senkou_b_period: int = int(config.get("senkou_b_period", 52))
        self.displacement: int = int(config.get("displacement", 26))
        self.min_periods: int = int(config.get("min_periods", 100))

        logger.info(
            "Ichimoku indicator initialized",
            tenkan_period=self.tenkan_period,
            kijun_period=self.kijun_period,
            senkou_b_period=self.senkou_b_period,
            displacement=self.displacement
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # No required fields - all have defaults
        # But validate values if provided

        if "tenkan_period" in self.config:
            if int(self.config["tenkan_period"]) < 1:
                raise ValueError("tenkan_period must be positive")

        if "kijun_period" in self.config:
            if int(self.config["kijun_period"]) < 1:
                raise ValueError("kijun_period must be positive")

        if "senkou_b_period" in self.config:
            if int(self.config["senkou_b_period"]) < 1:
                raise ValueError("senkou_b_period must be positive")

        if "displacement" in self.config:
            if int(self.config["displacement"]) < 0:
                raise ValueError("displacement must be non-negative")

        logger.debug("Ichimoku configuration validated")

    async def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate all Ichimoku components.

        Args:
            data: Polars DataFrame with columns: timestamp, open, high, low, close

        Returns:
            DataFrame with Ichimoku components and signals

        Raises:
            ValueError: If data is invalid or insufficient
        """
        try:
            # Validate input data
            self._validate_data(data)

            # Calculate all components
            result = data.clone()

            # Tenkan-sen (Conversion Line)
            result = await self._calculate_tenkan_sen(result)

            # Kijun-sen (Base Line)
            result = await self._calculate_kijun_sen(result)

            # Senkou Span A (Leading Span A)
            result = await self._calculate_senkou_span_a(result)

            # Senkou Span B (Leading Span B)
            result = await self._calculate_senkou_span_b(result)

            # Chikou Span (Lagging Span)
            result = await self._calculate_chikou_span(result)

            # Cloud metrics
            result = await self._calculate_cloud_metrics(result)

            # Generate signals
            result = await self._generate_signals(result)

            logger.info(
                "Ichimoku calculation completed",
                rows=len(result)
            )

            return result

        except Exception as e:
            logger.error("Ichimoku calculation failed", error=str(e))
            raise

    async def _calculate_tenkan_sen(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Tenkan-sen (Conversion Line).
        Formula: (Highest High + Lowest Low) / 2 over tenkan_period
        """
        try:
            result = data.with_columns([
                pl.col("high").rolling_max(window_size=self.tenkan_period).alias("tenkan_high"),
                pl.col("low").rolling_min(window_size=self.tenkan_period).alias("tenkan_low")
            ]).with_columns([
                ((pl.col("tenkan_high") + pl.col("tenkan_low")) / Decimal("2")).alias("tenkan_sen")
            ]).drop(["tenkan_high", "tenkan_low"])

            return result

        except Exception as e:
            logger.error("Tenkan-sen calculation failed", error=str(e))
            raise

    async def _calculate_kijun_sen(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Kijun-sen (Base Line).
        Formula: (Highest High + Lowest Low) / 2 over kijun_period
        """
        try:
            result = data.with_columns([
                pl.col("high").rolling_max(window_size=self.kijun_period).alias("kijun_high"),
                pl.col("low").rolling_min(window_size=self.kijun_period).alias("kijun_low")
            ]).with_columns([
                ((pl.col("kijun_high") + pl.col("kijun_low")) / Decimal("2")).alias("kijun_sen")
            ]).drop(["kijun_high", "kijun_low"])

            return result

        except Exception as e:
            logger.error("Kijun-sen calculation failed", error=str(e))
            raise

    async def _calculate_senkou_span_a(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Senkou Span A (Leading Span A).
        Formula: (Tenkan-sen + Kijun-sen) / 2, displaced forward
        """
        try:
            result = data.with_columns([
                ((pl.col("tenkan_sen") + pl.col("kijun_sen")) / Decimal("2")).alias("senkou_span_a_base")
            ])

            # Shift forward by displacement
            senkou_a_values = result["senkou_span_a_base"].to_list()
            displaced = [None] * self.displacement + senkou_a_values[:-self.displacement] if self.displacement > 0 else senkou_a_values

            result = result.with_columns([
                pl.Series("senkou_span_a", displaced)
            ]).drop("senkou_span_a_base")

            return result

        except Exception as e:
            logger.error("Senkou Span A calculation failed", error=str(e))
            raise

    async def _calculate_senkou_span_b(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Senkou Span B (Leading Span B).
        Formula: (Highest High + Lowest Low) / 2 over senkou_b_period, displaced forward
        """
        try:
            result = data.with_columns([
                pl.col("high").rolling_max(window_size=self.senkou_b_period).alias("senkou_b_high"),
                pl.col("low").rolling_min(window_size=self.senkou_b_period).alias("senkou_b_low")
            ]).with_columns([
                ((pl.col("senkou_b_high") + pl.col("senkou_b_low")) / Decimal("2")).alias("senkou_span_b_base")
            ]).drop(["senkou_b_high", "senkou_b_low"])

            # Shift forward by displacement
            senkou_b_values = result["senkou_span_b_base"].to_list()
            displaced = [None] * self.displacement + senkou_b_values[:-self.displacement] if self.displacement > 0 else senkou_b_values

            result = result.with_columns([
                pl.Series("senkou_span_b", displaced)
            ]).drop("senkou_span_b_base")

            return result

        except Exception as e:
            logger.error("Senkou Span B calculation failed", error=str(e))
            raise

    async def _calculate_chikou_span(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Chikou Span (Lagging Span).
        Formula: Close price, displaced backward
        """
        try:
            # Shift backward by displacement
            chikou_values = data["close"].to_list()
            displaced = chikou_values[self.displacement:] + [None] * self.displacement if self.displacement > 0 else chikou_values

            result = data.with_columns([
                pl.Series("chikou_span", displaced)
            ])

            return result

        except Exception as e:
            logger.error("Chikou Span calculation failed", error=str(e))
            raise

    async def _calculate_cloud_metrics(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate cloud top, bottom, and thickness."""
        try:
            result = data.with_columns([
                pl.max_horizontal("senkou_span_a", "senkou_span_b").alias("cloud_top"),
                pl.min_horizontal("senkou_span_a", "senkou_span_b").alias("cloud_bottom")
            ]).with_columns([
                (pl.col("cloud_top") - pl.col("cloud_bottom")).alias("cloud_thickness")
            ])

            return result

        except Exception as e:
            logger.error("Cloud metrics calculation failed", error=str(e))
            raise

    async def _generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Generate trading signals based on Ichimoku components.

        Signal logic:
        - STRONG_BULLISH: Price above cloud, Tenkan > Kijun, Chikou above price
        - BULLISH: Price above cloud
        - NEUTRAL: Price in cloud
        - BEARISH: Price below cloud
        - STRONG_BEARISH: Price below cloud, Tenkan < Kijun, Chikou below price
        """
        try:
            def determine_signal(row) -> str:
                try:
                    close = row["close"]
                    cloud_top = row["cloud_top"]
                    cloud_bottom = row["cloud_bottom"]
                    tenkan = row["tenkan_sen"]
                    kijun = row["kijun_sen"]
                    chikou = row["chikou_span"]

                    # Check for None values
                    if any(v is None for v in [close, cloud_top, cloud_bottom, tenkan, kijun]):
                        return IchimokuSignal.NEUTRAL.value

                    # Price position relative to cloud
                    if close > cloud_top:
                        # Bullish conditions
                        if chikou is not None and tenkan > kijun and chikou > close:
                            return IchimokuSignal.STRONG_BULLISH.value
                        return IchimokuSignal.BULLISH.value
                    elif close < cloud_bottom:
                        # Bearish conditions
                        if chikou is not None and tenkan < kijun and chikou < close:
                            return IchimokuSignal.STRONG_BEARISH.value
                        return IchimokuSignal.BEARISH.value
                    else:
                        # In cloud
                        return IchimokuSignal.NEUTRAL.value

                except Exception:
                    return IchimokuSignal.NEUTRAL.value

            # Apply signal generation
            signals = []
            for row in data.iter_rows(named=True):
                signals.append(determine_signal(row))

            result = data.with_columns([
                pl.Series("signal", signals)
            ])

            # Add signal strength (numeric)
            signal_strength_map = {
                IchimokuSignal.STRONG_BULLISH.value: Decimal("2"),
                IchimokuSignal.BULLISH.value: Decimal("1"),
                IchimokuSignal.NEUTRAL.value: Decimal("0"),
                IchimokuSignal.BEARISH.value: Decimal("-1"),
                IchimokuSignal.STRONG_BEARISH.value: Decimal("-2")
            }

            result = result.with_columns([
                pl.col("signal").map_elements(
                    lambda s: str(signal_strength_map.get(s, Decimal("0"))),
                    return_dtype=pl.Utf8
                ).alias("signal_strength")
            ])

            return result

        except Exception as e:
            logger.error("Signal generation failed", error=str(e))
            raise

    def _validate_data(self, data: pl.DataFrame) -> None:
        """Validate input data format and content."""
        required_columns = ["timestamp", "open", "high", "low", "close"]

        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        if len(data) < self.min_periods:
            raise ValueError(
                f"Insufficient data points: {len(data)} < {self.min_periods}"
            )

        # Check for null values in OHLC
        for col in ["open", "high", "low", "close"]:
            if data[col].null_count() > 0:
                raise ValueError(f"Data contains null values in '{col}' column")

    async def get_current_components(self, data: pl.DataFrame) -> Optional[IchimokuComponents]:
        """
        Get current Ichimoku components from calculated data.

        Args:
            data: DataFrame with calculated Ichimoku values

        Returns:
            IchimokuComponents dataclass or None
        """
        try:
            if len(data) == 0:
                return None

            last_row = data.row(-1, named=True)

            # Check if all components are available
            required = ["tenkan_sen", "kijun_sen", "senkou_span_a", "senkou_span_b",
                       "chikou_span", "cloud_top", "cloud_bottom", "cloud_thickness"]

            if any(last_row.get(k) is None for k in required):
                return None

            return IchimokuComponents(
                tenkan_sen=Decimal(str(last_row["tenkan_sen"])),
                kijun_sen=Decimal(str(last_row["kijun_sen"])),
                senkou_span_a=Decimal(str(last_row["senkou_span_a"])),
                senkou_span_b=Decimal(str(last_row["senkou_span_b"])),
                chikou_span=Decimal(str(last_row["chikou_span"])),
                cloud_top=Decimal(str(last_row["cloud_top"])),
                cloud_bottom=Decimal(str(last_row["cloud_bottom"])),
                cloud_thickness=Decimal(str(last_row["cloud_thickness"]))
            )

        except Exception as e:
            logger.error("Failed to get current components", error=str(e))
            return None


async def create_ichimoku_indicator(config: Dict[str, Any]) -> IchimokuIndicator:
    """
    Factory function to create Ichimoku indicator instance.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized Ichimoku indicator
    """
    return IchimokuIndicator(config)
