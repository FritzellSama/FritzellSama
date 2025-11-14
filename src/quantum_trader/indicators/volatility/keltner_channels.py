"""
Keltner Channels Indicator - Volatility-based channel indicator.

Keltner Channels use Average True Range (ATR) to set channel distance.
Useful for identifying overbought/oversold conditions and trend strength.
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


class KeltnerSignal(Enum):
    """Keltner Channels trading signals."""
    STRONG_OVERBOUGHT = "strong_overbought"
    OVERBOUGHT = "overbought"
    NEUTRAL = "neutral"
    OVERSOLD = "oversold"
    STRONG_OVERSOLD = "strong_oversold"


@dataclass
class KeltnerChannelValues:
    """Keltner Channels values at a point in time."""
    upper_band: Decimal
    middle_band: Decimal
    lower_band: Decimal
    atr: Decimal
    bandwidth: Decimal
    percent_b: Decimal


class KeltnerChannelsIndicator:
    """
    Keltner Channels volatility indicator.

    Keltner Channels consist of:
    - Middle Band: Exponential Moving Average (EMA)
    - Upper Band: EMA + (multiplier × ATR)
    - Lower Band: EMA - (multiplier × ATR)

    Attributes:
        config: Configuration dictionary
        ema_period: Period for middle band EMA
        atr_period: Period for ATR calculation
        multiplier: ATR multiplier for band width

    Example:
        >>> config = {
        ...     "ema_period": 20,
        ...     "atr_period": 10,
        ...     "multiplier": "2.0"
        ... }
        >>> keltner = KeltnerChannelsIndicator(config)
        >>> result = await keltner.calculate(ohlc_data)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize Keltner Channels indicator.

        Args:
            config: Configuration dictionary containing:
                - ema_period: EMA period for middle band
                - atr_period: ATR period for volatility
                - multiplier: ATR multiplier for bands
                - min_periods: Minimum periods required

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.ema_period: int = int(config["ema_period"])
        self.atr_period: int = int(config["atr_period"])
        self.multiplier: Decimal = Decimal(str(config["multiplier"]))
        self.min_periods: int = int(config.get("min_periods", 50))

        logger.info(
            "Keltner Channels indicator initialized",
            ema_period=self.ema_period,
            atr_period=self.atr_period,
            multiplier=str(self.multiplier)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_fields = ["ema_period", "atr_period", "multiplier"]

        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")

        if int(self.config["ema_period"]) < 1:
            raise ValueError("ema_period must be positive")

        if int(self.config["atr_period"]) < 1:
            raise ValueError("atr_period must be positive")

        if Decimal(str(self.config["multiplier"])) <= Decimal("0"):
            raise ValueError("multiplier must be positive")

        logger.debug("Keltner Channels configuration validated")

    async def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Keltner Channels.

        Args:
            data: Polars DataFrame with columns: timestamp, open, high, low, close

        Returns:
            DataFrame with Keltner Channels values and signals

        Raises:
            ValueError: If data is invalid or insufficient
        """
        try:
            # Validate input data
            self._validate_data(data)

            # Calculate middle band (EMA)
            result = await self._calculate_middle_band(data)

            # Calculate ATR
            result = await self._calculate_atr(result)

            # Calculate upper and lower bands
            result = await self._calculate_bands(result)

            # Calculate additional metrics
            result = await self._calculate_metrics(result)

            # Generate signals
            result = await self._generate_signals(result)

            logger.info(
                "Keltner Channels calculated",
                rows=len(result)
            )

            return result

        except Exception as e:
            logger.error("Keltner Channels calculation failed", error=str(e))
            raise

    async def _calculate_middle_band(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate middle band using Exponential Moving Average.

        Args:
            data: Input OHLC data

        Returns:
            DataFrame with middle_band column
        """
        try:
            # Calculate EMA using Polars
            # EMA formula: EMA(t) = α × Close(t) + (1-α) × EMA(t-1)
            # where α = 2 / (period + 1)

            alpha = Decimal("2") / (Decimal(str(self.ema_period)) + Decimal("1"))

            # Convert close to Decimal series
            close_values = [Decimal(str(x)) for x in data["close"].to_list()]

            # Calculate EMA
            ema_values = []
            ema = close_values[0]  # Initialize with first close

            for close in close_values:
                ema = alpha * close + (Decimal("1") - alpha) * ema
                ema_values.append(ema)

            # Add to DataFrame
            result = data.with_columns([
                pl.Series("middle_band", [str(v) for v in ema_values])
            ])

            return result

        except Exception as e:
            logger.error("Middle band calculation failed", error=str(e))
            raise

    async def _calculate_atr(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Average True Range (ATR).

        ATR measures volatility using:
        TR = max(high - low, |high - prev_close|, |low - prev_close|)

        Args:
            data: DataFrame with OHLC data

        Returns:
            DataFrame with ATR column
        """
        try:
            # Calculate True Range
            high_values = [Decimal(str(x)) for x in data["high"].to_list()]
            low_values = [Decimal(str(x)) for x in data["low"].to_list()]
            close_values = [Decimal(str(x)) for x in data["close"].to_list()]

            tr_values = []
            for i in range(len(data)):
                if i == 0:
                    # First period: just high - low
                    tr = high_values[i] - low_values[i]
                else:
                    tr = max(
                        high_values[i] - low_values[i],
                        abs(high_values[i] - close_values[i-1]),
                        abs(low_values[i] - close_values[i-1])
                    )
                tr_values.append(tr)

            # Calculate ATR using moving average
            atr_values = []
            alpha = Decimal("1") / Decimal(str(self.atr_period))

            atr = sum(tr_values[:self.atr_period]) / Decimal(str(self.atr_period))

            # Fill initial values with None
            atr_values = [None] * (self.atr_period - 1)
            atr_values.append(atr)

            # Calculate smoothed ATR
            for i in range(self.atr_period, len(tr_values)):
                atr = alpha * tr_values[i] + (Decimal("1") - alpha) * atr
                atr_values.append(atr)

            # Add to DataFrame
            result = data.with_columns([
                pl.Series("atr", [str(v) if v is not None else None for v in atr_values])
            ])

            return result

        except Exception as e:
            logger.error("ATR calculation failed", error=str(e))
            raise

    async def _calculate_bands(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate upper and lower Keltner Channel bands.

        Args:
            data: DataFrame with middle_band and atr

        Returns:
            DataFrame with upper_band and lower_band columns
        """
        try:
            # Calculate bands
            def calc_bands(row):
                middle = row["middle_band"]
                atr = row["atr"]

                if middle is None or atr is None:
                    return None, None

                middle_dec = Decimal(str(middle))
                atr_dec = Decimal(str(atr))

                upper = middle_dec + (self.multiplier * atr_dec)
                lower = middle_dec - (self.multiplier * atr_dec)

                return str(upper), str(lower)

            upper_bands = []
            lower_bands = []

            for row in data.iter_rows(named=True):
                upper, lower = calc_bands(row)
                upper_bands.append(upper)
                lower_bands.append(lower)

            result = data.with_columns([
                pl.Series("upper_band", upper_bands),
                pl.Series("lower_band", lower_bands)
            ])

            return result

        except Exception as e:
            logger.error("Band calculation failed", error=str(e))
            raise

    async def _calculate_metrics(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate additional Keltner Channel metrics.

        Metrics:
        - Bandwidth: (upper - lower) / middle
        - %B: (close - lower) / (upper - lower)

        Args:
            data: DataFrame with channel bands

        Returns:
            DataFrame with additional metrics
        """
        try:
            def calc_metrics(row):
                close = row["close"]
                upper = row["upper_band"]
                middle = row["middle_band"]
                lower = row["lower_band"]

                if any(v is None for v in [close, upper, middle, lower]):
                    return None, None

                close_dec = Decimal(str(close))
                upper_dec = Decimal(str(upper))
                middle_dec = Decimal(str(middle))
                lower_dec = Decimal(str(lower))

                # Bandwidth
                bandwidth = (upper_dec - lower_dec) / middle_dec

                # %B (position within bands)
                range_val = upper_dec - lower_dec
                if range_val == Decimal("0"):
                    percent_b = Decimal("0.5")
                else:
                    percent_b = (close_dec - lower_dec) / range_val

                return str(bandwidth), str(percent_b)

            bandwidths = []
            percent_bs = []

            for row in data.iter_rows(named=True):
                bandwidth, percent_b = calc_metrics(row)
                bandwidths.append(bandwidth)
                percent_bs.append(percent_b)

            result = data.with_columns([
                pl.Series("bandwidth", bandwidths),
                pl.Series("percent_b", percent_bs)
            ])

            return result

        except Exception as e:
            logger.error("Metrics calculation failed", error=str(e))
            raise

    async def _generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Generate trading signals based on Keltner Channels.

        Signal logic:
        - STRONG_OVERBOUGHT: Close > upper band, %B > 1.1
        - OVERBOUGHT: Close > upper band
        - NEUTRAL: Close within bands
        - OVERSOLD: Close < lower band
        - STRONG_OVERSOLD: Close < lower band, %B < -0.1

        Args:
            data: DataFrame with channel values

        Returns:
            DataFrame with signal column
        """
        try:
            def determine_signal(row) -> str:
                try:
                    close = row["close"]
                    upper = row["upper_band"]
                    lower = row["lower_band"]
                    percent_b = row["percent_b"]

                    if any(v is None for v in [close, upper, lower, percent_b]):
                        return KeltnerSignal.NEUTRAL.value

                    close_dec = Decimal(str(close))
                    upper_dec = Decimal(str(upper))
                    lower_dec = Decimal(str(lower))
                    percent_b_dec = Decimal(str(percent_b))

                    # Determine signal
                    if close_dec > upper_dec:
                        if percent_b_dec > Decimal("1.1"):
                            return KeltnerSignal.STRONG_OVERBOUGHT.value
                        return KeltnerSignal.OVERBOUGHT.value
                    elif close_dec < lower_dec:
                        if percent_b_dec < Decimal("-0.1"):
                            return KeltnerSignal.STRONG_OVERSOLD.value
                        return KeltnerSignal.OVERSOLD.value
                    else:
                        return KeltnerSignal.NEUTRAL.value

                except Exception:
                    return KeltnerSignal.NEUTRAL.value

            signals = []
            for row in data.iter_rows(named=True):
                signals.append(determine_signal(row))

            result = data.with_columns([
                pl.Series("signal", signals)
            ])

            # Add signal strength
            signal_strength_map = {
                KeltnerSignal.STRONG_OVERBOUGHT.value: Decimal("2"),
                KeltnerSignal.OVERBOUGHT.value: Decimal("1"),
                KeltnerSignal.NEUTRAL.value: Decimal("0"),
                KeltnerSignal.OVERSOLD.value: Decimal("-1"),
                KeltnerSignal.STRONG_OVERSOLD.value: Decimal("-2")
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

        # Check for null values
        for col in ["open", "high", "low", "close"]:
            if data[col].null_count() > 0:
                raise ValueError(f"Data contains null values in '{col}' column")

    async def get_current_values(
        self,
        data: pl.DataFrame
    ) -> Optional[KeltnerChannelValues]:
        """
        Get current Keltner Channel values.

        Args:
            data: DataFrame with calculated Keltner Channels

        Returns:
            KeltnerChannelValues dataclass or None
        """
        try:
            if len(data) == 0:
                return None

            last_row = data.row(-1, named=True)

            required = ["upper_band", "middle_band", "lower_band", "atr", "bandwidth", "percent_b"]
            if any(last_row.get(k) is None for k in required):
                return None

            return KeltnerChannelValues(
                upper_band=Decimal(str(last_row["upper_band"])),
                middle_band=Decimal(str(last_row["middle_band"])),
                lower_band=Decimal(str(last_row["lower_band"])),
                atr=Decimal(str(last_row["atr"])),
                bandwidth=Decimal(str(last_row["bandwidth"])),
                percent_b=Decimal(str(last_row["percent_b"]))
            )

        except Exception as e:
            logger.error("Failed to get current values", error=str(e))
            return None


async def create_keltner_channels_indicator(
    config: Dict[str, Any]
) -> KeltnerChannelsIndicator:
    """
    Factory function to create Keltner Channels indicator instance.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized Keltner Channels indicator
    """
    return KeltnerChannelsIndicator(config)
