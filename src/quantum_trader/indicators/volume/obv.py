"""
OBV (On-Balance Volume) Indicator - Volume-based momentum indicator.

On-Balance Volume uses volume flow to predict changes in price.
It's a cumulative indicator that adds volume on up days and subtracts on down days.
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


class OBVSignal(Enum):
    """OBV trading signals."""
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


@dataclass
class OBVValues:
    """OBV indicator values."""
    obv: Decimal
    obv_ema: Optional[Decimal]
    obv_trend: str


class OBVIndicator:
    """
    On-Balance Volume (OBV) indicator.

    OBV is calculated as:
    - If close > prev_close: OBV = prev_OBV + volume
    - If close < prev_close: OBV = prev_OBV - volume
    - If close == prev_close: OBV = prev_OBV

    Attributes:
        config: Configuration dictionary
        ema_period: Period for OBV EMA smoothing
        divergence_lookback: Lookback period for divergence detection

    Example:
        >>> config = {
        ...     "ema_period": 20,
        ...     "divergence_lookback": 50,
        ...     "enable_ema": True
        ... }
        >>> obv = OBVIndicator(config)
        >>> result = await obv.calculate(ohlcv_data)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize OBV indicator.

        Args:
            config: Configuration dictionary containing:
                - ema_period: Period for EMA smoothing (optional)
                - divergence_lookback: Lookback for divergence detection
                - enable_ema: Whether to calculate OBV EMA
                - min_periods: Minimum periods required

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.ema_period: int = int(config.get("ema_period", 20))
        self.divergence_lookback: int = int(config.get("divergence_lookback", 50))
        self.enable_ema: bool = config.get("enable_ema", True)
        self.min_periods: int = int(config.get("min_periods", 2))

        logger.info(
            "OBV indicator initialized",
            ema_period=self.ema_period,
            enable_ema=self.enable_ema
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if "ema_period" in self.config:
            if int(self.config["ema_period"]) < 1:
                raise ValueError("ema_period must be positive")

        if "divergence_lookback" in self.config:
            if int(self.config["divergence_lookback"]) < 1:
                raise ValueError("divergence_lookback must be positive")

        logger.debug("OBV configuration validated")

    async def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate On-Balance Volume.

        Args:
            data: Polars DataFrame with columns: timestamp, close, volume

        Returns:
            DataFrame with OBV values and signals

        Raises:
            ValueError: If data is invalid or insufficient
        """
        try:
            # Validate input data
            self._validate_data(data)

            # Calculate OBV
            result = await self._calculate_obv(data)

            # Calculate OBV EMA if enabled
            if self.enable_ema:
                result = await self._calculate_obv_ema(result)

            # Generate trend signals
            result = await self._generate_signals(result)

            # Detect divergences
            result = await self._detect_divergences(result)

            logger.info(
                "OBV calculated",
                rows=len(result)
            )

            return result

        except Exception as e:
            logger.error("OBV calculation failed", error=str(e))
            raise

    async def _calculate_obv(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate On-Balance Volume.

        Args:
            data: Input OHLCV data

        Returns:
            DataFrame with obv column
        """
        try:
            obv_values = []

            closes = [Decimal(str(x)) for x in data["close"].to_list()]
            volumes = [Decimal(str(x)) for x in data["volume"].to_list()]

            # Initialize OBV
            obv = Decimal("0")
            obv_values.append(str(obv))

            # Calculate OBV
            for i in range(1, len(closes)):
                curr_close = closes[i]
                prev_close = closes[i-1]
                volume = volumes[i]

                if curr_close > prev_close:
                    # Up day: add volume
                    obv += volume
                elif curr_close < prev_close:
                    # Down day: subtract volume
                    obv -= volume
                # else: no change in OBV

                obv_values.append(str(obv))

            result = data.with_columns([
                pl.Series("obv", obv_values)
            ])

            return result

        except Exception as e:
            logger.error("OBV calculation failed", error=str(e))
            raise

    async def _calculate_obv_ema(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate EMA of OBV for smoothing.

        Args:
            data: DataFrame with OBV

        Returns:
            DataFrame with obv_ema column
        """
        try:
            obv_values = [Decimal(str(x)) for x in data["obv"].to_list()]

            alpha = Decimal("2") / (Decimal(str(self.ema_period)) + Decimal("1"))
            ema_values = []

            if len(obv_values) < self.ema_period:
                result = data.with_columns([
                    pl.Series("obv_ema", [None] * len(obv_values))
                ])
                return result

            # Calculate initial SMA
            initial_sma = sum(obv_values[:self.ema_period]) / Decimal(str(self.ema_period))

            # Fill initial values with None
            ema_values = [None] * (self.ema_period - 1)
            ema_values.append(str(initial_sma))

            # Calculate EMA for remaining values
            ema = initial_sma
            for i in range(self.ema_period, len(obv_values)):
                ema = alpha * obv_values[i] + (Decimal("1") - alpha) * ema
                ema_values.append(str(ema))

            result = data.with_columns([
                pl.Series("obv_ema", ema_values)
            ])

            return result

        except Exception as e:
            logger.error("OBV EMA calculation failed", error=str(e))
            raise

    async def _generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Generate trading signals based on OBV.

        Signal logic:
        - OBV rising + price rising = BULLISH
        - OBV rising + OBV > OBV_EMA = STRONG_BULLISH
        - OBV falling + price falling = BEARISH
        - OBV falling + OBV < OBV_EMA = STRONG_BEARISH
        - Otherwise = NEUTRAL

        Args:
            data: DataFrame with OBV values

        Returns:
            DataFrame with signal column
        """
        try:
            signals = []

            obv_values = data["obv"].to_list()
            closes = data["close"].to_list()

            # Get OBV EMA if available
            has_ema = "obv_ema" in data.columns
            if has_ema:
                obv_ema_values = data["obv_ema"].to_list()
            else:
                obv_ema_values = [None] * len(obv_values)

            for i in range(len(data)):
                if i < 5:  # Need some history
                    signals.append(OBVSignal.NEUTRAL.value)
                    continue

                try:
                    # Get recent OBV and price trends
                    obv_curr = Decimal(str(obv_values[i]))
                    obv_prev = Decimal(str(obv_values[i-5]))

                    price_curr = Decimal(str(closes[i]))
                    price_prev = Decimal(str(closes[i-5]))

                    obv_trend = obv_curr - obv_prev
                    price_trend = price_curr - price_prev

                    # Check OBV vs EMA
                    above_ema = False
                    below_ema = False

                    if has_ema and obv_ema_values[i] is not None:
                        obv_ema = Decimal(str(obv_ema_values[i]))
                        above_ema = obv_curr > obv_ema
                        below_ema = obv_curr < obv_ema

                    # Determine signal
                    if obv_trend > Decimal("0") and price_trend > Decimal("0"):
                        # Both rising
                        if above_ema:
                            signals.append(OBVSignal.STRONG_BULLISH.value)
                        else:
                            signals.append(OBVSignal.BULLISH.value)
                    elif obv_trend < Decimal("0") and price_trend < Decimal("0"):
                        # Both falling
                        if below_ema:
                            signals.append(OBVSignal.STRONG_BEARISH.value)
                        else:
                            signals.append(OBVSignal.BEARISH.value)
                    else:
                        signals.append(OBVSignal.NEUTRAL.value)

                except Exception:
                    signals.append(OBVSignal.NEUTRAL.value)

            result = data.with_columns([
                pl.Series("signal", signals)
            ])

            # Add signal strength
            signal_strength_map = {
                OBVSignal.STRONG_BULLISH.value: Decimal("2"),
                OBVSignal.BULLISH.value: Decimal("1"),
                OBVSignal.NEUTRAL.value: Decimal("0"),
                OBVSignal.BEARISH.value: Decimal("-1"),
                OBVSignal.STRONG_BEARISH.value: Decimal("-2")
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

    async def _detect_divergences(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Detect bullish and bearish divergences between price and OBV.

        Bullish divergence: Price makes lower low, OBV makes higher low
        Bearish divergence: Price makes higher high, OBV makes lower high

        Args:
            data: DataFrame with OBV and price data

        Returns:
            DataFrame with divergence columns
        """
        try:
            bullish_div = []
            bearish_div = []

            closes = data["close"].to_list()
            obv_values = data["obv"].to_list()

            for i in range(len(data)):
                if i < self.divergence_lookback:
                    bullish_div.append(False)
                    bearish_div.append(False)
                    continue

                # Look for divergences in recent data
                lookback_start = i - self.divergence_lookback
                recent_closes = [Decimal(str(c)) for c in closes[lookback_start:i+1]]
                recent_obvs = [Decimal(str(o)) for o in obv_values[lookback_start:i+1]]

                # Simple divergence detection: compare current vs mid-point
                mid_idx = self.divergence_lookback // 2

                has_bullish = False
                has_bearish = False

                try:
                    # Bullish divergence: lower price low, higher OBV low
                    if (recent_closes[-1] < recent_closes[mid_idx] and
                        recent_obvs[-1] > recent_obvs[mid_idx]):
                        has_bullish = True

                    # Bearish divergence: higher price high, lower OBV high
                    if (recent_closes[-1] > recent_closes[mid_idx] and
                        recent_obvs[-1] < recent_obvs[mid_idx]):
                        has_bearish = True

                except Exception:
                    pass

                bullish_div.append(has_bullish)
                bearish_div.append(has_bearish)

            result = data.with_columns([
                pl.Series("bullish_divergence", bullish_div),
                pl.Series("bearish_divergence", bearish_div)
            ])

            return result

        except Exception as e:
            logger.error("Divergence detection failed", error=str(e))
            raise

    def _validate_data(self, data: pl.DataFrame) -> None:
        """Validate input data format and content."""
        required_columns = ["timestamp", "close", "volume"]

        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        if len(data) < self.min_periods:
            raise ValueError(
                f"Insufficient data points: {len(data)} < {self.min_periods}"
            )

        # Check for null values
        for col in ["close", "volume"]:
            if data[col].null_count() > 0:
                raise ValueError(f"Data contains null values in '{col}' column")

    async def get_current_values(
        self,
        data: pl.DataFrame
    ) -> Optional[OBVValues]:
        """
        Get current OBV values.

        Args:
            data: DataFrame with calculated OBV values

        Returns:
            OBVValues dataclass or None
        """
        try:
            if len(data) == 0:
                return None

            last_row = data.row(-1, named=True)

            if last_row.get("obv") is None:
                return None

            obv_ema = None
            if "obv_ema" in last_row and last_row["obv_ema"] is not None:
                obv_ema = Decimal(str(last_row["obv_ema"]))

            return OBVValues(
                obv=Decimal(str(last_row["obv"])),
                obv_ema=obv_ema,
                obv_trend=last_row.get("signal", OBVSignal.NEUTRAL.value)
            )

        except Exception as e:
            logger.error("Failed to get current values", error=str(e))
            return None

    async def calculate_obv_momentum(
        self,
        data: pl.DataFrame,
        period: int = 10
    ) -> pl.DataFrame:
        """
        Calculate OBV momentum (rate of change).

        Args:
            data: DataFrame with OBV values
            period: Period for momentum calculation

        Returns:
            DataFrame with obv_momentum column
        """
        try:
            if "obv" not in data.columns:
                raise ValueError("OBV must be calculated first")

            momentum_values = []
            obv_values = [Decimal(str(x)) for x in data["obv"].to_list()]

            for i in range(len(obv_values)):
                if i < period:
                    momentum_values.append(None)
                    continue

                current_obv = obv_values[i]
                past_obv = obv_values[i - period]

                if past_obv == Decimal("0"):
                    momentum = Decimal("0")
                else:
                    momentum = ((current_obv - past_obv) / abs(past_obv)) * Decimal("100")

                momentum_values.append(str(momentum))

            result = data.with_columns([
                pl.Series("obv_momentum", momentum_values)
            ])

            logger.debug("OBV momentum calculated", period=period)

            return result

        except Exception as e:
            logger.error("OBV momentum calculation failed", error=str(e))
            raise


async def create_obv_indicator(config: Dict[str, Any]) -> OBVIndicator:
    """
    Factory function to create OBV indicator instance.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized OBV indicator
    """
    return OBVIndicator(config)
