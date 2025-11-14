"""
MFI (Money Flow Index) Indicator - Volume-weighted RSI.

The Money Flow Index uses price and volume to identify overbought/oversold conditions.
It's more responsive than RSI alone as it incorporates volume data.
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


class MFISignal(Enum):
    """MFI trading signals."""
    STRONG_OVERBOUGHT = "strong_overbought"
    OVERBOUGHT = "overbought"
    NEUTRAL = "neutral"
    OVERSOLD = "oversold"
    STRONG_OVERSOLD = "strong_oversold"


@dataclass
class MFIValues:
    """MFI indicator values."""
    mfi: Decimal
    positive_flow: Decimal
    negative_flow: Decimal
    money_ratio: Decimal


class MFIIndicator:
    """
    Money Flow Index (MFI) indicator.

    MFI is calculated as:
    1. Typical Price = (High + Low + Close) / 3
    2. Money Flow = Typical Price × Volume
    3. Positive/Negative Money Flow based on price direction
    4. Money Ratio = Positive MF / Negative MF
    5. MFI = 100 - (100 / (1 + Money Ratio))

    Attributes:
        config: Configuration dictionary
        period: MFI calculation period (typically 14)
        overbought_level: Overbought threshold (typically 80)
        oversold_level: Oversold threshold (typically 20)

    Example:
        >>> config = {
        ...     "period": 14,
        ...     "overbought_level": "80",
        ...     "oversold_level": "20"
        ... }
        >>> mfi = MFIIndicator(config)
        >>> result = await mfi.calculate(ohlcv_data)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize MFI indicator.

        Args:
            config: Configuration dictionary containing:
                - period: MFI period
                - overbought_level: Overbought threshold
                - oversold_level: Oversold threshold
                - strong_overbought_level: Strong overbought threshold
                - strong_oversold_level: Strong oversold threshold
                - min_periods: Minimum periods required

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.period: int = int(config["period"])
        self.overbought_level: Decimal = Decimal(str(config.get("overbought_level", "80")))
        self.oversold_level: Decimal = Decimal(str(config.get("oversold_level", "20")))
        self.strong_overbought_level: Decimal = Decimal(str(
            config.get("strong_overbought_level", "90")
        ))
        self.strong_oversold_level: Decimal = Decimal(str(
            config.get("strong_oversold_level", "10")
        ))
        self.min_periods: int = int(config.get("min_periods", self.period + 1))

        logger.info(
            "MFI indicator initialized",
            period=self.period,
            overbought=str(self.overbought_level),
            oversold=str(self.oversold_level)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_fields = ["period"]

        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")

        if int(self.config["period"]) < 1:
            raise ValueError("period must be positive")

        # Validate threshold levels
        for field in ["overbought_level", "oversold_level"]:
            if field in self.config:
                level = Decimal(str(self.config[field]))
                if level < Decimal("0") or level > Decimal("100"):
                    raise ValueError(f"{field} must be between 0 and 100")

        logger.debug("MFI configuration validated")

    async def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Money Flow Index.

        Args:
            data: Polars DataFrame with columns: timestamp, high, low, close, volume

        Returns:
            DataFrame with MFI values and signals

        Raises:
            ValueError: If data is invalid or insufficient
        """
        try:
            # Validate input data
            self._validate_data(data)

            # Calculate typical price
            result = await self._calculate_typical_price(data)

            # Calculate raw money flow
            result = await self._calculate_money_flow(result)

            # Determine positive/negative flows
            result = await self._classify_money_flows(result)

            # Calculate MFI
            result = await self._calculate_mfi(result)

            # Generate signals
            result = await self._generate_signals(result)

            # Detect divergences
            result = await self._detect_divergences(result)

            logger.info(
                "MFI calculated",
                rows=len(result)
            )

            return result

        except Exception as e:
            logger.error("MFI calculation failed", error=str(e))
            raise

    async def _calculate_typical_price(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate typical price: (High + Low + Close) / 3

        Args:
            data: Input OHLCV data

        Returns:
            DataFrame with typical_price column
        """
        try:
            def calc_tp(row):
                high = Decimal(str(row["high"]))
                low = Decimal(str(row["low"]))
                close = Decimal(str(row["close"]))

                tp = (high + low + close) / Decimal("3")
                return str(tp)

            typical_prices = []
            for row in data.iter_rows(named=True):
                typical_prices.append(calc_tp(row))

            result = data.with_columns([
                pl.Series("typical_price", typical_prices)
            ])

            return result

        except Exception as e:
            logger.error("Typical price calculation failed", error=str(e))
            raise

    async def _calculate_money_flow(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate raw money flow: Typical Price × Volume

        Args:
            data: DataFrame with typical_price

        Returns:
            DataFrame with money_flow column
        """
        try:
            def calc_mf(row):
                tp = Decimal(str(row["typical_price"]))
                volume = Decimal(str(row["volume"]))

                mf = tp * volume
                return str(mf)

            money_flows = []
            for row in data.iter_rows(named=True):
                money_flows.append(calc_mf(row))

            result = data.with_columns([
                pl.Series("money_flow", money_flows)
            ])

            return result

        except Exception as e:
            logger.error("Money flow calculation failed", error=str(e))
            raise

    async def _classify_money_flows(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Classify money flows as positive or negative based on price direction.

        Args:
            data: DataFrame with typical_price and money_flow

        Returns:
            DataFrame with positive_flow and negative_flow columns
        """
        try:
            positive_flows = []
            negative_flows = []

            typical_prices = data["typical_price"].to_list()
            money_flows = data["money_flow"].to_list()

            for i in range(len(data)):
                if i == 0:
                    # First row: neutral
                    positive_flows.append("0")
                    negative_flows.append("0")
                    continue

                curr_tp = Decimal(str(typical_prices[i]))
                prev_tp = Decimal(str(typical_prices[i-1]))
                mf = Decimal(str(money_flows[i]))

                if curr_tp > prev_tp:
                    # Positive money flow
                    positive_flows.append(str(mf))
                    negative_flows.append("0")
                elif curr_tp < prev_tp:
                    # Negative money flow
                    positive_flows.append("0")
                    negative_flows.append(str(mf))
                else:
                    # No change: neutral
                    positive_flows.append("0")
                    negative_flows.append("0")

            result = data.with_columns([
                pl.Series("positive_flow", positive_flows),
                pl.Series("negative_flow", negative_flows)
            ])

            return result

        except Exception as e:
            logger.error("Money flow classification failed", error=str(e))
            raise

    async def _calculate_mfi(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Money Flow Index.

        MFI = 100 - (100 / (1 + Money Ratio))
        Money Ratio = Sum(Positive Flow, period) / Sum(Negative Flow, period)

        Args:
            data: DataFrame with positive_flow and negative_flow

        Returns:
            DataFrame with mfi, money_ratio columns
        """
        try:
            mfi_values = []
            money_ratios = []

            positive_flows = [Decimal(str(x)) for x in data["positive_flow"].to_list()]
            negative_flows = [Decimal(str(x)) for x in data["negative_flow"].to_list()]

            for i in range(len(data)):
                if i < self.period:
                    # Not enough data
                    mfi_values.append(None)
                    money_ratios.append(None)
                    continue

                # Sum positive and negative flows over period
                start_idx = i - self.period + 1
                positive_sum = sum(positive_flows[start_idx:i+1])
                negative_sum = sum(negative_flows[start_idx:i+1])

                # Calculate money ratio
                if negative_sum == Decimal("0"):
                    # All positive flow: MFI = 100
                    money_ratio = Decimal("999999999")  # Very large number
                    mfi = Decimal("100")
                else:
                    money_ratio = positive_sum / negative_sum
                    mfi = Decimal("100") - (Decimal("100") / (Decimal("1") + money_ratio))

                mfi_values.append(str(mfi))
                money_ratios.append(str(money_ratio))

            result = data.with_columns([
                pl.Series("mfi", mfi_values),
                pl.Series("money_ratio", money_ratios)
            ])

            return result

        except Exception as e:
            logger.error("MFI calculation failed", error=str(e))
            raise

    async def _generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Generate trading signals based on MFI levels.

        Signal logic:
        - STRONG_OVERBOUGHT: MFI > strong_overbought_level
        - OVERBOUGHT: MFI > overbought_level
        - NEUTRAL: Between oversold and overbought
        - OVERSOLD: MFI < oversold_level
        - STRONG_OVERSOLD: MFI < strong_oversold_level

        Args:
            data: DataFrame with MFI values

        Returns:
            DataFrame with signal column
        """
        try:
            def determine_signal(mfi_val) -> str:
                try:
                    if mfi_val is None:
                        return MFISignal.NEUTRAL.value

                    mfi = Decimal(str(mfi_val))

                    if mfi >= self.strong_overbought_level:
                        return MFISignal.STRONG_OVERBOUGHT.value
                    elif mfi >= self.overbought_level:
                        return MFISignal.OVERBOUGHT.value
                    elif mfi <= self.strong_oversold_level:
                        return MFISignal.STRONG_OVERSOLD.value
                    elif mfi <= self.oversold_level:
                        return MFISignal.OVERSOLD.value
                    else:
                        return MFISignal.NEUTRAL.value

                except Exception:
                    return MFISignal.NEUTRAL.value

            signals = []
            for mfi in data["mfi"].to_list():
                signals.append(determine_signal(mfi))

            result = data.with_columns([
                pl.Series("signal", signals)
            ])

            # Add signal strength
            signal_strength_map = {
                MFISignal.STRONG_OVERBOUGHT.value: Decimal("2"),
                MFISignal.OVERBOUGHT.value: Decimal("1"),
                MFISignal.NEUTRAL.value: Decimal("0"),
                MFISignal.OVERSOLD.value: Decimal("-1"),
                MFISignal.STRONG_OVERSOLD.value: Decimal("-2")
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
        Detect bullish and bearish divergences between price and MFI.

        Bullish divergence: Price makes lower low, MFI makes higher low
        Bearish divergence: Price makes higher high, MFI makes lower high

        Args:
            data: DataFrame with MFI and price data

        Returns:
            DataFrame with divergence columns
        """
        try:
            lookback = min(self.period * 2, 50)  # Lookback for finding peaks/troughs

            bullish_div = []
            bearish_div = []

            closes = data["close"].to_list()
            mfi_values = data["mfi"].to_list()

            for i in range(len(data)):
                if i < lookback or mfi_values[i] is None:
                    bullish_div.append(False)
                    bearish_div.append(False)
                    continue

                # Look for divergences in recent data
                recent_closes = [Decimal(str(c)) for c in closes[i-lookback:i+1]]
                recent_mfis = [
                    Decimal(str(m)) if m is not None else None
                    for m in mfi_values[i-lookback:i+1]
                ]

                # Filter out None values
                valid_indices = [
                    idx for idx, mfi in enumerate(recent_mfis)
                    if mfi is not None
                ]

                if len(valid_indices) < 4:
                    bullish_div.append(False)
                    bearish_div.append(False)
                    continue

                # Find local minima/maxima
                has_bullish = False
                has_bearish = False

                # Simple divergence detection: compare current vs lookback/2
                mid_idx = lookback // 2
                if mid_idx < len(recent_closes) and recent_mfis[mid_idx] is not None:
                    # Bullish divergence check
                    if (recent_closes[-1] < recent_closes[mid_idx] and
                        recent_mfis[-1] > recent_mfis[mid_idx]):
                        has_bullish = True

                    # Bearish divergence check
                    if (recent_closes[-1] > recent_closes[mid_idx] and
                        recent_mfis[-1] < recent_mfis[mid_idx]):
                        has_bearish = True

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
        required_columns = ["timestamp", "high", "low", "close", "volume"]

        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        if len(data) < self.min_periods:
            raise ValueError(
                f"Insufficient data points: {len(data)} < {self.min_periods}"
            )

        # Check for null values
        for col in ["high", "low", "close", "volume"]:
            if data[col].null_count() > 0:
                raise ValueError(f"Data contains null values in '{col}' column")

    async def get_current_values(
        self,
        data: pl.DataFrame
    ) -> Optional[MFIValues]:
        """
        Get current MFI values.

        Args:
            data: DataFrame with calculated MFI values

        Returns:
            MFIValues dataclass or None
        """
        try:
            if len(data) == 0:
                return None

            last_row = data.row(-1, named=True)

            required = ["mfi", "positive_flow", "negative_flow", "money_ratio"]
            if any(last_row.get(k) is None for k in required):
                return None

            # Get sum of recent flows
            positive_flows = data["positive_flow"].tail(self.period).to_list()
            negative_flows = data["negative_flow"].tail(self.period).to_list()

            pos_sum = sum(Decimal(str(x)) for x in positive_flows)
            neg_sum = sum(Decimal(str(x)) for x in negative_flows)

            return MFIValues(
                mfi=Decimal(str(last_row["mfi"])),
                positive_flow=pos_sum,
                negative_flow=neg_sum,
                money_ratio=Decimal(str(last_row["money_ratio"]))
            )

        except Exception as e:
            logger.error("Failed to get current values", error=str(e))
            return None


async def create_mfi_indicator(config: Dict[str, Any]) -> MFIIndicator:
    """
    Factory function to create MFI indicator instance.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized MFI indicator
    """
    return MFIIndicator(config)
