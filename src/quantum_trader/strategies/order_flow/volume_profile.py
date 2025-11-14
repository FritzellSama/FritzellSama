"""Volume Profile Order Flow Strategy.

Analyzes volume distribution at price levels to identify high-value areas,
institutional activity, and optimal entry/exit points based on order flow dynamics.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import polars as pl
from structlog import get_logger

from quantum_trader.models import Signal, SignalAction, Order, OrderSide, OrderType
from quantum_trader.strategies.base_strategy import BaseStrategy
from quantum_trader.risk.manager import RiskManager

logger = get_logger(__name__)


class VolumeProfileStrategy(BaseStrategy):
    """Volume Profile strategy using order flow analysis.

    Identifies key price levels based on volume distribution:
    - Point of Control (POC): Price level with highest volume
    - Value Area: Price range containing majority of volume
    - High/Low Volume Nodes: Support/resistance based on volume

    Attributes:
        config: Strategy configuration parameters
        risk_manager: Risk management instance
        name: Strategy identifier
        value_area_pct: Percentage of volume in value area from config
        num_price_levels: Number of price buckets from config
        min_volume_imbalance: Min imbalance for signal from config
        poc_zone_pct: POC zone width percentage from config
    """

    def __init__(self, config: Dict[str, Any], risk_manager: RiskManager) -> None:
        """Initialize volume profile strategy.

        Args:
            config: Configuration dictionary with strategy parameters
            risk_manager: Risk manager instance for validation

        Raises:
            ValueError: If required config parameters missing
        """
        super().__init__(config, risk_manager)
        self.name = "VolumeProfile"

        self._validate_config()

        # Load parameters from config
        self.value_area_pct = Decimal(str(config["indicators"]["value_area_pct"]))
        self.num_price_levels = config["indicators"]["num_price_levels"]
        self.min_volume_imbalance = Decimal(str(config["thresholds"]["min_volume_imbalance"]))
        self.poc_zone_pct = Decimal(str(config["thresholds"]["poc_zone_pct"]))
        self.min_confidence = Decimal(str(config["signal"]["min_confidence"]))
        self.lookback_bars = config.get("indicators", {}).get("lookback_bars", 100)

        logger.info(
            "volume_profile_initialized",
            value_area_pct=str(self.value_area_pct),
            num_levels=self.num_price_levels,
            min_imbalance=str(self.min_volume_imbalance)
        )

    def _validate_config(self) -> None:
        """Validate required configuration parameters.

        Raises:
            ValueError: If required parameters missing or invalid
        """
        required_keys = ["indicators", "thresholds", "signal"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        required_indicators = ["value_area_pct", "num_price_levels"]
        for key in required_indicators:
            if key not in self.config["indicators"]:
                raise ValueError(f"Missing required indicator config: {key}")

        required_thresholds = ["min_volume_imbalance", "poc_zone_pct"]
        for key in required_thresholds:
            if key not in self.config["thresholds"]:
                raise ValueError(f"Missing required threshold config: {key}")

        if "min_confidence" not in self.config["signal"]:
            raise ValueError("Missing required signal.min_confidence")

    def generate_signals(self, market_data: pl.DataFrame) -> List[Signal]:
        """Generate signals based on volume profile analysis.

        Identifies trading opportunities when price interacts with key
        volume levels (POC, value area boundaries, HVN/LVN).

        Args:
            market_data: Polars DataFrame with OHLCV data

        Returns:
            List of Signal objects for detected opportunities

        Raises:
            ValueError: If market data invalid or insufficient
        """
        try:
            if market_data.height < max(self.lookback_bars, 50):
                logger.warning(
                    "insufficient_data",
                    rows=market_data.height,
                    required=max(self.lookback_bars, 50)
                )
                return []

            # Calculate volume profile indicators
            indicators = self.calculate_indicators(market_data)

            signals = []

            # Get latest values
            current_price = indicators["current_price"]
            poc = indicators["poc"]
            value_area_high = indicators["value_area_high"]
            value_area_low = indicators["value_area_low"]
            volume_imbalance = indicators["volume_imbalance"]
            delta_volume = indicators["delta_volume"]

            symbol = market_data["symbol"][0] if "symbol" in market_data.columns else "UNKNOWN"
            timeframe = market_data["timeframe"][0] if "timeframe" in market_data.columns else "5m"

            # Calculate position relative to value area
            in_value_area = value_area_low <= current_price <= value_area_high
            near_poc = abs(current_price - poc) / poc <= (self.poc_zone_pct / Decimal("100"))

            # Bullish signal: Price below value area with positive delta
            if current_price < value_area_low and delta_volume > self.min_volume_imbalance:
                strength = self._calculate_signal_strength(
                    current_price, value_area_low, poc, delta_volume, "BUY"
                )
                confidence = self._calculate_confidence(
                    volume_imbalance, delta_volume, current_price, value_area_low, poc
                )

                if confidence >= self.min_confidence:
                    signal = Signal(
                        symbol=symbol,
                        action=SignalAction.BUY,
                        strength=strength,
                        confidence=confidence,
                        timestamp=datetime.now(timezone.utc),
                        strategy=self.name,
                        timeframe=timeframe,
                        indicators={
                            "current_price": current_price,
                            "poc": poc,
                            "value_area_high": value_area_high,
                            "value_area_low": value_area_low,
                            "volume_imbalance": volume_imbalance,
                            "delta_volume": delta_volume
                        },
                        metadata={
                            "signal_type": "value_area_support",
                            "target": value_area_low,
                            "poc_target": poc,
                            "volume_quality": "positive_delta",
                            "distance_to_va_pct": ((value_area_low - current_price) / current_price * Decimal("100"))
                        }
                    )

                    if self.validate_signal(signal):
                        signals.append(signal)
                        logger.info(
                            "bullish_volume_signal",
                            symbol=symbol,
                            price=str(current_price),
                            va_low=str(value_area_low),
                            delta=str(delta_volume),
                            confidence=str(confidence)
                        )

            # Bearish signal: Price above value area with negative delta
            elif current_price > value_area_high and delta_volume < -self.min_volume_imbalance:
                strength = self._calculate_signal_strength(
                    current_price, value_area_high, poc, abs(delta_volume), "SELL"
                )
                confidence = self._calculate_confidence(
                    volume_imbalance, abs(delta_volume), current_price, value_area_high, poc
                )

                if confidence >= self.min_confidence:
                    signal = Signal(
                        symbol=symbol,
                        action=SignalAction.SELL,
                        strength=strength,
                        confidence=confidence,
                        timestamp=datetime.now(timezone.utc),
                        strategy=self.name,
                        timeframe=timeframe,
                        indicators={
                            "current_price": current_price,
                            "poc": poc,
                            "value_area_high": value_area_high,
                            "value_area_low": value_area_low,
                            "volume_imbalance": volume_imbalance,
                            "delta_volume": delta_volume
                        },
                        metadata={
                            "signal_type": "value_area_resistance",
                            "target": value_area_high,
                            "poc_target": poc,
                            "volume_quality": "negative_delta",
                            "distance_to_va_pct": ((current_price - value_area_high) / current_price * Decimal("100"))
                        }
                    )

                    if self.validate_signal(signal):
                        signals.append(signal)
                        logger.info(
                            "bearish_volume_signal",
                            symbol=symbol,
                            price=str(current_price),
                            va_high=str(value_area_high),
                            delta=str(delta_volume),
                            confidence=str(confidence)
                        )

            # POC reversion signal
            elif near_poc and abs(delta_volume) > self.min_volume_imbalance:
                # Price tends to revert to POC
                action = SignalAction.BUY if current_price < poc else SignalAction.SELL

                signal = Signal(
                    symbol=symbol,
                    action=action,
                    strength=Decimal("0.6"),
                    confidence=Decimal("0.7"),
                    timestamp=datetime.now(timezone.utc),
                    strategy=self.name,
                    timeframe=timeframe,
                    indicators={
                        "current_price": current_price,
                        "poc": poc,
                        "delta_volume": delta_volume
                    },
                    metadata={
                        "signal_type": "poc_reversion",
                        "target": poc,
                        "distance_to_poc_pct": ((poc - current_price) / current_price * Decimal("100"))
                    }
                )

                if self.validate_signal(signal):
                    signals.append(signal)
                    logger.info(
                        "poc_reversion_signal",
                        symbol=symbol,
                        price=str(current_price),
                        poc=str(poc)
                    )

            return signals

        except Exception as e:
            logger.error("signal_generation_failed", error=str(e), exc_info=True)
            raise

    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate volume profile indicators.

        Args:
            data: Polars DataFrame with OHLCV data

        Returns:
            Dictionary of calculated indicator values

        Raises:
            ValueError: If data invalid or calculation fails
        """
        try:
            # Get data for lookback period
            lookback_data = data.tail(self.lookback_bars) if data.height > self.lookback_bars else data

            high_prices = lookback_data["high"].to_numpy()
            low_prices = lookback_data["low"].to_numpy()
            close_prices = lookback_data["close"].to_numpy()
            volumes = lookback_data["volume"].to_numpy()

            # Calculate price range
            price_min = float(low_prices.min())
            price_max = float(high_prices.max())
            price_range = price_max - price_min

            # Create price levels
            price_levels = [
                price_min + (price_range * i / self.num_price_levels)
                for i in range(self.num_price_levels + 1)
            ]

            # Build volume profile
            volume_profile = self._build_volume_profile(
                high_prices, low_prices, close_prices, volumes, price_levels
            )

            # Find POC (Point of Control)
            poc_index = volume_profile.index(max(volume_profile))
            poc = Decimal(str((price_levels[poc_index] + price_levels[poc_index + 1]) / 2))

            # Calculate Value Area
            value_area_high, value_area_low = self._calculate_value_area(
                price_levels, volume_profile, self.value_area_pct
            )

            # Calculate volume imbalance and delta
            buy_volume, sell_volume = self._calculate_volume_delta(
                lookback_data["close"].to_list(),
                lookback_data["volume"].to_list()
            )

            volume_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume + Decimal("1"))
            delta_volume = buy_volume - sell_volume

            current_price = Decimal(str(close_prices[-1]))

            return {
                "current_price": current_price,
                "poc": poc,
                "value_area_high": Decimal(str(value_area_high)),
                "value_area_low": Decimal(str(value_area_low)),
                "volume_imbalance": volume_imbalance,
                "delta_volume": delta_volume,
                "buy_volume": buy_volume,
                "sell_volume": sell_volume
            }

        except Exception as e:
            logger.error("indicator_calculation_failed", error=str(e))
            raise ValueError(f"Failed to calculate indicators: {e}")

    def _build_volume_profile(
        self,
        high: Any,
        low: Any,
        close: Any,
        volume: Any,
        price_levels: List[float]
    ) -> List[float]:
        """Build volume profile histogram.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            price_levels: Price level buckets

        Returns:
            List of volume at each price level
        """
        import numpy as np

        volume_at_level = [0.0] * (len(price_levels) - 1)

        for i in range(len(high)):
            bar_high = high[i]
            bar_low = low[i]
            bar_volume = volume[i]

            # Distribute volume across price levels within bar range
            for j in range(len(price_levels) - 1):
                level_low = price_levels[j]
                level_high = price_levels[j + 1]

                # Check if this price level overlaps with the bar
                if level_low <= bar_high and level_high >= bar_low:
                    # Calculate overlap ratio
                    overlap_low = max(level_low, bar_low)
                    overlap_high = min(level_high, bar_high)
                    overlap_range = overlap_high - overlap_low

                    bar_range = bar_high - bar_low
                    if bar_range > 0:
                        volume_ratio = overlap_range / bar_range
                        volume_at_level[j] += bar_volume * volume_ratio

        return volume_at_level

    def _calculate_value_area(
        self,
        price_levels: List[float],
        volume_profile: List[float],
        value_area_pct: Decimal
    ) -> Tuple[float, float]:
        """Calculate Value Area High and Low.

        Args:
            price_levels: Price level buckets
            volume_profile: Volume at each level
            value_area_pct: Percentage of volume in value area (e.g., 0.70 for 70%)

        Returns:
            Tuple of (value_area_high, value_area_low)
        """
        import numpy as np

        total_volume = sum(volume_profile)
        target_volume = total_volume * float(value_area_pct)

        # Start from POC and expand outward
        poc_index = volume_profile.index(max(volume_profile))

        value_area_volume = volume_profile[poc_index]
        lower_index = poc_index
        upper_index = poc_index

        while value_area_volume < target_volume:
            # Expand in direction with more volume
            lower_volume = volume_profile[lower_index - 1] if lower_index > 0 else 0
            upper_volume = volume_profile[upper_index + 1] if upper_index < len(volume_profile) - 1 else 0

            if lower_volume > upper_volume and lower_index > 0:
                lower_index -= 1
                value_area_volume += lower_volume
            elif upper_index < len(volume_profile) - 1:
                upper_index += 1
                value_area_volume += upper_volume
            else:
                break

        value_area_high = price_levels[upper_index + 1]
        value_area_low = price_levels[lower_index]

        return value_area_high, value_area_low

    def _calculate_volume_delta(
        self,
        close_prices: List,
        volumes: List
    ) -> Tuple[Decimal, Decimal]:
        """Calculate buy and sell volume based on price movement.

        Args:
            close_prices: List of close prices
            volumes: List of volumes

        Returns:
            Tuple of (buy_volume, sell_volume)
        """
        buy_volume = Decimal("0")
        sell_volume = Decimal("0")

        for i in range(1, len(close_prices)):
            vol = Decimal(str(volumes[i]))

            if close_prices[i] > close_prices[i-1]:
                buy_volume += vol
            elif close_prices[i] < close_prices[i-1]:
                sell_volume += vol
            else:
                # Split evenly on unchanged price
                buy_volume += vol / Decimal("2")
                sell_volume += vol / Decimal("2")

        return buy_volume, sell_volume

    def _calculate_signal_strength(
        self,
        price: Decimal,
        reference_level: Decimal,
        poc: Decimal,
        delta: Decimal,
        direction: str
    ) -> Decimal:
        """Calculate signal strength.

        Args:
            price: Current price
            reference_level: VA high or low
            poc: Point of Control
            delta: Volume delta magnitude
            direction: "BUY" or "SELL"

        Returns:
            Signal strength between 0 and 1
        """
        # Distance from reference level
        distance = abs(price - reference_level) / reference_level

        # Closer to level = stronger mean reversion signal
        distance_strength = Decimal("1.0") - min(distance * Decimal("10"), Decimal("0.5"))

        # Volume delta strength (normalized)
        delta_strength = min(delta / Decimal("1000000"), Decimal("1.0"))

        # Combined strength
        strength = (distance_strength * Decimal("0.6") + delta_strength * Decimal("0.4"))

        return min(max(strength, Decimal("0.3")), Decimal("1.0"))

    def _calculate_confidence(
        self,
        volume_imbalance: Decimal,
        delta_volume: Decimal,
        price: Decimal,
        reference_level: Decimal,
        poc: Decimal
    ) -> Decimal:
        """Calculate signal confidence.

        Args:
            volume_imbalance: Buy/sell volume imbalance
            delta_volume: Net volume delta
            price: Current price
            reference_level: VA boundary
            poc: Point of Control

        Returns:
            Confidence level between 0 and 1
        """
        # Strong volume imbalance = higher confidence
        imbalance_conf = min(abs(volume_imbalance) * Decimal("2"), Decimal("1.0"))

        # Price position confidence
        distance_to_poc = abs(price - poc) / poc
        position_conf = Decimal("1.0") - min(distance_to_poc * Decimal("5"), Decimal("0.4"))

        # Combined confidence
        confidence = (imbalance_conf * Decimal("0.7") + position_conf * Decimal("0.3"))

        return min(max(confidence, Decimal("0.5")), Decimal("1.0"))
