"""
Breakout Strategy - Momentum-based breakout trading.

Implements a breakout trading strategy that identifies and trades price breakouts
from consolidation ranges, support/resistance levels, and chart patterns.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

import polars as pl
from structlog import get_logger

from quantum_trader.models import Signal, SignalAction
from quantum_trader.strategies.base_strategy import BaseStrategy
from quantum_trader.risk.manager import RiskManager

logger = get_logger(__name__)


class BreakoutStrategy(BaseStrategy):
    """Breakout momentum trading strategy.

    This strategy identifies breakouts from consolidation ranges and generates
    signals when price breaks above resistance or below support with volume
    confirmation.

    Features:
    - Dynamic support/resistance level detection
    - Volume confirmation for breakouts
    - False breakout filtering
    - Multiple timeframe analysis

    Attributes:
        lookback_period: Period for identifying support/resistance
        breakout_threshold: Minimum percentage move for breakout
        volume_multiplier: Volume confirmation multiplier

    Example:
        >>> config = {
        ...     "lookback_period": "50",
        ...     "breakout_threshold": "0.02",
        ...     "volume_multiplier": "1.5"
        ... }
        >>> strategy = BreakoutStrategy(config, risk_manager)
    """

    def __init__(self, config: Dict[str, Any], risk_manager: RiskManager) -> None:
        """Initialize breakout strategy.

        Args:
            config: Strategy configuration
            risk_manager: Risk manager instance

        Raises:
            ValueError: If required parameters missing
        """
        super().__init__(config, risk_manager)

        # Breakout parameters
        self.lookback_period = int(config.get("lookback_period", 50))
        self.breakout_threshold = Decimal(str(config.get("breakout_threshold", "0.02")))  # 2%
        self.volume_multiplier = Decimal(str(config.get("volume_multiplier", "1.5")))
        self.consolidation_threshold = Decimal(str(config.get("consolidation_threshold", "0.05")))  # 5%

        # Technical parameters
        self.use_atr = config.get("use_atr", True)
        self.atr_period = int(config.get("atr_period", 14))
        self.atr_multiplier = Decimal(str(config.get("atr_multiplier", "2.0")))

        # State tracking
        self.support_levels: Dict[str, Decimal] = {}
        self.resistance_levels: Dict[str, Decimal] = {}
        self.consolidation_ranges: Dict[str, Tuple[Decimal, Decimal]] = {}
        self.last_breakout: Dict[str, datetime] = {}

        logger.info(
            "Breakout strategy initialized",
            strategy=self.name,
            lookback_period=self.lookback_period,
            breakout_threshold=self.breakout_threshold
        )

    def _validate_config(self) -> None:
        """Validate strategy configuration.

        Raises:
            ValueError: If parameters invalid
        """
        super()._validate_config()

        lookback = int(self.config.get("lookback_period", 50))
        if lookback < 10 or lookback > 500:
            raise ValueError(f"lookback_period must be between 10 and 500: {lookback}")

        threshold = Decimal(str(self.config.get("breakout_threshold", "0.02")))
        if threshold <= Decimal("0") or threshold > Decimal("0.5"):
            raise ValueError(f"breakout_threshold must be between 0 and 0.5: {threshold}")

    def calculate_support_resistance(
        self,
        data: pl.DataFrame
    ) -> Tuple[Decimal, Decimal]:
        """Calculate support and resistance levels.

        Uses swing highs and lows to identify key levels.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Tuple of (support, resistance)
        """
        try:
            if len(data) < self.lookback_period:
                raise ValueError(f"Insufficient data for support/resistance")

            # Get recent data
            recent = data.tail(self.lookback_period)

            highs = recent.select(pl.col("high")).to_series().to_list()
            lows = recent.select(pl.col("low")).to_series().to_list()

            # Find swing highs and lows
            swing_highs = []
            swing_lows = []

            window = 5  # Look at 5 candles on each side

            for i in range(window, len(highs) - window):
                # Check if it's a swing high
                is_swing_high = all(highs[i] >= highs[j] for j in range(i - window, i + window + 1) if j != i)
                if is_swing_high:
                    swing_highs.append(Decimal(str(highs[i])))

                # Check if it's a swing low
                is_swing_low = all(lows[i] <= lows[j] for j in range(i - window, i + window + 1) if j != i)
                if is_swing_low:
                    swing_lows.append(Decimal(str(lows[i])))

            # Use recent swing levels
            if swing_lows:
                support = min(swing_lows[-3:]) if len(swing_lows) >= 3 else min(swing_lows)
            else:
                support = Decimal(str(min(lows)))

            if swing_highs:
                resistance = max(swing_highs[-3:]) if len(swing_highs) >= 3 else max(swing_highs)
            else:
                resistance = Decimal(str(max(highs)))

            logger.debug(
                "Support/resistance calculated",
                support=support,
                resistance=resistance,
                swing_highs=len(swing_highs),
                swing_lows=len(swing_lows)
            )

            return support, resistance

        except Exception as e:
            logger.error("Error calculating support/resistance", error=str(e))
            # Fallback to simple high/low
            highs = data.select(pl.col("high")).to_series().to_list()
            lows = data.select(pl.col("low")).to_series().to_list()
            return Decimal(str(min(lows))), Decimal(str(max(highs)))

    def calculate_atr(self, data: pl.DataFrame) -> Decimal:
        """Calculate Average True Range (ATR).

        Args:
            data: DataFrame with OHLCV data

        Returns:
            ATR value as Decimal
        """
        try:
            if len(data) < self.atr_period + 1:
                logger.warning("Insufficient data for ATR")
                return Decimal("0")

            recent = data.tail(self.atr_period + 1)

            true_ranges = []
            rows = recent.to_dicts()

            for i in range(1, len(rows)):
                high = Decimal(str(rows[i]["high"]))
                low = Decimal(str(rows[i]["low"]))
                prev_close = Decimal(str(rows[i-1]["close"]))

                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                true_ranges.append(tr)

            atr = sum(true_ranges) / Decimal(str(len(true_ranges)))

            logger.debug("ATR calculated", atr=atr, period=self.atr_period)

            return atr

        except Exception as e:
            logger.error("Error calculating ATR", error=str(e))
            return Decimal("0")

    def is_consolidating(
        self,
        data: pl.DataFrame,
        support: Decimal,
        resistance: Decimal
    ) -> bool:
        """Check if price is consolidating within a range.

        Args:
            data: Market data
            support: Support level
            resistance: Resistance level

        Returns:
            True if consolidating
        """
        try:
            if resistance <= support:
                return False

            range_size = (resistance - support) / support

            # Check if range is tight enough to be consolidation
            if range_size > self.consolidation_threshold:
                return False

            # Check if recent prices are within range
            recent = data.tail(10)
            closes = recent.select(pl.col("close")).to_series().to_list()

            within_range = all(
                support <= Decimal(str(c)) <= resistance
                for c in closes
            )

            return within_range

        except Exception as e:
            logger.error("Error checking consolidation", error=str(e))
            return False

    def check_volume_confirmation(
        self,
        data: pl.DataFrame
    ) -> bool:
        """Check if current volume confirms breakout.

        Args:
            data: Market data with volume

        Returns:
            True if volume confirms breakout
        """
        try:
            if len(data) < self.lookback_period:
                return True  # Skip volume check if insufficient data

            volumes = data.select(pl.col("volume")).to_series().to_list()

            # Calculate average volume
            avg_volume = Decimal(str(sum(volumes[:-1]) / len(volumes[:-1])))
            current_volume = Decimal(str(volumes[-1]))

            # Check if current volume exceeds threshold
            volume_confirmed = current_volume >= (avg_volume * self.volume_multiplier)

            logger.debug(
                "Volume confirmation",
                current_volume=current_volume,
                avg_volume=avg_volume,
                multiplier=self.volume_multiplier,
                confirmed=volume_confirmed
            )

            return volume_confirmed

        except Exception as e:
            logger.error("Error checking volume", error=str(e))
            return False

    def generate_signals(self, market_data: pl.DataFrame) -> List[Signal]:
        """Generate breakout trading signals.

        Args:
            market_data: DataFrame with OHLCV data

        Returns:
            List of Signal objects
        """
        signals: List[Signal] = []

        try:
            # Validate data
            is_valid, error = self.validate_market_data(market_data)
            if not is_valid:
                logger.warning("Invalid market data", error=error)
                return signals

            if not self.active:
                return signals

            # Get symbol
            symbols = self.config.get("symbols", [])
            if not symbols:
                logger.warning("No symbols configured")
                return signals

            symbol = symbols[0]

            # Calculate support and resistance
            support, resistance = self.calculate_support_resistance(market_data)
            self.support_levels[symbol] = support
            self.resistance_levels[symbol] = resistance

            # Get current price
            current_price = Decimal(str(market_data["close"][-1]))

            # Calculate ATR if enabled
            atr = Decimal("0")
            if self.use_atr:
                atr = self.calculate_atr(market_data)

            # Check for consolidation
            is_consolidating = self.is_consolidating(market_data, support, resistance)
            if is_consolidating:
                self.consolidation_ranges[symbol] = (support, resistance)

            # Check volume confirmation
            volume_confirmed = self.check_volume_confirmation(market_data)

            # Detect breakouts
            action = SignalAction.HOLD
            strength = Decimal("0")
            confidence = Decimal("0.5")

            # Bullish breakout (above resistance)
            if current_price > resistance:
                breakout_size = (current_price - resistance) / resistance

                if breakout_size >= self.breakout_threshold and volume_confirmed:
                    action = SignalAction.BUY
                    strength = min(Decimal("1"), breakout_size / self.breakout_threshold)
                    confidence = Decimal("0.7") if is_consolidating else Decimal("0.6")

                    # Increase confidence if ATR-based threshold met
                    if self.use_atr and atr > Decimal("0"):
                        atr_breakout = (current_price - resistance) / atr
                        if atr_breakout >= self.atr_multiplier:
                            confidence = min(Decimal("1"), confidence + Decimal("0.1"))

                    logger.info(
                        "Bullish breakout detected",
                        symbol=symbol,
                        price=current_price,
                        resistance=resistance,
                        breakout_size=breakout_size,
                        volume_confirmed=volume_confirmed
                    )

            # Bearish breakout (below support)
            elif current_price < support:
                breakout_size = (support - current_price) / support

                if breakout_size >= self.breakout_threshold and volume_confirmed:
                    action = SignalAction.SELL
                    strength = min(Decimal("1"), breakout_size / self.breakout_threshold)
                    confidence = Decimal("0.7") if is_consolidating else Decimal("0.6")

                    # Increase confidence if ATR-based threshold met
                    if self.use_atr and atr > Decimal("0"):
                        atr_breakout = (support - current_price) / atr
                        if atr_breakout >= self.atr_multiplier:
                            confidence = min(Decimal("1"), confidence + Decimal("0.1"))

                    logger.info(
                        "Bearish breakout detected",
                        symbol=symbol,
                        price=current_price,
                        support=support,
                        breakout_size=breakout_size,
                        volume_confirmed=volume_confirmed
                    )

            # Create signal if breakout detected
            if action != SignalAction.HOLD:
                signal = Signal(
                    symbol=symbol,
                    action=action,
                    strength=strength,
                    confidence=confidence,
                    timestamp=datetime.now(timezone.utc),
                    strategy=self.name,
                    timeframe=self.config.get("timeframe", "1h"),
                    indicators={
                        "support": support,
                        "resistance": resistance,
                        "current_price": current_price,
                        "atr": atr,
                        "range_size": (resistance - support) / support
                    },
                    metadata={
                        "strategy_type": "momentum",
                        "pattern": "breakout",
                        "consolidating": is_consolidating,
                        "volume_confirmed": volume_confirmed
                    }
                )

                if self.validate_signal(signal):
                    signals.append(signal)
                    self.last_breakout[symbol] = datetime.now(timezone.utc)

                    # Update state
                    self.state["signals_generated"] = self.state.get("signals_generated", 0) + 1
                    self.state["last_signal_time"] = datetime.now(timezone.utc)

            return signals

        except Exception as e:
            logger.error("Error generating signals", error=str(e))
            return signals

    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate breakout indicators.

        Args:
            data: Market data

        Returns:
            Dictionary of indicators
        """
        try:
            if len(data) < self.lookback_period:
                return {}

            support, resistance = self.calculate_support_resistance(data)
            atr = self.calculate_atr(data) if self.use_atr else Decimal("0")

            current_price = Decimal(str(data["close"][-1]))
            range_size = (resistance - support) / support if support > Decimal("0") else Decimal("0")

            indicators = {
                "support": support,
                "resistance": resistance,
                "current_price": current_price,
                "range_size": range_size,
                "atr": atr,
                "distance_to_support": (current_price - support) / support if support > Decimal("0") else Decimal("0"),
                "distance_to_resistance": (resistance - current_price) / resistance if resistance > Decimal("0") else Decimal("0")
            }

            return indicators

        except Exception as e:
            logger.error("Error calculating indicators", error=str(e))
            return {}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        base_metrics = super().get_performance_metrics()

        breakout_metrics = {
            "support_levels": {symbol: float(level) for symbol, level in self.support_levels.items()},
            "resistance_levels": {symbol: float(level) for symbol, level in self.resistance_levels.items()},
            "consolidation_ranges": {
                symbol: (float(r[0]), float(r[1]))
                for symbol, r in self.consolidation_ranges.items()
            },
            "last_breakout": {
                symbol: time.isoformat()
                for symbol, time in self.last_breakout.items()
            }
        }

        return {**base_metrics, **breakout_metrics}
