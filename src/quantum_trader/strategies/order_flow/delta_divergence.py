"""
Delta Divergence Order Flow Strategy.

Implements order flow analysis based on cumulative delta divergence,
which compares buying and selling pressure to identify potential reversals.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from collections import deque

import polars as pl
from structlog import get_logger

from quantum_trader.models import Signal, SignalAction
from quantum_trader.strategies.base_strategy import BaseStrategy
from quantum_trader.risk.manager import RiskManager

logger = get_logger(__name__)


class DeltaDivergenceStrategy(BaseStrategy):
    """Delta divergence order flow strategy.

    This strategy analyzes cumulative delta (buying pressure - selling pressure)
    and identifies divergences between price action and order flow to predict
    potential reversals.

    Features:
    - Cumulative delta calculation from trades
    - Divergence detection (bullish/bearish)
    - Volume profile analysis
    - High-frequency order flow tracking

    Attributes:
        divergence_threshold: Minimum divergence for signal
        lookback_bars: Number of bars to analyze
        cumulative_delta: Running delta by symbol

    Example:
        >>> config = {
        ...     "divergence_threshold": "0.3",
        ...     "lookback_bars": "20",
        ...     "min_volume": "1000"
        ... }
        >>> strategy = DeltaDivergenceStrategy(config, risk_manager)
    """

    def __init__(self, config: Dict[str, Any], risk_manager: RiskManager) -> None:
        """Initialize delta divergence strategy.

        Args:
            config: Strategy configuration
            risk_manager: Risk manager instance

        Raises:
            ValueError: If required parameters missing
        """
        super().__init__(config, risk_manager)

        # Delta parameters
        self.divergence_threshold = Decimal(str(config.get("divergence_threshold", "0.3")))
        self.lookback_bars = int(config.get("lookback_bars", 20))
        self.min_volume = Decimal(str(config.get("min_volume", "1000")))

        # Delta tracking
        self.cumulative_delta: Dict[str, Decimal] = {}
        self.delta_history: Dict[str, deque] = {}
        self.price_history: Dict[str, deque] = {}

        # Divergence tracking
        self.last_divergence: Dict[str, Dict] = {}
        self.divergence_count: Dict[str, int] = {}

        logger.info(
            "Delta divergence strategy initialized",
            strategy=self.name,
            divergence_threshold=self.divergence_threshold,
            lookback_bars=self.lookback_bars
        )

    def _validate_config(self) -> None:
        """Validate strategy configuration.

        Raises:
            ValueError: If parameters invalid
        """
        super()._validate_config()

        threshold = Decimal(str(self.config.get("divergence_threshold", "0.3")))
        if threshold <= Decimal("0") or threshold > Decimal("1"):
            raise ValueError(f"divergence_threshold must be between 0 and 1: {threshold}")

        lookback = int(self.config.get("lookback_bars", 20))
        if lookback < 2 or lookback > 200:
            raise ValueError(f"lookback_bars must be between 2 and 200: {lookback}")

    def calculate_bar_delta(
        self,
        open_price: Decimal,
        close_price: Decimal,
        high_price: Decimal,
        low_price: Decimal,
        volume: Decimal
    ) -> Decimal:
        """Calculate delta for a single bar.

        Estimates buying vs selling pressure based on close position within range.

        Args:
            open_price: Open price
            close_price: Close price
            high_price: High price
            low_price: Low price
            volume: Total volume

        Returns:
            Delta (positive = buying pressure, negative = selling pressure)
        """
        try:
            if high_price == low_price:
                # No range, use close vs open
                if close_price > open_price:
                    return volume
                elif close_price < open_price:
                    return -volume
                else:
                    return Decimal("0")

            # Calculate close position within range [0, 1]
            close_position = (close_price - low_price) / (high_price - low_price)

            # Estimate buying pressure: volume weighted by close position
            # Close at high = 100% buying, close at low = 100% selling
            buying_volume = volume * close_position
            selling_volume = volume * (Decimal("1") - close_position)

            delta = buying_volume - selling_volume

            return delta

        except Exception as e:
            logger.error("Error calculating bar delta", error=str(e))
            return Decimal("0")

    def update_cumulative_delta(
        self,
        symbol: str,
        bar_delta: Decimal,
        price: Decimal
    ) -> None:
        """Update cumulative delta for symbol.

        Args:
            symbol: Trading symbol
            bar_delta: Delta for current bar
            price: Current price
        """
        try:
            # Initialize if needed
            if symbol not in self.cumulative_delta:
                self.cumulative_delta[symbol] = Decimal("0")

            if symbol not in self.delta_history:
                self.delta_history[symbol] = deque(maxlen=self.lookback_bars)

            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=self.lookback_bars)

            # Update cumulative delta
            self.cumulative_delta[symbol] += bar_delta

            # Update histories
            self.delta_history[symbol].append(self.cumulative_delta[symbol])
            self.price_history[symbol].append(price)

            logger.debug(
                "Cumulative delta updated",
                symbol=symbol,
                bar_delta=bar_delta,
                cumulative_delta=self.cumulative_delta[symbol]
            )

        except Exception as e:
            logger.error("Error updating cumulative delta", error=str(e))

    def detect_divergence(
        self,
        symbol: str
    ) -> Tuple[Optional[str], Decimal]:
        """Detect bullish or bearish divergence.

        Args:
            symbol: Trading symbol

        Returns:
            Tuple of (divergence_type, strength) where type is 'bullish', 'bearish', or None
        """
        try:
            if symbol not in self.delta_history or symbol not in self.price_history:
                return None, Decimal("0")

            delta_hist = list(self.delta_history[symbol])
            price_hist = list(self.price_history[symbol])

            if len(delta_hist) < self.lookback_bars or len(price_hist) < self.lookback_bars:
                return None, Decimal("0")

            # Calculate trends
            # Price trend: compare recent prices to earlier prices
            recent_prices = price_hist[-5:]
            earlier_prices = price_hist[:5]

            avg_recent_price = sum(recent_prices) / Decimal(str(len(recent_prices)))
            avg_earlier_price = sum(earlier_prices) / Decimal(str(len(earlier_prices)))

            price_trend = (avg_recent_price - avg_earlier_price) / avg_earlier_price

            # Delta trend: compare recent deltas to earlier deltas
            recent_deltas = delta_hist[-5:]
            earlier_deltas = delta_hist[:5]

            avg_recent_delta = sum(recent_deltas) / Decimal(str(len(recent_deltas)))
            avg_earlier_delta = sum(earlier_deltas) / Decimal(str(len(earlier_deltas)))

            # Normalize delta trend
            if avg_earlier_delta != Decimal("0"):
                delta_trend = (avg_recent_delta - avg_earlier_delta) / abs(avg_earlier_delta)
            else:
                delta_trend = Decimal("0")

            # Detect divergence
            # Bullish divergence: price down, delta up
            if price_trend < -self.divergence_threshold and delta_trend > self.divergence_threshold:
                strength = min(abs(price_trend), abs(delta_trend))
                logger.info(
                    "Bullish divergence detected",
                    symbol=symbol,
                    price_trend=price_trend,
                    delta_trend=delta_trend,
                    strength=strength
                )
                return "bullish", strength

            # Bearish divergence: price up, delta down
            elif price_trend > self.divergence_threshold and delta_trend < -self.divergence_threshold:
                strength = min(abs(price_trend), abs(delta_trend))
                logger.info(
                    "Bearish divergence detected",
                    symbol=symbol,
                    price_trend=price_trend,
                    delta_trend=delta_trend,
                    strength=strength
                )
                return "bearish", strength

            return None, Decimal("0")

        except Exception as e:
            logger.error("Error detecting divergence", error=str(e))
            return None, Decimal("0")

    def generate_signals(self, market_data: pl.DataFrame) -> List[Signal]:
        """Generate signals from delta divergence.

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

            # Process each bar to build delta history
            rows = market_data.to_dicts()

            for row in rows:
                open_price = Decimal(str(row["open"]))
                close_price = Decimal(str(row["close"]))
                high_price = Decimal(str(row["high"]))
                low_price = Decimal(str(row["low"]))
                volume = Decimal(str(row["volume"]))

                # Skip low volume bars
                if volume < self.min_volume:
                    continue

                # Calculate bar delta
                bar_delta = self.calculate_bar_delta(
                    open_price, close_price, high_price, low_price, volume
                )

                # Update cumulative delta
                self.update_cumulative_delta(symbol, bar_delta, close_price)

            # Detect divergence
            divergence_type, strength = self.detect_divergence(symbol)

            if divergence_type:
                action = SignalAction.BUY if divergence_type == "bullish" else SignalAction.SELL
                confidence = Decimal("0.7") + (strength * Decimal("0.2"))

                current_price = Decimal(str(market_data["close"][-1]))

                signal = Signal(
                    symbol=symbol,
                    action=action,
                    strength=strength,
                    confidence=min(Decimal("1"), confidence),
                    timestamp=datetime.now(timezone.utc),
                    strategy=self.name,
                    timeframe=self.config.get("timeframe", "1h"),
                    indicators={
                        "cumulative_delta": self.cumulative_delta[symbol],
                        "current_price": current_price,
                        "divergence_strength": strength
                    },
                    metadata={
                        "strategy_type": "order_flow",
                        "divergence_type": divergence_type,
                        "delta_bars": len(self.delta_history.get(symbol, []))
                    }
                )

                if self.validate_signal(signal):
                    signals.append(signal)

                    # Track divergence
                    self.last_divergence[symbol] = {
                        "type": divergence_type,
                        "strength": strength,
                        "timestamp": datetime.now(timezone.utc)
                    }

                    self.divergence_count[symbol] = self.divergence_count.get(symbol, 0) + 1

                    # Update state
                    self.state["signals_generated"] = self.state.get("signals_generated", 0) + 1
                    self.state["last_signal_time"] = datetime.now(timezone.utc)

            return signals

        except Exception as e:
            logger.error("Error generating signals", error=str(e))
            return signals

    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate delta indicators.

        Args:
            data: Market data

        Returns:
            Dictionary of indicators
        """
        try:
            indicators = {}

            symbols = self.config.get("symbols", [])
            for symbol in symbols:
                if symbol in self.cumulative_delta:
                    indicators[f"{symbol}_cumulative_delta"] = self.cumulative_delta[symbol]

                if symbol in self.delta_history:
                    delta_list = list(self.delta_history[symbol])
                    if delta_list:
                        indicators[f"{symbol}_avg_delta"] = sum(delta_list) / Decimal(str(len(delta_list)))
                        indicators[f"{symbol}_delta_trend"] = delta_list[-1] - delta_list[0] if len(delta_list) > 1 else Decimal("0")

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

        delta_metrics = {
            "cumulative_delta": {symbol: float(delta) for symbol, delta in self.cumulative_delta.items()},
            "divergence_count": self.divergence_count,
            "last_divergence": {
                symbol: {
                    "type": div["type"],
                    "strength": float(div["strength"]),
                    "timestamp": div["timestamp"].isoformat()
                }
                for symbol, div in self.last_divergence.items()
            }
        }

        return {**base_metrics, **delta_metrics}
