"""
Order Flow Analyzer Strategy.

Implements comprehensive order flow analysis including volume delta,
order book imbalance, and trade aggression metrics.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from collections import deque

import polars as pl
from structlog import get_logger

from quantum_trader.models import Signal, SignalAction, OrderBook
from quantum_trader.strategies.base_strategy import BaseStrategy
from quantum_trader.risk.manager import RiskManager

logger = get_logger(__name__)


class FlowAnalyzerStrategy(BaseStrategy):
    """Order flow analysis strategy.

    This strategy analyzes multiple dimensions of order flow:
    - Volume delta (buy vs sell pressure)
    - Order book imbalance
    - Trade aggression (market orders vs limit orders)
    - Large order detection
    - Absorption and exhaustion patterns

    Features:
    - Real-time order flow tracking
    - Multi-level order book analysis
    - Volume profile construction
    - Smart money detection

    Attributes:
        imbalance_threshold: Order book imbalance threshold
        volume_spike_multiplier: Volume spike detection multiplier
        aggression_threshold: Trade aggression threshold

    Example:
        >>> config = {
        ...     "imbalance_threshold": "0.3",
        ...     "volume_spike_multiplier": "2.0",
        ...     "aggression_threshold": "0.6"
        ... }
        >>> strategy = FlowAnalyzerStrategy(config, risk_manager)
    """

    def __init__(self, config: Dict[str, Any], risk_manager: RiskManager) -> None:
        """Initialize order flow analyzer strategy.

        Args:
            config: Strategy configuration
            risk_manager: Risk manager instance

        Raises:
            ValueError: If required parameters missing
        """
        super().__init__(config, risk_manager)

        # Flow analysis parameters
        self.imbalance_threshold = Decimal(str(config.get("imbalance_threshold", "0.3")))
        self.volume_spike_multiplier = Decimal(str(config.get("volume_spike_multiplier", "2.0")))
        self.aggression_threshold = Decimal(str(config.get("aggression_threshold", "0.6")))
        self.large_order_multiplier = Decimal(str(config.get("large_order_multiplier", "3.0")))

        # Analysis windows
        self.short_window = int(config.get("short_window", 10))
        self.medium_window = int(config.get("medium_window", 50))
        self.long_window = int(config.get("long_window", 200))

        # Order flow tracking
        self.buy_volume_history: Dict[str, deque] = {}
        self.sell_volume_history: Dict[str, deque] = {}
        self.imbalance_history: Dict[str, deque] = {}
        self.aggression_history: Dict[str, deque] = {}

        # Pattern detection
        self.absorption_detected: Dict[str, bool] = {}
        self.exhaustion_detected: Dict[str, bool] = {}
        self.last_orderbook: Dict[str, OrderBook] = {}

        logger.info(
            "Order flow analyzer initialized",
            strategy=self.name,
            imbalance_threshold=self.imbalance_threshold,
            volume_spike_multiplier=self.volume_spike_multiplier
        )

    def _validate_config(self) -> None:
        """Validate strategy configuration.

        Raises:
            ValueError: If parameters invalid
        """
        super()._validate_config()

        imbalance = Decimal(str(self.config.get("imbalance_threshold", "0.3")))
        if imbalance <= Decimal("0") or imbalance > Decimal("1"):
            raise ValueError(f"imbalance_threshold must be between 0 and 1: {imbalance}")

        spike = Decimal(str(self.config.get("volume_spike_multiplier", "2.0")))
        if spike <= Decimal("1"):
            raise ValueError(f"volume_spike_multiplier must be > 1: {spike}")

    def calculate_orderbook_imbalance(
        self,
        orderbook: OrderBook,
        levels: int = 5
    ) -> Decimal:
        """Calculate order book imbalance.

        Args:
            orderbook: Order book snapshot
            levels: Number of levels to analyze

        Returns:
            Imbalance ratio [-1, 1] where positive = bid pressure
        """
        try:
            # Sum bid and ask volumes for top N levels
            bid_volume = sum(size for _, size in orderbook.bids[:levels])
            ask_volume = sum(size for _, size in orderbook.asks[:levels])

            total_volume = bid_volume + ask_volume

            if total_volume == Decimal("0"):
                return Decimal("0")

            # Calculate imbalance: positive = more bids, negative = more asks
            imbalance = (bid_volume - ask_volume) / total_volume

            logger.debug(
                "Order book imbalance calculated",
                symbol=orderbook.symbol,
                bid_volume=bid_volume,
                ask_volume=ask_volume,
                imbalance=imbalance
            )

            return imbalance

        except Exception as e:
            logger.error("Error calculating imbalance", error=str(e))
            return Decimal("0")

    def calculate_volume_delta(
        self,
        data: pl.DataFrame
    ) -> Decimal:
        """Calculate cumulative volume delta from OHLCV data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Volume delta as Decimal
        """
        try:
            if data.is_empty():
                return Decimal("0")

            rows = data.to_dicts()

            total_buy_volume = Decimal("0")
            total_sell_volume = Decimal("0")

            for row in rows:
                close = Decimal(str(row["close"]))
                open_price = Decimal(str(row["open"]))
                volume = Decimal(str(row["volume"]))

                # Estimate buy/sell volume based on close vs open
                if close > open_price:
                    # Bullish bar - more buying
                    buy_volume = volume * Decimal("0.7")
                    sell_volume = volume * Decimal("0.3")
                elif close < open_price:
                    # Bearish bar - more selling
                    buy_volume = volume * Decimal("0.3")
                    sell_volume = volume * Decimal("0.7")
                else:
                    # Neutral - split evenly
                    buy_volume = volume * Decimal("0.5")
                    sell_volume = volume * Decimal("0.5")

                total_buy_volume += buy_volume
                total_sell_volume += sell_volume

            delta = total_buy_volume - total_sell_volume

            logger.debug(
                "Volume delta calculated",
                buy_volume=total_buy_volume,
                sell_volume=total_sell_volume,
                delta=delta
            )

            return delta

        except Exception as e:
            logger.error("Error calculating volume delta", error=str(e))
            return Decimal("0")

    def detect_volume_spike(
        self,
        symbol: str,
        current_volume: Decimal
    ) -> bool:
        """Detect abnormal volume spike.

        Args:
            symbol: Trading symbol
            current_volume: Current bar volume

        Returns:
            True if volume spike detected
        """
        try:
            if symbol not in self.buy_volume_history:
                return False

            # Calculate average volume
            recent_volumes = list(self.buy_volume_history[symbol])
            if len(recent_volumes) < self.short_window:
                return False

            avg_volume = sum(recent_volumes[-self.short_window:]) / Decimal(str(self.short_window))

            if avg_volume == Decimal("0"):
                return False

            # Check if current volume is significantly higher
            volume_ratio = current_volume / avg_volume

            is_spike = volume_ratio >= self.volume_spike_multiplier

            if is_spike:
                logger.info(
                    "Volume spike detected",
                    symbol=symbol,
                    current_volume=current_volume,
                    avg_volume=avg_volume,
                    ratio=volume_ratio
                )

            return is_spike

        except Exception as e:
            logger.error("Error detecting volume spike", error=str(e))
            return False

    def detect_absorption(
        self,
        symbol: str,
        orderbook: OrderBook
    ) -> bool:
        """Detect absorption pattern (large orders absorbing selling/buying pressure).

        Args:
            symbol: Trading symbol
            orderbook: Order book snapshot

        Returns:
            True if absorption detected
        """
        try:
            # Look for large orders at key levels
            if not orderbook.bids or not orderbook.asks:
                return False

            # Calculate average order size
            bid_sizes = [size for _, size in orderbook.bids[:10]]
            ask_sizes = [size for _, size in orderbook.asks[:10]]

            if not bid_sizes or not ask_sizes:
                return False

            avg_bid_size = sum(bid_sizes) / Decimal(str(len(bid_sizes)))
            avg_ask_size = sum(ask_sizes) / Decimal(str(len(ask_sizes)))

            # Check for large orders (absorption walls)
            max_bid = max(bid_sizes)
            max_ask = max(ask_sizes)

            bid_absorption = max_bid >= (avg_bid_size * self.large_order_multiplier)
            ask_absorption = max_ask >= (avg_ask_size * self.large_order_multiplier)

            absorption = bid_absorption or ask_absorption

            if absorption:
                logger.info(
                    "Absorption pattern detected",
                    symbol=symbol,
                    max_bid=max_bid,
                    avg_bid=avg_bid_size,
                    max_ask=max_ask,
                    avg_ask=avg_ask_size
                )

            return absorption

        except Exception as e:
            logger.error("Error detecting absorption", error=str(e))
            return False

    def calculate_trade_aggression(
        self,
        data: pl.DataFrame
    ) -> Decimal:
        """Calculate trade aggression (ratio of aggressive buyers to sellers).

        Args:
            data: Market data

        Returns:
            Aggression score [0, 1] where >0.5 = more aggressive buying
        """
        try:
            if data.is_empty():
                return Decimal("0.5")

            rows = data.to_dicts()

            aggressive_buy_count = 0
            aggressive_sell_count = 0

            for row in rows:
                close = Decimal(str(row["close"]))
                high = Decimal(str(row["high"]))
                low = Decimal(str(row["low"]))

                # Estimate aggression based on close position in range
                if high != low:
                    close_position = (close - low) / (high - low)

                    if close_position > Decimal("0.7"):
                        # Closed near high = aggressive buying
                        aggressive_buy_count += 1
                    elif close_position < Decimal("0.3"):
                        # Closed near low = aggressive selling
                        aggressive_sell_count += 1

            total = aggressive_buy_count + aggressive_sell_count
            if total == 0:
                return Decimal("0.5")

            aggression = Decimal(str(aggressive_buy_count)) / Decimal(str(total))

            return aggression

        except Exception as e:
            logger.error("Error calculating aggression", error=str(e))
            return Decimal("0.5")

    def generate_signals(self, market_data: pl.DataFrame) -> List[Signal]:
        """Generate signals from order flow analysis.

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

            # Initialize tracking structures
            if symbol not in self.buy_volume_history:
                self.buy_volume_history[symbol] = deque(maxlen=self.long_window)
                self.sell_volume_history[symbol] = deque(maxlen=self.long_window)
                self.imbalance_history[symbol] = deque(maxlen=self.medium_window)
                self.aggression_history[symbol] = deque(maxlen=self.medium_window)

            # Calculate volume delta
            volume_delta = self.calculate_volume_delta(market_data)

            # Calculate trade aggression
            aggression = self.calculate_trade_aggression(market_data)
            self.aggression_history[symbol].append(aggression)

            # Check for orderbook imbalance if available
            imbalance = Decimal("0")
            if symbol in self.last_orderbook:
                imbalance = self.calculate_orderbook_imbalance(self.last_orderbook[symbol])
                self.imbalance_history[symbol].append(imbalance)

            # Detect volume spike
            current_volume = Decimal(str(market_data["volume"][-1]))
            volume_spike = self.detect_volume_spike(symbol, current_volume)

            # Generate signals based on flow analysis
            action = SignalAction.HOLD
            strength = Decimal("0")
            confidence = Decimal("0.5")

            # Bullish flow conditions
            if (imbalance > self.imbalance_threshold and
                aggression > self.aggression_threshold and
                volume_delta > Decimal("0")):

                action = SignalAction.BUY
                strength = min(Decimal("1"), (imbalance + aggression) / Decimal("2"))
                confidence = Decimal("0.7")

                if volume_spike:
                    confidence += Decimal("0.1")

                logger.info(
                    "Bullish order flow detected",
                    symbol=symbol,
                    imbalance=imbalance,
                    aggression=aggression,
                    volume_delta=volume_delta
                )

            # Bearish flow conditions
            elif (imbalance < -self.imbalance_threshold and
                  aggression < (Decimal("1") - self.aggression_threshold) and
                  volume_delta < Decimal("0")):

                action = SignalAction.SELL
                strength = min(Decimal("1"), (abs(imbalance) + (Decimal("1") - aggression)) / Decimal("2"))
                confidence = Decimal("0.7")

                if volume_spike:
                    confidence += Decimal("0.1")

                logger.info(
                    "Bearish order flow detected",
                    symbol=symbol,
                    imbalance=imbalance,
                    aggression=aggression,
                    volume_delta=volume_delta
                )

            # Create signal if action determined
            if action != SignalAction.HOLD:
                signal = Signal(
                    symbol=symbol,
                    action=action,
                    strength=strength,
                    confidence=min(Decimal("1"), confidence),
                    timestamp=datetime.now(timezone.utc),
                    strategy=self.name,
                    timeframe=self.config.get("timeframe", "1m"),
                    indicators={
                        "volume_delta": volume_delta,
                        "imbalance": imbalance,
                        "aggression": aggression,
                        "volume_spike": Decimal("1" if volume_spike else "0")
                    },
                    metadata={
                        "strategy_type": "order_flow",
                        "analysis_type": "comprehensive",
                        "volume_spike": volume_spike
                    }
                )

                if self.validate_signal(signal):
                    signals.append(signal)

                    # Update state
                    self.state["signals_generated"] = self.state.get("signals_generated", 0) + 1
                    self.state["last_signal_time"] = datetime.now(timezone.utc)

            return signals

        except Exception as e:
            logger.error("Error generating signals", error=str(e))
            return signals

    def update_orderbook(self, symbol: str, orderbook: OrderBook) -> None:
        """Update order book for symbol.

        Args:
            symbol: Trading symbol
            orderbook: Order book snapshot
        """
        self.last_orderbook[symbol] = orderbook

    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate order flow indicators.

        Args:
            data: Market data

        Returns:
            Dictionary of indicators
        """
        try:
            volume_delta = self.calculate_volume_delta(data)
            aggression = self.calculate_trade_aggression(data)

            indicators = {
                "volume_delta": volume_delta,
                "trade_aggression": aggression
            }

            # Add imbalance if available
            symbols = self.config.get("symbols", [])
            for symbol in symbols:
                if symbol in self.last_orderbook:
                    imbalance = self.calculate_orderbook_imbalance(self.last_orderbook[symbol])
                    indicators[f"{symbol}_imbalance"] = imbalance

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

        flow_metrics = {
            "absorption_detected": self.absorption_detected,
            "exhaustion_detected": self.exhaustion_detected
        }

        # Add average metrics if history available
        for symbol in self.aggression_history:
            if self.aggression_history[symbol]:
                avg_aggression = sum(self.aggression_history[symbol]) / Decimal(str(len(self.aggression_history[symbol])))
                flow_metrics[f"{symbol}_avg_aggression"] = float(avg_aggression)

        return {**base_metrics, **flow_metrics}
