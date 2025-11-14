"""
Base Market Maker - Abstract base class for market making strategies.

This module provides the foundation for all market making strategies, including
order management, spread calculation, and inventory management.
"""

import asyncio
from abc import abstractmethod
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

import polars as pl
from structlog import get_logger

from quantum_trader.models import Signal, Order, OrderBook, Ticker, OrderSide, OrderType, SignalAction
from quantum_trader.strategies.base_strategy import BaseStrategy
from quantum_trader.risk.manager import RiskManager

logger = get_logger(__name__)


class BaseMarketMaker(BaseStrategy):
    """Abstract base class for market making strategies.

    Market makers provide liquidity by placing both buy and sell limit orders.
    This base class handles common market making functionality including:
    - Spread calculation and management
    - Inventory tracking and balancing
    - Order placement and cancellation
    - Quote management

    Attributes:
        bid_orders: Active bid orders
        ask_orders: Active ask orders
        inventory: Current inventory position
        target_inventory: Target inventory level
        last_orderbook: Most recent orderbook snapshot

    Example:
        >>> class MyMarketMaker(BaseMarketMaker):
        ...     def calculate_spread(self, orderbook, ticker):
        ...         return Decimal("0.001")  # 10 bps
        ...     def calculate_quote_size(self, side, inventory):
        ...         return Decimal("0.1")  # 0.1 BTC
    """

    def __init__(self, config: Dict[str, Any], risk_manager: RiskManager) -> None:
        """Initialize market maker strategy.

        Args:
            config: Strategy configuration
            risk_manager: Risk manager instance

        Raises:
            ValueError: If required config parameters missing
        """
        super().__init__(config, risk_manager)

        # Market making specific state
        self.bid_orders: List[Order] = []
        self.ask_orders: List[Order] = []
        self.inventory: Dict[str, Decimal] = {}
        self.target_inventory: Dict[str, Decimal] = {}
        self.last_orderbook: Optional[OrderBook] = None
        self.last_ticker: Optional[Ticker] = None

        # Configuration parameters
        self.min_spread = Decimal(str(config.get("min_spread", "0.0005")))  # 5 bps
        self.max_position = Decimal(str(config.get("max_position", "10.0")))
        self.order_levels = int(config.get("order_levels", 1))
        self.level_spacing = Decimal(str(config.get("level_spacing", "0.0001")))
        self.inventory_skew_factor = Decimal(str(config.get("inventory_skew_factor", "0.5")))
        self.quote_refresh_seconds = int(config.get("quote_refresh_seconds", 5))

        logger.info(
            "Market maker initialized",
            strategy=self.name,
            min_spread=self.min_spread,
            order_levels=self.order_levels
        )

    def _validate_config(self) -> None:
        """Validate market maker configuration.

        Raises:
            ValueError: If configuration invalid
        """
        super()._validate_config()

        # Validate market making specific parameters
        if "min_spread" in self.config:
            min_spread = Decimal(str(self.config["min_spread"]))
            if min_spread <= Decimal("0") or min_spread > Decimal("0.1"):
                raise ValueError(f"Invalid min_spread: {min_spread}")

        if "order_levels" in self.config:
            order_levels = int(self.config["order_levels"])
            if order_levels < 1 or order_levels > 10:
                raise ValueError(f"order_levels must be between 1 and 10: {order_levels}")

        if "max_position" in self.config:
            max_position = Decimal(str(self.config["max_position"]))
            if max_position <= Decimal("0"):
                raise ValueError(f"max_position must be positive: {max_position}")

    @abstractmethod
    def calculate_spread(self, orderbook: OrderBook, ticker: Ticker) -> Decimal:
        """Calculate bid-ask spread for market making.

        Args:
            orderbook: Current orderbook snapshot
            ticker: Current ticker data

        Returns:
            Spread as Decimal (e.g., 0.001 for 10 bps)

        Example:
            >>> spread = self.calculate_spread(orderbook, ticker)
            >>> Decimal("0.0015")
        """
        pass

    @abstractmethod
    def calculate_quote_size(self, side: OrderSide, inventory: Decimal) -> Decimal:
        """Calculate order size for quote.

        Args:
            side: Order side (BUY or SELL)
            inventory: Current inventory level

        Returns:
            Order size as Decimal

        Example:
            >>> size = self.calculate_quote_size(OrderSide.BUY, Decimal("5.0"))
            >>> Decimal("0.5")
        """
        pass

    def calculate_inventory_skew(self, symbol: str) -> Decimal:
        """Calculate inventory skew adjustment.

        Adjusts quotes based on current inventory position to drive inventory
        back toward target.

        Args:
            symbol: Trading symbol

        Returns:
            Skew adjustment as Decimal (positive = favor selling, negative = favor buying)
        """
        try:
            current_inventory = self.inventory.get(symbol, Decimal("0"))
            target = self.target_inventory.get(symbol, Decimal("0"))

            inventory_diff = current_inventory - target
            max_pos = self.max_position

            # Normalize difference to [-1, 1]
            normalized_diff = inventory_diff / max_pos if max_pos > Decimal("0") else Decimal("0")

            # Apply skew factor
            skew = normalized_diff * self.inventory_skew_factor

            logger.debug(
                "Inventory skew calculated",
                symbol=symbol,
                current=current_inventory,
                target=target,
                skew=skew
            )

            return skew

        except Exception as e:
            logger.error("Error calculating inventory skew", error=str(e), symbol=symbol)
            return Decimal("0")

    def generate_quotes(
        self,
        symbol: str,
        mid_price: Decimal,
        spread: Decimal,
        inventory: Decimal
    ) -> Tuple[List[Tuple[Decimal, Decimal]], List[Tuple[Decimal, Decimal]]]:
        """Generate bid and ask quotes for multiple levels.

        Args:
            symbol: Trading symbol
            mid_price: Current mid price
            spread: Bid-ask spread
            inventory: Current inventory

        Returns:
            Tuple of (bids, asks) where each is list of (price, size) tuples
        """
        try:
            bids: List[Tuple[Decimal, Decimal]] = []
            asks: List[Tuple[Decimal, Decimal]] = []

            # Calculate inventory skew
            skew = self.calculate_inventory_skew(symbol)

            # Half spread
            half_spread = spread / Decimal("2")

            for level in range(self.order_levels):
                # Calculate level offset
                level_offset = self.level_spacing * Decimal(str(level))

                # Calculate bid price (apply negative skew to widen when long)
                bid_price = mid_price - half_spread - level_offset - (skew * mid_price)
                bid_price = bid_price.quantize(Decimal("0.00000001"))

                # Calculate ask price (apply positive skew to widen when short)
                ask_price = mid_price + half_spread + level_offset + (skew * mid_price)
                ask_price = ask_price.quantize(Decimal("0.00000001"))

                # Calculate sizes (reduce size when inventory unfavorable)
                bid_size = self.calculate_quote_size(OrderSide.BUY, inventory)
                ask_size = self.calculate_quote_size(OrderSide.SELL, inventory)

                # Adjust sizes based on inventory
                if skew > Decimal("0"):  # Long inventory, reduce bids
                    bid_size = bid_size * (Decimal("1") - abs(skew))
                elif skew < Decimal("0"):  # Short inventory, reduce asks
                    ask_size = ask_size * (Decimal("1") - abs(skew))

                # Ensure minimum sizes
                min_size = Decimal(str(self.config.get("min_order_size", "0.001")))
                bid_size = max(bid_size, min_size).quantize(Decimal("0.00000001"))
                ask_size = max(ask_size, min_size).quantize(Decimal("0.00000001"))

                bids.append((bid_price, bid_size))
                asks.append((ask_price, ask_size))

            logger.debug(
                "Quotes generated",
                symbol=symbol,
                mid_price=mid_price,
                spread=spread,
                levels=self.order_levels,
                skew=skew
            )

            return bids, asks

        except Exception as e:
            logger.error("Error generating quotes", error=str(e), symbol=symbol)
            return [], []

    def generate_signals(self, market_data: pl.DataFrame) -> List[Signal]:
        """Generate trading signals from market data.

        For market makers, signals are primarily based on orderbook and ticker data
        rather than OHLCV data.

        Args:
            market_data: Market data (not typically used by market makers)

        Returns:
            List of Signal objects
        """
        signals: List[Signal] = []

        try:
            # Market makers primarily use orderbook updates
            # This is a placeholder that subclasses should override
            if not self.last_orderbook or not self.last_ticker:
                logger.debug("No orderbook/ticker data available for signal generation")
                return signals

            symbol = self.last_orderbook.symbol

            # Calculate current spread
            spread = self.calculate_spread(self.last_orderbook, self.last_ticker)

            # Check if spread is sufficient for market making
            if spread < self.min_spread:
                logger.debug(
                    "Spread below minimum",
                    symbol=symbol,
                    spread=spread,
                    min_spread=self.min_spread
                )
                return signals

            # Generate signal to update quotes
            signal = Signal(
                symbol=symbol,
                action=SignalAction.HOLD,  # Market makers don't take directional positions
                strength=Decimal("1.0"),
                confidence=Decimal("1.0"),
                timestamp=datetime.now(timezone.utc),
                strategy=self.name,
                timeframe="tick",
                indicators={
                    "spread": spread,
                    "mid_price": self.last_ticker.last,
                    "bid": self.last_ticker.bid,
                    "ask": self.last_ticker.ask
                },
                metadata={
                    "type": "market_making",
                    "order_levels": self.order_levels
                }
            )

            if self.validate_signal(signal):
                signals.append(signal)

            return signals

        except Exception as e:
            logger.error("Error generating signals", error=str(e))
            return signals

    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate technical indicators.

        Market makers typically use different metrics than directional strategies.

        Args:
            data: Market data

        Returns:
            Dictionary of indicators
        """
        indicators: Dict[str, Decimal] = {}

        try:
            if self.last_orderbook and self.last_ticker:
                # Calculate spread metrics
                bid = self.last_ticker.bid
                ask = self.last_ticker.ask
                mid = (bid + ask) / Decimal("2")

                spread_bps = ((ask - bid) / mid) * Decimal("10000")

                # Calculate orderbook depth
                bid_depth = sum(size for _, size in self.last_orderbook.bids[:5])
                ask_depth = sum(size for _, size in self.last_orderbook.asks[:5])

                # Calculate imbalance
                total_depth = bid_depth + ask_depth
                imbalance = (bid_depth - ask_depth) / total_depth if total_depth > Decimal("0") else Decimal("0")

                indicators = {
                    "spread_bps": spread_bps,
                    "bid_depth": bid_depth,
                    "ask_depth": ask_depth,
                    "imbalance": imbalance,
                    "mid_price": mid
                }

            return indicators

        except Exception as e:
            logger.error("Error calculating indicators", error=str(e))
            return indicators

    def update_inventory(self, symbol: str, quantity: Decimal, side: OrderSide) -> None:
        """Update inventory after order fill.

        Args:
            symbol: Trading symbol
            quantity: Fill quantity
            side: Order side (BUY or SELL)
        """
        try:
            if symbol not in self.inventory:
                self.inventory[symbol] = Decimal("0")

            if side == OrderSide.BUY:
                self.inventory[symbol] += quantity
            else:
                self.inventory[symbol] -= quantity

            logger.info(
                "Inventory updated",
                symbol=symbol,
                quantity=quantity,
                side=side.value,
                new_inventory=self.inventory[symbol]
            )

        except Exception as e:
            logger.error("Error updating inventory", error=str(e), symbol=symbol)

    def update_orderbook(self, orderbook: OrderBook) -> None:
        """Update last orderbook snapshot.

        Args:
            orderbook: New orderbook snapshot
        """
        self.last_orderbook = orderbook
        logger.debug("Orderbook updated", symbol=orderbook.symbol)

    def update_ticker(self, ticker: Ticker) -> None:
        """Update last ticker data.

        Args:
            ticker: New ticker data
        """
        self.last_ticker = ticker
        logger.debug("Ticker updated", symbol=ticker.symbol)

    def check_inventory_limits(self, symbol: str) -> bool:
        """Check if inventory within limits.

        Args:
            symbol: Trading symbol

        Returns:
            True if within limits, False otherwise
        """
        try:
            current = abs(self.inventory.get(symbol, Decimal("0")))
            max_pos = self.max_position

            within_limits = current <= max_pos

            if not within_limits:
                logger.warning(
                    "Inventory limit exceeded",
                    symbol=symbol,
                    current=current,
                    max=max_pos
                )

            return within_limits

        except Exception as e:
            logger.error("Error checking inventory limits", error=str(e), symbol=symbol)
            return False

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get market maker performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        base_metrics = super().get_performance_metrics()

        mm_metrics = {
            "inventory": {symbol: float(inv) for symbol, inv in self.inventory.items()},
            "active_bid_orders": len(self.bid_orders),
            "active_ask_orders": len(self.ask_orders),
            "total_active_orders": len(self.bid_orders) + len(self.ask_orders)
        }

        return {**base_metrics, **mm_metrics}
