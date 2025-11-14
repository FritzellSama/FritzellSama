"""
Quantum Trader AI - High-Performance Order Matching Engine
Production-grade matching engine with <1ms latency

CRITICAL: All numeric values use Decimal, never float
CRITICAL: Uses polars DataFrame for data operations
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml

import polars as pl

from quantum_trader.models import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    ExecutionResult,
    ExecutionStatus,
    AuditLog,
)


logger = logging.getLogger(__name__)


@dataclass
class OrderBookLevel:
    """Single level in the order book"""

    price: Decimal
    quantity: Decimal
    orders: List[Order] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Validate all numeric values are Decimal"""
        if not isinstance(self.price, Decimal):
            raise TypeError(f"Price must be Decimal, got {type(self.price)}")
        if not isinstance(self.quantity, Decimal):
            raise TypeError(f"Quantity must be Decimal, got {type(self.quantity)}")


@dataclass
class Match:
    """Represents a matched trade"""

    buy_order: Order
    sell_order: Order
    price: Decimal
    quantity: Decimal
    timestamp: datetime
    match_id: str

    def __post_init__(self) -> None:
        """Validate match data"""
        if not isinstance(self.price, Decimal):
            raise TypeError(f"Match price must be Decimal, got {type(self.price)}")
        if not isinstance(self.quantity, Decimal):
            raise TypeError(f"Match quantity must be Decimal, got {type(self.quantity)}")
        if self.quantity <= Decimal("0"):
            raise ValueError(f"Match quantity must be positive, got {self.quantity}")


class MatchingEngine:
    """
    High-performance order matching engine with price-time priority

    Features:
    - Price-time priority matching algorithm
    - <1ms matching latency for typical workloads
    - Order book management with optimal data structures
    - Partial fill handling
    - Match validation and verification
    - Thread-safe operations
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize matching engine

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)

        # Order books: {symbol: {side: {price: OrderBookLevel}}}
        self.order_books: Dict[str, Dict[OrderSide, Dict[Decimal, OrderBookLevel]]] = defaultdict(
            lambda: {OrderSide.BUY: {}, OrderSide.SELL: {}}
        )

        # Active orders: {order_id: Order}
        self.active_orders: Dict[str, Order] = {}

        # Order fills: {order_id: filled_quantity}
        self.order_fills: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))

        # Match history stored as polars DataFrame
        self.match_history = pl.DataFrame(
            schema={
                "match_id": pl.Utf8,
                "symbol": pl.Utf8,
                "price": pl.Utf8,  # Store as string to preserve Decimal precision
                "quantity": pl.Utf8,
                "buy_order_id": pl.Utf8,
                "sell_order_id": pl.Utf8,
                "timestamp": pl.Datetime,
            }
        )

        # Performance metrics
        self.match_count: int = 0
        self.total_volume: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))

        # Lock for thread safety
        self._lock = asyncio.Lock()

        logger.info(
            "MatchingEngine initialized",
            extra={
                "max_order_book_depth": self.config["max_order_book_depth"],
                "enable_partial_fills": self.config["enable_partial_fills"],
            },
        )

    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """
        Load configuration from YAML files

        Args:
            config_path: Override config path

        Returns:
            Configuration dictionary
        """
        default_config = {
            "max_order_book_depth": 1000,
            "enable_partial_fills": True,
            "match_validation_enabled": True,
            "max_price_deviation_bps": Decimal("100"),  # 1% max deviation
            "min_order_quantity": Decimal("0.00000001"),
            "tick_size": Decimal("0.01"),
            "lot_size": Decimal("0.00000001"),
        }

        if config_path is None:
            # Try to load from default locations
            base_path = Path(__file__).parent.parent.parent.parent
            config_files = [
                base_path / "config" / "bot" / "bot.yaml",
                base_path / "config" / "environments" / "production.yaml",
            ]
        else:
            config_files = [Path(config_path)]

        loaded_config = default_config.copy()

        for config_file in config_files:
            try:
                if config_file.exists():
                    with open(config_file, "r") as f:
                        file_config = yaml.safe_load(f)

                        # Extract relevant settings
                        if "bot" in file_config and "execution" in file_config["bot"]:
                            exec_config = file_config["bot"]["execution"]
                            loaded_config["max_concurrent_orders"] = exec_config.get(
                                "max_concurrent_orders", 50
                            )

                        if "trading" in file_config and "order_execution" in file_config["trading"]:
                            order_config = file_config["trading"]["order_execution"]
                            loaded_config["max_order_age_seconds"] = order_config.get(
                                "max_order_age_seconds", 300
                            )

                        if "performance" in file_config:
                            perf_config = file_config["performance"]
                            loaded_config["max_order_book_depth"] = perf_config.get(
                                "max_order_book_depth", 1000
                            )

            except Exception as e:
                logger.warning(f"Failed to load config from {config_file}: {e}")
                continue

        return loaded_config

    async def add_order(self, order: Order) -> bool:
        """
        Add order to the matching engine

        Args:
            order: Order to add

        Returns:
            True if order was added successfully

        Raises:
            ValueError: If order validation fails
        """
        async with self._lock:
            try:
                # Validate order
                self._validate_order(order)

                # Add to active orders
                if order.order_id is None:
                    raise ValueError("Order must have an order_id")

                self.active_orders[order.order_id] = order

                # Add to order book
                symbol_book = self.order_books[order.symbol]
                side_book = symbol_book[order.side]

                if order.price is None:
                    # Market orders are handled immediately
                    await self._match_market_order(order)
                else:
                    # Add limit order to book
                    price = order.price

                    if price not in side_book:
                        side_book[price] = OrderBookLevel(
                            price=price,
                            quantity=Decimal("0"),
                            orders=[],
                        )

                    level = side_book[price]
                    level.orders.append(order)
                    level.quantity += order.quantity

                    # Attempt to match
                    await self._match_order(order)

                # Create audit log
                await self._create_audit_log(
                    operation="add_order",
                    component="matching_engine",
                    severity="INFO",
                    details={
                        "order_id": order.order_id,
                        "symbol": order.symbol,
                        "side": order.side.value,
                        "quantity": str(order.quantity),
                        "price": str(order.price) if order.price else None,
                    },
                )

                return True

            except Exception as e:
                logger.error(f"Failed to add order: {e}", exc_info=True)
                await self._create_audit_log(
                    operation="add_order_failed",
                    component="matching_engine",
                    severity="ERROR",
                    details={"error": str(e), "order_id": order.order_id if order.order_id else "unknown"},
                )
                raise

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order

        Args:
            order_id: ID of order to cancel

        Returns:
            True if order was cancelled successfully
        """
        async with self._lock:
            try:
                if order_id not in self.active_orders:
                    logger.warning(f"Order {order_id} not found in active orders")
                    return False

                order = self.active_orders[order_id]

                # Remove from order book
                symbol_book = self.order_books[order.symbol]
                side_book = symbol_book[order.side]

                if order.price and order.price in side_book:
                    level = side_book[order.price]
                    if order in level.orders:
                        level.orders.remove(order)
                        level.quantity -= (order.quantity - self.order_fills[order_id])

                        # Clean up empty levels
                        if level.quantity <= Decimal("0"):
                            del side_book[order.price]

                # Remove from active orders
                del self.active_orders[order_id]

                await self._create_audit_log(
                    operation="cancel_order",
                    component="matching_engine",
                    severity="INFO",
                    details={"order_id": order_id, "symbol": order.symbol},
                )

                return True

            except Exception as e:
                logger.error(f"Failed to cancel order {order_id}: {e}", exc_info=True)
                return False

    async def _match_order(self, order: Order) -> List[Match]:
        """
        Attempt to match an order against the order book

        Args:
            order: Order to match

        Returns:
            List of matches generated
        """
        matches: List[Match] = []

        try:
            if order.order_id is None:
                raise ValueError("Order must have an order_id")

            # Get opposite side of order book
            symbol_book = self.order_books[order.symbol]
            opposite_side = OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY
            opposite_book = symbol_book[opposite_side]

            # Get filled quantity for this order
            filled = self.order_fills[order.order_id]
            remaining = order.quantity - filled

            if remaining <= Decimal("0"):
                return matches

            # Get sorted prices (best first)
            if opposite_side == OrderSide.SELL:
                # For buying, match against lowest sell prices
                sorted_prices = sorted(opposite_book.keys())
            else:
                # For selling, match against highest buy prices
                sorted_prices = sorted(opposite_book.keys(), reverse=True)

            # Match against each price level
            for price in sorted_prices:
                if remaining <= Decimal("0"):
                    break

                # Check if price is acceptable
                if not self._is_price_acceptable(order, price):
                    break

                level = opposite_book[price]

                # Match against orders at this level (FIFO - price-time priority)
                for opposite_order in level.orders[:]:
                    if remaining <= Decimal("0"):
                        break

                    if opposite_order.order_id is None:
                        continue

                    # Calculate match quantity
                    opposite_filled = self.order_fills[opposite_order.order_id]
                    opposite_remaining = opposite_order.quantity - opposite_filled

                    match_quantity = min(remaining, opposite_remaining)

                    if match_quantity <= Decimal("0"):
                        continue

                    # Create match
                    match = self._create_match(order, opposite_order, price, match_quantity)

                    if self._validate_match(match):
                        matches.append(match)

                        # Update fills
                        self.order_fills[order.order_id] += match_quantity
                        self.order_fills[opposite_order.order_id] += match_quantity

                        # Update remaining quantities
                        remaining -= match_quantity
                        level.quantity -= match_quantity

                        # Remove fully filled orders
                        if self.order_fills[opposite_order.order_id] >= opposite_order.quantity:
                            level.orders.remove(opposite_order)
                            if opposite_order.order_id in self.active_orders:
                                del self.active_orders[opposite_order.order_id]

                        # Record match
                        await self._record_match(match)

                        # Update metrics
                        self.match_count += 1
                        self.total_volume[order.symbol] += match_quantity

                # Clean up empty levels
                if level.quantity <= Decimal("0") and price in opposite_book:
                    del opposite_book[price]

            # Remove order if fully filled
            if order.order_id in self.active_orders:
                if self.order_fills[order.order_id] >= order.quantity:
                    # Remove from order book
                    if order.price and order.price in symbol_book[order.side]:
                        level = symbol_book[order.side][order.price]
                        if order in level.orders:
                            level.orders.remove(order)

                    del self.active_orders[order.order_id]

            return matches

        except Exception as e:
            logger.error(f"Error matching order: {e}", exc_info=True)
            return matches

    async def _match_market_order(self, order: Order) -> List[Match]:
        """
        Match a market order immediately at best available prices

        Args:
            order: Market order to match

        Returns:
            List of matches
        """
        # Market orders match at best available price
        # Temporarily set price to extreme value to ensure matching
        if order.side == OrderSide.BUY:
            order.price = Decimal("999999999999")  # High price for buy
        else:
            order.price = Decimal("0")  # Low price for sell

        matches = await self._match_order(order)

        # Reset price to None for market orders
        order.price = None

        return matches

    def _is_price_acceptable(self, order: Order, match_price: Decimal) -> bool:
        """
        Check if a match price is acceptable for the order

        Args:
            order: Order being matched
            match_price: Proposed match price

        Returns:
            True if price is acceptable
        """
        if order.price is None:
            # Market orders accept any price
            return True

        if order.side == OrderSide.BUY:
            # Buy orders match at prices <= limit price
            return match_price <= order.price
        else:
            # Sell orders match at prices >= limit price
            return match_price >= order.price

    def _create_match(
        self,
        order1: Order,
        order2: Order,
        price: Decimal,
        quantity: Decimal,
    ) -> Match:
        """
        Create a match between two orders

        Args:
            order1: First order
            order2: Second order
            price: Match price
            quantity: Match quantity

        Returns:
            Match object
        """
        # Determine buy and sell orders
        if order1.side == OrderSide.BUY:
            buy_order = order1
            sell_order = order2
        else:
            buy_order = order2
            sell_order = order1

        match_id = f"{buy_order.order_id}_{sell_order.order_id}_{datetime.utcnow().timestamp()}"

        return Match(
            buy_order=buy_order,
            sell_order=sell_order,
            price=price,
            quantity=quantity,
            timestamp=datetime.utcnow(),
            match_id=match_id,
        )

    def _validate_order(self, order: Order) -> None:
        """
        Validate order before adding to engine

        Args:
            order: Order to validate

        Raises:
            ValueError: If validation fails
        """
        if order.quantity < self.config["min_order_quantity"]:
            raise ValueError(
                f"Order quantity {order.quantity} below minimum {self.config['min_order_quantity']}"
            )

        if order.price is not None:
            # Validate price tick size
            tick_size = self.config["tick_size"]
            if order.price % tick_size != Decimal("0"):
                raise ValueError(f"Order price must be multiple of tick size {tick_size}")

        # Validate lot size
        lot_size = self.config["lot_size"]
        if order.quantity % lot_size != Decimal("0"):
            raise ValueError(f"Order quantity must be multiple of lot size {lot_size}")

    def _validate_match(self, match: Match) -> bool:
        """
        Validate a match before execution

        Args:
            match: Match to validate

        Returns:
            True if match is valid
        """
        if not self.config["match_validation_enabled"]:
            return True

        try:
            # Check match quantity is positive
            if match.quantity <= Decimal("0"):
                logger.error(f"Invalid match quantity: {match.quantity}")
                return False

            # Check price is reasonable
            if match.price <= Decimal("0"):
                logger.error(f"Invalid match price: {match.price}")
                return False

            # Validate against order prices
            if match.buy_order.price is not None:
                if match.price > match.buy_order.price:
                    logger.error(
                        f"Match price {match.price} exceeds buy order limit {match.buy_order.price}"
                    )
                    return False

            if match.sell_order.price is not None:
                if match.price < match.sell_order.price:
                    logger.error(
                        f"Match price {match.price} below sell order limit {match.sell_order.price}"
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"Match validation error: {e}", exc_info=True)
            return False

    async def _record_match(self, match: Match) -> None:
        """
        Record match in history

        Args:
            match: Match to record
        """
        try:
            # Create new row as polars DataFrame
            new_row = pl.DataFrame(
                {
                    "match_id": [match.match_id],
                    "symbol": [match.buy_order.symbol],
                    "price": [str(match.price)],
                    "quantity": [str(match.quantity)],
                    "buy_order_id": [match.buy_order.order_id or ""],
                    "sell_order_id": [match.sell_order.order_id or ""],
                    "timestamp": [match.timestamp],
                }
            )

            # Append to history
            self.match_history = pl.concat([self.match_history, new_row])

        except Exception as e:
            logger.error(f"Failed to record match: {e}", exc_info=True)

    async def _create_audit_log(
        self,
        operation: str,
        component: str,
        severity: str,
        details: Dict,
    ) -> None:
        """
        Create audit log entry

        Args:
            operation: Operation name
            component: Component name
            severity: Log severity
            details: Additional details
        """
        try:
            audit_log = AuditLog(
                timestamp=datetime.utcnow(),
                operation=operation,
                user_id="matching_engine",
                component=component,
                severity=severity,
                details=details,
            )

            # Log the audit entry
            logger.log(
                getattr(logging, severity),
                f"Audit: {operation}",
                extra={"audit_log": audit_log},
            )

        except Exception as e:
            logger.error(f"Failed to create audit log: {e}", exc_info=True)

    def get_order_book(
        self, symbol: str, depth: int = 10
    ) -> Tuple[List[Tuple[Decimal, Decimal]], List[Tuple[Decimal, Decimal]]]:
        """
        Get current order book snapshot

        Args:
            symbol: Trading symbol
            depth: Number of levels to return

        Returns:
            Tuple of (bids, asks) where each is list of (price, quantity)
        """
        symbol_book = self.order_books.get(symbol)
        if not symbol_book:
            return ([], [])

        # Get bids (sorted descending by price)
        buy_book = symbol_book[OrderSide.BUY]
        bids = [
            (price, level.quantity)
            for price, level in sorted(buy_book.items(), reverse=True)[:depth]
        ]

        # Get asks (sorted ascending by price)
        sell_book = symbol_book[OrderSide.SELL]
        asks = [
            (price, level.quantity)
            for price, level in sorted(sell_book.items())[:depth]
        ]

        return (bids, asks)

    def get_match_history(
        self, symbol: Optional[str] = None, limit: int = 100
    ) -> pl.DataFrame:
        """
        Get match history as polars DataFrame

        Args:
            symbol: Filter by symbol (optional)
            limit: Maximum number of matches to return

        Returns:
            DataFrame of matches
        """
        df = self.match_history

        if symbol:
            df = df.filter(pl.col("symbol") == symbol)

        # Get most recent matches
        df = df.sort("timestamp", descending=True).head(limit)

        return df

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """
        Get current status of an order

        Args:
            order_id: Order ID

        Returns:
            Order status dictionary or None if not found
        """
        if order_id not in self.active_orders:
            return None

        order = self.active_orders[order_id]
        filled = self.order_fills[order_id]
        remaining = order.quantity - filled

        if filled == Decimal("0"):
            status = OrderStatus.OPEN
        elif remaining > Decimal("0"):
            status = OrderStatus.PARTIAL
        else:
            status = OrderStatus.FILLED

        return {
            "order_id": order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": str(order.quantity),
            "filled": str(filled),
            "remaining": str(remaining),
            "status": status.value,
            "price": str(order.price) if order.price else None,
        }

    def get_metrics(self) -> Dict:
        """
        Get matching engine performance metrics

        Returns:
            Dictionary of metrics
        """
        total_active_orders = len(self.active_orders)
        total_symbols = len(self.order_books)

        return {
            "match_count": self.match_count,
            "active_orders": total_active_orders,
            "symbols": total_symbols,
            "total_volume": {
                symbol: str(volume) for symbol, volume in self.total_volume.items()
            },
        }
