"""
Order Refresher for Market Making.

Automatically refreshes and reprices limit orders based on market
conditions, queue position, and fill probability.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import deque
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


@dataclass
class ActiveOrder:
    """Active order being monitored."""
    order_id: str
    symbol: str
    side: OrderSide
    price: Decimal
    quantity: Decimal
    filled_quantity: Decimal
    status: OrderStatus
    placed_time: datetime
    last_update_time: datetime
    queue_position: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RefreshDecision:
    """Decision about whether to refresh an order."""
    should_refresh: bool
    new_price: Optional[Decimal] = None
    new_quantity: Optional[Decimal] = None
    reason: str = ""
    urgency: Decimal = Decimal("0")


class OrderRefresher:
    """
    Order Refresher for Market Making.

    Monitors active limit orders and decides when to cancel/replace
    them based on market movement, queue position, and time in market.

    Attributes:
        config: Configuration dictionary
        active_orders: Currently active orders
        refresh_history: History of refresh decisions
        fill_rate_tracker: Tracks order fill rates
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize Order Refresher.

        Args:
            config: Configuration dictionary with parameters

        Raises:
            ValueError: If required config parameters missing
        """
        self.config = config
        self._validate_config()

        # Load parameters from config
        self.max_order_age_seconds = Decimal(str(config["max_order_age_seconds"]))
        self.price_movement_threshold_bps = Decimal(str(config["price_movement_threshold_bps"]))
        self.min_queue_position = int(config.get("min_queue_position", 5))
        self.refresh_on_adverse_selection = config.get("refresh_on_adverse_selection", True)
        self.min_time_between_refreshes_ms = Decimal(str(config["min_time_between_refreshes_ms"]))
        self.aggressive_refresh_mode = config.get("aggressive_refresh_mode", False)

        # State tracking
        self.active_orders: Dict[str, ActiveOrder] = {}
        self.refresh_history: deque = deque(maxlen=1000)
        self.fill_rate_tracker: deque = deque(maxlen=100)
        self.last_market_price: Optional[Decimal] = None
        self.last_refresh_time: Dict[str, datetime] = {}

        logger.info(
            "order_refresher_initialized",
            max_age_seconds=float(self.max_order_age_seconds),
            price_threshold_bps=float(self.price_movement_threshold_bps)
        )

    def _validate_config(self) -> None:
        """Validate required configuration parameters."""
        required_params = [
            "max_order_age_seconds",
            "price_movement_threshold_bps",
            "min_time_between_refreshes_ms"
        ]

        missing = [p for p in required_params if p not in self.config]
        if missing:
            raise ValueError(f"Missing required config parameters: {missing}")

    async def register_order(
        self,
        order_id: str,
        symbol: str,
        side: OrderSide,
        price: Decimal,
        quantity: Decimal,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register an active order for monitoring.

        Args:
            order_id: Unique order identifier
            symbol: Trading symbol
            side: Order side
            price: Order price
            quantity: Order quantity
            metadata: Additional order metadata
        """
        now = datetime.now(timezone.utc)

        order = ActiveOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            price=price,
            quantity=quantity,
            filled_quantity=Decimal("0"),
            status=OrderStatus.OPEN,
            placed_time=now,
            last_update_time=now,
            metadata=metadata or {}
        )

        self.active_orders[order_id] = order

        logger.info(
            "order_registered",
            order_id=order_id,
            symbol=symbol,
            side=side.value,
            price=float(price),
            quantity=float(quantity)
        )

    async def evaluate_refresh_needed(
        self,
        order_id: str,
        current_market_data: pl.DataFrame
    ) -> RefreshDecision:
        """
        Evaluate if an order should be refreshed.

        Args:
            order_id: Order to evaluate
            current_market_data: Current market data with bid/ask

        Returns:
            RefreshDecision object

        Raises:
            ValueError: If order_id not found or market_data invalid
        """
        try:
            if order_id not in self.active_orders:
                raise ValueError(f"Order {order_id} not found in active orders")

            order = self.active_orders[order_id]

            if current_market_data.is_empty():
                return RefreshDecision(
                    should_refresh=False,
                    reason="No market data available"
                )

            # Get current market prices
            latest_row = current_market_data.sort("timestamp", descending=True).row(0, named=True)
            current_time = latest_row["timestamp"]

            bid_price = Decimal(str(latest_row.get("bid", latest_row.get("close", 0))))
            ask_price = Decimal(str(latest_row.get("ask", latest_row.get("close", 0))))
            mid_price = (bid_price + ask_price) / Decimal("2")

            self.last_market_price = mid_price

            # Check multiple refresh triggers
            refresh_reasons = []
            urgency = Decimal("0")

            # 1. Check order age
            age_seconds = (current_time - order.placed_time).total_seconds()
            if Decimal(str(age_seconds)) >= self.max_order_age_seconds:
                refresh_reasons.append("max_age_exceeded")
                urgency += Decimal("0.3")

            # 2. Check price movement
            if order.side == OrderSide.BUY:
                # Buy order - check if bid moved away
                price_distance_bps = ((bid_price - order.price) / order.price) * Decimal("10000")

                if abs(price_distance_bps) >= self.price_movement_threshold_bps:
                    refresh_reasons.append(f"price_moved_{price_distance_bps}bps")
                    urgency += Decimal("0.4")
            else:
                # Sell order - check if ask moved away
                price_distance_bps = ((order.price - ask_price) / order.price) * Decimal("10000")

                if abs(price_distance_bps) >= self.price_movement_threshold_bps:
                    refresh_reasons.append(f"price_moved_{price_distance_bps}bps")
                    urgency += Decimal("0.4")

            # 3. Check queue position (if available)
            if order.queue_position is not None:
                if order.queue_position > self.min_queue_position:
                    refresh_reasons.append(f"poor_queue_position_{order.queue_position}")
                    urgency += Decimal("0.2")

            # 4. Check for adverse selection
            if self.refresh_on_adverse_selection:
                adverse_selection = await self._detect_adverse_selection(order, mid_price)
                if adverse_selection:
                    refresh_reasons.append("adverse_selection_detected")
                    urgency += Decimal("0.5")

            # 5. Check if too soon since last refresh
            if order_id in self.last_refresh_time:
                time_since_refresh = (current_time - self.last_refresh_time[order_id]).total_seconds() * 1000
                if Decimal(str(time_since_refresh)) < self.min_time_between_refreshes_ms:
                    return RefreshDecision(
                        should_refresh=False,
                        reason="too_soon_since_last_refresh"
                    )

            # Decide if refresh needed
            should_refresh = len(refresh_reasons) > 0

            if should_refresh:
                # Calculate new price
                new_price = await self._calculate_new_price(
                    order=order,
                    bid_price=bid_price,
                    ask_price=ask_price,
                    mid_price=mid_price
                )

                # Calculate new quantity (may adjust based on market conditions)
                new_quantity = await self._calculate_new_quantity(
                    order=order,
                    current_market_data=current_market_data
                )

                decision = RefreshDecision(
                    should_refresh=True,
                    new_price=new_price,
                    new_quantity=new_quantity,
                    reason=", ".join(refresh_reasons),
                    urgency=min(Decimal("1.0"), urgency)
                )

                logger.info(
                    "refresh_decision_made",
                    order_id=order_id,
                    should_refresh=True,
                    reasons=refresh_reasons,
                    urgency=float(urgency),
                    new_price=float(new_price) if new_price else None
                )

                self.refresh_history.append({
                    "order_id": order_id,
                    "timestamp": current_time,
                    "decision": decision,
                    "old_price": order.price,
                    "new_price": new_price
                })

                return decision

            else:
                return RefreshDecision(
                    should_refresh=False,
                    reason="no_refresh_triggers"
                )

        except Exception as e:
            logger.error("refresh_evaluation_failed", order_id=order_id, error=str(e), exc_info=True)
            return RefreshDecision(should_refresh=False, reason=f"error: {str(e)}")

    async def _detect_adverse_selection(
        self,
        order: ActiveOrder,
        current_mid_price: Decimal
    ) -> bool:
        """
        Detect if order is experiencing adverse selection.

        Args:
            order: Active order
            current_mid_price: Current mid market price

        Returns:
            True if adverse selection detected
        """
        # Adverse selection: market is moving away from our order
        # For buy orders: price is going up (we're getting filled at worse prices)
        # For sell orders: price is going down

        if order.side == OrderSide.BUY:
            # If mid price is significantly above our order, we're being adversely selected
            price_diff_pct = (current_mid_price - order.price) / order.price

            # More than 0.2% above = adverse selection
            if price_diff_pct > Decimal("0.002"):
                return True

        else:  # SELL
            # If mid price is significantly below our order
            price_diff_pct = (order.price - current_mid_price) / order.price

            if price_diff_pct > Decimal("0.002"):
                return True

        return False

    async def _calculate_new_price(
        self,
        order: ActiveOrder,
        bid_price: Decimal,
        ask_price: Decimal,
        mid_price: Decimal
    ) -> Decimal:
        """
        Calculate new price for refreshed order.

        Args:
            order: Original order
            bid_price: Current best bid
            ask_price: Current best ask
            mid_price: Current mid price

        Returns:
            New order price
        """
        if order.side == OrderSide.BUY:
            # For buy orders, join the bid or slightly improve
            if self.aggressive_refresh_mode:
                # More aggressive: match or beat best bid
                tick_size = Decimal(str(self.config.get("tick_size", "0.01")))
                new_price = bid_price + tick_size
            else:
                # Conservative: match best bid
                new_price = bid_price

        else:  # SELL
            # For sell orders, join the ask or slightly improve
            if self.aggressive_refresh_mode:
                tick_size = Decimal(str(self.config.get("tick_size", "0.01")))
                new_price = ask_price - tick_size
            else:
                new_price = ask_price

        return new_price

    async def _calculate_new_quantity(
        self,
        order: ActiveOrder,
        current_market_data: pl.DataFrame
    ) -> Decimal:
        """
        Calculate new quantity for refreshed order.

        Args:
            order: Original order
            current_market_data: Current market data

        Returns:
            New order quantity
        """
        # Keep same quantity by default
        new_quantity = order.quantity

        # Adjust for any partial fills
        if order.filled_quantity > Decimal("0"):
            remaining_quantity = order.quantity - order.filled_quantity
            new_quantity = remaining_quantity

        # Could add logic to adjust size based on:
        # - Volatility (reduce size in high vol)
        # - Spread (larger size in wider spreads)
        # - Volume (adjust to market depth)

        return new_quantity

    async def update_order_status(
        self,
        order_id: str,
        filled_quantity: Optional[Decimal] = None,
        status: Optional[OrderStatus] = None,
        queue_position: Optional[int] = None
    ) -> None:
        """
        Update order status and metrics.

        Args:
            order_id: Order identifier
            filled_quantity: Filled quantity update
            status: Order status update
            queue_position: Queue position update
        """
        if order_id not in self.active_orders:
            logger.warning("update_for_unknown_order", order_id=order_id)
            return

        order = self.active_orders[order_id]

        if filled_quantity is not None:
            order.filled_quantity = filled_quantity

        if status is not None:
            order.status = status

        if queue_position is not None:
            order.queue_position = queue_position

        order.last_update_time = datetime.now(timezone.utc)

        # Track fill rate
        if status == OrderStatus.FILLED:
            self.fill_rate_tracker.append(True)
        elif status == OrderStatus.CANCELLED:
            self.fill_rate_tracker.append(False)

        logger.debug(
            "order_status_updated",
            order_id=order_id,
            filled=float(filled_quantity) if filled_quantity else None,
            status=status.value if status else None,
            queue_pos=queue_position
        )

    async def remove_order(self, order_id: str) -> None:
        """
        Remove order from active tracking.

        Args:
            order_id: Order to remove
        """
        if order_id in self.active_orders:
            del self.active_orders[order_id]

            if order_id in self.last_refresh_time:
                del self.last_refresh_time[order_id]

            logger.info("order_removed", order_id=order_id)

    async def mark_refreshed(self, order_id: str) -> None:
        """
        Mark that an order was refreshed.

        Args:
            order_id: Order identifier
        """
        self.last_refresh_time[order_id] = datetime.now(timezone.utc)

    async def get_active_order_count(self) -> int:
        """
        Get count of active orders.

        Returns:
            Number of active orders
        """
        return len(self.active_orders)

    async def get_fill_rate(self) -> Decimal:
        """
        Get historical fill rate.

        Returns:
            Fill rate (0.0 to 1.0)
        """
        if not self.fill_rate_tracker:
            return Decimal("0.5")  # Default assumption

        filled_count = sum(1 for filled in self.fill_rate_tracker if filled)
        total_count = len(self.fill_rate_tracker)

        return Decimal(str(filled_count)) / Decimal(str(total_count))

    async def get_refresh_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about refresh activity.

        Returns:
            Dictionary of refresh statistics
        """
        if not self.refresh_history:
            return {
                "total_refreshes": 0,
                "avg_urgency": 0.0,
                "common_reasons": []
            }

        total = len(self.refresh_history)

        # Calculate average urgency
        avg_urgency = sum(
            h["decision"].urgency for h in self.refresh_history
        ) / Decimal(str(total))

        # Count refresh reasons
        reason_counts: Dict[str, int] = {}
        for history in self.refresh_history:
            reasons = history["decision"].reason.split(", ")
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

        # Sort by frequency
        common_reasons = sorted(
            reason_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return {
            "total_refreshes": total,
            "avg_urgency": float(avg_urgency),
            "common_reasons": common_reasons,
            "fill_rate": float(await self.get_fill_rate())
        }
