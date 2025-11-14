"""
Quantum Trader AI - Centralized Order Management System
Production-grade order lifecycle management

CRITICAL: All numeric values use Decimal, never float
CRITICAL: Uses polars DataFrame for data operations
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set
from uuid import uuid4
import yaml

import polars as pl

from quantum_trader.models import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    ExecutionResult,
    ExecutionStatus,
    AuditLog,
)


logger = logging.getLogger(__name__)


class OrderLifecycleEvent(Enum):
    """Order lifecycle events"""

    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    MODIFIED = "MODIFIED"
    FAILED = "FAILED"


class OrderManager:
    """
    Centralized order management system

    Features:
    - Complete order lifecycle management (pending -> filled)
    - Order modification and cancellation with retry logic
    - Position tracking and updates
    - Fill aggregation and tracking
    - Order validation and risk checks
    - Order history management
    - Comprehensive audit trail
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize order manager

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)

        # Order tracking: {order_id: Order}
        self.orders: Dict[str, Order] = {}

        # Order status: {order_id: OrderStatus}
        self.order_status: Dict[str, OrderStatus] = {}

        # Order fills: {order_id: [(fill_price, fill_quantity, timestamp)]}
        self.order_fills: Dict[str, List[tuple]] = defaultdict(list)

        # Order lifecycle events stored as polars DataFrame
        self.lifecycle_events = pl.DataFrame(
            schema={
                "event_id": pl.Utf8,
                "order_id": pl.Utf8,
                "event_type": pl.Utf8,
                "timestamp": pl.Datetime,
                "details": pl.Utf8,
            }
        )

        # Position tracking: {symbol: {strategy: Position}}
        self.positions: Dict[str, Dict[str, Position]] = defaultdict(dict)

        # Order history stored as polars DataFrame
        self.order_history = pl.DataFrame(
            schema={
                "order_id": pl.Utf8,
                "symbol": pl.Utf8,
                "side": pl.Utf8,
                "order_type": pl.Utf8,
                "quantity": pl.Utf8,
                "price": pl.Utf8,
                "status": pl.Utf8,
                "filled_quantity": pl.Utf8,
                "average_fill_price": pl.Utf8,
                "exchange": pl.Utf8,
                "strategy": pl.Utf8,
                "created_at": pl.Datetime,
                "updated_at": pl.Datetime,
            }
        )

        # Execution results stored as polars DataFrame
        self.execution_results = pl.DataFrame(
            schema={
                "result_id": pl.Utf8,
                "order_id": pl.Utf8,
                "status": pl.Utf8,
                "filled_quantity": pl.Utf8,
                "average_price": pl.Utf8,
                "total_cost": pl.Utf8,
                "fees": pl.Utf8,
                "timestamp": pl.Datetime,
            }
        )

        # Pending cancellations
        self.pending_cancellations: Set[str] = set()

        # Pending modifications
        self.pending_modifications: Set[str] = set()

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Metrics
        self.metrics = {
            "total_orders": 0,
            "filled_orders": 0,
            "cancelled_orders": 0,
            "rejected_orders": 0,
            "total_volume": Decimal("0"),
            "total_fees": Decimal("0"),
        }

        logger.info(
            "OrderManager initialized",
            extra={
                "max_concurrent_orders": self.config["max_concurrent_orders"],
                "order_timeout_seconds": self.config["order_timeout_seconds"],
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
            "max_concurrent_orders": 50,
            "order_timeout_seconds": 30,
            "retry_attempts": 3,
            "retry_delay_ms": 1000,
            "retry_backoff_multiplier": Decimal("2.0"),
            "enable_position_tracking": True,
            "enable_fill_aggregation": True,
            "enable_order_validation": True,
            "auto_cancel_expired": True,
            "expiry_check_interval_seconds": 60,
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

                        # Extract execution settings
                        if "bot" in file_config and "execution" in file_config["bot"]:
                            exec_config = file_config["bot"]["execution"]
                            loaded_config["max_concurrent_orders"] = exec_config.get(
                                "max_concurrent_orders", 50
                            )
                            loaded_config["order_timeout_seconds"] = exec_config.get(
                                "order_timeout_seconds", 30
                            )
                            loaded_config["retry_attempts"] = exec_config.get("retry_attempts", 3)
                            loaded_config["retry_delay_ms"] = exec_config.get("retry_delay_ms", 1000)
                            retry_backoff = exec_config.get("retry_backoff_multiplier", 2.0)
                            loaded_config["retry_backoff_multiplier"] = Decimal(str(retry_backoff))

                        # Extract trading settings
                        if "bot" in file_config and "trading" in file_config["bot"]:
                            trading_config = file_config["bot"]["trading"]
                            loaded_config["max_positions"] = trading_config.get("max_positions", 20)

            except Exception as e:
                logger.warning(f"Failed to load config from {config_file}: {e}")
                continue

        return loaded_config

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType,
        exchange: str,
        strategy: str,
        price: Optional[Decimal] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[Order]:
        """
        Create a new order

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            order_type: Order type
            exchange: Exchange name
            strategy: Strategy name
            price: Limit price (required for LIMIT orders)
            metadata: Additional metadata

        Returns:
            Created Order or None if validation fails
        """
        async with self._lock:
            try:
                # Validate order parameters
                if not self._validate_order_params(symbol, quantity, price, order_type):
                    return None

                # Check concurrent order limit
                active_count = sum(
                    1
                    for status in self.order_status.values()
                    if status in [OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIAL]
                )

                if active_count >= self.config["max_concurrent_orders"]:
                    logger.error(
                        f"Maximum concurrent orders reached: {self.config['max_concurrent_orders']}"
                    )
                    return None

                # Generate order ID
                order_id = self._generate_order_id()

                # Create order
                order = Order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type=order_type,
                    exchange=exchange,
                    strategy=strategy,
                    timestamp=datetime.utcnow(),
                    price=price,
                    order_id=order_id,
                    metadata=metadata or {},
                )

                # Store order
                self.orders[order_id] = order
                self.order_status[order_id] = OrderStatus.PENDING

                # Record lifecycle event
                await self._record_lifecycle_event(
                    order_id=order_id,
                    event_type=OrderLifecycleEvent.CREATED,
                    details=f"Order created: {symbol} {side.value} {quantity} @ {price}",
                )

                # Update metrics
                self.metrics["total_orders"] += 1

                # Create audit log
                await self._create_audit_log(
                    operation="create_order",
                    component="order_manager",
                    severity="INFO",
                    details={
                        "order_id": order_id,
                        "symbol": symbol,
                        "side": side.value,
                        "quantity": str(quantity),
                        "price": str(price) if price else None,
                    },
                )

                logger.info(f"Order created: {order_id}")

                return order

            except Exception as e:
                logger.error(f"Failed to create order: {e}", exc_info=True)
                await self._create_audit_log(
                    operation="create_order_failed",
                    component="order_manager",
                    severity="ERROR",
                    details={"error": str(e)},
                )
                return None

    async def submit_order(self, order_id: str) -> bool:
        """
        Submit order to exchange with retry logic

        Args:
            order_id: Order ID

        Returns:
            True if submission successful
        """
        if order_id not in self.orders:
            logger.error(f"Order {order_id} not found")
            return False

        order = self.orders[order_id]

        # Retry logic
        max_retries = self.config["retry_attempts"]
        retry_delay = self.config["retry_delay_ms"] / 1000.0
        backoff = self.config["retry_backoff_multiplier"]

        for attempt in range(max_retries):
            try:
                # Simulate exchange submission (in production, call actual exchange API)
                success = await self._submit_to_exchange(order)

                if success:
                    async with self._lock:
                        self.order_status[order_id] = OrderStatus.OPEN

                        await self._record_lifecycle_event(
                            order_id=order_id,
                            event_type=OrderLifecycleEvent.SUBMITTED,
                            details=f"Order submitted to {order.exchange}",
                        )

                    logger.info(f"Order {order_id} submitted successfully")
                    return True
                else:
                    logger.warning(f"Order submission failed (attempt {attempt + 1}/{max_retries})")

            except Exception as e:
                logger.error(f"Order submission error: {e}", exc_info=True)

            # Wait before retry with exponential backoff
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= float(backoff)

        # All retries failed
        async with self._lock:
            self.order_status[order_id] = OrderStatus.FAILED
            self.metrics["rejected_orders"] += 1

            await self._record_lifecycle_event(
                order_id=order_id,
                event_type=OrderLifecycleEvent.FAILED,
                details=f"Order submission failed after {max_retries} attempts",
            )

        return False

    async def cancel_order(self, order_id: str, reason: str = "User requested") -> bool:
        """
        Cancel an order with retry logic

        Args:
            order_id: Order ID
            reason: Cancellation reason

        Returns:
            True if cancellation successful
        """
        if order_id not in self.orders:
            logger.error(f"Order {order_id} not found")
            return False

        if order_id in self.pending_cancellations:
            logger.warning(f"Order {order_id} cancellation already pending")
            return False

        async with self._lock:
            self.pending_cancellations.add(order_id)

        order = self.orders[order_id]

        # Retry logic
        max_retries = self.config["retry_attempts"]
        retry_delay = self.config["retry_delay_ms"] / 1000.0
        backoff = self.config["retry_backoff_multiplier"]

        for attempt in range(max_retries):
            try:
                # Simulate exchange cancellation (in production, call actual exchange API)
                success = await self._cancel_on_exchange(order)

                if success:
                    async with self._lock:
                        self.order_status[order_id] = OrderStatus.CANCELLED
                        self.pending_cancellations.discard(order_id)
                        self.metrics["cancelled_orders"] += 1

                        await self._record_lifecycle_event(
                            order_id=order_id,
                            event_type=OrderLifecycleEvent.CANCELLED,
                            details=f"Order cancelled: {reason}",
                        )

                    logger.info(f"Order {order_id} cancelled successfully")
                    return True

            except Exception as e:
                logger.error(f"Order cancellation error: {e}", exc_info=True)

            # Wait before retry with exponential backoff
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= float(backoff)

        # Cancellation failed
        async with self._lock:
            self.pending_cancellations.discard(order_id)

        logger.error(f"Failed to cancel order {order_id} after {max_retries} attempts")
        return False

    async def modify_order(
        self,
        order_id: str,
        new_quantity: Optional[Decimal] = None,
        new_price: Optional[Decimal] = None,
    ) -> bool:
        """
        Modify an existing order

        Args:
            order_id: Order ID
            new_quantity: New quantity (optional)
            new_price: New price (optional)

        Returns:
            True if modification successful
        """
        if order_id not in self.orders:
            logger.error(f"Order {order_id} not found")
            return False

        if order_id in self.pending_modifications:
            logger.warning(f"Order {order_id} modification already pending")
            return False

        async with self._lock:
            self.pending_modifications.add(order_id)

        order = self.orders[order_id]

        # Validate new parameters
        if new_quantity is not None and new_quantity <= Decimal("0"):
            logger.error(f"Invalid new quantity: {new_quantity}")
            async with self._lock:
                self.pending_modifications.discard(order_id)
            return False

        # Retry logic
        max_retries = self.config["retry_attempts"]
        retry_delay = self.config["retry_delay_ms"] / 1000.0
        backoff = self.config["retry_backoff_multiplier"]

        for attempt in range(max_retries):
            try:
                # Simulate exchange modification (in production, call actual exchange API)
                success = await self._modify_on_exchange(order, new_quantity, new_price)

                if success:
                    async with self._lock:
                        # Update order
                        if new_quantity is not None:
                            order.quantity = new_quantity
                        if new_price is not None:
                            order.price = new_price

                        self.pending_modifications.discard(order_id)

                        await self._record_lifecycle_event(
                            order_id=order_id,
                            event_type=OrderLifecycleEvent.MODIFIED,
                            details=f"Order modified: qty={new_quantity}, price={new_price}",
                        )

                    logger.info(f"Order {order_id} modified successfully")
                    return True

            except Exception as e:
                logger.error(f"Order modification error: {e}", exc_info=True)

            # Wait before retry with exponential backoff
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= float(backoff)

        # Modification failed
        async with self._lock:
            self.pending_modifications.discard(order_id)

        logger.error(f"Failed to modify order {order_id} after {max_retries} attempts")
        return False

    async def process_fill(
        self,
        order_id: str,
        fill_quantity: Decimal,
        fill_price: Decimal,
        fees: Decimal = Decimal("0"),
        exchange_order_id: Optional[str] = None,
    ) -> None:
        """
        Process an order fill

        Args:
            order_id: Order ID
            fill_quantity: Filled quantity
            fill_price: Fill price
            fees: Trading fees
            exchange_order_id: Exchange order ID
        """
        if order_id not in self.orders:
            logger.error(f"Order {order_id} not found for fill processing")
            return

        async with self._lock:
            try:
                order = self.orders[order_id]

                # Record fill
                fill_record = (fill_price, fill_quantity, datetime.utcnow())
                self.order_fills[order_id].append(fill_record)

                # Calculate total filled quantity
                total_filled = sum(Decimal(str(f[1])) for f in self.order_fills[order_id])

                # Calculate average fill price
                total_cost = sum(Decimal(str(f[0])) * Decimal(str(f[1])) for f in self.order_fills[order_id])
                avg_price = total_cost / total_filled if total_filled > Decimal("0") else Decimal("0")

                # Update order status
                if total_filled >= order.quantity:
                    self.order_status[order_id] = OrderStatus.FILLED
                    event_type = OrderLifecycleEvent.FILLED
                    self.metrics["filled_orders"] += 1
                else:
                    self.order_status[order_id] = OrderStatus.PARTIAL
                    event_type = OrderLifecycleEvent.PARTIAL_FILL

                # Update metrics
                self.metrics["total_volume"] += fill_quantity
                self.metrics["total_fees"] += fees

                # Record lifecycle event
                await self._record_lifecycle_event(
                    order_id=order_id,
                    event_type=event_type,
                    details=f"Fill: {fill_quantity} @ {fill_price}, Total: {total_filled}/{order.quantity}",
                )

                # Create execution result
                execution_result = ExecutionResult(
                    order_id=order_id,
                    status=(
                        ExecutionStatus.SUCCESS
                        if total_filled >= order.quantity
                        else ExecutionStatus.PARTIAL
                    ),
                    symbol=order.symbol,
                    side=order.side,
                    filled_quantity=fill_quantity,
                    average_price=fill_price,
                    total_cost=fill_quantity * fill_price,
                    fees=fees,
                    exchange=order.exchange,
                    timestamp=datetime.utcnow(),
                    exchange_order_id=exchange_order_id,
                )

                # Record execution result
                await self._record_execution_result(execution_result)

                # Update position tracking
                if self.config["enable_position_tracking"]:
                    await self._update_position(order, fill_quantity, fill_price)

                logger.info(
                    f"Processed fill for order {order_id}: {fill_quantity} @ {fill_price}, "
                    f"Total filled: {total_filled}/{order.quantity}"
                )

            except Exception as e:
                logger.error(f"Failed to process fill for order {order_id}: {e}", exc_info=True)

    async def _update_position(
        self, order: Order, fill_quantity: Decimal, fill_price: Decimal
    ) -> None:
        """
        Update position tracking based on fill

        Args:
            order: Filled order
            fill_quantity: Filled quantity
            fill_price: Fill price
        """
        try:
            symbol = order.symbol
            strategy = order.strategy

            if symbol not in self.positions or strategy not in self.positions[symbol]:
                # Create new position
                if order.side == OrderSide.BUY:
                    quantity = fill_quantity
                else:
                    quantity = -fill_quantity

                position = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=fill_price,
                    current_price=fill_price,
                    exchange=order.exchange,
                    strategy=strategy,
                    opened_at=datetime.utcnow(),
                    position_id=str(uuid4()),
                )

                self.positions[symbol][strategy] = position
                logger.info(f"Created new position: {position.position_id}")

            else:
                # Update existing position
                position = self.positions[symbol][strategy]

                if order.side == OrderSide.BUY:
                    # Add to long position
                    new_quantity = position.quantity + fill_quantity
                    new_entry_price = (
                        (position.entry_price * abs(position.quantity)) + (fill_price * fill_quantity)
                    ) / abs(new_quantity)
                else:
                    # Reduce long or add to short position
                    new_quantity = position.quantity - fill_quantity
                    if new_quantity != Decimal("0"):
                        new_entry_price = position.entry_price
                    else:
                        new_entry_price = fill_price

                position.quantity = new_quantity
                position.entry_price = new_entry_price
                position.current_price = fill_price

                # Close position if quantity is zero
                if position.quantity == Decimal("0"):
                    del self.positions[symbol][strategy]
                    logger.info(f"Closed position: {position.position_id}")
                else:
                    logger.info(f"Updated position: {position.position_id}")

        except Exception as e:
            logger.error(f"Failed to update position: {e}", exc_info=True)

    async def _record_lifecycle_event(
        self, order_id: str, event_type: OrderLifecycleEvent, details: str
    ) -> None:
        """
        Record order lifecycle event

        Args:
            order_id: Order ID
            event_type: Event type
            details: Event details
        """
        try:
            event_id = str(uuid4())

            new_event = pl.DataFrame(
                {
                    "event_id": [event_id],
                    "order_id": [order_id],
                    "event_type": [event_type.value],
                    "timestamp": [datetime.utcnow()],
                    "details": [details],
                }
            )

            self.lifecycle_events = pl.concat([self.lifecycle_events, new_event])

        except Exception as e:
            logger.error(f"Failed to record lifecycle event: {e}", exc_info=True)

    async def _record_execution_result(self, result: ExecutionResult) -> None:
        """
        Record execution result

        Args:
            result: Execution result
        """
        try:
            result_id = str(uuid4())

            new_result = pl.DataFrame(
                {
                    "result_id": [result_id],
                    "order_id": [result.order_id],
                    "status": [result.status.value],
                    "filled_quantity": [str(result.filled_quantity)],
                    "average_price": [str(result.average_price)],
                    "total_cost": [str(result.total_cost)],
                    "fees": [str(result.fees)],
                    "timestamp": [result.timestamp],
                }
            )

            self.execution_results = pl.concat([self.execution_results, new_result])

        except Exception as e:
            logger.error(f"Failed to record execution result: {e}", exc_info=True)

    async def _submit_to_exchange(self, order: Order) -> bool:
        """
        Submit order to exchange (placeholder for actual implementation)

        Args:
            order: Order to submit

        Returns:
            True if successful
        """
        # In production, this would call the actual exchange API
        # For now, simulate successful submission
        await asyncio.sleep(0.01)  # Simulate network delay
        return True

    async def _cancel_on_exchange(self, order: Order) -> bool:
        """
        Cancel order on exchange (placeholder for actual implementation)

        Args:
            order: Order to cancel

        Returns:
            True if successful
        """
        # In production, this would call the actual exchange API
        await asyncio.sleep(0.01)  # Simulate network delay
        return True

    async def _modify_on_exchange(
        self, order: Order, new_quantity: Optional[Decimal], new_price: Optional[Decimal]
    ) -> bool:
        """
        Modify order on exchange (placeholder for actual implementation)

        Args:
            order: Order to modify
            new_quantity: New quantity
            new_price: New price

        Returns:
            True if successful
        """
        # In production, this would call the actual exchange API
        await asyncio.sleep(0.01)  # Simulate network delay
        return True

    def _validate_order_params(
        self, symbol: str, quantity: Decimal, price: Optional[Decimal], order_type: OrderType
    ) -> bool:
        """
        Validate order parameters

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            price: Order price
            order_type: Order type

        Returns:
            True if valid
        """
        if not self.config["enable_order_validation"]:
            return True

        if quantity <= Decimal("0"):
            logger.error(f"Invalid quantity: {quantity}")
            return False

        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if price is None or price <= Decimal("0"):
                logger.error(f"Invalid price for {order_type.value}: {price}")
                return False

        return True

    def _generate_order_id(self) -> str:
        """
        Generate unique order ID

        Returns:
            Order ID
        """
        return f"ORD_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}"

    async def _create_audit_log(
        self, operation: str, component: str, severity: str, details: Dict
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
                user_id="order_manager",
                component=component,
                severity=severity,
                details=details,
            )

            logger.log(
                getattr(logging, severity),
                f"Audit: {operation}",
                extra={"audit_log": audit_log},
            )

        except Exception as e:
            logger.error(f"Failed to create audit log: {e}", exc_info=True)

    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID

        Args:
            order_id: Order ID

        Returns:
            Order or None if not found
        """
        return self.orders.get(order_id)

    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """
        Get order status

        Args:
            order_id: Order ID

        Returns:
            Order status or None if not found
        """
        return self.order_status.get(order_id)

    def get_order_fills(self, order_id: str) -> List[tuple]:
        """
        Get order fills

        Args:
            order_id: Order ID

        Returns:
            List of fills (price, quantity, timestamp)
        """
        return self.order_fills.get(order_id, [])

    def get_position(self, symbol: str, strategy: str) -> Optional[Position]:
        """
        Get position for symbol and strategy

        Args:
            symbol: Trading symbol
            strategy: Strategy name

        Returns:
            Position or None if not found
        """
        return self.positions.get(symbol, {}).get(strategy)

    def get_all_positions(self) -> List[Position]:
        """
        Get all active positions

        Returns:
            List of positions
        """
        positions = []
        for symbol_positions in self.positions.values():
            positions.extend(symbol_positions.values())
        return positions

    def get_lifecycle_events(self, order_id: str) -> pl.DataFrame:
        """
        Get lifecycle events for an order

        Args:
            order_id: Order ID

        Returns:
            DataFrame of lifecycle events
        """
        return self.lifecycle_events.filter(pl.col("order_id") == order_id)

    def get_execution_results(self, order_id: str) -> pl.DataFrame:
        """
        Get execution results for an order

        Args:
            order_id: Order ID

        Returns:
            DataFrame of execution results
        """
        return self.execution_results.filter(pl.col("order_id") == order_id)

    def get_metrics(self) -> Dict:
        """
        Get order manager metrics

        Returns:
            Dictionary of metrics
        """
        active_orders = sum(
            1
            for status in self.order_status.values()
            if status in [OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIAL]
        )

        total_positions = sum(len(positions) for positions in self.positions.values())

        return {
            "total_orders": self.metrics["total_orders"],
            "filled_orders": self.metrics["filled_orders"],
            "cancelled_orders": self.metrics["cancelled_orders"],
            "rejected_orders": self.metrics["rejected_orders"],
            "active_orders": active_orders,
            "total_positions": total_positions,
            "total_volume": str(self.metrics["total_volume"]),
            "total_fees": str(self.metrics["total_fees"]),
        }
