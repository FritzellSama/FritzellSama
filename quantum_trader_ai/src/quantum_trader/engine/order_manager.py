"""
Order Lifecycle Management System
Production-ready order management with comprehensive tracking and validation
"""

from decimal import Decimal
from typing import Optional, Dict, List, Any
import logging
import asyncio
from datetime import datetime
from enum import Enum
import polars as pl
from dataclasses import dataclass, asdict
import uuid

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enumeration"""
    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "BUY"
    SELL = "SELL"


class TimeInForce(Enum):
    """Time in force enumeration"""
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    DAY = "DAY"  # Day order


@dataclass
class Order:
    """Order data structure"""
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: Decimal
    price: Optional[Decimal]
    stop_price: Optional[Decimal]
    time_in_force: str
    status: str
    created_at: str
    updated_at: str
    exchange: str
    user_id: str
    filled_quantity: Decimal = Decimal('0')
    average_fill_price: Optional[Decimal] = None
    remaining_quantity: Optional[Decimal] = None
    client_order_id: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class OrderValidationResult:
    """Order validation result"""
    valid: bool
    order_id: str
    errors: List[str]
    warnings: List[str]


class OrderManager:
    """Order lifecycle management system"""

    def __init__(self) -> None:
        """Initialize order manager with configuration"""
        self.config = get_config()
        self._load_config()
        self._orders: Dict[str, Order] = {}
        self._order_history: List[Dict[str, Any]] = []
        self._validation_rules: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        logger.info("OrderManager initialized")

    def _load_config(self) -> None:
        """Load configuration from engine.yaml"""
        self.routing_enabled = self.config.get_bool('engine', 'order_routing.enabled')
        self.routing_strategy = self.config.get_string('engine', 'order_routing.strategy')

        # Load exchanges
        exchanges = self.config.get_list('engine', 'order_routing.exchanges')
        self.exchanges = [str(ex) for ex in exchanges]

        self.price_improvement_min_bps = self.config.get_int('engine', 'order_routing.price_improvement_min_bps')
        self.latency_threshold_ms = self.config.get_int('engine', 'order_routing.latency_threshold_ms')

        # Set up validation rules
        self._setup_validation_rules()

        logger.info(
            f"OrderManager configured: routing={self.routing_enabled}, "
            f"strategy={self.routing_strategy}, exchanges={len(self.exchanges)}"
        )

    def _setup_validation_rules(self) -> None:
        """Set up order validation rules"""
        self._validation_rules = {
            'min_quantity': Decimal('0.00000001'),
            'max_quantity': Decimal('1000000'),
            'min_price': Decimal('0.00000001'),
            'max_price': Decimal('1000000000'),
            'supported_order_types': [ot.value for ot in OrderType],
            'supported_sides': [s.value for s in OrderSide],
            'supported_tif': [tif.value for tif in TimeInForce],
            'supported_exchanges': self.exchanges
        }

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        time_in_force: str = "GTC",
        exchange: str = "BINANCE",
        user_id: str = "SYSTEM",
        client_order_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Order:
        """
        Create a new order

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            order_type: Order type (MARKET/LIMIT/etc)
            quantity: Order quantity
            price: Limit price (optional)
            stop_price: Stop price (optional)
            time_in_force: Time in force (GTC/IOC/FOK/DAY)
            exchange: Target exchange
            user_id: User ID
            client_order_id: Client-provided order ID (optional)
            metadata: Additional metadata (optional)

        Returns:
            Created Order object

        Raises:
            ValueError: If order parameters are invalid
        """
        async with self._lock:
            # Generate order ID
            order_id = str(uuid.uuid4())

            # Create order object
            now = datetime.utcnow().isoformat()
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                status=OrderStatus.CREATED.value,
                created_at=now,
                updated_at=now,
                exchange=exchange,
                user_id=user_id,
                client_order_id=client_order_id,
                metadata=metadata or {}
            )

            # Calculate remaining quantity
            order.remaining_quantity = quantity

            # Store order
            self._orders[order_id] = order

            logger.info(
                f"Order created: {order_id} - {side} {quantity} {symbol} @ {price} on {exchange}"
            )

            return order

    async def validate_order(self, order_id: str) -> OrderValidationResult:
        """
        Validate an order

        Args:
            order_id: Order ID to validate

        Returns:
            OrderValidationResult with validation details

        Raises:
            ValueError: If order not found
        """
        async with self._lock:
            if order_id not in self._orders:
                raise ValueError(f"Order {order_id} not found")

            order = self._orders[order_id]

            errors: List[str] = []
            warnings: List[str] = []

            # Validate order type
            if order.order_type not in self._validation_rules['supported_order_types']:
                errors.append(f"Unsupported order type: {order.order_type}")

            # Validate side
            if order.side not in self._validation_rules['supported_sides']:
                errors.append(f"Unsupported order side: {order.side}")

            # Validate time in force
            if order.time_in_force not in self._validation_rules['supported_tif']:
                errors.append(f"Unsupported time in force: {order.time_in_force}")

            # Validate exchange
            if order.exchange not in self._validation_rules['supported_exchanges']:
                errors.append(f"Unsupported exchange: {order.exchange}")

            # Validate quantity
            if order.quantity < self._validation_rules['min_quantity']:
                errors.append(f"Quantity {order.quantity} below minimum {self._validation_rules['min_quantity']}")

            if order.quantity > self._validation_rules['max_quantity']:
                errors.append(f"Quantity {order.quantity} exceeds maximum {self._validation_rules['max_quantity']}")

            # Validate price for limit orders
            if order.order_type in ["LIMIT", "STOP_LIMIT"]:
                if order.price is None:
                    errors.append(f"Price required for {order.order_type} orders")
                elif order.price < self._validation_rules['min_price']:
                    errors.append(f"Price {order.price} below minimum {self._validation_rules['min_price']}")
                elif order.price > self._validation_rules['max_price']:
                    errors.append(f"Price {order.price} exceeds maximum {self._validation_rules['max_price']}")

            # Validate stop price for stop orders
            if order.order_type in ["STOP_LOSS", "STOP_LIMIT", "TRAILING_STOP"]:
                if order.stop_price is None:
                    errors.append(f"Stop price required for {order.order_type} orders")

            # Validate symbol format
            if not order.symbol or '/' not in order.symbol:
                errors.append(f"Invalid symbol format: {order.symbol}")

            # Check for warnings
            if order.order_type == "MARKET":
                warnings.append("Market orders may experience slippage")

            if order.time_in_force == "IOC":
                warnings.append("IOC orders may be partially filled or cancelled")

            # Determine if valid
            valid = len(errors) == 0

            # Update order status
            if valid:
                order.status = OrderStatus.VALIDATED.value
                order.updated_at = datetime.utcnow().isoformat()
            else:
                order.status = OrderStatus.REJECTED.value
                order.error_message = "; ".join(errors)
                order.updated_at = datetime.utcnow().isoformat()

            result = OrderValidationResult(
                valid=valid,
                order_id=order_id,
                errors=errors,
                warnings=warnings
            )

            logger.info(
                f"Order {order_id} validation: "
                f"valid={valid}, errors={len(errors)}, warnings={len(warnings)}"
            )

            return result

    async def track_order(
        self,
        order_id: str,
        status: str,
        filled_quantity: Optional[Decimal] = None,
        average_fill_price: Optional[Decimal] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update order tracking information

        Args:
            order_id: Order ID to update
            status: New order status
            filled_quantity: Filled quantity (optional)
            average_fill_price: Average fill price (optional)
            error_message: Error message (optional)

        Returns:
            bool: True if update successful

        Raises:
            ValueError: If order not found
        """
        async with self._lock:
            if order_id not in self._orders:
                raise ValueError(f"Order {order_id} not found")

            order = self._orders[order_id]

            # Update status
            old_status = order.status
            order.status = status
            order.updated_at = datetime.utcnow().isoformat()

            # Update filled quantity
            if filled_quantity is not None:
                order.filled_quantity = filled_quantity
                order.remaining_quantity = order.quantity - filled_quantity

            # Update average fill price
            if average_fill_price is not None:
                order.average_fill_price = average_fill_price

            # Update error message
            if error_message is not None:
                order.error_message = error_message

            # Archive completed orders
            if status in [
                OrderStatus.FILLED.value,
                OrderStatus.CANCELLED.value,
                OrderStatus.REJECTED.value,
                OrderStatus.EXPIRED.value,
                OrderStatus.FAILED.value
            ]:
                self._archive_order(order)

            logger.info(
                f"Order {order_id} updated: {old_status} -> {status}, "
                f"filled={filled_quantity}/{order.quantity}"
            )

            return True

    def _archive_order(self, order: Order) -> None:
        """Archive completed order to history"""
        order_dict = asdict(order)

        # Convert Decimal to string for storage
        for key, value in order_dict.items():
            if isinstance(value, Decimal):
                order_dict[key] = str(value)

        self._order_history.append(order_dict)

    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> pl.DataFrame:
        """
        Get order history with optional filters

        Args:
            symbol: Filter by symbol (optional)
            user_id: Filter by user ID (optional)
            status: Filter by status (optional)
            limit: Maximum number of records to return

        Returns:
            Polars DataFrame with order history
        """
        async with self._lock:
            if not self._order_history:
                return pl.DataFrame()

            # Create DataFrame from history
            df = pl.DataFrame(self._order_history)

            # Apply filters
            if symbol:
                df = df.filter(pl.col('symbol') == symbol)

            if user_id:
                df = df.filter(pl.col('user_id') == user_id)

            if status:
                df = df.filter(pl.col('status') == status)

            # Limit results
            df = df.tail(limit)

            logger.info(f"Retrieved {len(df)} order history records")

            return df

    async def get_active_orders(
        self,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None
    ) -> List[Order]:
        """
        Get active (non-terminal) orders

        Args:
            symbol: Filter by symbol (optional)
            exchange: Filter by exchange (optional)

        Returns:
            List of active Order objects
        """
        async with self._lock:
            active_statuses = [
                OrderStatus.CREATED.value,
                OrderStatus.VALIDATED.value,
                OrderStatus.SUBMITTED.value,
                OrderStatus.ACKNOWLEDGED.value,
                OrderStatus.PARTIALLY_FILLED.value
            ]

            active_orders = [
                order for order in self._orders.values()
                if order.status in active_statuses
            ]

            # Apply filters
            if symbol:
                active_orders = [o for o in active_orders if o.symbol == symbol]

            if exchange:
                active_orders = [o for o in active_orders if o.exchange == exchange]

            logger.info(f"Retrieved {len(active_orders)} active orders")

            return active_orders

    async def cancel_order(self, order_id: str, reason: str = "User requested") -> bool:
        """
        Cancel an order

        Args:
            order_id: Order ID to cancel
            reason: Cancellation reason

        Returns:
            bool: True if cancellation successful

        Raises:
            ValueError: If order not found or already terminal
        """
        async with self._lock:
            if order_id not in self._orders:
                raise ValueError(f"Order {order_id} not found")

            order = self._orders[order_id]

            # Check if order can be cancelled
            terminal_statuses = [
                OrderStatus.FILLED.value,
                OrderStatus.CANCELLED.value,
                OrderStatus.REJECTED.value,
                OrderStatus.EXPIRED.value
            ]

            if order.status in terminal_statuses:
                raise ValueError(f"Order {order_id} cannot be cancelled (status: {order.status})")

            # Update order
            order.status = OrderStatus.CANCELLED.value
            order.error_message = reason
            order.updated_at = datetime.utcnow().isoformat()

            # Archive order
            self._archive_order(order)

            logger.info(f"Order {order_id} cancelled: {reason}")

            return True

    async def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID

        Args:
            order_id: Order ID to retrieve

        Returns:
            Order object or None if not found
        """
        async with self._lock:
            return self._orders.get(order_id)

    def get_order_metrics(self) -> Dict[str, Any]:
        """
        Get order management metrics

        Returns:
            Dictionary with metrics
        """
        total_orders = len(self._order_history) + len(self._orders)

        status_counts = {}
        for order in list(self._orders.values()) + [
            Order(**{k: Decimal(v) if k in ['quantity', 'price', 'stop_price', 'filled_quantity', 'remaining_quantity', 'average_fill_price'] and v and isinstance(v, str) else v for k, v in oh.items()})
            for oh in self._order_history
        ]:
            status = order.status
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            'total_orders': total_orders,
            'active_orders': len(self._orders),
            'historical_orders': len(self._order_history),
            'status_counts': status_counts
        }
