"""Database models for trading orders.

This module provides SQLAlchemy ORM models for persisting trading orders
to TimescaleDB with support for high-frequency trading operations.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, DateTime, Numeric, Enum as SQLEnum, Index, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from structlog import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class OrderModel(Base):
    """Database model for trading orders.

    Stores all order information with support for high-frequency inserts
    and time-series queries using TimescaleDB hypertables.

    Attributes:
        id: Unique order identifier (UUID)
        order_id: Internal order ID from the system
        exchange_order_id: Order ID from the exchange
        symbol: Trading pair (e.g., 'BTC/USDT')
        side: Order side (BUY/SELL)
        order_type: Type of order (MARKET/LIMIT/etc)
        quantity: Order quantity in base currency
        price: Limit price (None for market orders)
        filled_quantity: Amount filled so far
        average_fill_price: Average price of fills
        status: Current order status
        exchange: Exchange name
        strategy: Strategy that generated the order
        timestamp: Order creation timestamp (UTC)
        updated_at: Last update timestamp (UTC)
        filled_at: Timestamp when fully filled (UTC)
        cancelled_at: Timestamp when cancelled (UTC)
        fee: Trading fee paid
        fee_currency: Currency of the fee
        metadata: Additional order context (JSONB)

    Example:
        >>> order = OrderModel(
        ...     symbol="BTC/USDT",
        ...     side="BUY",
        ...     order_type="LIMIT",
        ...     quantity=Decimal("0.1"),
        ...     price=Decimal("50000.00"),
        ...     exchange="BINANCE",
        ...     strategy="momentum_v1"
        ... )
    """

    __tablename__ = "orders"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Order identifiers
    order_id = Column(String(100), unique=True, nullable=True, index=True)
    exchange_order_id = Column(String(100), unique=True, nullable=True, index=True)

    # Order details
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)
    order_type = Column(String(20), nullable=False)
    quantity = Column(Numeric(precision=28, scale=18), nullable=False)
    price = Column(Numeric(precision=28, scale=18), nullable=True)

    # Execution details
    filled_quantity = Column(
        Numeric(precision=28, scale=18),
        nullable=False,
        default=Decimal("0")
    )
    average_fill_price = Column(Numeric(precision=28, scale=18), nullable=True)
    status = Column(String(20), nullable=False, default="PENDING", index=True)

    # Context
    exchange = Column(String(20), nullable=False, index=True)
    strategy = Column(String(50), nullable=False, index=True)

    # Timestamps
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), nullable=True)
    filled_at = Column(DateTime(timezone=True), nullable=True)
    cancelled_at = Column(DateTime(timezone=True), nullable=True)

    # Fees
    fee = Column(Numeric(precision=28, scale=18), nullable=True)
    fee_currency = Column(String(10), nullable=True)

    # Additional metadata
    metadata = Column(JSONB, nullable=False, default=dict)

    # Indexes for performance
    __table_args__ = (
        Index("idx_orders_timestamp_exchange", "timestamp", "exchange"),
        Index("idx_orders_strategy_status", "strategy", "status"),
        Index("idx_orders_symbol_timestamp", "symbol", "timestamp"),
    )

    def __repr__(self) -> str:
        """String representation of order."""
        return (
            f"<OrderModel(id={self.id}, symbol={self.symbol}, "
            f"side={self.side}, quantity={self.quantity}, "
            f"status={self.status})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary.

        Returns:
            Dictionary representation of the order

        Example:
            >>> order.to_dict()
            {
                'id': '123e4567-e89b-12d3-a456-426614174000',
                'symbol': 'BTC/USDT',
                'side': 'BUY',
                ...
            }
        """
        return {
            "id": str(self.id),
            "order_id": self.order_id,
            "exchange_order_id": self.exchange_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": str(self.quantity) if self.quantity else None,
            "price": str(self.price) if self.price else None,
            "filled_quantity": str(self.filled_quantity) if self.filled_quantity else None,
            "average_fill_price": str(self.average_fill_price) if self.average_fill_price else None,
            "status": self.status,
            "exchange": self.exchange,
            "strategy": self.strategy,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "fee": str(self.fee) if self.fee else None,
            "fee_currency": self.fee_currency,
            "metadata": self.metadata,
        }

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == "FILLED"

    @property
    def is_active(self) -> bool:
        """Check if order is active (not filled, cancelled, or rejected)."""
        return self.status in ("PENDING", "OPEN", "PARTIAL")

    @property
    def fill_percentage(self) -> Decimal:
        """Calculate fill percentage.

        Returns:
            Percentage of order filled (0-100)
        """
        if not self.quantity or self.quantity == 0:
            return Decimal("0")
        return (self.filled_quantity / self.quantity) * Decimal("100")
