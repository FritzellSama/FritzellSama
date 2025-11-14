"""Database models for trading positions.

This module provides SQLAlchemy ORM models for persisting trading positions
to TimescaleDB with support for real-time position tracking and P&L calculations.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, DateTime, Numeric, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from structlog import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class PositionModel(Base):
    """Database model for trading positions.

    Stores all position information with support for real-time updates
    and time-series queries using TimescaleDB hypertables.

    Attributes:
        id: Unique position identifier (UUID)
        position_id: Internal position ID
        symbol: Trading pair (e.g., 'BTC/USDT')
        exchange: Exchange name
        strategy: Strategy managing this position

        # Position Details
        quantity: Position size (positive=long, negative=short)
        entry_price: Average entry price
        current_price: Current market price
        liquidation_price: Liquidation price (for leveraged positions)

        # P&L
        unrealized_pnl: Current unrealized profit/loss
        realized_pnl: Realized P&L from partial closes
        total_pnl: Total P&L (realized + unrealized)

        # Risk
        stop_loss: Stop loss price
        take_profit: Take profit price
        leverage: Leverage used for position

        # Status
        is_open: Whether position is currently open
        opened_at: Position open timestamp (UTC)
        closed_at: Position close timestamp (UTC)
        updated_at: Last update timestamp (UTC)

        # Fees
        total_fees: Total fees paid for this position
        fee_currency: Currency of fees

        # Additional context
        metadata: Additional position context (JSONB)

    Example:
        >>> position = PositionModel(
        ...     symbol="BTC/USDT",
        ...     quantity=Decimal("0.5"),
        ...     entry_price=Decimal("50000.00"),
        ...     current_price=Decimal("51000.00"),
        ...     exchange="BINANCE",
        ...     strategy="momentum_v1"
        ... )
    """

    __tablename__ = "positions"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Position identifiers
    position_id = Column(String(100), unique=True, nullable=False, index=True)

    # Position details
    symbol = Column(String(20), nullable=False, index=True)
    exchange = Column(String(20), nullable=False, index=True)
    strategy = Column(String(50), nullable=False, index=True)

    # Quantity and prices
    quantity = Column(Numeric(precision=28, scale=18), nullable=False)
    entry_price = Column(Numeric(precision=28, scale=18), nullable=False)
    current_price = Column(Numeric(precision=28, scale=18), nullable=False)
    liquidation_price = Column(Numeric(precision=28, scale=18), nullable=True)

    # P&L
    unrealized_pnl = Column(
        Numeric(precision=28, scale=18),
        nullable=False,
        default=Decimal("0")
    )
    realized_pnl = Column(
        Numeric(precision=28, scale=18),
        nullable=False,
        default=Decimal("0")
    )
    total_pnl = Column(
        Numeric(precision=28, scale=18),
        nullable=False,
        default=Decimal("0")
    )

    # Risk management
    stop_loss = Column(Numeric(precision=28, scale=18), nullable=True)
    take_profit = Column(Numeric(precision=28, scale=18), nullable=True)
    leverage = Column(
        Numeric(precision=10, scale=2),
        nullable=False,
        default=Decimal("1.0")
    )

    # Status
    is_open = Column(Boolean, nullable=False, default=True, index=True)

    # Timestamps
    opened_at = Column(DateTime(timezone=True), nullable=False, index=True)
    closed_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False)

    # Fees
    total_fees = Column(
        Numeric(precision=28, scale=18),
        nullable=False,
        default=Decimal("0")
    )
    fee_currency = Column(String(10), nullable=True)

    # Additional metadata
    metadata = Column(JSONB, nullable=False, default=dict)

    # Indexes for performance
    __table_args__ = (
        Index("idx_positions_symbol_is_open", "symbol", "is_open"),
        Index("idx_positions_strategy_is_open", "strategy", "is_open"),
        Index("idx_positions_opened_at_exchange", "opened_at", "exchange"),
    )

    def __repr__(self) -> str:
        """String representation of position."""
        return (
            f"<PositionModel(id={self.position_id}, symbol={self.symbol}, "
            f"quantity={self.quantity}, pnl={self.total_pnl})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary.

        Returns:
            Dictionary representation of the position

        Example:
            >>> position.to_dict()
            {
                'position_id': 'BTC_USDT_123',
                'symbol': 'BTC/USDT',
                'quantity': '0.5',
                ...
            }
        """
        return {
            "id": str(self.id),
            "position_id": self.position_id,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "strategy": self.strategy,
            "quantity": str(self.quantity) if self.quantity is not None else None,
            "entry_price": str(self.entry_price) if self.entry_price is not None else None,
            "current_price": str(self.current_price) if self.current_price is not None else None,
            "liquidation_price": str(self.liquidation_price) if self.liquidation_price is not None else None,
            "unrealized_pnl": str(self.unrealized_pnl) if self.unrealized_pnl is not None else None,
            "realized_pnl": str(self.realized_pnl) if self.realized_pnl is not None else None,
            "total_pnl": str(self.total_pnl) if self.total_pnl is not None else None,
            "stop_loss": str(self.stop_loss) if self.stop_loss is not None else None,
            "take_profit": str(self.take_profit) if self.take_profit is not None else None,
            "leverage": str(self.leverage) if self.leverage is not None else None,
            "is_open": self.is_open,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "total_fees": str(self.total_fees) if self.total_fees is not None else None,
            "fee_currency": self.fee_currency,
            "metadata": self.metadata,
        }

    @property
    def pnl_percent(self) -> Decimal:
        """Calculate P&L percentage.

        Returns:
            P&L as percentage of entry value
        """
        if not self.entry_price or self.entry_price == 0:
            return Decimal("0")

        price_diff = self.current_price - self.entry_price
        return (price_diff / self.entry_price) * Decimal("100")

    @property
    def position_value(self) -> Decimal:
        """Calculate current position value.

        Returns:
            Current market value of the position
        """
        return abs(self.quantity) * self.current_price

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0

    @property
    def is_profitable(self) -> bool:
        """Check if position is currently profitable."""
        return self.total_pnl > 0

    def update_pnl(self, current_price: Decimal) -> None:
        """Update P&L based on current price.

        Args:
            current_price: Current market price

        Example:
            >>> position.update_pnl(Decimal("51500.00"))
        """
        self.current_price = current_price
        price_diff = self.current_price - self.entry_price
        self.unrealized_pnl = price_diff * self.quantity
        self.total_pnl = self.realized_pnl + self.unrealized_pnl
        self.updated_at = datetime.utcnow()
