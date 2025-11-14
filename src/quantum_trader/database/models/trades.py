"""Database models for executed trades.

This module provides SQLAlchemy ORM models for persisting executed trades
to TimescaleDB with support for high-frequency trade storage and analysis.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, DateTime, Numeric, Index, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from structlog import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class TradeModel(Base):
    """Database model for executed trades.

    Stores all executed trade information with support for high-frequency
    inserts and time-series queries using TimescaleDB hypertables.

    Attributes:
        id: Unique trade identifier (UUID)
        trade_id: Internal trade ID
        exchange_trade_id: Trade ID from the exchange
        order_id: Related order ID
        position_id: Related position ID (if applicable)

        # Trade Details
        symbol: Trading pair (e.g., 'BTC/USDT')
        side: Trade side (BUY/SELL)
        quantity: Trade quantity
        price: Execution price
        quote_quantity: Quote currency amount (quantity * price)

        # Context
        exchange: Exchange name
        strategy: Strategy that executed the trade
        timestamp: Trade execution timestamp (UTC)

        # Fees
        fee: Trading fee paid
        fee_currency: Currency of the fee

        # P&L (for closing trades)
        realized_pnl: Realized P&L (if closing a position)
        pnl_percentage: P&L as percentage

        # Trade type
        is_maker: Whether this was a maker trade
        is_liquidation: Whether this was a liquidation
        is_closing: Whether this trade closes a position

        # Additional context
        metadata: Additional trade context (JSONB)

    Example:
        >>> trade = TradeModel(
        ...     symbol="BTC/USDT",
        ...     side="BUY",
        ...     quantity=Decimal("0.1"),
        ...     price=Decimal("50000.00"),
        ...     exchange="BINANCE",
        ...     strategy="momentum_v1"
        ... )
    """

    __tablename__ = "trades"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Trade identifiers
    trade_id = Column(String(100), unique=True, nullable=False, index=True)
    exchange_trade_id = Column(String(100), nullable=True, index=True)
    order_id = Column(String(100), nullable=True, index=True)
    position_id = Column(String(100), nullable=True, index=True)

    # Trade details
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)
    quantity = Column(Numeric(precision=28, scale=18), nullable=False)
    price = Column(Numeric(precision=28, scale=18), nullable=False)
    quote_quantity = Column(Numeric(precision=28, scale=18), nullable=False)

    # Context
    exchange = Column(String(20), nullable=False, index=True)
    strategy = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    # Fees
    fee = Column(Numeric(precision=28, scale=18), nullable=False, default=Decimal("0"))
    fee_currency = Column(String(10), nullable=True)

    # P&L
    realized_pnl = Column(Numeric(precision=28, scale=18), nullable=True)
    pnl_percentage = Column(Numeric(precision=28, scale=18), nullable=True)

    # Trade type
    is_maker = Column(Boolean, nullable=False, default=False)
    is_liquidation = Column(Boolean, nullable=False, default=False)
    is_closing = Column(Boolean, nullable=False, default=False)

    # Additional metadata
    metadata = Column(JSONB, nullable=False, default=dict)

    # Indexes for performance
    __table_args__ = (
        Index("idx_trades_timestamp_exchange", "timestamp", "exchange"),
        Index("idx_trades_strategy_symbol", "strategy", "symbol"),
        Index("idx_trades_symbol_timestamp", "symbol", "timestamp"),
    )

    def __repr__(self) -> str:
        """String representation of trade."""
        return (
            f"<TradeModel(id={self.trade_id}, symbol={self.symbol}, "
            f"side={self.side}, quantity={self.quantity}, price={self.price})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary.

        Returns:
            Dictionary representation of the trade

        Example:
            >>> trade.to_dict()
            {
                'trade_id': 'BTC_USDT_12345',
                'symbol': 'BTC/USDT',
                'side': 'BUY',
                ...
            }
        """
        return {
            "id": str(self.id),
            "trade_id": self.trade_id,
            "exchange_trade_id": self.exchange_trade_id,
            "order_id": self.order_id,
            "position_id": self.position_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": str(self.quantity) if self.quantity is not None else None,
            "price": str(self.price) if self.price is not None else None,
            "quote_quantity": str(self.quote_quantity) if self.quote_quantity is not None else None,
            "exchange": self.exchange,
            "strategy": self.strategy,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "fee": str(self.fee) if self.fee is not None else None,
            "fee_currency": self.fee_currency,
            "realized_pnl": str(self.realized_pnl) if self.realized_pnl is not None else None,
            "pnl_percentage": str(self.pnl_percentage) if self.pnl_percentage is not None else None,
            "is_maker": self.is_maker,
            "is_liquidation": self.is_liquidation,
            "is_closing": self.is_closing,
            "metadata": self.metadata,
        }

    @property
    def total_cost(self) -> Decimal:
        """Calculate total cost including fees.

        Returns:
            Total cost (quote_quantity + fee in quote currency)
        """
        fee_in_quote = self.fee if self.fee else Decimal("0")
        # If fee is in base currency, convert it
        if self.fee_currency and self.fee_currency != self.symbol.split("/")[1]:
            # Fee is in base currency, convert to quote
            fee_in_quote = self.fee * self.price

        return self.quote_quantity + fee_in_quote

    @property
    def is_buy(self) -> bool:
        """Check if trade is a buy."""
        return self.side.upper() == "BUY"

    @property
    def is_sell(self) -> bool:
        """Check if trade is a sell."""
        return self.side.upper() == "SELL"

    @property
    def is_profitable(self) -> bool:
        """Check if trade was profitable (for closing trades).

        Returns:
            True if realized P&L is positive, False otherwise
        """
        if self.realized_pnl is None:
            return False
        return self.realized_pnl > 0
