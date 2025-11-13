"""
Core Data Models for Quantum Trader
All data models used across the trading system
"""

from decimal import Decimal
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class OrderSide(Enum):
    """Order side (buy/sell)"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"


class SignalAction(Enum):
    """Trading signal actions"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class Exchange(Enum):
    """Supported exchanges"""
    BINANCE = "BINANCE"
    BYBIT = "BYBIT"
    OKX = "OKX"
    KUCOIN = "KUCOIN"
    BITGET = "BITGET"


class TimeFrame(Enum):
    """Trading timeframes"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Order:
    """Trading order - immutable after creation"""
    symbol: str  # Trading pair e.g. 'BTC/USDT'
    side: OrderSide  # BUY or SELL
    quantity: Decimal  # Order amount (NEVER float)
    order_type: OrderType  # MARKET, LIMIT, etc.
    exchange: str  # Exchange name
    strategy: str  # Strategy that generated order
    timestamp: datetime  # UTC timestamp
    price: Optional[Decimal] = None  # Limit price, None = market order
    order_id: Optional[str] = None  # Internal order ID
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional context

    def __post_init__(self):
        """Validate order after creation"""
        if self.quantity <= 0:
            raise ValueError("Order quantity must be positive")
        if self.price is not None and self.price <= 0:
            raise ValueError("Order price must be positive")
        if not isinstance(self.timestamp.tzinfo, type(timezone.utc)):
            # Ensure UTC timezone
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)


@dataclass
class Position:
    """Open trading position"""
    symbol: str
    quantity: Decimal  # Positive=long, negative=short
    entry_price: Decimal
    current_price: Decimal
    exchange: str
    strategy: str
    opened_at: datetime  # UTC timestamp
    position_id: str

    @property
    def pnl(self) -> Decimal:
        """Unrealized P&L"""
        return (self.current_price - self.entry_price) * self.quantity

    @property
    def pnl_percent(self) -> Decimal:
        """P&L percentage"""
        if self.entry_price == 0:
            return Decimal('0')
        return ((self.current_price - self.entry_price) / self.entry_price) * Decimal('100')

    @property
    def value(self) -> Decimal:
        """Current position value"""
        return abs(self.quantity * self.current_price)

    def update_price(self, new_price: Decimal) -> None:
        """Update current price"""
        self.current_price = new_price


@dataclass
class Signal:
    """Trading signal from strategy"""
    symbol: str
    action: SignalAction
    strength: Decimal  # 0.0 to 1.0
    confidence: Decimal  # 0.0 to 1.0
    timestamp: datetime  # UTC timestamp
    strategy: str
    timeframe: str  # e.g. '1m', '5m', '1h'
    indicators: Dict[str, Decimal] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate signal"""
        if not (Decimal('0') <= self.strength <= Decimal('1')):
            raise ValueError("Signal strength must be between 0 and 1")
        if not (Decimal('0') <= self.confidence <= Decimal('1')):
            raise ValueError("Signal confidence must be between 0 and 1")


@dataclass
class ExecutionResult:
    """Result of order execution"""
    order_id: str
    symbol: str
    side: OrderSide
    status: OrderStatus
    filled_quantity: Decimal
    remaining_quantity: Decimal
    average_price: Decimal
    total_cost: Decimal
    fees: Decimal
    exchange: str
    executed_at: datetime
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_fully_filled(self) -> bool:
        """Check if order is fully filled"""
        return self.remaining_quantity == 0 and self.status == OrderStatus.FILLED

    @property
    def fill_percent(self) -> Decimal:
        """Percentage of order filled"""
        total_qty = self.filled_quantity + self.remaining_quantity
        if total_qty == 0:
            return Decimal('0')
        return (self.filled_quantity / total_qty) * Decimal('100')


@dataclass
class Trade:
    """Executed trade record"""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    fee: Decimal
    fee_currency: str
    exchange: str
    executed_at: datetime
    is_maker: bool = False  # Maker or taker
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def value(self) -> Decimal:
        """Trade value"""
        return self.quantity * self.price

    @property
    def total_cost(self) -> Decimal:
        """Total cost including fees"""
        return self.value + self.fee


@dataclass
class MarketData:
    """Market data snapshot"""
    symbol: str
    exchange: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    quote_volume: Optional[Decimal] = None
    trades_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderBook:
    """Order book snapshot"""
    symbol: str
    exchange: str
    timestamp: datetime
    bids: list  # List of [price, quantity] tuples
    asks: list  # List of [price, quantity] tuples
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def best_bid(self) -> Optional[Decimal]:
        """Best bid price"""
        return Decimal(str(self.bids[0][0])) if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        """Best ask price"""
        return Decimal(str(self.asks[0][0])) if self.asks else None

    @property
    def spread(self) -> Optional[Decimal]:
        """Bid-ask spread"""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def mid_price(self) -> Optional[Decimal]:
        """Mid-market price"""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / Decimal('2')
        return None


@dataclass
class Balance:
    """Account balance"""
    currency: str
    exchange: str
    total: Decimal
    available: Decimal
    locked: Decimal
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def available_percent(self) -> Decimal:
        """Percentage of balance available"""
        if self.total == 0:
            return Decimal('0')
        return (self.available / self.total) * Decimal('100')


# Export all models and enums
__all__ = [
    # Enums
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'SignalAction',
    'Exchange',
    'TimeFrame',
    # Data Models
    'Order',
    'Position',
    'Signal',
    'ExecutionResult',
    'Trade',
    'MarketData',
    'OrderBook',
    'Balance',
]
