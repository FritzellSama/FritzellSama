"""
Quantum Trader AI - Core Data Models
Production-grade dataclasses for trading system

CRITICAL: All numeric values use Decimal, never float
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional


# ===== ENUMS =====

class OrderSide(Enum):
    """Order direction"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type specification"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """Order lifecycle status"""
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"


class SignalAction(Enum):
    """Trading signal action type"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class ExecutionStatus(Enum):
    """Execution result status"""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"


# ===== DATA MODELS =====

@dataclass
class Order:
    """Trading order - immutable after creation

    CRITICAL: Never use float for quantity or price - always Decimal
    """
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    exchange: str
    strategy: str
    timestamp: datetime
    price: Optional[Decimal] = None
    order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate order data"""
        if not isinstance(self.quantity, Decimal):
            raise TypeError(f"Order quantity must be Decimal, got {type(self.quantity)}")
        if self.price is not None and not isinstance(self.price, Decimal):
            raise TypeError(f"Order price must be Decimal, got {type(self.price)}")
        if self.quantity <= Decimal('0'):
            raise ValueError(f"Order quantity must be positive, got {self.quantity}")


@dataclass
class Position:
    """Open trading position

    CRITICAL: All monetary values are Decimal
    """
    symbol: str
    quantity: Decimal  # Positive=long, negative=short
    entry_price: Decimal
    current_price: Decimal
    exchange: str
    strategy: str
    opened_at: datetime
    position_id: str

    def __post_init__(self) -> None:
        """Validate position data"""
        for field_name in ['quantity', 'entry_price', 'current_price']:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                raise TypeError(f"Position {field_name} must be Decimal, got {type(value)}")

    @property
    def pnl(self) -> Decimal:
        """Unrealized P&L in absolute terms"""
        return (self.current_price - self.entry_price) * abs(self.quantity)

    @property
    def pnl_percent(self) -> Decimal:
        """P&L as percentage of entry value"""
        if self.entry_price == Decimal('0'):
            return Decimal('0')
        return ((self.current_price - self.entry_price) / self.entry_price) * Decimal('100')


@dataclass
class Signal:
    """Trading signal from strategy

    CRITICAL: strength and confidence are Decimal 0.0 to 1.0
    """
    symbol: str
    action: SignalAction
    strength: Decimal  # 0.0 to 1.0
    confidence: Decimal  # 0.0 to 1.0
    timestamp: datetime
    strategy: str
    timeframe: str
    indicators: Dict[str, Decimal] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate signal data"""
        if not isinstance(self.strength, Decimal) or not isinstance(self.confidence, Decimal):
            raise TypeError("Signal strength and confidence must be Decimal")
        if not (Decimal('0') <= self.strength <= Decimal('1')):
            raise ValueError(f"Signal strength must be 0-1, got {self.strength}")
        if not (Decimal('0') <= self.confidence <= Decimal('1')):
            raise ValueError(f"Signal confidence must be 0-1, got {self.confidence}")


@dataclass
class ExecutionResult:
    """Result of order execution

    CRITICAL: All monetary and quantity values are Decimal
    """
    order_id: str
    status: ExecutionStatus
    symbol: str
    side: OrderSide
    filled_quantity: Decimal
    average_price: Decimal
    total_cost: Decimal
    fees: Decimal
    exchange: str
    timestamp: datetime
    exchange_order_id: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate execution result"""
        for field_name in ['filled_quantity', 'average_price', 'total_cost', 'fees']:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                raise TypeError(f"ExecutionResult {field_name} must be Decimal, got {type(value)}")


@dataclass
class RiskMetrics:
    """Portfolio risk metrics

    CRITICAL: All values are Decimal
    """
    portfolio_value: Decimal
    cash_balance: Decimal
    total_exposure: Decimal
    var_95: Decimal  # Value at Risk 95%
    cvar_95: Decimal  # Conditional VaR
    max_drawdown: Decimal
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    beta: Decimal
    daily_pnl: Decimal
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate risk metrics"""
        for field_name in ['portfolio_value', 'cash_balance', 'total_exposure',
                           'var_95', 'cvar_95', 'max_drawdown', 'sharpe_ratio',
                           'sortino_ratio', 'beta', 'daily_pnl']:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                raise TypeError(f"RiskMetrics {field_name} must be Decimal, got {type(value)}")


@dataclass
class MarketData:
    """Real-time market data snapshot

    CRITICAL: All prices and volumes are Decimal
    """
    symbol: str
    timestamp: datetime
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume: Decimal
    high_24h: Decimal
    low_24h: Decimal
    exchange: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate market data"""
        for field_name in ['bid', 'ask', 'last', 'volume', 'high_24h', 'low_24h']:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                raise TypeError(f"MarketData {field_name} must be Decimal, got {type(value)}")


@dataclass
class AuditLog:
    """Audit trail entry for compliance

    CRITICAL: Immutable record of all critical operations
    """
    timestamp: datetime
    operation: str
    user_id: str
    component: str
    severity: str  # INFO, WARNING, ERROR, CRITICAL
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    result: Optional[str] = None

    def __post_init__(self) -> None:
        """Ensure severity is valid"""
        valid_severities = {'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if self.severity not in valid_severities:
            raise ValueError(f"Invalid severity: {self.severity}")
