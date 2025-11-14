"""
Types - Custom type definitions and type aliases.

This module defines custom types, type aliases, and type helpers
for improved type safety and code clarity.
"""

from typing import (
    TypeVar, Generic, Union, Optional, Dict, List, Tuple,
    Callable, Awaitable, Any, TypedDict, Literal
)
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass
import polars as pl


# Type variables
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

# Numeric types
Price = Decimal
Quantity = Decimal
Balance = Decimal
Percentage = Decimal
Fee = Decimal

# Time types
Timestamp = datetime
Timeframe = Literal['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']

# Symbol types
Symbol = str  # Format: 'BTC/USDT'
Currency = str  # Format: 'USDT', 'BTC'
ExchangeName = str  # Format: 'binance', 'bybit'

# Order types
OrderID = str
OrderSide = Literal['BUY', 'SELL']
OrderType = Literal['MARKET', 'LIMIT', 'STOP_LOSS', 'TAKE_PROFIT']
OrderStatus = Literal['PENDING', 'FILLED', 'CANCELLED', 'REJECTED']

# Position types
PositionID = str
PositionSide = Literal['LONG', 'SHORT', 'FLAT']

# Strategy types
StrategyName = str
SignalStrength = Decimal  # 0.0 to 1.0
Confidence = Decimal  # 0.0 to 1.0

# Data types
DataFrame = pl.DataFrame
LazyFrame = pl.LazyFrame

# Config types
ConfigDict = Dict[str, Any]
ParameterGrid = Dict[str, List[Any]]

# Result types
SuccessResult = Tuple[bool, Optional[str]]  # (success, error_message)


class OrderDict(TypedDict, total=False):
    """TypedDict for order data."""

    order_id: str
    symbol: str
    side: str
    quantity: str  # Decimal as string
    price: Optional[str]  # Decimal as string
    order_type: str
    exchange: str
    strategy: str
    timestamp: datetime
    metadata: Dict[str, Any]


class PositionDict(TypedDict):
    """TypedDict for position data."""

    position_id: str
    symbol: str
    quantity: str  # Decimal as string
    entry_price: str  # Decimal as string
    current_price: str  # Decimal as string
    exchange: str
    strategy: str
    opened_at: datetime


class SignalDict(TypedDict):
    """TypedDict for signal data."""

    symbol: str
    action: str
    strength: str  # Decimal as string
    confidence: str  # Decimal as string
    timestamp: datetime
    strategy: str
    timeframe: str
    indicators: Dict[str, str]  # Decimal values as strings
    metadata: Dict[str, Any]


class ExecutionResultDict(TypedDict):
    """TypedDict for execution result data."""

    order_id: str
    status: str
    filled_quantity: str  # Decimal as string
    average_price: str  # Decimal as string
    exchange: str
    timestamp: datetime
    fees: str  # Decimal as string
    exchange_order_id: str


class TickerDict(TypedDict):
    """TypedDict for ticker data."""

    symbol: str
    bid: str  # Decimal as string
    ask: str  # Decimal as string
    last: str  # Decimal as string
    volume: str  # Decimal as string
    timestamp: datetime


class OrderBookDict(TypedDict):
    """TypedDict for orderbook data."""

    symbol: str
    bids: List[Tuple[str, str]]  # [(price, quantity)]
    asks: List[Tuple[str, str]]  # [(price, quantity)]
    timestamp: datetime


class BalanceDict(TypedDict):
    """TypedDict for balance data."""

    currency: str
    free: str  # Decimal as string
    locked: str  # Decimal as string
    total: str  # Decimal as string


class PerformanceMetricsDict(TypedDict, total=False):
    """TypedDict for performance metrics."""

    total_return: str  # Decimal as string
    sharpe_ratio: str
    sortino_ratio: str
    max_drawdown: str
    win_rate: str
    profit_factor: str
    total_trades: int
    winning_trades: int
    losing_trades: int


class RiskMetricsDict(TypedDict):
    """TypedDict for risk metrics."""

    current_exposure: str  # Decimal as string
    max_exposure: str
    current_drawdown: str
    max_drawdown: str
    var_95: str  # Value at Risk 95%
    positions_count: int


# Async function type
AsyncFunc = Callable[..., Awaitable[Any]]

# Callback types
OrderCallback = Callable[[OrderDict], None]
SignalCallback = Callable[[SignalDict], None]
EventCallback = Callable[[Dict[str, Any]], None]


@dataclass
class Result(Generic[T]):
    """
    Generic result type for operations that may fail.

    Attributes:
        success: Whether operation succeeded
        value: Result value if successful
        error: Error message if failed
        details: Additional error details

    Example:
        >>> result = Result(success=True, value=Decimal('100.50'))
        >>> if result.success:
        ...     print(f"Value: {result.value}")
    """

    success: bool
    value: Optional[T] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    @staticmethod
    def ok(value: T) -> 'Result[T]':
        """
        Create successful result.

        Args:
            value: Result value

        Returns:
            Result with success=True
        """
        return Result(success=True, value=value)

    @staticmethod
    def fail(error: str, details: Optional[Dict[str, Any]] = None) -> 'Result[T]':
        """
        Create failed result.

        Args:
            error: Error message
            details: Additional error details

        Returns:
            Result with success=False
        """
        return Result(success=False, error=error, details=details)

    def unwrap(self) -> T:
        """
        Unwrap result value.

        Returns:
            Result value

        Raises:
            ValueError: If result is not successful
        """
        if not self.success:
            raise ValueError(f"Cannot unwrap failed result: {self.error}")
        return self.value

    def unwrap_or(self, default: T) -> T:
        """
        Unwrap result or return default.

        Args:
            default: Default value

        Returns:
            Result value or default
        """
        return self.value if self.success else default


@dataclass
class Paginated(Generic[T]):
    """
    Generic paginated result.

    Attributes:
        items: List of items
        total: Total number of items
        page: Current page number
        page_size: Items per page
        has_more: Whether there are more pages

    Example:
        >>> results = Paginated(
        ...     items=[order1, order2],
        ...     total=100,
        ...     page=1,
        ...     page_size=20
        ... )
    """

    items: List[T]
    total: int
    page: int
    page_size: int

    @property
    def has_more(self) -> bool:
        """Check if there are more pages."""
        return (self.page * self.page_size) < self.total

    @property
    def total_pages(self) -> int:
        """Get total number of pages."""
        return (self.total + self.page_size - 1) // self.page_size


# Range types
@dataclass
class PriceRange:
    """Price range."""

    low: Price
    high: Price

    def contains(self, price: Price) -> bool:
        """Check if price is in range."""
        return self.low <= price <= self.high

    def midpoint(self) -> Price:
        """Get midpoint of range."""
        return (self.low + self.high) / Decimal('2')


@dataclass
class TimeRange:
    """Time range."""

    start: Timestamp
    end: Timestamp

    def contains(self, timestamp: Timestamp) -> bool:
        """Check if timestamp is in range."""
        return self.start <= timestamp <= self.end

    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        return (self.end - self.start).total_seconds()


# Union types for flexibility
NumericValue = Union[int, float, Decimal]
DateTimeValue = Union[datetime, int, str]  # datetime, timestamp ms, or ISO string
ConfigValue = Union[str, int, float, bool, List, Dict]


# Validator function types
ValidatorFunc = Callable[[Any], bool]
TransformFunc = Callable[[Any], Any]


# WebSocket message types
class WSMessage(TypedDict):
    """WebSocket message structure."""

    type: str
    data: Dict[str, Any]
    timestamp: datetime


# API response types
class APIResponse(TypedDict, total=False):
    """API response structure."""

    success: bool
    data: Any
    error: Optional[str]
    code: Optional[int]
    timestamp: datetime
