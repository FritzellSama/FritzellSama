"""
Position schemas for Quantum Trader AI API.

Pydantic models for position request/response validation.
"""

from decimal import Decimal
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class PositionSide(str, Enum):
    """Position side enumeration."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class PositionStatus(str, Enum):
    """Position status enumeration."""
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"


class PositionResponse(BaseModel):
    """Position response.

    Attributes:
        position_id: Position identifier
        symbol: Trading pair
        side: Position side (LONG/SHORT)
        quantity: Position quantity
        entry_price: Average entry price
        current_price: Current market price
        unrealized_pnl: Unrealized profit/loss
        unrealized_pnl_percent: Unrealized P&L percentage
        realized_pnl: Realized profit/loss
        total_pnl: Total profit/loss
        exchange: Exchange name
        strategy: Strategy name
        opened_at: Position open timestamp
        updated_at: Last update timestamp
        metadata: Additional position metadata
    """
    position_id: str = Field(..., description="Position identifier")
    symbol: str = Field(..., description="Trading pair")
    side: PositionSide = Field(..., description="Position side")
    quantity: str = Field(..., description="Position quantity")
    entry_price: str = Field(..., description="Average entry price")
    current_price: str = Field(..., description="Current market price")
    unrealized_pnl: str = Field(..., description="Unrealized profit/loss")
    unrealized_pnl_percent: str = Field(..., description="Unrealized P&L percentage")
    realized_pnl: str = Field(..., description="Realized profit/loss")
    total_pnl: str = Field(..., description="Total profit/loss")
    exchange: str = Field(..., description="Exchange name")
    strategy: Optional[str] = Field(None, description="Strategy name")
    status: PositionStatus = Field(..., description="Position status")
    opened_at: datetime = Field(..., description="Position open timestamp (UTC)")
    updated_at: datetime = Field(..., description="Last update timestamp (UTC)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        schema_extra = {
            "example": {
                "position_id": "pos_1234567890",
                "symbol": "BTC/USDT",
                "side": "LONG",
                "quantity": "0.5",
                "entry_price": "45000.00",
                "current_price": "46000.00",
                "unrealized_pnl": "500.00",
                "unrealized_pnl_percent": "2.22",
                "realized_pnl": "0.00",
                "total_pnl": "500.00",
                "exchange": "BINANCE",
                "strategy": "momentum_strategy",
                "status": "OPEN",
                "opened_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T12:00:00Z",
                "metadata": {}
            }
        }


class ClosePositionRequest(BaseModel):
    """Close position request.

    Attributes:
        quantity: Quantity to close (optional, defaults to all)
        price: Limit price for closing order (optional)
    """
    quantity: Optional[str] = Field(None, description="Quantity to close (None = close all)")
    price: Optional[str] = Field(None, description="Limit price (None = market order)")

    @validator("quantity")
    def validate_quantity(cls, v: Optional[str]) -> Optional[str]:
        """Validate quantity if provided."""
        if v is None:
            return v
        try:
            qty = Decimal(v)
            if qty <= Decimal("0"):
                raise ValueError("Quantity must be positive")
            return v
        except Exception:
            raise ValueError("Invalid quantity format")

    @validator("price")
    def validate_price(cls, v: Optional[str]) -> Optional[str]:
        """Validate price if provided."""
        if v is None:
            return v
        try:
            price = Decimal(v)
            if price <= Decimal("0"):
                raise ValueError("Price must be positive")
            return v
        except Exception:
            raise ValueError("Invalid price format")

    class Config:
        schema_extra = {
            "example": {
                "quantity": "0.25",
                "price": "46500.00"
            }
        }


class PositionListQuery(BaseModel):
    """Position list query parameters.

    Attributes:
        symbol: Filter by symbol
        exchange: Filter by exchange
        strategy: Filter by strategy
        status: Filter by status
        side: Filter by side
        limit: Maximum results
        offset: Results offset
    """
    symbol: Optional[str] = Field(None, description="Filter by symbol")
    exchange: Optional[str] = Field(None, description="Filter by exchange")
    strategy: Optional[str] = Field(None, description="Filter by strategy")
    status: Optional[PositionStatus] = Field(None, description="Filter by status")
    side: Optional[PositionSide] = Field(None, description="Filter by side")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Results offset")


class PositionSummary(BaseModel):
    """Position summary statistics.

    Attributes:
        total_positions: Total number of positions
        open_positions: Number of open positions
        long_positions: Number of long positions
        short_positions: Number of short positions
        total_unrealized_pnl: Total unrealized P&L
        total_realized_pnl: Total realized P&L
        total_pnl: Total P&L
        largest_position: Largest position by value
        largest_winner: Position with highest unrealized P&L
        largest_loser: Position with lowest unrealized P&L
    """
    total_positions: int = Field(..., description="Total number of positions")
    open_positions: int = Field(..., description="Number of open positions")
    long_positions: int = Field(..., description="Number of long positions")
    short_positions: int = Field(..., description="Number of short positions")
    total_unrealized_pnl: str = Field(..., description="Total unrealized P&L")
    total_realized_pnl: str = Field(..., description="Total realized P&L")
    total_pnl: str = Field(..., description="Total P&L")
    largest_position: Optional[Dict[str, Any]] = Field(None, description="Largest position")
    largest_winner: Optional[Dict[str, Any]] = Field(None, description="Largest winner")
    largest_loser: Optional[Dict[str, Any]] = Field(None, description="Largest loser")

    class Config:
        schema_extra = {
            "example": {
                "total_positions": 15,
                "open_positions": 12,
                "long_positions": 8,
                "short_positions": 4,
                "total_unrealized_pnl": "2500.00",
                "total_realized_pnl": "1500.00",
                "total_pnl": "4000.00",
                "largest_position": {
                    "symbol": "BTC/USDT",
                    "quantity": "1.0",
                    "value": "46000.00"
                },
                "largest_winner": {
                    "symbol": "ETH/USDT",
                    "unrealized_pnl": "800.00"
                },
                "largest_loser": {
                    "symbol": "SOL/USDT",
                    "unrealized_pnl": "-200.00"
                }
            }
        }


class PositionHistoryEntry(BaseModel):
    """Position history entry.

    Attributes:
        position_id: Position identifier
        symbol: Trading pair
        side: Position side
        entry_quantity: Entry quantity
        exit_quantity: Exit quantity
        entry_price: Entry price
        exit_price: Exit price
        realized_pnl: Realized P&L
        realized_pnl_percent: Realized P&L percentage
        commission: Total commission
        exchange: Exchange name
        strategy: Strategy name
        opened_at: Position opened timestamp
        closed_at: Position closed timestamp
        duration_seconds: Position duration in seconds
    """
    position_id: str = Field(..., description="Position identifier")
    symbol: str = Field(..., description="Trading pair")
    side: PositionSide = Field(..., description="Position side")
    entry_quantity: str = Field(..., description="Entry quantity")
    exit_quantity: str = Field(..., description="Exit quantity")
    entry_price: str = Field(..., description="Entry price")
    exit_price: str = Field(..., description="Exit price")
    realized_pnl: str = Field(..., description="Realized P&L")
    realized_pnl_percent: str = Field(..., description="Realized P&L percentage")
    commission: str = Field(..., description="Total commission")
    exchange: str = Field(..., description="Exchange name")
    strategy: Optional[str] = Field(None, description="Strategy name")
    opened_at: datetime = Field(..., description="Position opened timestamp (UTC)")
    closed_at: datetime = Field(..., description="Position closed timestamp (UTC)")
    duration_seconds: int = Field(..., description="Position duration in seconds")

    class Config:
        schema_extra = {
            "example": {
                "position_id": "pos_1234567890",
                "symbol": "BTC/USDT",
                "side": "LONG",
                "entry_quantity": "0.5",
                "exit_quantity": "0.5",
                "entry_price": "45000.00",
                "exit_price": "46000.00",
                "realized_pnl": "500.00",
                "realized_pnl_percent": "2.22",
                "commission": "10.00",
                "exchange": "BINANCE",
                "strategy": "momentum_strategy",
                "opened_at": "2024-01-01T00:00:00Z",
                "closed_at": "2024-01-01T12:00:00Z",
                "duration_seconds": 43200
            }
        }


class PositionRiskMetrics(BaseModel):
    """Position risk metrics.

    Attributes:
        position_id: Position identifier
        symbol: Trading pair
        var_95: Value at Risk (95% confidence)
        var_99: Value at Risk (99% confidence)
        max_drawdown: Maximum drawdown
        sharpe_ratio: Sharpe ratio
        leverage: Current leverage
        liquidation_price: Estimated liquidation price
        margin_ratio: Margin ratio
    """
    position_id: str = Field(..., description="Position identifier")
    symbol: str = Field(..., description="Trading pair")
    var_95: str = Field(..., description="Value at Risk 95%")
    var_99: str = Field(..., description="Value at Risk 99%")
    max_drawdown: str = Field(..., description="Maximum drawdown")
    sharpe_ratio: str = Field(..., description="Sharpe ratio")
    leverage: str = Field(..., description="Current leverage")
    liquidation_price: Optional[str] = Field(None, description="Liquidation price")
    margin_ratio: str = Field(..., description="Margin ratio")

    class Config:
        schema_extra = {
            "example": {
                "position_id": "pos_1234567890",
                "symbol": "BTC/USDT",
                "var_95": "500.00",
                "var_99": "750.00",
                "max_drawdown": "-200.00",
                "sharpe_ratio": "1.5",
                "leverage": "2.0",
                "liquidation_price": "40000.00",
                "margin_ratio": "0.4"
            }
        }
