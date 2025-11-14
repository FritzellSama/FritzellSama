"""
Pydantic schemas for position-related API endpoints.

Defines request and response models for position querying,
modification, and analytics with validation.
"""

from decimal import Decimal
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator
from structlog import get_logger

logger = get_logger(__name__)


class PositionSideEnum(str, Enum):
    """Position side enumeration."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class PositionStatusEnum(str, Enum):
    """Position status enumeration."""
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"


# Request Schemas

class ClosePositionRequest(BaseModel):
    """Request to close a position."""

    quantity: Optional[str] = Field(None, description="Quantity to close (None = close all)")
    reason: Optional[str] = Field(None, description="Reason for closing")
    emergency: bool = Field(False, description="Emergency close (market order)")

    @validator("quantity")
    def validate_quantity(cls, v: Optional[str]) -> Optional[str]:
        """Validate quantity is positive decimal."""
        if v is None:
            return v

        try:
            qty = Decimal(v)
            if qty <= 0:
                raise ValueError("Quantity must be positive")
            return v
        except (ValueError, ArithmeticError) as e:
            raise ValueError(f"Invalid quantity: {e}")

    class Config:
        json_schema_extra = {
            "example": {
                "quantity": "0.5",
                "reason": "Take profit target reached",
                "emergency": False
            }
        }


class ModifyPositionRequest(BaseModel):
    """Request to modify position parameters."""

    stop_loss: Optional[str] = Field(None, description="New stop loss price")
    take_profit: Optional[str] = Field(None, description="New take profit price")

    @validator("stop_loss", "take_profit")
    def validate_price(cls, v: Optional[str]) -> Optional[str]:
        """Validate price is positive decimal."""
        if v is None:
            return v

        try:
            price = Decimal(v)
            if price <= 0:
                raise ValueError("Price must be positive")
            return v
        except (ValueError, ArithmeticError) as e:
            raise ValueError(f"Invalid price: {e}")

    class Config:
        json_schema_extra = {
            "example": {
                "stop_loss": "48000.00",
                "take_profit": "52000.00"
            }
        }


# Response Schemas

class PositionResponse(BaseModel):
    """Position information response."""

    position_id: str = Field(..., description="Position identifier")
    symbol: str = Field(..., description="Trading pair")
    side: PositionSideEnum = Field(..., description="Position side (LONG/SHORT)")
    quantity: str = Field(..., description="Position quantity (abs value)")
    entry_price: str = Field(..., description="Average entry price")
    current_price: str = Field(..., description="Current market price")
    unrealized_pnl: str = Field(..., description="Unrealized P&L")
    unrealized_pnl_percent: str = Field(..., description="Unrealized P&L percentage")
    realized_pnl: str = Field(..., description="Realized P&L")
    total_pnl: str = Field(..., description="Total P&L (realized + unrealized)")
    exchange: str = Field(..., description="Exchange name")
    strategy: str = Field(..., description="Strategy identifier")
    leverage: Optional[str] = Field(None, description="Position leverage")
    liquidation_price: Optional[str] = Field(None, description="Liquidation price")
    stop_loss: Optional[str] = Field(None, description="Stop loss price")
    take_profit: Optional[str] = Field(None, description="Take profit price")
    status: PositionStatusEnum = Field(..., description="Position status")
    opened_at: str = Field(..., description="Position open timestamp (ISO 8601)")
    updated_at: str = Field(..., description="Last update timestamp (ISO 8601)")
    closed_at: Optional[str] = Field(None, description="Position close timestamp (ISO 8601)")
    holding_time_seconds: float = Field(..., description="Time position has been held (seconds)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Position metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "position_id": "pos_1234567890",
                "symbol": "BTC/USDT",
                "side": "LONG",
                "quantity": "0.1",
                "entry_price": "50000.00",
                "current_price": "51000.00",
                "unrealized_pnl": "100.00",
                "unrealized_pnl_percent": "2.00",
                "realized_pnl": "0.00",
                "total_pnl": "100.00",
                "exchange": "binance",
                "strategy": "momentum_strategy",
                "leverage": "1",
                "liquidation_price": None,
                "stop_loss": "48000.00",
                "take_profit": "52000.00",
                "status": "OPEN",
                "opened_at": "2025-01-15T10:00:00Z",
                "updated_at": "2025-01-15T11:00:00Z",
                "closed_at": None,
                "holding_time_seconds": 3600.0
            }
        }


class ClosePositionResponse(BaseModel):
    """Response after closing a position."""

    position_id: str = Field(..., description="Closed position ID")
    symbol: str = Field(..., description="Trading pair")
    closed_quantity: str = Field(..., description="Quantity closed")
    exit_price: str = Field(..., description="Average exit price")
    realized_pnl: str = Field(..., description="Realized P&L")
    realized_pnl_percent: str = Field(..., description="Realized P&L percentage")
    fees: str = Field(..., description="Total fees paid")
    status: PositionStatusEnum = Field(..., description="Final position status")
    closed_at: str = Field(..., description="Close timestamp (ISO 8601)")
    holding_time_seconds: float = Field(..., description="Total holding time (seconds)")
    message: str = Field(..., description="Success message")

    class Config:
        json_schema_extra = {
            "example": {
                "position_id": "pos_1234567890",
                "symbol": "BTC/USDT",
                "closed_quantity": "0.1",
                "exit_price": "51000.00",
                "realized_pnl": "95.00",
                "realized_pnl_percent": "1.90",
                "fees": "5.00",
                "status": "CLOSED",
                "closed_at": "2025-01-15T12:00:00Z",
                "holding_time_seconds": 7200.0,
                "message": "Position closed successfully"
            }
        }


class PositionListResponse(BaseModel):
    """Response with list of positions."""

    positions: List[PositionResponse] = Field(..., description="List of positions")
    total: int = Field(..., description="Total number of positions")
    open_positions: int = Field(..., description="Number of open positions")
    total_unrealized_pnl: str = Field(..., description="Total unrealized P&L")
    total_realized_pnl: str = Field(..., description="Total realized P&L")
    skip: int = Field(..., description="Number of positions skipped")
    limit: int = Field(..., description="Maximum positions returned")

    class Config:
        json_schema_extra = {
            "example": {
                "positions": [
                    {
                        "position_id": "pos_1234567890",
                        "symbol": "BTC/USDT",
                        "side": "LONG",
                        "quantity": "0.1",
                        "entry_price": "50000.00",
                        "current_price": "51000.00",
                        "unrealized_pnl": "100.00",
                        "unrealized_pnl_percent": "2.00",
                        "realized_pnl": "0.00",
                        "total_pnl": "100.00",
                        "exchange": "binance",
                        "strategy": "momentum_strategy",
                        "status": "OPEN",
                        "opened_at": "2025-01-15T10:00:00Z",
                        "updated_at": "2025-01-15T11:00:00Z",
                        "holding_time_seconds": 3600.0
                    }
                ],
                "total": 1,
                "open_positions": 1,
                "total_unrealized_pnl": "100.00",
                "total_realized_pnl": "0.00",
                "skip": 0,
                "limit": 100
            }
        }


class PositionSummary(BaseModel):
    """Summary of positions for a symbol or strategy."""

    symbol: Optional[str] = Field(None, description="Trading pair (if filtered)")
    strategy: Optional[str] = Field(None, description="Strategy (if filtered)")
    total_positions: int = Field(..., description="Total number of positions")
    open_positions: int = Field(..., description="Number of open positions")
    closed_positions: int = Field(..., description="Number of closed positions")
    long_positions: int = Field(..., description="Number of long positions")
    short_positions: int = Field(..., description="Number of short positions")
    total_quantity: str = Field(..., description="Total quantity held")
    average_entry_price: str = Field(..., description="Average entry price")
    total_unrealized_pnl: str = Field(..., description="Total unrealized P&L")
    total_realized_pnl: str = Field(..., description="Total realized P&L")
    total_pnl: str = Field(..., description="Total P&L")
    win_rate: str = Field(..., description="Win rate percentage")
    average_holding_time_seconds: float = Field(..., description="Average holding time")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTC/USDT",
                "strategy": "momentum_strategy",
                "total_positions": 50,
                "open_positions": 5,
                "closed_positions": 45,
                "long_positions": 30,
                "short_positions": 20,
                "total_quantity": "5.0",
                "average_entry_price": "49500.00",
                "total_unrealized_pnl": "250.00",
                "total_realized_pnl": "1250.00",
                "total_pnl": "1500.00",
                "win_rate": "65.00",
                "average_holding_time_seconds": 5400.0
            }
        }


class PositionPerformance(BaseModel):
    """Performance metrics for positions."""

    total_trades: int = Field(..., description="Total number of closed positions")
    winning_trades: int = Field(..., description="Number of winning trades")
    losing_trades: int = Field(..., description="Number of losing trades")
    win_rate: str = Field(..., description="Win rate percentage")
    total_pnl: str = Field(..., description="Total P&L")
    average_win: str = Field(..., description="Average winning trade P&L")
    average_loss: str = Field(..., description="Average losing trade P&L")
    largest_win: str = Field(..., description="Largest winning trade")
    largest_loss: str = Field(..., description="Largest losing trade")
    profit_factor: str = Field(..., description="Profit factor (gross profit / gross loss)")
    average_holding_time_seconds: float = Field(..., description="Average holding time")
    sharpe_ratio: Optional[str] = Field(None, description="Sharpe ratio")
    max_drawdown: str = Field(..., description="Maximum drawdown")
    recovery_factor: str = Field(..., description="Recovery factor (net profit / max drawdown)")

    class Config:
        json_schema_extra = {
            "example": {
                "total_trades": 100,
                "winning_trades": 65,
                "losing_trades": 35,
                "win_rate": "65.00",
                "total_pnl": "5000.00",
                "average_win": "150.00",
                "average_loss": "85.00",
                "largest_win": "500.00",
                "largest_loss": "300.00",
                "profit_factor": "2.35",
                "average_holding_time_seconds": 4800.0,
                "sharpe_ratio": "1.85",
                "max_drawdown": "1200.00",
                "recovery_factor": "4.17"
            }
        }


class PositionRiskMetrics(BaseModel):
    """Risk metrics for positions."""

    position_id: str = Field(..., description="Position identifier")
    symbol: str = Field(..., description="Trading pair")
    exposure: str = Field(..., description="Total exposure (quantity * price)")
    exposure_percent: str = Field(..., description="Exposure as % of portfolio")
    risk_amount: str = Field(..., description="Amount at risk")
    risk_percent: str = Field(..., description="Risk as % of portfolio")
    reward_to_risk_ratio: Optional[str] = Field(None, description="Reward to risk ratio")
    max_loss: str = Field(..., description="Maximum potential loss")
    var_95: str = Field(..., description="Value at Risk (95%)")
    expected_shortfall: str = Field(..., description="Expected Shortfall (CVaR)")
    distance_to_stop_loss_percent: Optional[str] = Field(None, description="% distance to stop loss")
    distance_to_take_profit_percent: Optional[str] = Field(None, description="% distance to take profit")

    class Config:
        json_schema_extra = {
            "example": {
                "position_id": "pos_1234567890",
                "symbol": "BTC/USDT",
                "exposure": "5100.00",
                "exposure_percent": "5.10",
                "risk_amount": "200.00",
                "risk_percent": "0.20",
                "reward_to_risk_ratio": "2.00",
                "max_loss": "200.00",
                "var_95": "150.00",
                "expected_shortfall": "180.00",
                "distance_to_stop_loss_percent": "4.00",
                "distance_to_take_profit_percent": "2.00"
            }
        }


class PortfolioPositionsSummary(BaseModel):
    """Summary of all portfolio positions."""

    total_positions: int = Field(..., description="Total number of positions")
    open_positions: int = Field(..., description="Number of open positions")
    total_exposure: str = Field(..., description="Total portfolio exposure")
    total_unrealized_pnl: str = Field(..., description="Total unrealized P&L")
    total_realized_pnl: str = Field(..., description="Total realized P&L")
    portfolio_value: str = Field(..., description="Total portfolio value")
    available_balance: str = Field(..., description="Available balance")
    margin_used: str = Field(..., description="Margin used")
    margin_available: str = Field(..., description="Margin available")
    positions_by_exchange: Dict[str, int] = Field(..., description="Position count by exchange")
    positions_by_strategy: Dict[str, int] = Field(..., description="Position count by strategy")
    top_positions: List[PositionResponse] = Field(..., description="Top 5 positions by exposure")

    class Config:
        json_schema_extra = {
            "example": {
                "total_positions": 10,
                "open_positions": 8,
                "total_exposure": "50000.00",
                "total_unrealized_pnl": "1250.00",
                "total_realized_pnl": "3500.00",
                "portfolio_value": "104750.00",
                "available_balance": "54750.00",
                "margin_used": "25000.00",
                "margin_available": "75000.00",
                "positions_by_exchange": {
                    "binance": 5,
                    "bybit": 3
                },
                "positions_by_strategy": {
                    "momentum_strategy": 6,
                    "mean_reversion": 2
                },
                "top_positions": []
            }
        }
