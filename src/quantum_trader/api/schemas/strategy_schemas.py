"""
Strategy Schemas - Pydantic models for strategy API validation.

This module defines all request/response schemas for strategy management endpoints.
All monetary values use string representation of Decimal for precision.
"""

from decimal import Decimal
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator
from structlog import get_logger

logger = get_logger(__name__)


class StrategyType(str, Enum):
    """Supported strategy types."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    TREND_FOLLOWING = "trend_following"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    ML_BASED = "ml_based"
    CUSTOM = "custom"


class StrategyStatus(str, Enum):
    """Strategy execution status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"
    ERROR = "error"
    INITIALIZING = "initializing"


class StrategyCreateRequest(BaseModel):
    """Request schema for creating a new strategy.

    Example:
        {
            "name": "Momentum Strategy V1",
            "strategy_type": "momentum",
            "description": "BTC momentum trading",
            "config": {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "lookback_period": "24"
            },
            "risk_limits": {
                "max_position_size": "10000.00",
                "max_daily_loss": "1000.00"
            }
        }
    """
    name: str = Field(..., min_length=1, max_length=100, description="Strategy name")
    strategy_type: StrategyType = Field(..., description="Type of strategy")
    description: Optional[str] = Field(None, max_length=500, description="Strategy description")
    config: Dict[str, Any] = Field(..., description="Strategy configuration parameters")
    risk_limits: Dict[str, str] = Field(..., description="Risk management limits (Decimal as string)")
    exchange: str = Field(..., description="Target exchange")
    symbols: List[str] = Field(..., min_items=1, description="Trading symbols")
    enabled: bool = Field(True, description="Enable strategy on creation")

    @validator('risk_limits')
    def validate_risk_limits(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate risk limits are valid Decimal strings."""
        required_limits = ['max_position_size', 'max_daily_loss', 'max_drawdown']
        for limit in required_limits:
            if limit not in v:
                raise ValueError(f"Missing required risk limit: {limit}")
            try:
                decimal_value = Decimal(v[limit])
                if decimal_value <= Decimal('0'):
                    raise ValueError(f"Risk limit {limit} must be positive")
            except Exception as e:
                raise ValueError(f"Invalid decimal value for {limit}: {e}")
        return v

    @validator('config')
    def validate_config(cls, v: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate strategy config based on strategy type."""
        if not v:
            raise ValueError("Config cannot be empty")

        # All strategies must have timeframe
        if 'timeframe' not in v:
            raise ValueError("Config must include 'timeframe'")

        return v

    class Config:
        schema_extra = {
            "example": {
                "name": "BTC Momentum Strategy",
                "strategy_type": "momentum",
                "description": "High-frequency momentum trading on BTC",
                "config": {
                    "timeframe": "1h",
                    "lookback_period": "24",
                    "momentum_threshold": "0.05"
                },
                "risk_limits": {
                    "max_position_size": "50000.00",
                    "max_daily_loss": "5000.00",
                    "max_drawdown": "10000.00"
                },
                "exchange": "BINANCE",
                "symbols": ["BTC/USDT"],
                "enabled": True
            }
        }


class StrategyUpdateRequest(BaseModel):
    """Request schema for updating a strategy.

    All fields are optional to allow partial updates.
    """
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    config: Optional[Dict[str, Any]] = None
    risk_limits: Optional[Dict[str, str]] = None
    enabled: Optional[bool] = None

    @validator('risk_limits')
    def validate_risk_limits(cls, v: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        """Validate risk limits if provided."""
        if v is None:
            return v

        for limit_name, limit_value in v.items():
            try:
                decimal_value = Decimal(limit_value)
                if decimal_value <= Decimal('0'):
                    raise ValueError(f"Risk limit {limit_name} must be positive")
            except Exception as e:
                raise ValueError(f"Invalid decimal value for {limit_name}: {e}")
        return v

    class Config:
        schema_extra = {
            "example": {
                "description": "Updated strategy description",
                "config": {
                    "momentum_threshold": "0.06"
                },
                "enabled": False
            }
        }


class StrategyResponse(BaseModel):
    """Response schema for strategy details.

    Example:
        {
            "strategy_id": "mom_btc_001",
            "name": "BTC Momentum",
            "strategy_type": "momentum",
            "status": "active",
            "is_active": true,
            "created_at": "2025-01-15T10:00:00Z",
            "updated_at": "2025-01-15T12:00:00Z"
        }
    """
    strategy_id: str = Field(..., description="Unique strategy identifier")
    name: str = Field(..., description="Strategy name")
    strategy_type: StrategyType = Field(..., description="Strategy type")
    description: Optional[str] = Field(None, description="Strategy description")
    status: StrategyStatus = Field(..., description="Current execution status")
    is_active: bool = Field(..., description="Whether strategy is actively trading")
    exchange: str = Field(..., description="Target exchange")
    symbols: List[str] = Field(..., description="Trading symbols")
    config: Dict[str, Any] = Field(..., description="Strategy configuration")
    risk_limits: Dict[str, str] = Field(..., description="Risk limits")
    created_at: datetime = Field(..., description="Creation timestamp (UTC)")
    updated_at: datetime = Field(..., description="Last update timestamp (UTC)")
    started_at: Optional[datetime] = Field(None, description="Last start timestamp (UTC)")
    stopped_at: Optional[datetime] = Field(None, description="Last stop timestamp (UTC)")

    class Config:
        schema_extra = {
            "example": {
                "strategy_id": "mom_btc_001",
                "name": "BTC Momentum Strategy",
                "strategy_type": "momentum",
                "description": "High-frequency momentum trading",
                "status": "active",
                "is_active": True,
                "exchange": "BINANCE",
                "symbols": ["BTC/USDT"],
                "config": {"timeframe": "1h"},
                "risk_limits": {"max_position_size": "50000.00"},
                "created_at": "2025-01-15T10:00:00Z",
                "updated_at": "2025-01-15T12:00:00Z",
                "started_at": "2025-01-15T10:30:00Z",
                "stopped_at": None
            }
        }


class StrategyListResponse(BaseModel):
    """Response schema for strategy listing."""
    strategies: List[StrategyResponse] = Field(..., description="List of strategies")
    total: int = Field(..., description="Total number of strategies")
    active: int = Field(..., description="Number of active strategies")

    class Config:
        schema_extra = {
            "example": {
                "strategies": [],
                "total": 10,
                "active": 6
            }
        }


class StrategyPerformanceResponse(BaseModel):
    """Response schema for strategy performance metrics.

    All monetary values are Decimal represented as strings.

    Example:
        {
            "strategy_id": "mom_btc_001",
            "total_pnl": "12345.67",
            "sharpe_ratio": "2.34",
            "win_rate": "0.65",
            "total_trades": 150,
            "profitable_trades": 98
        }
    """
    strategy_id: str = Field(..., description="Strategy identifier")
    total_pnl: str = Field(..., description="Total profit/loss (Decimal)")
    total_pnl_percent: str = Field(..., description="Total P&L percentage (Decimal)")
    sharpe_ratio: str = Field(..., description="Sharpe ratio (Decimal)")
    sortino_ratio: str = Field(..., description="Sortino ratio (Decimal)")
    max_drawdown: str = Field(..., description="Maximum drawdown (Decimal)")
    win_rate: str = Field(..., description="Win rate 0-1 (Decimal)")
    total_trades: int = Field(..., description="Total number of trades")
    profitable_trades: int = Field(..., description="Number of profitable trades")
    losing_trades: int = Field(..., description="Number of losing trades")
    avg_win: str = Field(..., description="Average winning trade (Decimal)")
    avg_loss: str = Field(..., description="Average losing trade (Decimal)")
    avg_trade_duration: str = Field(..., description="Average trade duration in seconds")
    period_start: datetime = Field(..., description="Performance period start (UTC)")
    period_end: datetime = Field(..., description="Performance period end (UTC)")

    @validator('total_pnl', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'avg_win', 'avg_loss')
    def validate_decimal_strings(cls, v: str) -> str:
        """Validate that string values are valid Decimals."""
        try:
            Decimal(v)
            return v
        except Exception as e:
            raise ValueError(f"Invalid decimal string: {e}")

    class Config:
        schema_extra = {
            "example": {
                "strategy_id": "mom_btc_001",
                "total_pnl": "12345.67",
                "total_pnl_percent": "12.34",
                "sharpe_ratio": "2.34",
                "sortino_ratio": "3.12",
                "max_drawdown": "2345.00",
                "win_rate": "0.65",
                "total_trades": 150,
                "profitable_trades": 98,
                "losing_trades": 52,
                "avg_win": "234.56",
                "avg_loss": "123.45",
                "avg_trade_duration": "3600",
                "period_start": "2025-01-01T00:00:00Z",
                "period_end": "2025-01-15T23:59:59Z"
            }
        }


class StrategyConfigResponse(BaseModel):
    """Response schema for strategy configuration."""
    strategy_id: str = Field(..., description="Strategy identifier")
    config: Dict[str, Any] = Field(..., description="Strategy configuration")
    risk_limits: Dict[str, str] = Field(..., description="Risk limits (Decimal as string)")
    updated_at: datetime = Field(..., description="Last config update (UTC)")

    class Config:
        schema_extra = {
            "example": {
                "strategy_id": "mom_btc_001",
                "config": {
                    "timeframe": "1h",
                    "lookback_period": "24",
                    "momentum_threshold": "0.05"
                },
                "risk_limits": {
                    "max_position_size": "50000.00",
                    "max_daily_loss": "5000.00",
                    "max_drawdown": "10000.00"
                },
                "updated_at": "2025-01-15T12:00:00Z"
            }
        }
