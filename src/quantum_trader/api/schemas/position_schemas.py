"""Position Pydantic schemas."""
from pydantic import BaseModel, Field
from decimal import Decimal
from typing import Optional
from datetime import datetime


class PositionResponse(BaseModel):
    position_id: str
    symbol: str
    quantity: str = Field(..., description="Position size (+ long, - short)")
    entry_price: str
    current_price: str
    unrealized_pnl: str
    unrealized_pnl_percent: str
    exchange: str
    strategy: str
    opened_at: datetime
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class PositionSummary(BaseModel):
    total_positions: int
    long_positions: int
    short_positions: int
    total_value: str
    total_pnl: str
    
    class Config:
        json_encoders = {
            Decimal: str
        }


class ClosePositionRequest(BaseModel):
    position_id: str
    quantity: Optional[Decimal] = None  # None = close all
    reason: Optional[str] = None


class PositionUpdate(BaseModel):
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
