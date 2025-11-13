"""Order Pydantic schemas."""
from pydantic import BaseModel, Field, validator
from decimal import Decimal
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


class OrderCreate(BaseModel):
    symbol: str = Field(..., description="Trading pair (e.g., BTC/USDT)")
    side: OrderSide
    quantity: Decimal = Field(..., gt=0, description="Order quantity")
    price: Optional[Decimal] = Field(None, gt=0, description="Limit price")
    order_type: OrderType
    exchange: str = Field(..., description="Exchange name")
    strategy: str = Field(..., description="Strategy name")
    
    @validator('quantity', 'price')
    def validate_decimal(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Must be positive")
        return v


class OrderResponse(BaseModel):
    order_id: str
    symbol: str
    side: OrderSide
    quantity: str
    price: Optional[str]
    order_type: OrderType
    exchange: str
    status: str
    created_at: datetime
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class OrderUpdate(BaseModel):
    quantity: Optional[Decimal] = None
    price: Optional[Decimal] = None


class OrderCancel(BaseModel):
    order_id: str
    reason: Optional[str] = None
