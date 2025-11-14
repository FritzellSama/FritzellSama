"""
Pydantic schemas for order-related API endpoints.

Defines request and response models for order creation,
modification, and querying with validation.
"""

from decimal import Decimal
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator
from structlog import get_logger

logger = get_logger(__name__)


class OrderSideEnum(str, Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderTypeEnum(str, Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatusEnum(str, Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"


class TimeInForceEnum(str, Enum):
    """Time in force enumeration."""
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    GTD = "GTD"  # Good Till Date


# Request Schemas

class CreateOrderRequest(BaseModel):
    """Request to create a new order."""

    symbol: str = Field(..., description="Trading pair (e.g., 'BTC/USDT')")
    side: OrderSideEnum = Field(..., description="Order side (BUY/SELL)")
    quantity: str = Field(..., description="Order quantity (as decimal string)")
    order_type: OrderTypeEnum = Field(..., description="Order type")
    price: Optional[str] = Field(None, description="Limit price (required for LIMIT orders)")
    stop_price: Optional[str] = Field(None, description="Stop price (for STOP orders)")
    exchange: str = Field(..., description="Exchange name")
    strategy: str = Field(..., description="Strategy identifier")
    time_in_force: TimeInForceEnum = Field(TimeInForceEnum.GTC, description="Time in force")
    client_order_id: Optional[str] = Field(None, description="Client-assigned order ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional order metadata")

    @validator("quantity")
    def validate_quantity(cls, v: str) -> str:
        """Validate quantity is positive decimal."""
        try:
            qty = Decimal(v)
            if qty <= 0:
                raise ValueError("Quantity must be positive")
            return v
        except (ValueError, ArithmeticError) as e:
            raise ValueError(f"Invalid quantity: {e}")

    @validator("price", "stop_price")
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

    @root_validator
    def validate_order_params(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate order parameters are consistent."""
        order_type = values.get("order_type")
        price = values.get("price")
        stop_price = values.get("stop_price")

        # LIMIT orders require price
        if order_type == OrderTypeEnum.LIMIT and not price:
            raise ValueError("LIMIT orders require price")

        # STOP orders require stop_price
        if order_type in [OrderTypeEnum.STOP_LOSS, OrderTypeEnum.STOP_LIMIT] and not stop_price:
            raise ValueError("STOP orders require stop_price")

        # STOP_LIMIT orders require both
        if order_type == OrderTypeEnum.STOP_LIMIT and not price:
            raise ValueError("STOP_LIMIT orders require both price and stop_price")

        # MARKET orders should not have price
        if order_type == OrderTypeEnum.MARKET and (price or stop_price):
            raise ValueError("MARKET orders should not have price or stop_price")

        return values

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTC/USDT",
                "side": "BUY",
                "quantity": "0.1",
                "order_type": "LIMIT",
                "price": "50000.00",
                "exchange": "binance",
                "strategy": "momentum_strategy",
                "time_in_force": "GTC",
                "client_order_id": "custom_id_123"
            }
        }


class ModifyOrderRequest(BaseModel):
    """Request to modify an existing order."""

    quantity: Optional[str] = Field(None, description="New order quantity")
    price: Optional[str] = Field(None, description="New limit price")

    @validator("quantity", "price")
    def validate_decimal(cls, v: Optional[str]) -> Optional[str]:
        """Validate decimal fields."""
        if v is None:
            return v

        try:
            decimal_val = Decimal(v)
            if decimal_val <= 0:
                raise ValueError("Value must be positive")
            return v
        except (ValueError, ArithmeticError) as e:
            raise ValueError(f"Invalid decimal value: {e}")

    class Config:
        json_schema_extra = {
            "example": {
                "quantity": "0.15",
                "price": "51000.00"
            }
        }


class CancelOrderRequest(BaseModel):
    """Request to cancel an order."""

    reason: Optional[str] = Field(None, description="Cancellation reason")

    class Config:
        json_schema_extra = {
            "example": {
                "reason": "Strategy signal changed"
            }
        }


# Response Schemas

class OrderResponse(BaseModel):
    """Order information response."""

    order_id: str = Field(..., description="Order identifier")
    client_order_id: Optional[str] = Field(None, description="Client-assigned order ID")
    symbol: str = Field(..., description="Trading pair")
    side: OrderSideEnum = Field(..., description="Order side")
    quantity: str = Field(..., description="Order quantity")
    filled_quantity: str = Field(..., description="Filled quantity")
    remaining_quantity: str = Field(..., description="Remaining quantity")
    order_type: OrderTypeEnum = Field(..., description="Order type")
    price: Optional[str] = Field(None, description="Order price")
    stop_price: Optional[str] = Field(None, description="Stop price")
    average_fill_price: Optional[str] = Field(None, description="Average fill price")
    status: OrderStatusEnum = Field(..., description="Order status")
    exchange: str = Field(..., description="Exchange name")
    strategy: str = Field(..., description="Strategy identifier")
    time_in_force: TimeInForceEnum = Field(..., description="Time in force")
    created_at: str = Field(..., description="Creation timestamp (ISO 8601)")
    updated_at: str = Field(..., description="Last update timestamp (ISO 8601)")
    filled_at: Optional[str] = Field(None, description="Fill timestamp (ISO 8601)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Order metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "order_id": "ord_1234567890",
                "client_order_id": "custom_id_123",
                "symbol": "BTC/USDT",
                "side": "BUY",
                "quantity": "0.1",
                "filled_quantity": "0.05",
                "remaining_quantity": "0.05",
                "order_type": "LIMIT",
                "price": "50000.00",
                "average_fill_price": "49995.00",
                "status": "PARTIAL",
                "exchange": "binance",
                "strategy": "momentum_strategy",
                "time_in_force": "GTC",
                "created_at": "2025-01-15T10:00:00Z",
                "updated_at": "2025-01-15T10:05:00Z",
                "filled_at": None
            }
        }


class CreateOrderResponse(BaseModel):
    """Response after creating an order."""

    order: OrderResponse = Field(..., description="Created order details")
    message: str = Field(..., description="Success message")

    class Config:
        json_schema_extra = {
            "example": {
                "order": {
                    "order_id": "ord_1234567890",
                    "symbol": "BTC/USDT",
                    "side": "BUY",
                    "quantity": "0.1",
                    "filled_quantity": "0.0",
                    "remaining_quantity": "0.1",
                    "order_type": "LIMIT",
                    "price": "50000.00",
                    "status": "OPEN",
                    "exchange": "binance",
                    "strategy": "momentum_strategy",
                    "time_in_force": "GTC",
                    "created_at": "2025-01-15T10:00:00Z",
                    "updated_at": "2025-01-15T10:00:00Z"
                },
                "message": "Order created successfully"
            }
        }


class CancelOrderResponse(BaseModel):
    """Response after cancelling an order."""

    order_id: str = Field(..., description="Cancelled order ID")
    status: OrderStatusEnum = Field(..., description="Final order status")
    message: str = Field(..., description="Cancellation message")
    cancelled_at: str = Field(..., description="Cancellation timestamp (ISO 8601)")

    class Config:
        json_schema_extra = {
            "example": {
                "order_id": "ord_1234567890",
                "status": "CANCELLED",
                "message": "Order cancelled successfully",
                "cancelled_at": "2025-01-15T10:30:00Z"
            }
        }


class OrderListResponse(BaseModel):
    """Response with list of orders."""

    orders: List[OrderResponse] = Field(..., description="List of orders")
    total: int = Field(..., description="Total number of orders")
    skip: int = Field(..., description="Number of orders skipped")
    limit: int = Field(..., description="Maximum orders returned")

    class Config:
        json_schema_extra = {
            "example": {
                "orders": [
                    {
                        "order_id": "ord_1234567890",
                        "symbol": "BTC/USDT",
                        "side": "BUY",
                        "quantity": "0.1",
                        "filled_quantity": "0.1",
                        "remaining_quantity": "0.0",
                        "order_type": "LIMIT",
                        "price": "50000.00",
                        "average_fill_price": "49995.00",
                        "status": "FILLED",
                        "exchange": "binance",
                        "strategy": "momentum_strategy",
                        "time_in_force": "GTC",
                        "created_at": "2025-01-15T10:00:00Z",
                        "updated_at": "2025-01-15T10:05:00Z",
                        "filled_at": "2025-01-15T10:05:00Z"
                    }
                ],
                "total": 1,
                "skip": 0,
                "limit": 100
            }
        }


class OrderExecutionReport(BaseModel):
    """Order execution report with fill details."""

    order_id: str = Field(..., description="Order identifier")
    symbol: str = Field(..., description="Trading pair")
    side: OrderSideEnum = Field(..., description="Order side")
    total_quantity: str = Field(..., description="Total order quantity")
    filled_quantity: str = Field(..., description="Total filled quantity")
    average_price: str = Field(..., description="Average execution price")
    total_fees: str = Field(..., description="Total fees paid")
    fee_currency: str = Field(..., description="Fee currency")
    fills: List[Dict[str, Any]] = Field(..., description="Individual fill details")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    status: OrderStatusEnum = Field(..., description="Final order status")
    created_at: str = Field(..., description="Order creation timestamp")
    completed_at: Optional[str] = Field(None, description="Order completion timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "order_id": "ord_1234567890",
                "symbol": "BTC/USDT",
                "side": "BUY",
                "total_quantity": "0.1",
                "filled_quantity": "0.1",
                "average_price": "49995.00",
                "total_fees": "5.00",
                "fee_currency": "USDT",
                "fills": [
                    {
                        "price": "49990.00",
                        "quantity": "0.05",
                        "timestamp": "2025-01-15T10:05:00Z"
                    },
                    {
                        "price": "50000.00",
                        "quantity": "0.05",
                        "timestamp": "2025-01-15T10:05:01Z"
                    }
                ],
                "execution_time_ms": 1250.5,
                "status": "FILLED",
                "created_at": "2025-01-15T10:00:00Z",
                "completed_at": "2025-01-15T10:05:01Z"
            }
        }


class OrderStatistics(BaseModel):
    """Order statistics for a strategy or timeframe."""

    total_orders: int = Field(..., description="Total number of orders")
    filled_orders: int = Field(..., description="Number of filled orders")
    cancelled_orders: int = Field(..., description="Number of cancelled orders")
    rejected_orders: int = Field(..., description="Number of rejected orders")
    total_volume: str = Field(..., description="Total traded volume")
    total_fees: str = Field(..., description="Total fees paid")
    average_fill_time_ms: float = Field(..., description="Average fill time in milliseconds")
    fill_rate: str = Field(..., description="Order fill rate percentage")

    class Config:
        json_schema_extra = {
            "example": {
                "total_orders": 100,
                "filled_orders": 85,
                "cancelled_orders": 10,
                "rejected_orders": 5,
                "total_volume": "10.5",
                "total_fees": "525.00",
                "average_fill_time_ms": 850.5,
                "fill_rate": "85.00"
            }
        }
