"""
Order schemas for Quantum Trader AI API.

Pydantic models for order request/response validation.
"""

from decimal import Decimal
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"


class TimeInForce(str, Enum):
    """Time in force enumeration."""
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    GTD = "GTD"  # Good Till Date


class CreateOrderRequest(BaseModel):
    """Create order request.

    Attributes:
        symbol: Trading pair (e.g., 'BTC/USDT')
        side: Order side (BUY/SELL)
        order_type: Order type
        quantity: Order quantity
        price: Limit price (required for LIMIT orders)
        stop_price: Stop price (required for STOP orders)
        exchange: Exchange name
        time_in_force: Time in force
        client_order_id: Client-provided order ID (optional)
    """
    symbol: str = Field(..., description="Trading pair (e.g., 'BTC/USDT')")
    side: OrderSide = Field(..., description="Order side")
    order_type: OrderType = Field(..., description="Order type")
    quantity: str = Field(..., description="Order quantity (as string to preserve precision)")
    price: Optional[str] = Field(None, description="Limit price")
    stop_price: Optional[str] = Field(None, description="Stop price")
    exchange: str = Field(..., description="Exchange name")
    time_in_force: TimeInForce = Field(default=TimeInForce.GTC, description="Time in force")
    client_order_id: Optional[str] = Field(None, description="Client order ID")

    @validator("quantity")
    def validate_quantity(cls, v: str) -> str:
        """Validate quantity is positive."""
        try:
            qty = Decimal(v)
            if qty <= Decimal("0"):
                raise ValueError("Quantity must be positive")
            return v
        except Exception:
            raise ValueError("Invalid quantity format")

    @validator("price")
    def validate_price(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Validate price for LIMIT orders."""
        if v is None:
            # Price required for LIMIT and STOP_LIMIT orders
            order_type = values.get("order_type")
            if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                raise ValueError("Price required for LIMIT and STOP_LIMIT orders")
            return v

        try:
            price = Decimal(v)
            if price <= Decimal("0"):
                raise ValueError("Price must be positive")
            return v
        except Exception:
            raise ValueError("Invalid price format")

    @validator("stop_price")
    def validate_stop_price(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Validate stop price for STOP orders."""
        if v is None:
            # Stop price required for STOP orders
            order_type = values.get("order_type")
            if order_type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT, OrderType.STOP_LIMIT]:
                raise ValueError("Stop price required for STOP orders")
            return v

        try:
            stop_price = Decimal(v)
            if stop_price <= Decimal("0"):
                raise ValueError("Stop price must be positive")
            return v
        except Exception:
            raise ValueError("Invalid stop price format")

    class Config:
        schema_extra = {
            "example": {
                "symbol": "BTC/USDT",
                "side": "BUY",
                "order_type": "LIMIT",
                "quantity": "0.01",
                "price": "45000.00",
                "exchange": "BINANCE",
                "time_in_force": "GTC"
            }
        }


class ModifyOrderRequest(BaseModel):
    """Modify order request.

    Attributes:
        quantity: New order quantity (optional)
        price: New limit price (optional)
    """
    quantity: Optional[str] = Field(None, description="New order quantity")
    price: Optional[str] = Field(None, description="New limit price")

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
                "quantity": "0.02",
                "price": "46000.00"
            }
        }


class OrderResponse(BaseModel):
    """Order response.

    Attributes:
        order_id: Internal order ID
        client_order_id: Client order ID (if provided)
        exchange_order_id: Exchange order ID
        symbol: Trading pair
        side: Order side
        order_type: Order type
        quantity: Order quantity
        filled_quantity: Filled quantity
        price: Order price
        avg_fill_price: Average fill price
        status: Order status
        exchange: Exchange name
        created_at: Order creation timestamp
        updated_at: Order last update timestamp
        metadata: Additional order metadata
    """
    order_id: str = Field(..., description="Internal order ID")
    client_order_id: Optional[str] = Field(None, description="Client order ID")
    exchange_order_id: Optional[str] = Field(None, description="Exchange order ID")
    symbol: str = Field(..., description="Trading pair")
    side: OrderSide = Field(..., description="Order side")
    order_type: OrderType = Field(..., description="Order type")
    quantity: str = Field(..., description="Order quantity")
    filled_quantity: str = Field(..., description="Filled quantity")
    price: Optional[str] = Field(None, description="Order price")
    avg_fill_price: Optional[str] = Field(None, description="Average fill price")
    status: OrderStatus = Field(..., description="Order status")
    exchange: str = Field(..., description="Exchange name")
    created_at: datetime = Field(..., description="Order creation timestamp (UTC)")
    updated_at: datetime = Field(..., description="Order last update timestamp (UTC)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        schema_extra = {
            "example": {
                "order_id": "ord_1234567890",
                "client_order_id": "my_order_1",
                "exchange_order_id": "123456789",
                "symbol": "BTC/USDT",
                "side": "BUY",
                "order_type": "LIMIT",
                "quantity": "0.01",
                "filled_quantity": "0.005",
                "price": "45000.00",
                "avg_fill_price": "44950.00",
                "status": "PARTIAL",
                "exchange": "BINANCE",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:01:00Z",
                "metadata": {}
            }
        }


class CancelOrderResponse(BaseModel):
    """Cancel order response.

    Attributes:
        order_id: Order ID
        status: Order status after cancellation
        cancelled_at: Cancellation timestamp
    """
    order_id: str = Field(..., description="Order ID")
    status: OrderStatus = Field(..., description="Order status")
    cancelled_at: datetime = Field(..., description="Cancellation timestamp (UTC)")

    class Config:
        schema_extra = {
            "example": {
                "order_id": "ord_1234567890",
                "status": "CANCELLED",
                "cancelled_at": "2024-01-01T00:02:00Z"
            }
        }


class OrderListQuery(BaseModel):
    """Order list query parameters.

    Attributes:
        symbol: Filter by symbol
        exchange: Filter by exchange
        status: Filter by status
        side: Filter by side
        start_date: Start date filter
        end_date: End date filter
        limit: Maximum results
        offset: Results offset
    """
    symbol: Optional[str] = Field(None, description="Filter by symbol")
    exchange: Optional[str] = Field(None, description="Filter by exchange")
    status: Optional[OrderStatus] = Field(None, description="Filter by status")
    side: Optional[OrderSide] = Field(None, description="Filter by side")
    start_date: Optional[datetime] = Field(None, description="Start date filter (UTC)")
    end_date: Optional[datetime] = Field(None, description="End date filter (UTC)")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Results offset")


class OrderExecutionReport(BaseModel):
    """Order execution report.

    Attributes:
        order_id: Order ID
        execution_id: Execution ID
        symbol: Trading pair
        side: Order side
        quantity: Executed quantity
        price: Execution price
        commission: Commission paid
        executed_at: Execution timestamp
    """
    order_id: str = Field(..., description="Order ID")
    execution_id: str = Field(..., description="Execution ID")
    symbol: str = Field(..., description="Trading pair")
    side: OrderSide = Field(..., description="Order side")
    quantity: str = Field(..., description="Executed quantity")
    price: str = Field(..., description="Execution price")
    commission: str = Field(..., description="Commission paid")
    executed_at: datetime = Field(..., description="Execution timestamp (UTC)")

    class Config:
        schema_extra = {
            "example": {
                "order_id": "ord_1234567890",
                "execution_id": "exec_1234567890",
                "symbol": "BTC/USDT",
                "side": "BUY",
                "quantity": "0.005",
                "price": "44950.00",
                "commission": "0.224",
                "executed_at": "2024-01-01T00:01:00Z"
            }
        }
