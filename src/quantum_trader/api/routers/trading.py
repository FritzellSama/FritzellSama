"""
Trading Router - FastAPI endpoints for trading operations.

This module provides REST API endpoints for submitting orders, managing positions,
and monitoring trade execution.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends, status, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from structlog import get_logger

from quantum_trader.api.services.trading_service import TradingService
from quantum_trader.core.exceptions import (
    OrderValidationError,
    InsufficientBalanceError,
    PositionNotFoundError,
    RiskLimitExceededError
)

logger = get_logger(__name__)

router = APIRouter(prefix="/trading", tags=["trading"])


class OrderRequest(BaseModel):
    """Order submission request schema."""
    symbol: str = Field(..., description="Trading pair (e.g., BTC/USDT)")
    side: str = Field(..., description="Order side: BUY or SELL")
    order_type: str = Field(..., description="Order type: MARKET, LIMIT, etc")
    quantity: str = Field(..., description="Order quantity (Decimal as string)")
    price: Optional[str] = Field(None, description="Limit price (Decimal as string)")
    exchange: str = Field(..., description="Target exchange")
    strategy: str = Field(..., description="Strategy identifier")
    time_in_force: Optional[str] = Field("GTC", description="Time in force")

    @validator('quantity', 'price')
    def validate_decimal_fields(cls, v: Optional[str]) -> Optional[str]:
        """Validate decimal string fields."""
        if v is None:
            return v
        try:
            decimal_val = Decimal(v)
            if decimal_val <= Decimal('0'):
                raise ValueError("Value must be positive")
            return v
        except Exception as e:
            raise ValueError(f"Invalid decimal value: {e}")

    class Config:
        schema_extra = {
            "example": {
                "symbol": "BTC/USDT",
                "side": "BUY",
                "order_type": "LIMIT",
                "quantity": "0.5",
                "price": "45000.00",
                "exchange": "BINANCE",
                "strategy": "momentum_v1",
                "time_in_force": "GTC"
            }
        }


class OrderResponse(BaseModel):
    """Order response schema."""
    order_id: str = Field(..., description="Unique order identifier")
    symbol: str = Field(..., description="Trading pair")
    side: str = Field(..., description="Order side")
    order_type: str = Field(..., description="Order type")
    quantity: str = Field(..., description="Order quantity")
    price: Optional[str] = Field(None, description="Order price")
    status: str = Field(..., description="Order status")
    filled_quantity: str = Field(..., description="Filled quantity")
    average_price: Optional[str] = Field(None, description="Average fill price")
    exchange: str = Field(..., description="Exchange")
    strategy: str = Field(..., description="Strategy identifier")
    created_at: datetime = Field(..., description="Creation timestamp (UTC)")
    updated_at: datetime = Field(..., description="Last update timestamp (UTC)")

    class Config:
        schema_extra = {
            "example": {
                "order_id": "ord_abc123",
                "symbol": "BTC/USDT",
                "side": "BUY",
                "order_type": "LIMIT",
                "quantity": "0.5",
                "price": "45000.00",
                "status": "FILLED",
                "filled_quantity": "0.5",
                "average_price": "44998.50",
                "exchange": "BINANCE",
                "strategy": "momentum_v1",
                "created_at": "2025-01-15T10:00:00Z",
                "updated_at": "2025-01-15T10:00:05Z"
            }
        }


class PositionResponse(BaseModel):
    """Position response schema."""
    position_id: str = Field(..., description="Position identifier")
    symbol: str = Field(..., description="Trading pair")
    quantity: str = Field(..., description="Position quantity (Decimal)")
    entry_price: str = Field(..., description="Entry price (Decimal)")
    current_price: str = Field(..., description="Current price (Decimal)")
    unrealized_pnl: str = Field(..., description="Unrealized P&L (Decimal)")
    unrealized_pnl_percent: str = Field(..., description="Unrealized P&L % (Decimal)")
    exchange: str = Field(..., description="Exchange")
    strategy: str = Field(..., description="Strategy identifier")
    opened_at: datetime = Field(..., description="Position open time (UTC)")

    class Config:
        schema_extra = {
            "example": {
                "position_id": "pos_xyz789",
                "symbol": "BTC/USDT",
                "quantity": "0.5",
                "entry_price": "45000.00",
                "current_price": "46000.00",
                "unrealized_pnl": "500.00",
                "unrealized_pnl_percent": "2.22",
                "exchange": "BINANCE",
                "strategy": "momentum_v1",
                "opened_at": "2025-01-15T09:00:00Z"
            }
        }


class ClosePositionRequest(BaseModel):
    """Close position request schema."""
    position_id: str = Field(..., description="Position to close")
    quantity: Optional[str] = Field(None, description="Partial close quantity")

    @validator('quantity')
    def validate_quantity(cls, v: Optional[str]) -> Optional[str]:
        """Validate quantity if provided."""
        if v is not None:
            try:
                decimal_val = Decimal(v)
                if decimal_val <= Decimal('0'):
                    raise ValueError("Quantity must be positive")
            except Exception as e:
                raise ValueError(f"Invalid quantity: {e}")
        return v


async def get_trading_service() -> TradingService:
    """Dependency injection for TradingService.

    Returns:
        TradingService: Initialized trading service
    """
    service = TradingService()
    await service.initialize()
    return service


@router.post("/orders", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
async def submit_order(
    request: OrderRequest,
    service: TradingService = Depends(get_trading_service)
) -> OrderResponse:
    """Submit a new trading order.

    Args:
        request: Order parameters
        service: Injected trading service

    Returns:
        OrderResponse: Created order details

    Raises:
        HTTPException: 400 for validation errors, 402 for insufficient balance

    Example:
        >>> POST /trading/orders
        {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "quantity": "0.5",
            ...
        }
    """
    try:
        logger.info(
            "Submitting order",
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity
        )

        order = await service.submit_order(request)

        logger.info("Order submitted successfully", order_id=order.order_id)

        return order

    except OrderValidationError as e:
        logger.warning("Order validation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except InsufficientBalanceError as e:
        logger.warning("Insufficient balance", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=str(e)
        )
    except RiskLimitExceededError as e:
        logger.warning("Risk limit exceeded", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Order submission failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Order submission failed: {str(e)}"
        )


@router.get("/orders/{order_id}", response_model=OrderResponse)
async def get_order(
    order_id: str,
    service: TradingService = Depends(get_trading_service)
) -> OrderResponse:
    """Get order details by ID.

    Args:
        order_id: Order identifier
        service: Injected trading service

    Returns:
        OrderResponse: Order details

    Raises:
        HTTPException: 404 if order not found
    """
    try:
        logger.debug("Retrieving order", order_id=order_id)

        order = await service.get_order(order_id)

        if not order:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Order {order_id} not found"
            )

        return order

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retrieve order", error=str(e), order_id=order_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve order: {str(e)}"
        )


@router.get("/orders", response_model=List[OrderResponse])
async def list_orders(
    strategy: Optional[str] = Query(None, description="Filter by strategy"),
    exchange: Optional[str] = Query(None, description="Filter by exchange"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
    service: TradingService = Depends(get_trading_service)
) -> List[OrderResponse]:
    """List orders with optional filtering.

    Args:
        strategy: Filter by strategy
        exchange: Filter by exchange
        status_filter: Filter by order status
        limit: Maximum results
        service: Injected trading service

    Returns:
        List of orders
    """
    try:
        logger.debug(
            "Listing orders",
            strategy=strategy,
            exchange=exchange,
            status=status_filter,
            limit=limit
        )

        orders = await service.list_orders(
            strategy=strategy,
            exchange=exchange,
            status_filter=status_filter,
            limit=limit
        )

        return orders

    except Exception as e:
        logger.error("Failed to list orders", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list orders: {str(e)}"
        )


@router.delete("/orders/{order_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_order(
    order_id: str,
    service: TradingService = Depends(get_trading_service)
) -> None:
    """Cancel an active order.

    Args:
        order_id: Order identifier to cancel
        service: Injected trading service

    Raises:
        HTTPException: 404 if order not found
    """
    try:
        logger.info("Cancelling order", order_id=order_id)

        await service.cancel_order(order_id)

        logger.info("Order cancelled successfully", order_id=order_id)

    except Exception as e:
        logger.error("Failed to cancel order", error=str(e), order_id=order_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel order: {str(e)}"
        )


@router.get("/positions", response_model=List[PositionResponse])
async def list_positions(
    strategy: Optional[str] = Query(None, description="Filter by strategy"),
    exchange: Optional[str] = Query(None, description="Filter by exchange"),
    service: TradingService = Depends(get_trading_service)
) -> List[PositionResponse]:
    """List all open positions.

    Args:
        strategy: Filter by strategy
        exchange: Filter by exchange
        service: Injected trading service

    Returns:
        List of open positions
    """
    try:
        logger.debug("Listing positions", strategy=strategy, exchange=exchange)

        positions = await service.list_positions(
            strategy=strategy,
            exchange=exchange
        )

        return positions

    except Exception as e:
        logger.error("Failed to list positions", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list positions: {str(e)}"
        )


@router.get("/positions/{position_id}", response_model=PositionResponse)
async def get_position(
    position_id: str,
    service: TradingService = Depends(get_trading_service)
) -> PositionResponse:
    """Get position details by ID.

    Args:
        position_id: Position identifier
        service: Injected trading service

    Returns:
        PositionResponse: Position details

    Raises:
        HTTPException: 404 if position not found
    """
    try:
        logger.debug("Retrieving position", position_id=position_id)

        position = await service.get_position(position_id)

        if not position:
            raise PositionNotFoundError(f"Position {position_id} not found")

        return position

    except PositionNotFoundError as e:
        logger.warning("Position not found", position_id=position_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to retrieve position", error=str(e), position_id=position_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve position: {str(e)}"
        )


@router.post("/positions/{position_id}/close", response_model=OrderResponse)
async def close_position(
    position_id: str,
    request: ClosePositionRequest,
    service: TradingService = Depends(get_trading_service)
) -> OrderResponse:
    """Close an open position.

    Args:
        position_id: Position to close
        request: Close parameters
        service: Injected trading service

    Returns:
        OrderResponse: Closing order details

    Raises:
        HTTPException: 404 if position not found
    """
    try:
        logger.info("Closing position", position_id=position_id)

        order = await service.close_position(position_id, request.quantity)

        logger.info("Position closed successfully", position_id=position_id, order_id=order.order_id)

        return order

    except PositionNotFoundError as e:
        logger.warning("Position not found", position_id=position_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to close position", error=str(e), position_id=position_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to close position: {str(e)}"
        )
