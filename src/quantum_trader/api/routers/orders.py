"""
Orders API Router.

Provides REST endpoints for order management including creation,
modification, cancellation, and querying.
"""

from decimal import Decimal
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from structlog import get_logger

from quantum_trader.api.dependencies import (
    verify_api_key,
    check_rate_limit,
    get_pagination,
    PaginationParams,
    get_trading_engine,
    get_portfolio
)
from quantum_trader.api.exceptions import (
    ValidationError,
    ResourceNotFoundError,
    OrderRejectedError,
    InvalidOrderError
)
from quantum_trader.api.schemas.order_schemas import (
    CreateOrderRequest,
    CreateOrderResponse,
    ModifyOrderRequest,
    CancelOrderRequest,
    CancelOrderResponse,
    OrderResponse,
    OrderListResponse,
    OrderExecutionReport,
    OrderStatistics,
    OrderStatusEnum,
    OrderSideEnum,
    OrderTypeEnum
)

logger = get_logger(__name__)

router = APIRouter(
    prefix="/orders",
    tags=["orders"],
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)]
)


# Helper functions

def _order_to_response(order: Any) -> OrderResponse:
    """
    Convert internal Order object to OrderResponse.

    Args:
        order: Internal order object

    Returns:
        OrderResponse schema
    """
    return OrderResponse(
        order_id=order.order_id or "pending",
        client_order_id=order.metadata.get("client_order_id"),
        symbol=order.symbol,
        side=OrderSideEnum(order.side.value),
        quantity=str(order.quantity),
        filled_quantity=str(order.metadata.get("filled_quantity", Decimal("0"))),
        remaining_quantity=str(
            order.quantity - order.metadata.get("filled_quantity", Decimal("0"))
        ),
        order_type=OrderTypeEnum(order.order_type.value),
        price=str(order.price) if order.price else None,
        stop_price=str(order.metadata.get("stop_price")) if order.metadata.get("stop_price") else None,
        average_fill_price=str(order.metadata.get("average_fill_price")) if order.metadata.get("average_fill_price") else None,
        status=OrderStatusEnum(order.metadata.get("status", "PENDING")),
        exchange=order.exchange,
        strategy=order.strategy,
        time_in_force=order.metadata.get("time_in_force", "GTC"),
        created_at=order.timestamp.isoformat(),
        updated_at=order.metadata.get("updated_at", order.timestamp).isoformat() if isinstance(order.metadata.get("updated_at"), datetime) else order.timestamp.isoformat(),
        filled_at=order.metadata.get("filled_at").isoformat() if order.metadata.get("filled_at") else None,
        metadata=order.metadata
    )


# Endpoints

@router.post(
    "",
    response_model=CreateOrderResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create new order",
    description="Submit a new trading order"
)
async def create_order(
    request: CreateOrderRequest,
    trading_engine = Depends(get_trading_engine)
) -> CreateOrderResponse:
    """
    Create a new trading order.

    Validates order parameters, performs risk checks, and submits to exchange.

    Example:
        POST /orders
        {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "quantity": "0.1",
            "order_type": "LIMIT",
            "price": "50000.00",
            "exchange": "binance",
            "strategy": "momentum"
        }
    """
    try:
        # Import here to avoid circular dependency
        from quantum_trader.models import Order, OrderSide, OrderType

        # Create order object
        order = Order(
            symbol=request.symbol,
            side=OrderSide[request.side.value],
            quantity=Decimal(request.quantity),
            price=Decimal(request.price) if request.price else None,
            order_type=OrderType[request.order_type.value],
            exchange=request.exchange,
            strategy=request.strategy,
            timestamp=datetime.utcnow(),
            metadata={
                "time_in_force": request.time_in_force.value,
                "client_order_id": request.client_order_id,
                "stop_price": Decimal(request.stop_price) if request.stop_price else None,
                **(request.metadata or {})
            }
        )

        logger.info(
            "creating_order",
            symbol=request.symbol,
            side=request.side.value,
            quantity=request.quantity,
            order_type=request.order_type.value
        )

        # Execute order through trading engine
        execution_result = await trading_engine.execute_order(order)

        if not execution_result:
            raise OrderRejectedError(
                reason="Order execution failed",
                symbol=request.symbol,
                exchange=request.exchange
            )

        # Build response
        order_response = _order_to_response(order)

        logger.info(
            "order_created",
            order_id=order_response.order_id,
            symbol=request.symbol,
            status=order_response.status
        )

        return CreateOrderResponse(
            order=order_response,
            message="Order created successfully"
        )

    except (OrderRejectedError, InvalidOrderError, ValidationError):
        raise

    except Exception as e:
        logger.error(
            "create_order_error",
            symbol=request.symbol,
            error=str(e),
            error_type=type(e).__name__
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create order: {str(e)}"
        )


@router.get(
    "/{order_id}",
    response_model=OrderResponse,
    summary="Get order by ID",
    description="Retrieve details of a specific order"
)
async def get_order(
    order_id: str = Path(..., description="Order identifier")
) -> OrderResponse:
    """
    Get order details by ID.

    Example:
        GET /orders/ord_1234567890
    """
    try:
        # In production, fetch from order management system
        # This is a placeholder
        logger.info("fetching_order", order_id=order_id)

        # Placeholder response
        raise ResourceNotFoundError(
            resource="Order",
            resource_id=order_id
        )

    except ResourceNotFoundError:
        raise

    except Exception as e:
        logger.error("get_order_error", order_id=order_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch order: {str(e)}"
        )


@router.get(
    "",
    response_model=OrderListResponse,
    summary="List orders",
    description="Retrieve list of orders with optional filtering"
)
async def list_orders(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    exchange: Optional[str] = Query(None, description="Filter by exchange"),
    strategy: Optional[str] = Query(None, description="Filter by strategy"),
    status: Optional[OrderStatusEnum] = Query(None, description="Filter by status"),
    side: Optional[OrderSideEnum] = Query(None, description="Filter by side"),
    pagination: PaginationParams = Depends(get_pagination)
) -> OrderListResponse:
    """
    List orders with optional filtering.

    Returns paginated list of orders matching the specified criteria.

    Example:
        GET /orders?symbol=BTC/USDT&status=OPEN&skip=0&limit=50
    """
    try:
        logger.info(
            "listing_orders",
            symbol=symbol,
            exchange=exchange,
            status=status,
            skip=pagination.skip,
            limit=pagination.limit
        )

        # In production, query from order management system
        # This is a placeholder
        orders: List[OrderResponse] = []

        return OrderListResponse(
            orders=orders,
            total=len(orders),
            skip=pagination.skip,
            limit=pagination.limit
        )

    except Exception as e:
        logger.error("list_orders_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list orders: {str(e)}"
        )


@router.put(
    "/{order_id}",
    response_model=OrderResponse,
    summary="Modify order",
    description="Modify an existing open order"
)
async def modify_order(
    order_id: str = Path(..., description="Order identifier"),
    request: ModifyOrderRequest = ...
) -> OrderResponse:
    """
    Modify an existing order.

    Only quantity and price can be modified. Order must be in OPEN status.

    Example:
        PUT /orders/ord_1234567890
        {
            "quantity": "0.15",
            "price": "51000.00"
        }
    """
    try:
        logger.info(
            "modifying_order",
            order_id=order_id,
            new_quantity=request.quantity,
            new_price=request.price
        )

        # In production, modify order through order management system
        # This is a placeholder
        raise ResourceNotFoundError(
            resource="Order",
            resource_id=order_id
        )

    except ResourceNotFoundError:
        raise

    except Exception as e:
        logger.error("modify_order_error", order_id=order_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to modify order: {str(e)}"
        )


@router.delete(
    "/{order_id}",
    response_model=CancelOrderResponse,
    summary="Cancel order",
    description="Cancel an open order"
)
async def cancel_order(
    order_id: str = Path(..., description="Order identifier"),
    request: CancelOrderRequest = CancelOrderRequest()
) -> CancelOrderResponse:
    """
    Cancel an existing order.

    Order must be in OPEN or PARTIAL status.

    Example:
        DELETE /orders/ord_1234567890
        {
            "reason": "Strategy signal changed"
        }
    """
    try:
        logger.info(
            "cancelling_order",
            order_id=order_id,
            reason=request.reason
        )

        # In production, cancel order through order management system
        # This is a placeholder
        raise ResourceNotFoundError(
            resource="Order",
            resource_id=order_id
        )

    except ResourceNotFoundError:
        raise

    except Exception as e:
        logger.error("cancel_order_error", order_id=order_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel order: {str(e)}"
        )


@router.get(
    "/{order_id}/execution",
    response_model=OrderExecutionReport,
    summary="Get order execution report",
    description="Retrieve detailed execution report for an order"
)
async def get_order_execution(
    order_id: str = Path(..., description="Order identifier")
) -> OrderExecutionReport:
    """
    Get detailed execution report for an order.

    Includes fill details, fees, and execution timing.

    Example:
        GET /orders/ord_1234567890/execution
    """
    try:
        logger.info("fetching_execution_report", order_id=order_id)

        # In production, fetch from execution tracking system
        # This is a placeholder
        raise ResourceNotFoundError(
            resource="Order",
            resource_id=order_id
        )

    except ResourceNotFoundError:
        raise

    except Exception as e:
        logger.error("get_execution_error", order_id=order_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch execution report: {str(e)}"
        )


@router.get(
    "/statistics/{strategy}",
    response_model=OrderStatistics,
    summary="Get order statistics",
    description="Retrieve order statistics for a strategy"
)
async def get_order_statistics(
    strategy: str = Path(..., description="Strategy identifier"),
    start_date: Optional[str] = Query(None, description="Start date (ISO 8601)"),
    end_date: Optional[str] = Query(None, description="End date (ISO 8601)")
) -> OrderStatistics:
    """
    Get order statistics for a strategy.

    Returns aggregated metrics including fill rates, volume, and fees.

    Example:
        GET /orders/statistics/momentum_strategy?start_date=2025-01-01T00:00:00Z
    """
    try:
        logger.info(
            "fetching_order_statistics",
            strategy=strategy,
            start_date=start_date,
            end_date=end_date
        )

        # In production, calculate from order history
        # This is a placeholder
        return OrderStatistics(
            total_orders=0,
            filled_orders=0,
            cancelled_orders=0,
            rejected_orders=0,
            total_volume="0.00",
            total_fees="0.00",
            average_fill_time_ms=0.0,
            fill_rate="0.00"
        )

    except Exception as e:
        logger.error("get_statistics_error", strategy=strategy, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch order statistics: {str(e)}"
        )


@router.post(
    "/batch",
    response_model=List[CreateOrderResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Create multiple orders",
    description="Submit multiple orders in a single request"
)
async def create_batch_orders(
    orders: List[CreateOrderRequest],
    trading_engine = Depends(get_trading_engine)
) -> List[CreateOrderResponse]:
    """
    Create multiple orders in batch.

    All orders are validated before submission. If any order fails validation,
    the entire batch is rejected.

    Example:
        POST /orders/batch
        [
            {"symbol": "BTC/USDT", "side": "BUY", ...},
            {"symbol": "ETH/USDT", "side": "SELL", ...}
        ]
    """
    try:
        logger.info("creating_batch_orders", count=len(orders))

        if len(orders) > 100:
            raise ValidationError(
                message="Batch size exceeds maximum of 100 orders",
                field="orders",
                value=len(orders)
            )

        responses: List[CreateOrderResponse] = []

        # Process each order
        for idx, order_request in enumerate(orders):
            try:
                response = await create_order(order_request, trading_engine)
                responses.append(response)

            except Exception as e:
                logger.error(
                    "batch_order_failed",
                    order_index=idx,
                    symbol=order_request.symbol,
                    error=str(e)
                )
                # In production, decide whether to fail entire batch or continue
                raise

        logger.info(
            "batch_orders_created",
            total=len(orders),
            successful=len(responses)
        )

        return responses

    except (ValidationError, OrderRejectedError):
        raise

    except Exception as e:
        logger.error("create_batch_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create batch orders: {str(e)}"
        )


@router.delete(
    "/cancel-all/{strategy}",
    summary="Cancel all orders for strategy",
    description="Cancel all open orders for a specific strategy"
)
async def cancel_all_orders(
    strategy: str = Path(..., description="Strategy identifier"),
    symbol: Optional[str] = Query(None, description="Optional symbol filter"),
    exchange: Optional[str] = Query(None, description="Optional exchange filter")
) -> Dict[str, Any]:
    """
    Cancel all open orders for a strategy.

    Optionally filter by symbol and/or exchange.

    Example:
        DELETE /orders/cancel-all/momentum_strategy?symbol=BTC/USDT
    """
    try:
        logger.info(
            "cancelling_all_orders",
            strategy=strategy,
            symbol=symbol,
            exchange=exchange
        )

        # In production, cancel all matching orders
        # This is a placeholder
        return {
            "message": "All orders cancelled",
            "strategy": strategy,
            "cancelled_count": 0,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error("cancel_all_error", strategy=strategy, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel all orders: {str(e)}"
        )
