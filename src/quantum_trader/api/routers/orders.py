"""
Orders router for Quantum Trader AI API.

Provides endpoints for order management.
"""

from decimal import Decimal
from typing import Optional, List, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query
from structlog import get_logger

from quantum_trader.api.schemas.order_schemas import (
    CreateOrderRequest,
    ModifyOrderRequest,
    OrderResponse,
    CancelOrderResponse,
    OrderListQuery,
    OrderExecutionReport,
    OrderStatus
)
from quantum_trader.api.dependencies import (
    get_current_user,
    get_db_pool,
    require_permissions
)
from quantum_trader.api.exceptions import (
    OrderException,
    OrderRejected,
    NotFoundException,
    ValidationException
)

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/",
    response_model=OrderResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create order",
    description="Create a new trading order",
    dependencies=[Depends(require_permissions(["trade"]))]
)
async def create_order(
    request: CreateOrderRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_pool = Depends(get_db_pool)
) -> OrderResponse:
    """Create new order.

    Args:
        request: Order creation request
        current_user: Current authenticated user
        db_pool: Database pool

    Returns:
        Created order information

    Raises:
        HTTPException: If order creation fails
    """
    try:
        logger.info(
            "Create order request",
            user_id=current_user["user_id"],
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type
        )

        # Generate order ID
        import secrets
        order_id = f"ord_{secrets.token_hex(16)}"

        # Store order in database
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO orders (
                    order_id, user_id, symbol, side, order_type,
                    quantity, price, stop_price, exchange,
                    status, created_at, updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """,
                order_id,
                current_user["user_id"],
                request.symbol,
                request.side.value,
                request.order_type.value,
                Decimal(request.quantity),
                Decimal(request.price) if request.price else None,
                Decimal(request.stop_price) if request.stop_price else None,
                request.exchange,
                OrderStatus.PENDING.value,
                datetime.utcnow(),
                datetime.utcnow()
            )

        logger.info("Order created", order_id=order_id)

        return OrderResponse(
            order_id=order_id,
            client_order_id=request.client_order_id,
            exchange_order_id=None,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            filled_quantity="0",
            price=request.price,
            avg_fill_price=None,
            status=OrderStatus.PENDING,
            exchange=request.exchange,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata={}
        )

    except ValidationException as e:
        logger.warning("Order validation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Order creation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Order creation failed"
        )


@router.get(
    "/",
    response_model=List[OrderResponse],
    summary="Get orders",
    description="Get list of orders with optional filters"
)
async def get_orders(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    exchange: Optional[str] = Query(None, description="Filter by exchange"),
    status: Optional[OrderStatus] = Query(None, description="Filter by status"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(default=0, ge=0, description="Results offset"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_pool = Depends(get_db_pool)
) -> List[OrderResponse]:
    """Get orders.

    Args:
        symbol: Symbol filter
        exchange: Exchange filter
        status: Status filter
        limit: Maximum results
        offset: Results offset
        current_user: Current user
        db_pool: Database pool

    Returns:
        List of orders

    Raises:
        HTTPException: If request fails
    """
    try:
        logger.info("Get orders request", user_id=current_user["user_id"])

        # Build query
        query = """
            SELECT
                order_id, symbol, side, order_type, quantity,
                filled_quantity, price, avg_fill_price, status,
                exchange, created_at, updated_at
            FROM orders
            WHERE user_id = $1
        """
        params: List[Any] = [current_user["user_id"]]
        param_count = 1

        if symbol:
            param_count += 1
            query += f" AND symbol = ${param_count}"
            params.append(symbol)

        if exchange:
            param_count += 1
            query += f" AND exchange = ${param_count}"
            params.append(exchange)

        if status:
            param_count += 1
            query += f" AND status = ${param_count}"
            params.append(status.value)

        query += f" ORDER BY created_at DESC LIMIT ${param_count + 1} OFFSET ${param_count + 2}"
        params.extend([limit, offset])

        async with db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        orders = []
        for row in rows:
            orders.append(OrderResponse(
                order_id=row["order_id"],
                client_order_id=None,
                exchange_order_id=None,
                symbol=row["symbol"],
                side=row["side"],
                order_type=row["order_type"],
                quantity=str(row["quantity"]),
                filled_quantity=str(row["filled_quantity"]) if row["filled_quantity"] else "0",
                price=str(row["price"]) if row["price"] else None,
                avg_fill_price=str(row["avg_fill_price"]) if row["avg_fill_price"] else None,
                status=row["status"],
                exchange=row["exchange"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                metadata={}
            ))

        return orders

    except Exception as e:
        logger.error("Get orders failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get orders"
        )


@router.get(
    "/{order_id}",
    response_model=OrderResponse,
    summary="Get order",
    description="Get specific order by ID"
)
async def get_order(
    order_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_pool = Depends(get_db_pool)
) -> OrderResponse:
    """Get order by ID.

    Args:
        order_id: Order ID
        current_user: Current user
        db_pool: Database pool

    Returns:
        Order information

    Raises:
        HTTPException: If order not found
    """
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    order_id, symbol, side, order_type, quantity,
                    filled_quantity, price, avg_fill_price, status,
                    exchange, created_at, updated_at
                FROM orders
                WHERE order_id = $1 AND user_id = $2
                """,
                order_id,
                current_user["user_id"]
            )

            if not row:
                raise NotFoundException("Order", order_id)

        return OrderResponse(
            order_id=row["order_id"],
            client_order_id=None,
            exchange_order_id=None,
            symbol=row["symbol"],
            side=row["side"],
            order_type=row["order_type"],
            quantity=str(row["quantity"]),
            filled_quantity=str(row["filled_quantity"]) if row["filled_quantity"] else "0",
            price=str(row["price"]) if row["price"] else None,
            avg_fill_price=str(row["avg_fill_price"]) if row["avg_fill_price"] else None,
            status=row["status"],
            exchange=row["exchange"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata={}
        )

    except NotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Get order failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get order"
        )


@router.delete(
    "/{order_id}",
    response_model=CancelOrderResponse,
    summary="Cancel order",
    description="Cancel an existing order",
    dependencies=[Depends(require_permissions(["trade"]))]
)
async def cancel_order(
    order_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_pool = Depends(get_db_pool)
) -> CancelOrderResponse:
    """Cancel order.

    Args:
        order_id: Order ID
        current_user: Current user
        db_pool: Database pool

    Returns:
        Cancellation confirmation

    Raises:
        HTTPException: If cancellation fails
    """
    try:
        logger.info("Cancel order request", order_id=order_id, user_id=current_user["user_id"])

        async with db_pool.acquire() as conn:
            # Check order exists and belongs to user
            row = await conn.fetchrow(
                "SELECT status FROM orders WHERE order_id = $1 AND user_id = $2",
                order_id,
                current_user["user_id"]
            )

            if not row:
                raise NotFoundException("Order", order_id)

            # Check if order can be cancelled
            if row["status"] in [OrderStatus.FILLED.value, OrderStatus.CANCELLED.value]:
                raise OrderException(f"Cannot cancel order in {row['status']} status")

            # Update order status
            await conn.execute(
                """
                UPDATE orders
                SET status = $1, updated_at = $2
                WHERE order_id = $3
                """,
                OrderStatus.CANCELLED.value,
                datetime.utcnow(),
                order_id
            )

        logger.info("Order cancelled", order_id=order_id)

        return CancelOrderResponse(
            order_id=order_id,
            status=OrderStatus.CANCELLED,
            cancelled_at=datetime.utcnow()
        )

    except NotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except OrderException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Cancel order failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Cancel order failed"
        )


@router.patch(
    "/{order_id}",
    response_model=OrderResponse,
    summary="Modify order",
    description="Modify an existing order",
    dependencies=[Depends(require_permissions(["trade"]))]
)
async def modify_order(
    order_id: str,
    request: ModifyOrderRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_pool = Depends(get_db_pool)
) -> OrderResponse:
    """Modify order.

    Args:
        order_id: Order ID
        request: Modification request
        current_user: Current user
        db_pool: Database pool

    Returns:
        Modified order information

    Raises:
        HTTPException: If modification fails
    """
    try:
        logger.info("Modify order request", order_id=order_id)

        async with db_pool.acquire() as conn:
            # Get current order
            row = await conn.fetchrow(
                """
                SELECT status, order_type FROM orders
                WHERE order_id = $1 AND user_id = $2
                """,
                order_id,
                current_user["user_id"]
            )

            if not row:
                raise NotFoundException("Order", order_id)

            # Check if order can be modified
            if row["status"] not in [OrderStatus.OPEN.value, OrderStatus.PENDING.value]:
                raise OrderException(f"Cannot modify order in {row['status']} status")

            # Build update query
            updates = []
            params = []
            param_count = 0

            if request.quantity:
                param_count += 1
                updates.append(f"quantity = ${param_count}")
                params.append(Decimal(request.quantity))

            if request.price:
                param_count += 1
                updates.append(f"price = ${param_count}")
                params.append(Decimal(request.price))

            if not updates:
                raise ValidationException("No fields to update")

            param_count += 1
            updates.append(f"updated_at = ${param_count}")
            params.append(datetime.utcnow())

            params.extend([order_id, current_user["user_id"]])

            query = f"""
                UPDATE orders
                SET {', '.join(updates)}
                WHERE order_id = ${param_count + 1} AND user_id = ${param_count + 2}
                RETURNING *
            """

            updated_row = await conn.fetchrow(query, *params)

        logger.info("Order modified", order_id=order_id)

        return OrderResponse(
            order_id=updated_row["order_id"],
            client_order_id=None,
            exchange_order_id=None,
            symbol=updated_row["symbol"],
            side=updated_row["side"],
            order_type=updated_row["order_type"],
            quantity=str(updated_row["quantity"]),
            filled_quantity=str(updated_row["filled_quantity"]) if updated_row["filled_quantity"] else "0",
            price=str(updated_row["price"]) if updated_row["price"] else None,
            avg_fill_price=str(updated_row["avg_fill_price"]) if updated_row["avg_fill_price"] else None,
            status=updated_row["status"],
            exchange=updated_row["exchange"],
            created_at=updated_row["created_at"],
            updated_at=updated_row["updated_at"],
            metadata={}
        )

    except (NotFoundException, OrderException, ValidationException) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Modify order failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Modify order failed"
        )
