"""Order repository for database operations.

This module provides repository pattern implementation for order data
with support for high-performance CRUD operations and complex queries.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, List, Any
import polars as pl
from sqlalchemy import select, update, delete, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from structlog import get_logger

from quantum_trader.database.models.orders import OrderModel

logger = get_logger(__name__)


class OrderRepository:
    """Repository for order operations.

    Provides CRUD operations and complex queries for trading orders
    with support for high-frequency operations.

    Attributes:
        session: Async database session

    Example:
        >>> repo = OrderRepository(session)
        >>> order = await repo.create_order(order_data)
        >>> await repo.update_order_status(order_id, "FILLED")
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize order repository.

        Args:
            session: Async database session

        Example:
            >>> async with db.get_session() as session:
            ...     repo = OrderRepository(session)
        """
        self.session = session
        logger.debug("OrderRepository initialized")

    async def create_order(self, order_data: Dict[str, Any]) -> OrderModel:
        """Create a new order.

        Args:
            order_data: Order data dictionary

        Returns:
            Created order model

        Example:
            >>> order = await repo.create_order({
            ...     'symbol': 'BTC/USDT',
            ...     'side': 'BUY',
            ...     'quantity': Decimal('0.1'),
            ...     'price': Decimal('50000.00'),
            ...     'exchange': 'BINANCE',
            ...     'strategy': 'momentum_v1'
            ... })
        """
        try:
            order = OrderModel(**order_data)
            self.session.add(order)
            await self.session.flush()

            logger.info(
                "Order created",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side
            )
            return order

        except Exception as e:
            logger.error("Failed to create order", error=str(e))
            raise

    async def get_order(self, order_id: str) -> Optional[OrderModel]:
        """Get order by ID.

        Args:
            order_id: Internal order ID

        Returns:
            Order model or None if not found

        Example:
            >>> order = await repo.get_order("ORD_123")
        """
        try:
            stmt = select(OrderModel).where(OrderModel.order_id == order_id)
            result = await self.session.execute(stmt)
            order = result.scalar_one_or_none()

            if order:
                logger.debug("Order retrieved", order_id=order_id)
            else:
                logger.warning("Order not found", order_id=order_id)

            return order

        except Exception as e:
            logger.error("Failed to get order", error=str(e), order_id=order_id)
            raise

    async def update_order_status(
        self,
        order_id: str,
        status: str,
        filled_quantity: Optional[Decimal] = None,
        average_fill_price: Optional[Decimal] = None,
    ) -> bool:
        """Update order status and execution details.

        Args:
            order_id: Internal order ID
            status: New order status
            filled_quantity: Filled quantity (optional)
            average_fill_price: Average fill price (optional)

        Returns:
            True if updated successfully

        Example:
            >>> success = await repo.update_order_status(
            ...     "ORD_123",
            ...     "FILLED",
            ...     Decimal("0.1"),
            ...     Decimal("50500.00")
            ... )
        """
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow(),
            }

            if filled_quantity is not None:
                update_data["filled_quantity"] = filled_quantity

            if average_fill_price is not None:
                update_data["average_fill_price"] = average_fill_price

            if status == "FILLED":
                update_data["filled_at"] = datetime.utcnow()
            elif status == "CANCELLED":
                update_data["cancelled_at"] = datetime.utcnow()

            stmt = (
                update(OrderModel)
                .where(OrderModel.order_id == order_id)
                .values(**update_data)
            )

            result = await self.session.execute(stmt)
            success = result.rowcount > 0

            if success:
                logger.info("Order status updated", order_id=order_id, status=status)
            else:
                logger.warning("Order not found for update", order_id=order_id)

            return success

        except Exception as e:
            logger.error("Failed to update order status", error=str(e), order_id=order_id)
            raise

    async def get_active_orders(
        self,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> List[OrderModel]:
        """Get active orders (PENDING, OPEN, PARTIAL).

        Args:
            symbol: Optional symbol filter
            exchange: Optional exchange filter
            strategy: Optional strategy filter

        Returns:
            List of active orders

        Example:
            >>> orders = await repo.get_active_orders(symbol="BTC/USDT")
        """
        try:
            stmt = select(OrderModel).where(
                OrderModel.status.in_(["PENDING", "OPEN", "PARTIAL"])
            )

            if symbol:
                stmt = stmt.where(OrderModel.symbol == symbol)
            if exchange:
                stmt = stmt.where(OrderModel.exchange == exchange)
            if strategy:
                stmt = stmt.where(OrderModel.strategy == strategy)

            stmt = stmt.order_by(OrderModel.timestamp.desc())

            result = await self.session.execute(stmt)
            orders = result.scalars().all()

            logger.info("Active orders retrieved", count=len(orders))
            return list(orders)

        except Exception as e:
            logger.error("Failed to get active orders", error=str(e))
            raise

    async def get_order_history(
        self,
        start_time: datetime,
        end_time: datetime,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        strategy: Optional[str] = None,
        status: Optional[str] = None,
    ) -> pl.DataFrame:
        """Get order history as polars DataFrame.

        Args:
            start_time: Start timestamp (UTC)
            end_time: End timestamp (UTC)
            symbol: Optional symbol filter
            exchange: Optional exchange filter
            strategy: Optional strategy filter
            status: Optional status filter

        Returns:
            Polars DataFrame with order history

        Example:
            >>> df = await repo.get_order_history(
            ...     datetime(2024, 1, 1),
            ...     datetime(2024, 1, 2),
            ...     symbol="BTC/USDT"
            ... )
        """
        try:
            stmt = select(OrderModel).where(
                and_(
                    OrderModel.timestamp >= start_time,
                    OrderModel.timestamp < end_time,
                )
            )

            if symbol:
                stmt = stmt.where(OrderModel.symbol == symbol)
            if exchange:
                stmt = stmt.where(OrderModel.exchange == exchange)
            if strategy:
                stmt = stmt.where(OrderModel.strategy == strategy)
            if status:
                stmt = stmt.where(OrderModel.status == status)

            stmt = stmt.order_by(OrderModel.timestamp.desc())

            result = await self.session.execute(stmt)
            orders = result.scalars().all()

            # Convert to polars DataFrame
            if not orders:
                return pl.DataFrame()

            data = [order.to_dict() for order in orders]
            df = pl.DataFrame(data)

            logger.info(
                "Order history retrieved",
                count=len(df),
                start=start_time,
                end=end_time
            )
            return df

        except Exception as e:
            logger.error("Failed to get order history", error=str(e))
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order.

        Args:
            order_id: Internal order ID

        Returns:
            True if cancelled successfully

        Example:
            >>> success = await repo.cancel_order("ORD_123")
        """
        return await self.update_order_status(order_id, "CANCELLED")

    async def delete_old_orders(self, days: int) -> int:
        """Delete orders older than specified days.

        Args:
            days: Number of days to keep

        Returns:
            Number of deleted orders

        Example:
            >>> deleted = await repo.delete_old_orders(90)
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            stmt = delete(OrderModel).where(
                and_(
                    OrderModel.timestamp < cutoff_date,
                    OrderModel.status.in_(["FILLED", "CANCELLED", "REJECTED", "EXPIRED"])
                )
            )

            result = await self.session.execute(stmt)
            deleted_count = result.rowcount

            logger.info("Old orders deleted", count=deleted_count, days=days)
            return deleted_count

        except Exception as e:
            logger.error("Failed to delete old orders", error=str(e))
            raise

    async def get_order_statistics(
        self,
        start_time: datetime,
        end_time: datetime,
        strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get order statistics for a time period.

        Args:
            start_time: Start timestamp (UTC)
            end_time: End timestamp (UTC)
            strategy: Optional strategy filter

        Returns:
            Dictionary with order statistics

        Example:
            >>> stats = await repo.get_order_statistics(
            ...     datetime(2024, 1, 1),
            ...     datetime(2024, 1, 2)
            ... )
        """
        try:
            stmt = select(OrderModel).where(
                and_(
                    OrderModel.timestamp >= start_time,
                    OrderModel.timestamp < end_time,
                )
            )

            if strategy:
                stmt = stmt.where(OrderModel.strategy == strategy)

            result = await self.session.execute(stmt)
            orders = result.scalars().all()

            total_orders = len(orders)
            filled_orders = len([o for o in orders if o.status == "FILLED"])
            cancelled_orders = len([o for o in orders if o.status == "CANCELLED"])
            rejected_orders = len([o for o in orders if o.status == "REJECTED"])

            stats = {
                "total_orders": total_orders,
                "filled_orders": filled_orders,
                "cancelled_orders": cancelled_orders,
                "rejected_orders": rejected_orders,
                "fill_rate": (Decimal(filled_orders) / Decimal(total_orders) * Decimal("100"))
                if total_orders > 0 else Decimal("0"),
                "cancel_rate": (Decimal(cancelled_orders) / Decimal(total_orders) * Decimal("100"))
                if total_orders > 0 else Decimal("0"),
            }

            logger.info("Order statistics calculated", stats=stats)
            return stats

        except Exception as e:
            logger.error("Failed to get order statistics", error=str(e))
            raise
