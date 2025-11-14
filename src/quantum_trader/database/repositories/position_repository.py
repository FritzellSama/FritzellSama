"""Position repository for database operations.

This module provides repository pattern implementation for position data
with support for real-time P&L tracking and complex queries.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, List, Any
import polars as pl
from sqlalchemy import select, update, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from structlog import get_logger

from quantum_trader.database.models.positions import PositionModel

logger = get_logger(__name__)


class PositionRepository:
    """Repository for position operations.

    Provides CRUD operations and complex queries for trading positions
    with support for real-time P&L updates.

    Attributes:
        session: Async database session

    Example:
        >>> repo = PositionRepository(session)
        >>> position = await repo.create_position(position_data)
        >>> await repo.update_position_price(position_id, Decimal("51000"))
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize position repository.

        Args:
            session: Async database session

        Example:
            >>> async with db.get_session() as session:
            ...     repo = PositionRepository(session)
        """
        self.session = session
        logger.debug("PositionRepository initialized")

    async def create_position(
        self,
        position_data: Dict[str, Any]
    ) -> PositionModel:
        """Create a new position.

        Args:
            position_data: Position data dictionary

        Returns:
            Created position model

        Example:
            >>> position = await repo.create_position({
            ...     'position_id': 'BTC_USDT_LONG_123',
            ...     'symbol': 'BTC/USDT',
            ...     'quantity': Decimal('0.5'),
            ...     'entry_price': Decimal('50000.00'),
            ...     'current_price': Decimal('50000.00'),
            ...     'exchange': 'BINANCE',
            ...     'strategy': 'momentum_v1'
            ... })
        """
        try:
            position = PositionModel(**position_data)
            self.session.add(position)
            await self.session.flush()

            logger.info(
                "Position created",
                position_id=position.position_id,
                symbol=position.symbol,
                quantity=position.quantity
            )
            return position

        except Exception as e:
            logger.error("Failed to create position", error=str(e))
            raise

    async def get_position(
        self,
        position_id: str
    ) -> Optional[PositionModel]:
        """Get position by ID.

        Args:
            position_id: Position ID

        Returns:
            Position model or None if not found

        Example:
            >>> position = await repo.get_position("BTC_USDT_LONG_123")
        """
        try:
            stmt = select(PositionModel).where(
                PositionModel.position_id == position_id
            )
            result = await self.session.execute(stmt)
            position = result.scalar_one_or_none()

            if position:
                logger.debug("Position retrieved", position_id=position_id)
            else:
                logger.warning("Position not found", position_id=position_id)

            return position

        except Exception as e:
            logger.error("Failed to get position", error=str(e), position_id=position_id)
            raise

    async def update_position_price(
        self,
        position_id: str,
        current_price: Decimal
    ) -> bool:
        """Update position current price and recalculate P&L.

        Args:
            position_id: Position ID
            current_price: Current market price

        Returns:
            True if updated successfully

        Example:
            >>> success = await repo.update_position_price(
            ...     "BTC_USDT_LONG_123",
            ...     Decimal("51000.00")
            ... )
        """
        try:
            position = await self.get_position(position_id)
            if not position:
                return False

            # Update P&L
            position.update_pnl(current_price)

            await self.session.flush()

            logger.info(
                "Position price updated",
                position_id=position_id,
                current_price=current_price,
                pnl=position.total_pnl
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to update position price",
                error=str(e),
                position_id=position_id
            )
            raise

    async def close_position(
        self,
        position_id: str,
        close_price: Decimal,
        realized_pnl: Decimal
    ) -> bool:
        """Close a position.

        Args:
            position_id: Position ID
            close_price: Closing price
            realized_pnl: Final realized P&L

        Returns:
            True if closed successfully

        Example:
            >>> success = await repo.close_position(
            ...     "BTC_USDT_LONG_123",
            ...     Decimal("51500.00"),
            ...     Decimal("750.00")
            ... )
        """
        try:
            stmt = (
                update(PositionModel)
                .where(PositionModel.position_id == position_id)
                .values(
                    is_open=False,
                    closed_at=datetime.utcnow(),
                    current_price=close_price,
                    realized_pnl=realized_pnl,
                    total_pnl=realized_pnl,
                    updated_at=datetime.utcnow(),
                )
            )

            result = await self.session.execute(stmt)
            success = result.rowcount > 0

            if success:
                logger.info(
                    "Position closed",
                    position_id=position_id,
                    realized_pnl=realized_pnl
                )
            else:
                logger.warning("Position not found for closing", position_id=position_id)

            return success

        except Exception as e:
            logger.error("Failed to close position", error=str(e), position_id=position_id)
            raise

    async def get_open_positions(
        self,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> List[PositionModel]:
        """Get all open positions.

        Args:
            symbol: Optional symbol filter
            exchange: Optional exchange filter
            strategy: Optional strategy filter

        Returns:
            List of open positions

        Example:
            >>> positions = await repo.get_open_positions(symbol="BTC/USDT")
        """
        try:
            stmt = select(PositionModel).where(PositionModel.is_open == True)

            if symbol:
                stmt = stmt.where(PositionModel.symbol == symbol)
            if exchange:
                stmt = stmt.where(PositionModel.exchange == exchange)
            if strategy:
                stmt = stmt.where(PositionModel.strategy == strategy)

            stmt = stmt.order_by(PositionModel.opened_at.desc())

            result = await self.session.execute(stmt)
            positions = result.scalars().all()

            logger.info("Open positions retrieved", count=len(positions))
            return list(positions)

        except Exception as e:
            logger.error("Failed to get open positions", error=str(e))
            raise

    async def get_position_history(
        self,
        start_time: datetime,
        end_time: datetime,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> pl.DataFrame:
        """Get position history as polars DataFrame.

        Args:
            start_time: Start timestamp (UTC)
            end_time: End timestamp (UTC)
            symbol: Optional symbol filter
            exchange: Optional exchange filter
            strategy: Optional strategy filter

        Returns:
            Polars DataFrame with position history

        Example:
            >>> df = await repo.get_position_history(
            ...     datetime(2024, 1, 1),
            ...     datetime(2024, 1, 2)
            ... )
        """
        try:
            stmt = select(PositionModel).where(
                and_(
                    PositionModel.opened_at >= start_time,
                    PositionModel.opened_at < end_time,
                )
            )

            if symbol:
                stmt = stmt.where(PositionModel.symbol == symbol)
            if exchange:
                stmt = stmt.where(PositionModel.exchange == exchange)
            if strategy:
                stmt = stmt.where(PositionModel.strategy == strategy)

            stmt = stmt.order_by(PositionModel.opened_at.desc())

            result = await self.session.execute(stmt)
            positions = result.scalars().all()

            # Convert to polars DataFrame
            if not positions:
                return pl.DataFrame()

            data = [p.to_dict() for p in positions]
            df = pl.DataFrame(data)

            logger.info(
                "Position history retrieved",
                count=len(df),
                start=start_time,
                end=end_time
            )
            return df

        except Exception as e:
            logger.error("Failed to get position history", error=str(e))
            raise

    async def get_position_pnl_summary(
        self,
        strategy: Optional[str] = None,
        exchange: Optional[str] = None,
    ) -> Dict[str, Decimal]:
        """Get P&L summary for positions.

        Args:
            strategy: Optional strategy filter
            exchange: Optional exchange filter

        Returns:
            Dictionary with P&L summary

        Example:
            >>> summary = await repo.get_position_pnl_summary(strategy="momentum_v1")
        """
        try:
            stmt = select(PositionModel)

            if strategy:
                stmt = stmt.where(PositionModel.strategy == strategy)
            if exchange:
                stmt = stmt.where(PositionModel.exchange == exchange)

            result = await self.session.execute(stmt)
            positions = result.scalars().all()

            total_pnl = Decimal("0")
            realized_pnl = Decimal("0")
            unrealized_pnl = Decimal("0")
            total_fees = Decimal("0")

            for position in positions:
                total_pnl += position.total_pnl
                realized_pnl += position.realized_pnl
                unrealized_pnl += position.unrealized_pnl
                total_fees += position.total_fees

            summary = {
                "total_pnl": total_pnl,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "total_fees": total_fees,
                "net_pnl": total_pnl - total_fees,
                "position_count": len(positions),
                "open_positions": len([p for p in positions if p.is_open]),
                "closed_positions": len([p for p in positions if not p.is_open]),
            }

            logger.info("Position P&L summary calculated", summary=summary)
            return summary

        except Exception as e:
            logger.error("Failed to get position P&L summary", error=str(e))
            raise

    async def update_stop_loss_take_profit(
        self,
        position_id: str,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
    ) -> bool:
        """Update position stop loss and take profit levels.

        Args:
            position_id: Position ID
            stop_loss: New stop loss price (optional)
            take_profit: New take profit price (optional)

        Returns:
            True if updated successfully

        Example:
            >>> success = await repo.update_stop_loss_take_profit(
            ...     "BTC_USDT_LONG_123",
            ...     stop_loss=Decimal("49000.00"),
            ...     take_profit=Decimal("52000.00")
            ... )
        """
        try:
            update_data = {"updated_at": datetime.utcnow()}

            if stop_loss is not None:
                update_data["stop_loss"] = stop_loss
            if take_profit is not None:
                update_data["take_profit"] = take_profit

            stmt = (
                update(PositionModel)
                .where(PositionModel.position_id == position_id)
                .values(**update_data)
            )

            result = await self.session.execute(stmt)
            success = result.rowcount > 0

            if success:
                logger.info(
                    "Position risk levels updated",
                    position_id=position_id,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
            else:
                logger.warning("Position not found for update", position_id=position_id)

            return success

        except Exception as e:
            logger.error(
                "Failed to update stop loss/take profit",
                error=str(e),
                position_id=position_id
            )
            raise

    async def get_positions_by_symbol(
        self,
        symbol: str,
        is_open: Optional[bool] = None
    ) -> List[PositionModel]:
        """Get all positions for a symbol.

        Args:
            symbol: Trading pair
            is_open: Optional filter for open/closed positions

        Returns:
            List of positions

        Example:
            >>> positions = await repo.get_positions_by_symbol("BTC/USDT", is_open=True)
        """
        try:
            stmt = select(PositionModel).where(PositionModel.symbol == symbol)

            if is_open is not None:
                stmt = stmt.where(PositionModel.is_open == is_open)

            stmt = stmt.order_by(PositionModel.opened_at.desc())

            result = await self.session.execute(stmt)
            positions = result.scalars().all()

            logger.info(
                "Positions by symbol retrieved",
                symbol=symbol,
                count=len(positions)
            )
            return list(positions)

        except Exception as e:
            logger.error("Failed to get positions by symbol", error=str(e), symbol=symbol)
            raise
