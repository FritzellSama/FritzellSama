"""Trade repository for database operations.

This module provides repository pattern implementation for trade data
with support for high-frequency inserts and time-series queries.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, List, Any
import polars as pl
from sqlalchemy import select, text, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from structlog import get_logger

from quantum_trader.database.models.trades import TradeModel

logger = get_logger(__name__)


class TradeRepository:
    """Repository for trade operations.

    Provides high-performance operations for executed trades with support
    for batch inserts and complex time-series queries.

    Attributes:
        session: Async database session

    Example:
        >>> repo = TradeRepository(session)
        >>> trade = await repo.create_trade(trade_data)
        >>> trades_df = await repo.get_trades(start_time, end_time)
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize trade repository.

        Args:
            session: Async database session

        Example:
            >>> async with db.get_session() as session:
            ...     repo = TradeRepository(session)
        """
        self.session = session
        logger.debug("TradeRepository initialized")

    async def create_trade(self, trade_data: Dict[str, Any]) -> TradeModel:
        """Create a new trade record.

        Args:
            trade_data: Trade data dictionary

        Returns:
            Created trade model

        Example:
            >>> trade = await repo.create_trade({
            ...     'trade_id': 'TRD_123',
            ...     'symbol': 'BTC/USDT',
            ...     'side': 'BUY',
            ...     'quantity': Decimal('0.1'),
            ...     'price': Decimal('50000.00'),
            ...     'quote_quantity': Decimal('5000.00'),
            ...     'exchange': 'BINANCE',
            ...     'strategy': 'momentum_v1',
            ...     'timestamp': datetime.utcnow()
            ... })
        """
        try:
            trade = TradeModel(**trade_data)
            self.session.add(trade)
            await self.session.flush()

            logger.info(
                "Trade created",
                trade_id=trade.trade_id,
                symbol=trade.symbol,
                side=trade.side,
                quantity=trade.quantity
            )
            return trade

        except Exception as e:
            logger.error("Failed to create trade", error=str(e))
            raise

    async def create_trades_batch(
        self,
        trades_data: List[Dict[str, Any]]
    ) -> List[TradeModel]:
        """Create multiple trades in batch.

        Args:
            trades_data: List of trade data dictionaries

        Returns:
            List of created trade models

        Example:
            >>> trades = await repo.create_trades_batch([
            ...     {'symbol': 'BTC/USDT', 'side': 'BUY', ...},
            ...     {'symbol': 'ETH/USDT', 'side': 'SELL', ...},
            ... ])
        """
        try:
            trades = [TradeModel(**data) for data in trades_data]
            self.session.add_all(trades)
            await self.session.flush()

            logger.info("Trades batch created", count=len(trades))
            return trades

        except Exception as e:
            logger.error("Failed to create trades batch", error=str(e))
            raise

    async def get_trade(self, trade_id: str) -> Optional[TradeModel]:
        """Get trade by ID.

        Args:
            trade_id: Trade ID

        Returns:
            Trade model or None if not found

        Example:
            >>> trade = await repo.get_trade("TRD_123")
        """
        try:
            stmt = select(TradeModel).where(TradeModel.trade_id == trade_id)
            result = await self.session.execute(stmt)
            trade = result.scalar_one_or_none()

            if trade:
                logger.debug("Trade retrieved", trade_id=trade_id)
            else:
                logger.warning("Trade not found", trade_id=trade_id)

            return trade

        except Exception as e:
            logger.error("Failed to get trade", error=str(e), trade_id=trade_id)
            raise

    async def get_trades(
        self,
        start_time: datetime,
        end_time: datetime,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        strategy: Optional[str] = None,
        side: Optional[str] = None,
    ) -> pl.DataFrame:
        """Get trades as polars DataFrame.

        Args:
            start_time: Start timestamp (UTC)
            end_time: End timestamp (UTC)
            symbol: Optional symbol filter
            exchange: Optional exchange filter
            strategy: Optional strategy filter
            side: Optional side filter (BUY/SELL)

        Returns:
            Polars DataFrame with trades

        Example:
            >>> df = await repo.get_trades(
            ...     datetime(2024, 1, 1),
            ...     datetime(2024, 1, 2),
            ...     symbol="BTC/USDT"
            ... )
        """
        try:
            stmt = select(TradeModel).where(
                and_(
                    TradeModel.timestamp >= start_time,
                    TradeModel.timestamp < end_time,
                )
            )

            if symbol:
                stmt = stmt.where(TradeModel.symbol == symbol)
            if exchange:
                stmt = stmt.where(TradeModel.exchange == exchange)
            if strategy:
                stmt = stmt.where(TradeModel.strategy == strategy)
            if side:
                stmt = stmt.where(TradeModel.side == side)

            stmt = stmt.order_by(TradeModel.timestamp.asc())

            result = await self.session.execute(stmt)
            trades = result.scalars().all()

            # Convert to polars DataFrame
            if not trades:
                return pl.DataFrame()

            data = [t.to_dict() for t in trades]
            df = pl.DataFrame(data)

            logger.info(
                "Trades retrieved",
                count=len(df),
                start=start_time,
                end=end_time
            )
            return df

        except Exception as e:
            logger.error("Failed to get trades", error=str(e))
            raise

    async def get_trades_by_order(self, order_id: str) -> List[TradeModel]:
        """Get all trades for an order.

        Args:
            order_id: Order ID

        Returns:
            List of trades

        Example:
            >>> trades = await repo.get_trades_by_order("ORD_123")
        """
        try:
            stmt = select(TradeModel).where(
                TradeModel.order_id == order_id
            ).order_by(TradeModel.timestamp.asc())

            result = await self.session.execute(stmt)
            trades = result.scalars().all()

            logger.info("Trades by order retrieved", order_id=order_id, count=len(trades))
            return list(trades)

        except Exception as e:
            logger.error("Failed to get trades by order", error=str(e), order_id=order_id)
            raise

    async def get_trades_by_position(
        self,
        position_id: str
    ) -> List[TradeModel]:
        """Get all trades for a position.

        Args:
            position_id: Position ID

        Returns:
            List of trades

        Example:
            >>> trades = await repo.get_trades_by_position("BTC_USDT_LONG_123")
        """
        try:
            stmt = select(TradeModel).where(
                TradeModel.position_id == position_id
            ).order_by(TradeModel.timestamp.asc())

            result = await self.session.execute(stmt)
            trades = result.scalars().all()

            logger.info(
                "Trades by position retrieved",
                position_id=position_id,
                count=len(trades)
            )
            return list(trades)

        except Exception as e:
            logger.error(
                "Failed to get trades by position",
                error=str(e),
                position_id=position_id
            )
            raise

    async def get_trade_statistics(
        self,
        start_time: datetime,
        end_time: datetime,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get trade statistics for a time period.

        Args:
            start_time: Start timestamp (UTC)
            end_time: End timestamp (UTC)
            symbol: Optional symbol filter
            strategy: Optional strategy filter

        Returns:
            Dictionary with trade statistics

        Example:
            >>> stats = await repo.get_trade_statistics(
            ...     datetime(2024, 1, 1),
            ...     datetime(2024, 1, 2)
            ... )
        """
        try:
            stmt = select(TradeModel).where(
                and_(
                    TradeModel.timestamp >= start_time,
                    TradeModel.timestamp < end_time,
                )
            )

            if symbol:
                stmt = stmt.where(TradeModel.symbol == symbol)
            if strategy:
                stmt = stmt.where(TradeModel.strategy == strategy)

            result = await self.session.execute(stmt)
            trades = result.scalars().all()

            total_trades = len(trades)
            buy_trades = len([t for t in trades if t.is_buy])
            sell_trades = len([t for t in trades if t.is_sell])

            total_volume = sum(t.quantity for t in trades)
            total_quote_volume = sum(t.quote_quantity for t in trades)
            total_fees = sum(t.fee for t in trades)

            profitable_trades = len([t for t in trades if t.is_profitable])

            stats = {
                "total_trades": total_trades,
                "buy_trades": buy_trades,
                "sell_trades": sell_trades,
                "total_volume": total_volume,
                "total_quote_volume": total_quote_volume,
                "total_fees": total_fees,
                "profitable_trades": profitable_trades,
                "avg_trade_size": total_volume / Decimal(total_trades) if total_trades > 0 else Decimal("0"),
                "maker_trades": len([t for t in trades if t.is_maker]),
                "taker_trades": len([t for t in trades if not t.is_maker]),
            }

            logger.info("Trade statistics calculated", stats=stats)
            return stats

        except Exception as e:
            logger.error("Failed to get trade statistics", error=str(e))
            raise

    async def get_realized_pnl(
        self,
        start_time: datetime,
        end_time: datetime,
        strategy: Optional[str] = None,
        exchange: Optional[str] = None,
    ) -> Decimal:
        """Get total realized P&L from closing trades.

        Args:
            start_time: Start timestamp (UTC)
            end_time: End timestamp (UTC)
            strategy: Optional strategy filter
            exchange: Optional exchange filter

        Returns:
            Total realized P&L

        Example:
            >>> pnl = await repo.get_realized_pnl(
            ...     datetime(2024, 1, 1),
            ...     datetime(2024, 1, 2)
            ... )
        """
        try:
            stmt = select(TradeModel).where(
                and_(
                    TradeModel.timestamp >= start_time,
                    TradeModel.timestamp < end_time,
                    TradeModel.is_closing == True,
                    TradeModel.realized_pnl.isnot(None),
                )
            )

            if strategy:
                stmt = stmt.where(TradeModel.strategy == strategy)
            if exchange:
                stmt = stmt.where(TradeModel.exchange == exchange)

            result = await self.session.execute(stmt)
            trades = result.scalars().all()

            total_pnl = sum(t.realized_pnl for t in trades if t.realized_pnl)

            logger.info(
                "Realized P&L calculated",
                pnl=total_pnl,
                trades=len(trades)
            )
            return total_pnl

        except Exception as e:
            logger.error("Failed to get realized P&L", error=str(e))
            raise

    async def get_recent_trades(
        self,
        symbol: str,
        limit: int,
        exchange: Optional[str] = None,
    ) -> List[TradeModel]:
        """Get recent trades for a symbol.

        Args:
            symbol: Trading pair
            limit: Maximum number of trades to return
            exchange: Optional exchange filter

        Returns:
            List of recent trades

        Example:
            >>> trades = await repo.get_recent_trades("BTC/USDT", 100)
        """
        try:
            stmt = select(TradeModel).where(TradeModel.symbol == symbol)

            if exchange:
                stmt = stmt.where(TradeModel.exchange == exchange)

            stmt = stmt.order_by(TradeModel.timestamp.desc()).limit(limit)

            result = await self.session.execute(stmt)
            trades = result.scalars().all()

            logger.info(
                "Recent trades retrieved",
                symbol=symbol,
                count=len(trades)
            )
            return list(trades)

        except Exception as e:
            logger.error("Failed to get recent trades", error=str(e), symbol=symbol)
            raise

    async def get_volume_by_exchange(
        self,
        start_time: datetime,
        end_time: datetime,
        symbol: Optional[str] = None,
    ) -> Dict[str, Decimal]:
        """Get trading volume grouped by exchange.

        Args:
            start_time: Start timestamp (UTC)
            end_time: End timestamp (UTC)
            symbol: Optional symbol filter

        Returns:
            Dictionary mapping exchange to total volume

        Example:
            >>> volumes = await repo.get_volume_by_exchange(
            ...     datetime(2024, 1, 1),
            ...     datetime(2024, 1, 2)
            ... )
        """
        try:
            stmt = select(
                TradeModel.exchange,
                func.sum(TradeModel.quantity).label("total_volume")
            ).where(
                and_(
                    TradeModel.timestamp >= start_time,
                    TradeModel.timestamp < end_time,
                )
            )

            if symbol:
                stmt = stmt.where(TradeModel.symbol == symbol)

            stmt = stmt.group_by(TradeModel.exchange)

            result = await self.session.execute(stmt)
            rows = result.fetchall()

            volumes = {row[0]: Decimal(str(row[1])) for row in rows}

            logger.info("Volume by exchange calculated", exchanges=list(volumes.keys()))
            return volumes

        except Exception as e:
            logger.error("Failed to get volume by exchange", error=str(e))
            raise

    async def get_trade_count_by_strategy(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, int]:
        """Get trade count grouped by strategy.

        Args:
            start_time: Start timestamp (UTC)
            end_time: End timestamp (UTC)

        Returns:
            Dictionary mapping strategy to trade count

        Example:
            >>> counts = await repo.get_trade_count_by_strategy(
            ...     datetime(2024, 1, 1),
            ...     datetime(2024, 1, 2)
            ... )
        """
        try:
            stmt = select(
                TradeModel.strategy,
                func.count(TradeModel.id).label("trade_count")
            ).where(
                and_(
                    TradeModel.timestamp >= start_time,
                    TradeModel.timestamp < end_time,
                )
            ).group_by(TradeModel.strategy)

            result = await self.session.execute(stmt)
            rows = result.fetchall()

            counts = {row[0]: int(row[1]) for row in rows}

            logger.info("Trade count by strategy calculated", strategies=list(counts.keys()))
            return counts

        except Exception as e:
            logger.error("Failed to get trade count by strategy", error=str(e))
            raise
