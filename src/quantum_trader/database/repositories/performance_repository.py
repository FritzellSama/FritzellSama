"""Performance repository for database operations.

This module provides repository pattern implementation for performance metrics
with support for time-series queries and aggregations.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, List, Any
import polars as pl
from sqlalchemy import select, text, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from structlog import get_logger

from quantum_trader.database.models.performance import PerformanceMetricsModel

logger = get_logger(__name__)


class PerformanceRepository:
    """Repository for performance metrics operations.

    Provides CRUD operations and complex queries for strategy performance
    metrics with support for time-series analysis.

    Attributes:
        session: Async database session

    Example:
        >>> repo = PerformanceRepository(session)
        >>> metrics = await repo.save_metrics(metrics_data)
        >>> performance = await repo.get_strategy_performance("momentum_v1")
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize performance repository.

        Args:
            session: Async database session

        Example:
            >>> async with db.get_session() as session:
            ...     repo = PerformanceRepository(session)
        """
        self.session = session
        logger.debug("PerformanceRepository initialized")

    async def save_metrics(
        self,
        metrics_data: Dict[str, Any]
    ) -> PerformanceMetricsModel:
        """Save performance metrics.

        Args:
            metrics_data: Performance metrics data

        Returns:
            Created metrics model

        Example:
            >>> metrics = await repo.save_metrics({
            ...     'strategy': 'momentum_v1',
            ...     'exchange': 'BINANCE',
            ...     'timestamp': datetime.utcnow(),
            ...     'total_pnl': Decimal('15000.50'),
            ...     'sharpe_ratio': Decimal('2.5'),
            ... })
        """
        try:
            metrics = PerformanceMetricsModel(**metrics_data)
            self.session.add(metrics)
            await self.session.flush()

            logger.info(
                "Performance metrics saved",
                strategy=metrics.strategy,
                pnl=metrics.total_pnl
            )
            return metrics

        except Exception as e:
            logger.error("Failed to save performance metrics", error=str(e))
            raise

    async def get_latest_metrics(
        self,
        strategy: str,
        exchange: Optional[str] = None
    ) -> Optional[PerformanceMetricsModel]:
        """Get latest performance metrics for a strategy.

        Args:
            strategy: Strategy name
            exchange: Optional exchange filter

        Returns:
            Latest metrics or None if not found

        Example:
            >>> metrics = await repo.get_latest_metrics("momentum_v1", "BINANCE")
        """
        try:
            stmt = select(PerformanceMetricsModel).where(
                PerformanceMetricsModel.strategy == strategy
            )

            if exchange:
                stmt = stmt.where(PerformanceMetricsModel.exchange == exchange)

            stmt = stmt.order_by(PerformanceMetricsModel.timestamp.desc()).limit(1)

            result = await self.session.execute(stmt)
            metrics = result.scalar_one_or_none()

            if metrics:
                logger.debug("Latest metrics retrieved", strategy=strategy)
            else:
                logger.warning("No metrics found", strategy=strategy)

            return metrics

        except Exception as e:
            logger.error("Failed to get latest metrics", error=str(e), strategy=strategy)
            raise

    async def get_metrics_history(
        self,
        strategy: str,
        start_time: datetime,
        end_time: datetime,
        exchange: Optional[str] = None,
    ) -> pl.DataFrame:
        """Get performance metrics history as polars DataFrame.

        Args:
            strategy: Strategy name
            start_time: Start timestamp (UTC)
            end_time: End timestamp (UTC)
            exchange: Optional exchange filter

        Returns:
            Polars DataFrame with metrics history

        Example:
            >>> df = await repo.get_metrics_history(
            ...     "momentum_v1",
            ...     datetime(2024, 1, 1),
            ...     datetime(2024, 1, 2)
            ... )
        """
        try:
            stmt = select(PerformanceMetricsModel).where(
                and_(
                    PerformanceMetricsModel.strategy == strategy,
                    PerformanceMetricsModel.timestamp >= start_time,
                    PerformanceMetricsModel.timestamp < end_time,
                )
            )

            if exchange:
                stmt = stmt.where(PerformanceMetricsModel.exchange == exchange)

            stmt = stmt.order_by(PerformanceMetricsModel.timestamp.asc())

            result = await self.session.execute(stmt)
            metrics = result.scalars().all()

            # Convert to polars DataFrame
            if not metrics:
                return pl.DataFrame()

            data = [m.to_dict() for m in metrics]
            df = pl.DataFrame(data)

            logger.info(
                "Metrics history retrieved",
                strategy=strategy,
                count=len(df)
            )
            return df

        except Exception as e:
            logger.error("Failed to get metrics history", error=str(e), strategy=strategy)
            raise

    async def get_strategy_performance(
        self,
        strategy: str,
        exchange: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get overall strategy performance summary.

        Args:
            strategy: Strategy name
            exchange: Optional exchange filter

        Returns:
            Dictionary with performance summary

        Example:
            >>> perf = await repo.get_strategy_performance("momentum_v1")
        """
        try:
            # Get latest metrics
            latest = await self.get_latest_metrics(strategy, exchange)

            if not latest:
                return {
                    "strategy": strategy,
                    "exchange": exchange,
                    "total_pnl": Decimal("0"),
                    "sharpe_ratio": None,
                    "max_drawdown": None,
                    "win_rate": None,
                    "total_trades": 0,
                }

            # Calculate additional metrics from history
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=30)

            history_df = await self.get_metrics_history(
                strategy, start_time, end_time, exchange
            )

            # Calculate metrics from history
            avg_sharpe = None
            max_dd = None

            if len(history_df) > 0:
                if "sharpe_ratio" in history_df.columns:
                    sharpe_values = history_df["sharpe_ratio"].drop_nulls()
                    if len(sharpe_values) > 0:
                        avg_sharpe = Decimal(str(sharpe_values.mean()))

                if "max_drawdown" in history_df.columns:
                    dd_values = history_df["max_drawdown"].drop_nulls()
                    if len(dd_values) > 0:
                        max_dd = Decimal(str(dd_values.max()))

            performance = {
                "strategy": strategy,
                "exchange": exchange or "ALL",
                "total_pnl": latest.total_pnl,
                "realized_pnl": latest.realized_pnl,
                "unrealized_pnl": latest.unrealized_pnl,
                "sharpe_ratio": avg_sharpe or latest.sharpe_ratio,
                "max_drawdown": max_dd or latest.max_drawdown,
                "win_rate": latest.win_rate,
                "profit_factor": latest.profit_factor,
                "total_trades": int(latest.total_trades) if latest.total_trades else 0,
                "winning_trades": int(latest.winning_trades) if latest.winning_trades else 0,
                "last_updated": latest.timestamp,
            }

            logger.info("Strategy performance retrieved", strategy=strategy)
            return performance

        except Exception as e:
            logger.error("Failed to get strategy performance", error=str(e), strategy=strategy)
            raise

    async def compare_strategies(
        self,
        strategies: List[str],
        start_time: datetime,
        end_time: datetime,
    ) -> pl.DataFrame:
        """Compare performance of multiple strategies.

        Args:
            strategies: List of strategy names
            start_time: Start timestamp (UTC)
            end_time: End timestamp (UTC)

        Returns:
            Polars DataFrame with comparative metrics

        Example:
            >>> df = await repo.compare_strategies(
            ...     ["momentum_v1", "mean_reversion_v1"],
            ...     datetime(2024, 1, 1),
            ...     datetime(2024, 1, 31)
            ... )
        """
        try:
            comparison_data = []

            for strategy in strategies:
                # Get latest metrics for period
                stmt = select(PerformanceMetricsModel).where(
                    and_(
                        PerformanceMetricsModel.strategy == strategy,
                        PerformanceMetricsModel.timestamp >= start_time,
                        PerformanceMetricsModel.timestamp < end_time,
                    )
                ).order_by(PerformanceMetricsModel.timestamp.desc()).limit(1)

                result = await self.session.execute(stmt)
                metrics = result.scalar_one_or_none()

                if metrics:
                    comparison_data.append({
                        "strategy": strategy,
                        "total_pnl": str(metrics.total_pnl),
                        "sharpe_ratio": str(metrics.sharpe_ratio) if metrics.sharpe_ratio else None,
                        "max_drawdown": str(metrics.max_drawdown) if metrics.max_drawdown else None,
                        "win_rate": str(metrics.win_rate) if metrics.win_rate else None,
                        "total_trades": int(metrics.total_trades) if metrics.total_trades else 0,
                    })

            if not comparison_data:
                return pl.DataFrame()

            df = pl.DataFrame(comparison_data)
            logger.info("Strategy comparison completed", strategies=strategies)
            return df

        except Exception as e:
            logger.error("Failed to compare strategies", error=str(e))
            raise

    async def get_top_performing_strategies(
        self,
        limit: int,
        metric: str = "total_pnl",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get top performing strategies by metric.

        Args:
            limit: Number of strategies to return
            metric: Metric to sort by (total_pnl, sharpe_ratio, win_rate)
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            List of strategy performance dictionaries

        Example:
            >>> top_strategies = await repo.get_top_performing_strategies(
            ...     limit=5,
            ...     metric="sharpe_ratio"
            ... )
        """
        try:
            # Build query to get latest metrics for each strategy
            if start_time and end_time:
                time_filter = and_(
                    PerformanceMetricsModel.timestamp >= start_time,
                    PerformanceMetricsModel.timestamp < end_time,
                )
            else:
                # Get recent metrics (last 24 hours)
                time_filter = PerformanceMetricsModel.timestamp >= (
                    datetime.utcnow() - timedelta(days=1)
                )

            # Get latest metrics per strategy
            subquery = (
                select(
                    PerformanceMetricsModel.strategy,
                    func.max(PerformanceMetricsModel.timestamp).label("max_timestamp")
                )
                .where(time_filter)
                .group_by(PerformanceMetricsModel.strategy)
                .subquery()
            )

            stmt = (
                select(PerformanceMetricsModel)
                .join(
                    subquery,
                    and_(
                        PerformanceMetricsModel.strategy == subquery.c.strategy,
                        PerformanceMetricsModel.timestamp == subquery.c.max_timestamp,
                    ),
                )
            )

            # Sort by metric
            if metric == "total_pnl":
                stmt = stmt.order_by(PerformanceMetricsModel.total_pnl.desc())
            elif metric == "sharpe_ratio":
                stmt = stmt.order_by(PerformanceMetricsModel.sharpe_ratio.desc())
            elif metric == "win_rate":
                stmt = stmt.order_by(PerformanceMetricsModel.win_rate.desc())

            stmt = stmt.limit(limit)

            result = await self.session.execute(stmt)
            metrics = result.scalars().all()

            top_strategies = [m.to_dict() for m in metrics]

            logger.info(
                "Top performing strategies retrieved",
                count=len(top_strategies),
                metric=metric
            )
            return top_strategies

        except Exception as e:
            logger.error("Failed to get top performing strategies", error=str(e))
            raise
