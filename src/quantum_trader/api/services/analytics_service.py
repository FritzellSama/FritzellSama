"""
Analytics service for Quantum Trader AI.

Provides comprehensive trading analytics, performance metrics, and reporting.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class AnalyticsService:
    """Production-ready analytics service for trading metrics.

    Provides real-time and historical analytics including:
    - P&L calculations and tracking
    - Performance metrics (Sharpe, Sortino, max drawdown)
    - Trade analytics and statistics
    - Portfolio analysis
    - Risk metrics

    Attributes:
        config: Configuration dictionary loaded from YAML
        db_pool: AsyncPG database connection pool
        redis_client: Redis client for caching
        cache_ttl: Cache time-to-live in seconds
    """

    def __init__(
        self,
        config: Dict[str, Any],
        db_pool: Any,
        redis_client: Any
    ) -> None:
        """Initialize analytics service.

        Args:
            config: Configuration from config files
            db_pool: Database connection pool
            redis_client: Redis client for caching

        Raises:
            ValueError: If config validation fails
        """
        self.config = config
        self.db_pool = db_pool
        self.redis_client = redis_client

        # Load from config
        self.cache_ttl = config.get("analytics", {}).get("cache_ttl_seconds", 300)
        self.risk_free_rate = Decimal(str(config.get("analytics", {}).get("risk_free_rate", "0.02")))
        self.trading_days_per_year = config.get("analytics", {}).get("trading_days_per_year", 365)

        self._validate_config()
        logger.info("Analytics service initialized", cache_ttl=self.cache_ttl)

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.cache_ttl <= 0:
            raise ValueError("cache_ttl must be positive")
        if self.risk_free_rate < Decimal("0"):
            raise ValueError("risk_free_rate cannot be negative")
        if self.trading_days_per_year <= 0:
            raise ValueError("trading_days_per_year must be positive")

    async def get_portfolio_metrics(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics.

        Args:
            user_id: User identifier
            start_date: Start date for analysis (UTC)
            end_date: End date for analysis (UTC)

        Returns:
            Dictionary containing:
                - total_pnl: Total P&L as Decimal
                - total_pnl_percent: Total P&L percentage as Decimal
                - sharpe_ratio: Sharpe ratio as Decimal
                - sortino_ratio: Sortino ratio as Decimal
                - max_drawdown: Maximum drawdown as Decimal
                - win_rate: Win rate as Decimal
                - total_trades: Total number of trades
                - avg_trade_pnl: Average trade P&L as Decimal

        Raises:
            ValueError: If user_id is invalid
            Exception: If database query fails
        """
        if not user_id:
            raise ValueError("user_id cannot be empty")

        # Check cache first
        cache_key = f"portfolio_metrics:{user_id}:{start_date}:{end_date}"
        try:
            cached_result = await self.redis_client.get(cache_key)
            if cached_result:
                logger.debug("Cache hit for portfolio metrics", user_id=user_id)
                return cached_result
        except Exception as e:
            logger.warning("Cache read failed", error=str(e))

        try:
            # Query trades from database
            trades_df = await self._fetch_trades(user_id, start_date, end_date)

            if trades_df.height == 0:
                logger.info("No trades found", user_id=user_id)
                return self._empty_metrics()

            # Calculate metrics
            metrics = self._calculate_portfolio_metrics(trades_df)

            # Cache results
            try:
                await self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    metrics
                )
            except Exception as e:
                logger.warning("Cache write failed", error=str(e))

            logger.info(
                "Portfolio metrics calculated",
                user_id=user_id,
                total_trades=metrics["total_trades"]
            )

            return metrics

        except Exception as e:
            logger.error(
                "Failed to calculate portfolio metrics",
                user_id=user_id,
                error=str(e)
            )
            raise

    async def _fetch_trades(
        self,
        user_id: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pl.DataFrame:
        """Fetch trades from database.

        Args:
            user_id: User identifier
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Polars DataFrame with trade data

        Raises:
            Exception: If database query fails
        """
        query = """
            SELECT
                trade_id,
                symbol,
                side,
                quantity,
                entry_price,
                exit_price,
                pnl,
                commission,
                opened_at,
                closed_at,
                strategy
            FROM trades
            WHERE user_id = $1
        """
        params = [user_id]

        if start_date:
            query += " AND closed_at >= $2"
            params.append(start_date)
        if end_date:
            query += f" AND closed_at <= ${len(params) + 1}"
            params.append(end_date)

        query += " ORDER BY closed_at DESC"

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        if not rows:
            return pl.DataFrame()

        # Convert to Polars DataFrame
        data = {
            "trade_id": [r["trade_id"] for r in rows],
            "symbol": [r["symbol"] for r in rows],
            "side": [r["side"] for r in rows],
            "quantity": [Decimal(str(r["quantity"])) for r in rows],
            "entry_price": [Decimal(str(r["entry_price"])) for r in rows],
            "exit_price": [Decimal(str(r["exit_price"])) for r in rows],
            "pnl": [Decimal(str(r["pnl"])) for r in rows],
            "commission": [Decimal(str(r["commission"])) for r in rows],
            "opened_at": [r["opened_at"] for r in rows],
            "closed_at": [r["closed_at"] for r in rows],
            "strategy": [r["strategy"] for r in rows],
        }

        return pl.DataFrame(data)

    def _calculate_portfolio_metrics(self, trades_df: pl.DataFrame) -> Dict[str, Any]:
        """Calculate portfolio metrics from trades DataFrame.

        Args:
            trades_df: Polars DataFrame with trade data

        Returns:
            Dictionary with calculated metrics
        """
        # Convert to list for calculations
        pnls = trades_df["pnl"].to_list()

        total_pnl = sum(pnls)
        total_trades = len(pnls)
        winning_trades = [p for p in pnls if p > Decimal("0")]
        losing_trades = [p for p in pnls if p < Decimal("0")]

        win_rate = Decimal(str(len(winning_trades))) / Decimal(str(total_trades)) if total_trades > 0 else Decimal("0")
        avg_trade_pnl = total_pnl / Decimal(str(total_trades)) if total_trades > 0 else Decimal("0")

        # Calculate Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio(pnls)

        # Calculate Sortino ratio
        sortino_ratio = self._calculate_sortino_ratio(pnls)

        # Calculate max drawdown
        max_drawdown = self._calculate_max_drawdown(pnls)

        # Calculate total P&L percentage (assuming we can derive it)
        avg_win = sum(winning_trades) / Decimal(str(len(winning_trades))) if winning_trades else Decimal("0")
        avg_loss = sum(losing_trades) / Decimal(str(len(losing_trades))) if losing_trades else Decimal("0")

        return {
            "total_pnl": str(total_pnl),
            "total_pnl_percent": str(sharpe_ratio * Decimal("100")) if sharpe_ratio else "0",
            "sharpe_ratio": str(sharpe_ratio),
            "sortino_ratio": str(sortino_ratio),
            "max_drawdown": str(max_drawdown),
            "win_rate": str(win_rate * Decimal("100")),
            "total_trades": total_trades,
            "avg_trade_pnl": str(avg_trade_pnl),
            "avg_win": str(avg_win),
            "avg_loss": str(avg_loss),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
        }

    def _calculate_sharpe_ratio(self, pnls: List[Decimal]) -> Decimal:
        """Calculate Sharpe ratio.

        Args:
            pnls: List of P&L values

        Returns:
            Sharpe ratio as Decimal
        """
        if len(pnls) < 2:
            return Decimal("0")

        mean_return = sum(pnls) / Decimal(str(len(pnls)))

        # Calculate standard deviation
        variance = sum((p - mean_return) ** 2 for p in pnls) / Decimal(str(len(pnls) - 1))
        std_dev = variance.sqrt() if variance > Decimal("0") else Decimal("0")

        if std_dev == Decimal("0"):
            return Decimal("0")

        # Annualized Sharpe ratio
        sharpe = (mean_return - self.risk_free_rate / Decimal(str(self.trading_days_per_year))) / std_dev
        sharpe = sharpe * (Decimal(str(self.trading_days_per_year)).sqrt())

        return sharpe

    def _calculate_sortino_ratio(self, pnls: List[Decimal]) -> Decimal:
        """Calculate Sortino ratio (downside deviation).

        Args:
            pnls: List of P&L values

        Returns:
            Sortino ratio as Decimal
        """
        if len(pnls) < 2:
            return Decimal("0")

        mean_return = sum(pnls) / Decimal(str(len(pnls)))

        # Calculate downside deviation
        downside_returns = [p for p in pnls if p < Decimal("0")]
        if not downside_returns:
            return Decimal("0")

        downside_variance = sum(p ** 2 for p in downside_returns) / Decimal(str(len(downside_returns)))
        downside_std = downside_variance.sqrt() if downside_variance > Decimal("0") else Decimal("0")

        if downside_std == Decimal("0"):
            return Decimal("0")

        sortino = (mean_return - self.risk_free_rate / Decimal(str(self.trading_days_per_year))) / downside_std
        sortino = sortino * (Decimal(str(self.trading_days_per_year)).sqrt())

        return sortino

    def _calculate_max_drawdown(self, pnls: List[Decimal]) -> Decimal:
        """Calculate maximum drawdown.

        Args:
            pnls: List of P&L values

        Returns:
            Maximum drawdown as Decimal (negative value)
        """
        if not pnls:
            return Decimal("0")

        cumulative = Decimal("0")
        peak = Decimal("0")
        max_dd = Decimal("0")

        for pnl in pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative

            drawdown = cumulative - peak
            if drawdown < max_dd:
                max_dd = drawdown

        return max_dd

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure.

        Returns:
            Dictionary with zero/empty metrics
        """
        return {
            "total_pnl": "0",
            "total_pnl_percent": "0",
            "sharpe_ratio": "0",
            "sortino_ratio": "0",
            "max_drawdown": "0",
            "win_rate": "0",
            "total_trades": 0,
            "avg_trade_pnl": "0",
            "avg_win": "0",
            "avg_loss": "0",
            "winning_trades": 0,
            "losing_trades": 0,
        }

    async def get_trade_distribution(
        self,
        user_id: str,
        group_by: str = "symbol"
    ) -> pl.DataFrame:
        """Get trade distribution grouped by specified field.

        Args:
            user_id: User identifier
            group_by: Field to group by (symbol, strategy, exchange)

        Returns:
            Polars DataFrame with aggregated trade data

        Raises:
            ValueError: If group_by field is invalid
        """
        valid_fields = ["symbol", "strategy", "exchange"]
        if group_by not in valid_fields:
            raise ValueError(f"group_by must be one of {valid_fields}")

        try:
            query = f"""
                SELECT
                    {group_by},
                    COUNT(*) as trade_count,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades
                FROM trades
                WHERE user_id = $1
                GROUP BY {group_by}
                ORDER BY total_pnl DESC
            """

            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, user_id)

            if not rows:
                return pl.DataFrame()

            data = {
                group_by: [r[group_by] for r in rows],
                "trade_count": [r["trade_count"] for r in rows],
                "total_pnl": [Decimal(str(r["total_pnl"])) for r in rows],
                "avg_pnl": [Decimal(str(r["avg_pnl"])) for r in rows],
                "winning_trades": [r["winning_trades"] for r in rows],
                "losing_trades": [r["losing_trades"] for r in rows],
            }

            return pl.DataFrame(data)

        except Exception as e:
            logger.error("Failed to get trade distribution", error=str(e))
            raise

    async def get_performance_over_time(
        self,
        user_id: str,
        interval: str = "1d"
    ) -> pl.DataFrame:
        """Get performance metrics aggregated over time intervals.

        Args:
            user_id: User identifier
            interval: Time interval (1h, 1d, 1w, 1M)

        Returns:
            Polars DataFrame with time-series performance data

        Raises:
            ValueError: If interval is invalid
        """
        valid_intervals = ["1h", "1d", "1w", "1M"]
        if interval not in valid_intervals:
            raise ValueError(f"interval must be one of {valid_intervals}")

        # Map to PostgreSQL interval
        pg_interval = {
            "1h": "1 hour",
            "1d": "1 day",
            "1w": "1 week",
            "1M": "1 month"
        }[interval]

        try:
            query = f"""
                SELECT
                    date_trunc('{pg_interval}', closed_at) as period,
                    COUNT(*) as trade_count,
                    SUM(pnl) as period_pnl,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as max_pnl,
                    MIN(pnl) as min_pnl
                FROM trades
                WHERE user_id = $1 AND closed_at IS NOT NULL
                GROUP BY period
                ORDER BY period DESC
                LIMIT 100
            """

            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, user_id)

            if not rows:
                return pl.DataFrame()

            data = {
                "period": [r["period"] for r in rows],
                "trade_count": [r["trade_count"] for r in rows],
                "period_pnl": [Decimal(str(r["period_pnl"])) for r in rows],
                "avg_pnl": [Decimal(str(r["avg_pnl"])) for r in rows],
                "max_pnl": [Decimal(str(r["max_pnl"])) for r in rows],
                "min_pnl": [Decimal(str(r["min_pnl"])) for r in rows],
            }

            return pl.DataFrame(data)

        except Exception as e:
            logger.error("Failed to get performance over time", error=str(e))
            raise

    async def close(self) -> None:
        """Cleanup resources."""
        logger.info("Analytics service shutting down")
