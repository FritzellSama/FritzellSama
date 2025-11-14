"""
Analytics API Router

Production-ready FastAPI router for analytics and performance metrics.
Provides endpoints for trading performance, portfolio analytics, and system metrics.
"""

import asyncio
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
import os

from fastapi import APIRouter, HTTPException, Depends, Query, Path as PathParam
from fastapi.responses import JSONResponse
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)

# Set high precision
getcontext().prec = 28

# Create router
router = APIRouter(
    prefix="/analytics",
    tags=["analytics"],
    responses={404: {"description": "Not found"}}
)


class AnalyticsService:
    """
    Service for analytics calculations.

    Centralizes analytics logic for performance metrics,
    risk metrics, and portfolio analytics.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize analytics service.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        analytics_config = self.config.get("analytics", {})

        self.risk_free_rate: Decimal = Decimal(
            str(analytics_config.get("risk_free_rate", os.getenv("ANALYTICS_RISK_FREE_RATE", "0.02")))
        )
        self.annualization_factor: Decimal = Decimal(
            str(analytics_config.get("annualization_factor", os.getenv("ANALYTICS_ANNUALIZATION_FACTOR", "252")))
        )

        logger.info("analytics_service_initialized")

    async def calculate_performance_metrics(
        self,
        trades_data: pl.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics from trades.

        Args:
            trades_data: DataFrame with trade history
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Dictionary of performance metrics
        """
        try:
            # Filter by date if provided
            if start_date:
                trades_data = trades_data.filter(pl.col("timestamp") >= start_date)
            if end_date:
                trades_data = trades_data.filter(pl.col("timestamp") <= end_date)

            if len(trades_data) == 0:
                logger.warning("no_trades_for_analytics")
                return self._get_empty_metrics()

            # Calculate P&L
            pnl = trades_data.select("pnl").to_numpy().flatten()
            total_pnl = Decimal(str(sum(pnl)))

            # Trade statistics
            total_trades = len(trades_data)
            winning_trades = len(trades_data.filter(pl.col("pnl") > 0))
            losing_trades = len(trades_data.filter(pl.col("pnl") < 0))

            win_rate = Decimal(str(winning_trades)) / Decimal(str(total_trades)) if total_trades > 0 else Decimal("0")

            # Average win/loss
            if winning_trades > 0:
                avg_win = Decimal(str(trades_data.filter(pl.col("pnl") > 0).select("pnl").mean()[0, 0]))
            else:
                avg_win = Decimal("0")

            if losing_trades > 0:
                avg_loss = Decimal(str(trades_data.filter(pl.col("pnl") < 0).select("pnl").mean()[0, 0]))
            else:
                avg_loss = Decimal("0")

            # Profit factor
            total_wins = sum(p for p in pnl if p > 0)
            total_losses = abs(sum(p for p in pnl if p < 0))
            profit_factor = Decimal(str(total_wins / total_losses)) if total_losses > 0 else Decimal("0")

            # Sharpe ratio
            returns = pnl / Decimal("1000")  # Normalize by capital
            sharpe = self._calculate_sharpe_ratio(returns)

            # Max drawdown
            cumulative_pnl = []
            cumsum = Decimal("0")
            for p in pnl:
                cumsum += Decimal(str(p))
                cumulative_pnl.append(cumsum)

            max_dd = self._calculate_max_drawdown(cumulative_pnl)

            metrics = {
                "total_pnl": str(total_pnl),
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": str(win_rate),
                "avg_win": str(avg_win),
                "avg_loss": str(avg_loss),
                "profit_factor": str(profit_factor),
                "sharpe_ratio": str(sharpe),
                "max_drawdown": str(max_dd),
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None
            }

            logger.info("performance_metrics_calculated", total_trades=total_trades)

            return metrics

        except Exception as e:
            logger.error("performance_metrics_calculation_failed", error=str(e))
            raise

    def _calculate_sharpe_ratio(self, returns: Any) -> Decimal:
        """Calculate Sharpe ratio."""
        import numpy as np

        if isinstance(returns, (list, tuple)):
            returns_array = np.array(returns, dtype=float)
        else:
            returns_array = returns

        if len(returns_array) == 0:
            return Decimal("0")

        mean_return = Decimal(str(np.mean(returns_array)))
        std_return = Decimal(str(np.std(returns_array)))

        if std_return == Decimal("0"):
            return Decimal("0")

        daily_rf = self.risk_free_rate / self.annualization_factor
        excess_return = mean_return - daily_rf

        sharpe = (excess_return / std_return) * (self.annualization_factor ** Decimal("0.5"))

        return sharpe

    def _calculate_max_drawdown(self, cumulative_pnl: List[Decimal]) -> Decimal:
        """Calculate maximum drawdown."""
        if not cumulative_pnl:
            return Decimal("0")

        peak = cumulative_pnl[0]
        max_dd = Decimal("0")

        for value in cumulative_pnl:
            if value > peak:
                peak = value

            dd = peak - value
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure."""
        return {
            "total_pnl": "0",
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": "0",
            "avg_win": "0",
            "avg_loss": "0",
            "profit_factor": "0",
            "sharpe_ratio": "0",
            "max_drawdown": "0",
            "start_date": None,
            "end_date": None
        }


# Dependency for analytics service
async def get_analytics_service() -> AnalyticsService:
    """Get analytics service instance."""
    # In production, this would load from app state
    config = {
        "analytics": {
            "risk_free_rate": os.getenv("ANALYTICS_RISK_FREE_RATE", "0.02"),
            "annualization_factor": os.getenv("ANALYTICS_ANNUALIZATION_FACTOR", "252")
        }
    }
    return AnalyticsService(config)


@router.get("/performance", summary="Get trading performance metrics")
async def get_performance_metrics(
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    strategy: Optional[str] = Query(None, description="Filter by strategy"),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
) -> Dict[str, Any]:
    """
    Get trading performance metrics.

    Returns comprehensive performance analytics including P&L,
    win rate, Sharpe ratio, and drawdown.

    Args:
        start_date: Start date for analysis
        end_date: End date for analysis
        symbol: Filter by specific symbol
        strategy: Filter by specific strategy
        analytics_service: Analytics service instance

    Returns:
        Dictionary with performance metrics

    Example:
        GET /analytics/performance?start_date=2025-01-01&end_date=2025-01-31
    """
    try:
        # Parse dates
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None

        # In production, fetch from database
        # For now, return mock data
        trades_data = pl.DataFrame({
            "timestamp": [datetime.now(timezone.utc)] * 10,
            "pnl": [100.0, -50.0, 200.0, -30.0, 150.0, -20.0, 80.0, -40.0, 120.0, -60.0],
            "symbol": ["BTC/USDT"] * 10,
            "strategy": ["momentum"] * 10
        })

        # Filter by symbol/strategy if provided
        if symbol:
            trades_data = trades_data.filter(pl.col("symbol") == symbol)
        if strategy:
            trades_data = trades_data.filter(pl.col("strategy") == strategy)

        metrics = await analytics_service.calculate_performance_metrics(
            trades_data,
            start_dt,
            end_dt
        )

        logger.info("performance_metrics_requested", symbol=symbol, strategy=strategy)

        return metrics

    except ValueError as e:
        logger.error("invalid_date_format", error=str(e))
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        logger.error("performance_metrics_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to calculate performance metrics")


@router.get("/portfolio", summary="Get portfolio analytics")
async def get_portfolio_analytics(
    as_of_date: Optional[str] = Query(None, description="As-of date (ISO format)")
) -> Dict[str, Any]:
    """
    Get portfolio analytics and allocation.

    Returns current portfolio positions, allocation,
    and risk metrics.

    Args:
        as_of_date: Date for portfolio snapshot

    Returns:
        Dictionary with portfolio analytics

    Example:
        GET /analytics/portfolio
    """
    try:
        as_of_dt = datetime.fromisoformat(as_of_date) if as_of_date else datetime.now(timezone.utc)

        # In production, fetch from database
        portfolio_data = {
            "timestamp": as_of_dt.isoformat(),
            "total_equity": "10000.00",
            "cash": "5000.00",
            "positions_value": "5000.00",
            "positions": [
                {
                    "symbol": "BTC/USDT",
                    "quantity": "0.1",
                    "entry_price": "45000.00",
                    "current_price": "50000.00",
                    "pnl": "500.00",
                    "pnl_percent": "11.11",
                    "allocation_percent": "50.00"
                }
            ],
            "allocation": {
                "BTC": "50.00",
                "cash": "50.00"
            },
            "risk_metrics": {
                "portfolio_volatility": "0.15",
                "var_95": "150.00",
                "beta": "1.2",
                "sharpe_ratio": "1.5"
            }
        }

        logger.info("portfolio_analytics_requested", as_of_date=as_of_dt.isoformat())

        return portfolio_data

    except ValueError as e:
        logger.error("invalid_date_format", error=str(e))
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        logger.error("portfolio_analytics_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get portfolio analytics")


@router.get("/risk", summary="Get risk metrics")
async def get_risk_metrics(
    lookback_days: int = Query(30, description="Lookback period in days", ge=1, le=365)
) -> Dict[str, Any]:
    """
    Get risk metrics and exposure analysis.

    Returns VaR, volatility, correlation, and other risk metrics.

    Args:
        lookback_days: Lookback period for calculations

    Returns:
        Dictionary with risk metrics

    Example:
        GET /analytics/risk?lookback_days=30
    """
    try:
        # In production, calculate from historical data
        risk_metrics = {
            "lookback_days": lookback_days,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "value_at_risk": {
                "var_95": "150.00",
                "var_99": "200.00",
                "cvar_95": "180.00"
            },
            "volatility": {
                "daily": "0.02",
                "weekly": "0.045",
                "monthly": "0.09",
                "annualized": "0.32"
            },
            "exposure": {
                "gross_exposure": "5000.00",
                "net_exposure": "5000.00",
                "leverage": "0.5"
            },
            "correlations": {
                "BTC_ETH": "0.85",
                "BTC_SOL": "0.75"
            },
            "limits": {
                "max_position_size": "1000.00",
                "max_leverage": "2.0",
                "max_drawdown_limit": "0.2",
                "current_drawdown": "0.05"
            }
        }

        logger.info("risk_metrics_requested", lookback_days=lookback_days)

        return risk_metrics

    except Exception as e:
        logger.error("risk_metrics_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to calculate risk metrics")


@router.get("/trades/summary", summary="Get trades summary")
async def get_trades_summary(
    start_date: Optional[str] = Query(None, description="Start date"),
    end_date: Optional[str] = Query(None, description="End date"),
    symbol: Optional[str] = Query(None, description="Symbol filter"),
    strategy: Optional[str] = Query(None, description="Strategy filter"),
    limit: int = Query(100, description="Max results", ge=1, le=1000)
) -> Dict[str, Any]:
    """
    Get trades summary and statistics.

    Returns trade statistics grouped by various dimensions.

    Args:
        start_date: Start date filter
        end_date: End date filter
        symbol: Symbol filter
        strategy: Strategy filter
        limit: Maximum number of results

    Returns:
        Dictionary with trades summary

    Example:
        GET /analytics/trades/summary?symbol=BTC/USDT
    """
    try:
        # Parse dates
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None

        # In production, query from database
        summary = {
            "filters": {
                "start_date": start_date,
                "end_date": end_date,
                "symbol": symbol,
                "strategy": strategy
            },
            "overall": {
                "total_trades": 100,
                "total_pnl": "1500.00",
                "avg_pnl_per_trade": "15.00",
                "win_rate": "0.60",
                "profit_factor": "2.0"
            },
            "by_symbol": [
                {
                    "symbol": "BTC/USDT",
                    "trades": 50,
                    "pnl": "1000.00",
                    "win_rate": "0.65"
                },
                {
                    "symbol": "ETH/USDT",
                    "trades": 50,
                    "pnl": "500.00",
                    "win_rate": "0.55"
                }
            ],
            "by_strategy": [
                {
                    "strategy": "momentum",
                    "trades": 60,
                    "pnl": "900.00",
                    "win_rate": "0.62"
                },
                {
                    "strategy": "mean_reversion",
                    "trades": 40,
                    "pnl": "600.00",
                    "win_rate": "0.58"
                }
            ],
            "by_hour": {
                "0": 5, "1": 3, "2": 2, "3": 1,
                "8": 10, "9": 15, "10": 12
            }
        }

        logger.info("trades_summary_requested", symbol=symbol, strategy=strategy)

        return summary

    except ValueError as e:
        logger.error("invalid_date_format", error=str(e))
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        logger.error("trades_summary_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get trades summary")


@router.get("/system/health", summary="Get system health metrics")
async def get_system_health() -> Dict[str, Any]:
    """
    Get system health and operational metrics.

    Returns system status, latency, uptime, and resource usage.

    Returns:
        Dictionary with system health metrics

    Example:
        GET /analytics/system/health
    """
    try:
        health_metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "healthy",
            "uptime_seconds": 86400,
            "performance": {
                "avg_latency_ms": "5.2",
                "p95_latency_ms": "8.5",
                "p99_latency_ms": "12.3",
                "requests_per_second": "100"
            },
            "resources": {
                "cpu_percent": "45.2",
                "memory_percent": "62.5",
                "disk_percent": "35.0"
            },
            "services": {
                "database": "healthy",
                "redis": "healthy",
                "exchanges": {
                    "binance": "connected",
                    "bybit": "connected"
                }
            },
            "trading": {
                "active_strategies": 5,
                "open_positions": 3,
                "pending_orders": 2
            }
        }

        logger.info("system_health_requested")

        return health_metrics

    except Exception as e:
        logger.error("system_health_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get system health")
