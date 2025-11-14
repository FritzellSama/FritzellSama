"""
Analytics Service

Production-ready service for trading analytics, performance metrics,
and risk calculations. Centralized analytics logic for the API layer.
"""

import asyncio
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import os

import polars as pl
import numpy as np
from structlog import get_logger

logger = get_logger(__name__)

# Set high precision
getcontext().prec = 28


class AnalyticsService:
    """
    Centralized analytics service for performance and risk calculations.

    Provides methods for calculating trading metrics, portfolio analytics,
    risk metrics, and system performance.

    Attributes:
        config: Configuration dictionary
        risk_free_rate: Annual risk-free rate for Sharpe calculation
        annualization_factor: Factor for annualizing returns (252 for daily)

    Example:
        >>> config = {"analytics": {"risk_free_rate": "0.02"}}
        >>> service = AnalyticsService(config)
        >>> metrics = await service.calculate_performance_metrics(trades_df)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize analytics service.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        analytics_config = self.config.get("analytics", {})

        # Risk and return parameters
        self.risk_free_rate: Decimal = Decimal(
            str(analytics_config.get("risk_free_rate", os.getenv("ANALYTICS_RISK_FREE_RATE", "0.02")))
        )
        self.annualization_factor: Decimal = Decimal(
            str(analytics_config.get("annualization_factor", os.getenv("ANALYTICS_ANNUALIZATION_FACTOR", "252")))
        )

        # VaR confidence levels
        self.var_confidence_levels: List[float] = analytics_config.get(
            "var_confidence_levels",
            [float(x) for x in os.getenv("VAR_CONFIDENCE_LEVELS", "0.95,0.99").split(",")]
        )

        # Drawdown thresholds
        self.warning_drawdown: Decimal = Decimal(
            str(analytics_config.get("warning_drawdown", os.getenv("WARNING_DRAWDOWN", "0.1")))
        )
        self.critical_drawdown: Decimal = Decimal(
            str(analytics_config.get("critical_drawdown", os.getenv("CRITICAL_DRAWDOWN", "0.2")))
        )

        logger.info(
            "analytics_service_initialized",
            risk_free_rate=str(self.risk_free_rate),
            annualization_factor=str(self.annualization_factor)
        )

    def _validate_config(self) -> None:
        """Validate configuration."""
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

    async def calculate_performance_metrics(
        self,
        trades_data: pl.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        initial_capital: Optional[Decimal] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.

        Args:
            trades_data: DataFrame with trade history
                Required columns: timestamp, pnl, symbol, strategy
            start_date: Start date for analysis
            end_date: End date for analysis
            initial_capital: Initial capital for return calculations

        Returns:
            Dictionary with performance metrics

        Raises:
            ValueError: If data is invalid

        Example:
            >>> metrics = await service.calculate_performance_metrics(
            ...     trades_df,
            ...     initial_capital=Decimal("10000")
            ... )
        """
        try:
            # Validate data
            required_cols = ["timestamp", "pnl"]
            if not all(col in trades_data.columns for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")

            # Filter by date
            if start_date:
                trades_data = trades_data.filter(pl.col("timestamp") >= start_date)
            if end_date:
                trades_data = trades_data.filter(pl.col("timestamp") <= end_date)

            if len(trades_data) == 0:
                logger.warning("no_trades_for_performance_calculation")
                return self._get_empty_performance_metrics()

            # Extract P&L data
            pnl_array = trades_data.select("pnl").to_numpy().flatten()
            pnl_decimals = [Decimal(str(p)) for p in pnl_array]

            # Basic metrics
            total_pnl = sum(pnl_decimals)
            total_trades = len(pnl_decimals)
            winning_trades = sum(1 for p in pnl_decimals if p > 0)
            losing_trades = sum(1 for p in pnl_decimals if p < 0)
            breakeven_trades = total_trades - winning_trades - losing_trades

            # Win/loss statistics
            win_rate = Decimal(str(winning_trades)) / Decimal(str(total_trades)) if total_trades > 0 else Decimal("0")

            winning_pnls = [p for p in pnl_decimals if p > 0]
            losing_pnls = [p for p in pnl_decimals if p < 0]

            avg_win = sum(winning_pnls) / Decimal(str(len(winning_pnls))) if winning_pnls else Decimal("0")
            avg_loss = sum(losing_pnls) / Decimal(str(len(losing_pnls))) if losing_pnls else Decimal("0")
            avg_trade = total_pnl / Decimal(str(total_trades)) if total_trades > 0 else Decimal("0")

            # Profit factor
            total_wins = sum(winning_pnls) if winning_pnls else Decimal("0")
            total_losses = abs(sum(losing_pnls)) if losing_pnls else Decimal("1")
            profit_factor = total_wins / total_losses if total_losses > 0 else Decimal("0")

            # Expectancy
            expectancy = (win_rate * avg_win) + ((Decimal("1") - win_rate) * avg_loss)

            # Returns and risk metrics
            if initial_capital:
                total_return_pct = (total_pnl / initial_capital) * Decimal("100")

                # Calculate returns for each trade
                returns = [p / initial_capital for p in pnl_decimals]
                sharpe = await self._calculate_sharpe_ratio(returns)
                sortino = await self._calculate_sortino_ratio(returns)
            else:
                total_return_pct = Decimal("0")
                sharpe = Decimal("0")
                sortino = Decimal("0")

            # Drawdown metrics
            cumulative_pnl = []
            running_sum = Decimal("0")
            for p in pnl_decimals:
                running_sum += p
                cumulative_pnl.append(running_sum)

            max_dd, max_dd_pct = await self._calculate_max_drawdown(
                cumulative_pnl,
                initial_capital
            )

            # Longest winning/losing streaks
            win_streak, loss_streak = self._calculate_streaks(pnl_decimals)

            # Time metrics
            if "timestamp" in trades_data.columns:
                timestamps = trades_data.select("timestamp").to_numpy().flatten()
                trading_period_days = (timestamps[-1] - timestamps[0]).days if len(timestamps) > 1 else 0
            else:
                trading_period_days = 0

            metrics = {
                "total_pnl": str(total_pnl),
                "total_return_pct": str(total_return_pct),
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "breakeven_trades": breakeven_trades,
                "win_rate": str(win_rate),
                "avg_win": str(avg_win),
                "avg_loss": str(avg_loss),
                "avg_trade": str(avg_trade),
                "profit_factor": str(profit_factor),
                "expectancy": str(expectancy),
                "sharpe_ratio": str(sharpe),
                "sortino_ratio": str(sortino),
                "max_drawdown": str(max_dd),
                "max_drawdown_pct": str(max_dd_pct),
                "longest_win_streak": win_streak,
                "longest_loss_streak": loss_streak,
                "trading_period_days": trading_period_days,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None
            }

            logger.info(
                "performance_metrics_calculated",
                total_trades=total_trades,
                win_rate=str(win_rate),
                sharpe=str(sharpe)
            )

            return metrics

        except Exception as e:
            logger.error("performance_metrics_calculation_failed", error=str(e))
            raise

    async def _calculate_sharpe_ratio(self, returns: List[Decimal]) -> Decimal:
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 2:
            return Decimal("0")

        returns_array = np.array([float(r) for r in returns])

        mean_return = Decimal(str(np.mean(returns_array)))
        std_return = Decimal(str(np.std(returns_array)))

        if std_return == Decimal("0"):
            return Decimal("0")

        daily_rf = self.risk_free_rate / self.annualization_factor
        excess_return = mean_return - daily_rf

        sharpe = (excess_return / std_return) * (self.annualization_factor ** Decimal("0.5"))

        return sharpe

    async def _calculate_sortino_ratio(self, returns: List[Decimal]) -> Decimal:
        """Calculate Sortino ratio (downside deviation)."""
        if not returns or len(returns) < 2:
            return Decimal("0")

        returns_array = np.array([float(r) for r in returns])

        mean_return = Decimal(str(np.mean(returns_array)))

        # Downside deviation (only negative returns)
        downside_returns = [r for r in returns if r < Decimal("0")]
        if not downside_returns:
            return Decimal("0")

        downside_array = np.array([float(r) for r in downside_returns])
        downside_std = Decimal(str(np.std(downside_array)))

        if downside_std == Decimal("0"):
            return Decimal("0")

        daily_rf = self.risk_free_rate / self.annualization_factor
        excess_return = mean_return - daily_rf

        sortino = (excess_return / downside_std) * (self.annualization_factor ** Decimal("0.5"))

        return sortino

    async def _calculate_max_drawdown(
        self,
        cumulative_pnl: List[Decimal],
        initial_capital: Optional[Decimal] = None
    ) -> Tuple[Decimal, Decimal]:
        """Calculate maximum drawdown."""
        if not cumulative_pnl:
            return Decimal("0"), Decimal("0")

        peak = cumulative_pnl[0]
        max_dd = Decimal("0")

        for value in cumulative_pnl:
            if value > peak:
                peak = value

            dd = peak - value
            if dd > max_dd:
                max_dd = dd

        # Calculate percentage drawdown
        if initial_capital and initial_capital > 0:
            max_dd_pct = (max_dd / initial_capital) * Decimal("100")
        else:
            max_dd_pct = Decimal("0")

        return max_dd, max_dd_pct

    def _calculate_streaks(self, pnl: List[Decimal]) -> Tuple[int, int]:
        """Calculate longest winning and losing streaks."""
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0

        for p in pnl:
            if p > 0:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            elif p < 0:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
            else:
                current_win_streak = 0
                current_loss_streak = 0

        return max_win_streak, max_loss_streak

    async def calculate_risk_metrics(
        self,
        positions_data: pl.DataFrame,
        market_data: pl.DataFrame,
        portfolio_value: Decimal
    ) -> Dict[str, Any]:
        """
        Calculate portfolio risk metrics.

        Args:
            positions_data: Current positions
            market_data: Historical market data
            portfolio_value: Total portfolio value

        Returns:
            Dictionary with risk metrics
        """
        try:
            if len(positions_data) == 0:
                return self._get_empty_risk_metrics()

            # Calculate VaR
            var_metrics = await self._calculate_var(market_data, portfolio_value)

            # Portfolio volatility
            volatility = await self._calculate_portfolio_volatility(market_data)

            # Exposure metrics
            exposure = self._calculate_exposure(positions_data, portfolio_value)

            # Concentration risk
            concentration = self._calculate_concentration(positions_data)

            metrics = {
                "value_at_risk": var_metrics,
                "volatility": volatility,
                "exposure": exposure,
                "concentration": concentration,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            logger.info("risk_metrics_calculated")

            return metrics

        except Exception as e:
            logger.error("risk_metrics_calculation_failed", error=str(e))
            raise

    async def _calculate_var(
        self,
        market_data: pl.DataFrame,
        portfolio_value: Decimal
    ) -> Dict[str, str]:
        """Calculate Value at Risk."""
        # Simplified VaR calculation
        if "returns" in market_data.columns:
            returns = market_data.select("returns").to_numpy().flatten()
        else:
            returns = np.array([])

        var_metrics = {}

        for confidence in self.var_confidence_levels:
            if len(returns) > 0:
                var_percentile = np.percentile(returns, (1 - confidence) * 100)
                var_amount = abs(Decimal(str(var_percentile))) * portfolio_value
            else:
                var_amount = Decimal("0")

            confidence_key = f"var_{int(confidence * 100)}"
            var_metrics[confidence_key] = str(var_amount)

        return var_metrics

    async def _calculate_portfolio_volatility(
        self,
        market_data: pl.DataFrame
    ) -> Dict[str, str]:
        """Calculate portfolio volatility."""
        volatility_metrics = {
            "daily": "0",
            "weekly": "0",
            "monthly": "0",
            "annualized": "0"
        }

        if "returns" in market_data.columns and len(market_data) > 1:
            returns = market_data.select("returns").to_numpy().flatten()

            daily_vol = Decimal(str(np.std(returns)))
            volatility_metrics["daily"] = str(daily_vol)
            volatility_metrics["weekly"] = str(daily_vol * Decimal("7") ** Decimal("0.5"))
            volatility_metrics["monthly"] = str(daily_vol * Decimal("30") ** Decimal("0.5"))
            volatility_metrics["annualized"] = str(daily_vol * self.annualization_factor ** Decimal("0.5"))

        return volatility_metrics

    def _calculate_exposure(
        self,
        positions_data: pl.DataFrame,
        portfolio_value: Decimal
    ) -> Dict[str, str]:
        """Calculate exposure metrics."""
        if "position_value" not in positions_data.columns:
            return {
                "gross_exposure": "0",
                "net_exposure": "0",
                "leverage": "0"
            }

        position_values = positions_data.select("position_value").to_numpy().flatten()
        gross_exposure = sum(abs(Decimal(str(v))) for v in position_values)
        net_exposure = sum(Decimal(str(v)) for v in position_values)

        leverage = gross_exposure / portfolio_value if portfolio_value > 0 else Decimal("0")

        return {
            "gross_exposure": str(gross_exposure),
            "net_exposure": str(net_exposure),
            "leverage": str(leverage)
        }

    def _calculate_concentration(self, positions_data: pl.DataFrame) -> Dict[str, str]:
        """Calculate concentration metrics."""
        if "position_value" not in positions_data.columns:
            return {"herfindahl_index": "0", "max_position_pct": "0"}

        position_values = positions_data.select("position_value").to_numpy().flatten()
        total_value = sum(abs(Decimal(str(v))) for v in position_values)

        if total_value == 0:
            return {"herfindahl_index": "0", "max_position_pct": "0"}

        # Herfindahl index
        weights = [abs(Decimal(str(v))) / total_value for v in position_values]
        herfindahl = sum(w ** 2 for w in weights)

        # Max position percentage
        max_position_pct = max(weights) * Decimal("100") if weights else Decimal("0")

        return {
            "herfindahl_index": str(herfindahl),
            "max_position_pct": str(max_position_pct)
        }

    def _get_empty_performance_metrics(self) -> Dict[str, Any]:
        """Return empty performance metrics."""
        return {
            "total_pnl": "0",
            "total_return_pct": "0",
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "breakeven_trades": 0,
            "win_rate": "0",
            "avg_win": "0",
            "avg_loss": "0",
            "avg_trade": "0",
            "profit_factor": "0",
            "expectancy": "0",
            "sharpe_ratio": "0",
            "sortino_ratio": "0",
            "max_drawdown": "0",
            "max_drawdown_pct": "0",
            "longest_win_streak": 0,
            "longest_loss_streak": 0,
            "trading_period_days": 0
        }

    def _get_empty_risk_metrics(self) -> Dict[str, Any]:
        """Return empty risk metrics."""
        return {
            "value_at_risk": {f"var_{int(c * 100)}": "0" for c in self.var_confidence_levels},
            "volatility": {
                "daily": "0",
                "weekly": "0",
                "monthly": "0",
                "annualized": "0"
            },
            "exposure": {
                "gross_exposure": "0",
                "net_exposure": "0",
                "leverage": "0"
            },
            "concentration": {
                "herfindahl_index": "0",
                "max_position_pct": "0"
            }
        }
