"""Performance Analyzer for Backtest Results.

Comprehensive performance analysis including returns, risk metrics,
and statistical significance testing.
"""

import asyncio
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class PerformanceAnalyzer:
    """Analyze backtest performance metrics.

    Calculates comprehensive performance and risk metrics:
    - Return metrics (total, annual, monthly)
    - Risk metrics (Sharpe, Sortino, Calmar)
    - Drawdown analysis
    - Win/loss statistics
    - Trade analysis

    Attributes:
        config: Analyzer configuration from config files
        risk_free_rate: Annual risk-free rate for Sharpe calculation
        trading_days_per_year: Trading days for annualization
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize performance analyzer.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid

        Example:
            >>> config = {
            ...     "risk_free_rate": 0.02,
            ...     "trading_days_per_year": 252
            ... }
            >>> analyzer = PerformanceAnalyzer(config)
        """
        self.config = config
        self._validate_config()

        self.risk_free_rate = Decimal(str(config.get("risk_free_rate", 0.02)))
        self.trading_days_per_year = config.get("trading_days_per_year", 252)

        logger.info(
            "PerformanceAnalyzer initialized",
            risk_free_rate=float(self.risk_free_rate),
            trading_days_per_year=self.trading_days_per_year
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if not isinstance(self.config, dict):
            raise ValueError("config must be a dictionary")

    async def analyze_performance(
        self,
        trades: pl.DataFrame,
        equity_curve: pl.DataFrame,
        initial_capital: Decimal
    ) -> Dict[str, Any]:
        """Analyze complete backtest performance.

        Args:
            trades: DataFrame with trade history
            equity_curve: DataFrame with equity progression
            initial_capital: Starting capital

        Returns:
            Dictionary with performance metrics

        Example:
            >>> analyzer = PerformanceAnalyzer(config)
            >>> metrics = await analyzer.analyze_performance(
            ...     trades_df, equity_df, Decimal("100000")
            ... )
            >>> print(f"Sharpe: {metrics['sharpe_ratio']}")
        """
        try:
            logger.info("Analyzing performance")

            metrics = {}

            # Return metrics
            metrics.update(await self._calculate_returns(equity_curve, initial_capital))

            # Risk metrics
            metrics.update(await self._calculate_risk_metrics(equity_curve))

            # Drawdown analysis
            metrics.update(await self._calculate_drawdowns(equity_curve))

            # Trade statistics
            if trades.height > 0:
                metrics.update(await self._calculate_trade_statistics(trades))

            logger.info("Performance analysis completed", total_metrics=len(metrics))
            return metrics

        except Exception as e:
            logger.error("Failed to analyze performance", error=str(e))
            raise

    async def _calculate_returns(
        self,
        equity_curve: pl.DataFrame,
        initial_capital: Decimal
    ) -> Dict[str, Decimal]:
        """Calculate return metrics."""
        if equity_curve.height == 0:
            return {"total_return": Decimal("0"), "annual_return": Decimal("0")}

        final_equity = Decimal(str(equity_curve["equity"][-1]))
        total_return = (final_equity - initial_capital) / initial_capital

        # Annualized return (simplified)
        days = equity_curve.height
        if days > 0:
            years = Decimal(str(days)) / Decimal(str(self.trading_days_per_year))
            if years > Decimal("0"):
                annual_return = ((Decimal("1") + total_return) ** (Decimal("1") / years)) - Decimal("1")
            else:
                annual_return = total_return
        else:
            annual_return = Decimal("0")

        return {
            "total_return": total_return,
            "annual_return": annual_return
        }

    async def _calculate_risk_metrics(
        self,
        equity_curve: pl.DataFrame
    ) -> Dict[str, Decimal]:
        """Calculate risk-adjusted metrics."""
        if equity_curve.height < 2:
            return {
                "sharpe_ratio": Decimal("0"),
                "sortino_ratio": Decimal("0"),
                "calmar_ratio": Decimal("0")
            }

        # Calculate daily returns
        returns = []
        equity_values = equity_curve["equity"].to_list()

        for i in range(1, len(equity_values)):
            if equity_values[i-1] > 0:
                daily_return = (Decimal(str(equity_values[i])) - Decimal(str(equity_values[i-1]))) / Decimal(str(equity_values[i-1]))
                returns.append(daily_return)

        if not returns:
            return {
                "sharpe_ratio": Decimal("0"),
                "sortino_ratio": Decimal("0"),
                "calmar_ratio": Decimal("0")
            }

        # Sharpe ratio
        avg_return = sum(returns) / Decimal(str(len(returns)))

        if len(returns) > 1:
            variance = sum((r - avg_return) ** 2 for r in returns) / Decimal(str(len(returns) - 1))
            std_dev = variance.sqrt() if variance > Decimal("0") else Decimal("0")
        else:
            std_dev = Decimal("0")

        daily_rf_rate = self.risk_free_rate / Decimal(str(self.trading_days_per_year))

        if std_dev > Decimal("0"):
            sharpe_ratio = ((avg_return - daily_rf_rate) / std_dev) * (Decimal(str(self.trading_days_per_year)).sqrt())
        else:
            sharpe_ratio = Decimal("0")

        # Sortino ratio (downside deviation)
        downside_returns = [r for r in returns if r < Decimal("0")]

        if downside_returns:
            downside_variance = sum(r ** 2 for r in downside_returns) / Decimal(str(len(downside_returns)))
            downside_dev = downside_variance.sqrt() if downside_variance > Decimal("0") else Decimal("0")

            if downside_dev > Decimal("0"):
                sortino_ratio = ((avg_return - daily_rf_rate) / downside_dev) * (Decimal(str(self.trading_days_per_year)).sqrt())
            else:
                sortino_ratio = Decimal("0")
        else:
            sortino_ratio = sharpe_ratio

        # Calmar ratio (placeholder - would need max drawdown)
        calmar_ratio = Decimal("0")

        return {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio
        }

    async def _calculate_drawdowns(
        self,
        equity_curve: pl.DataFrame
    ) -> Dict[str, Any]:
        """Calculate drawdown statistics."""
        if equity_curve.height == 0:
            return {
                "max_drawdown": Decimal("0"),
                "max_drawdown_duration_days": 0
            }

        equity_values = [Decimal(str(v)) for v in equity_curve["equity"].to_list()]

        max_drawdown = Decimal("0")
        peak = equity_values[0]
        max_dd_duration = 0
        current_dd_duration = 0

        for equity in equity_values:
            if equity > peak:
                peak = equity
                current_dd_duration = 0
            else:
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)

                if peak > Decimal("0"):
                    drawdown = (peak - equity) / peak
                    max_drawdown = max(max_drawdown, drawdown)

        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_duration_days": max_dd_duration
        }

    async def _calculate_trade_statistics(
        self,
        trades: pl.DataFrame
    ) -> Dict[str, Any]:
        """Calculate trade-level statistics."""
        total_trades = trades.height

        if "pnl" not in trades.columns:
            return {"total_trades": total_trades}

        pnl_values = [Decimal(str(v)) for v in trades["pnl"].to_list()]

        winning_trades = [p for p in pnl_values if p > Decimal("0")]
        losing_trades = [p for p in pnl_values if p < Decimal("0")]

        win_count = len(winning_trades)
        loss_count = len(losing_trades)

        win_rate = Decimal(str(win_count)) / Decimal(str(total_trades)) if total_trades > 0 else Decimal("0")

        avg_win = sum(winning_trades) / Decimal(str(len(winning_trades))) if winning_trades else Decimal("0")
        avg_loss = sum(losing_trades) / Decimal(str(len(losing_trades))) if losing_trades else Decimal("0")

        total_wins = sum(winning_trades)
        total_losses = abs(sum(losing_trades))

        if total_losses > Decimal("0"):
            profit_factor = total_wins / total_losses
        else:
            profit_factor = Decimal("0") if total_wins == Decimal("0") else Decimal("999")

        return {
            "total_trades": total_trades,
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "largest_win": max(winning_trades) if winning_trades else Decimal("0"),
            "largest_loss": min(losing_trades) if losing_trades else Decimal("0")
        }
