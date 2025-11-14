"""Risk Analyzer for Backtesting.

Comprehensive risk analysis including VaR, CVaR, drawdown analysis,
and risk-adjusted performance metrics.
"""

import asyncio
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class RiskAnalyzer:
    """Analyze trading risk metrics.

    Provides comprehensive risk analysis including:
    - Value at Risk (VaR)
    - Conditional Value at Risk (CVaR)
    - Drawdown analysis
    - Risk-adjusted metrics
    - Volatility analysis

    Attributes:
        config: Analyzer configuration from config files
        confidence_level: Confidence level for VaR calculation
        risk_free_rate: Risk-free rate for calculations
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize risk analyzer.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid

        Example:
            >>> config = {
            ...     "var_confidence_level": 0.95,
            ...     "risk_free_rate": 0.02,
            ...     "trading_days_per_year": 252
            ... }
            >>> analyzer = RiskAnalyzer(config)
        """
        self.config = config
        self._validate_config()

        self.confidence_level = Decimal(str(config.get("var_confidence_level", 0.95)))
        self.risk_free_rate = Decimal(str(config.get("risk_free_rate", 0.02)))
        self.trading_days_per_year = config.get("trading_days_per_year", 252)

        logger.info(
            "RiskAnalyzer initialized",
            confidence_level=float(self.confidence_level),
            risk_free_rate=float(self.risk_free_rate)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("config must be a dictionary")

    async def analyze_risk(
        self,
        equity_curve: pl.DataFrame,
        trades: Optional[pl.DataFrame] = None
    ) -> Dict[str, Any]:
        """Analyze risk comprehensively.

        Args:
            equity_curve: DataFrame with equity progression
            trades: Optional DataFrame with trade history

        Returns:
            Dictionary with risk analysis

        Raises:
            ValueError: If equity_curve is invalid

        Example:
            >>> analyzer = RiskAnalyzer(config)
            >>> analysis = await analyzer.analyze_risk(equity_df, trades_df)
            >>> print(f"VaR 95%: {analysis['var_95']}")
        """
        try:
            self._validate_equity_curve(equity_curve)

            logger.info("Analyzing risk", data_points=equity_curve.height)

            analysis = {}

            # Calculate returns for risk analysis
            daily_returns = await self._calculate_returns(equity_curve)

            # VaR analysis
            analysis.update(await self._calculate_var(daily_returns))

            # Drawdown analysis
            analysis.update(await self._calculate_drawdown_metrics(equity_curve))

            # Volatility metrics
            analysis.update(await self._calculate_volatility_metrics(daily_returns))

            # Risk-adjusted metrics
            analysis.update(await self._calculate_risk_adjusted_metrics(daily_returns))

            # Trade-based risk if provided
            if trades is not None and trades.height > 0:
                analysis.update(await self._analyze_trade_risk(trades))

            logger.info("Risk analysis completed", metrics_count=len(analysis))
            return analysis

        except Exception as e:
            logger.error("Failed to analyze risk", error=str(e))
            raise

    def _validate_equity_curve(self, equity_curve: pl.DataFrame) -> None:
        """Validate equity curve data."""
        if not isinstance(equity_curve, pl.DataFrame):
            raise ValueError("equity_curve must be a Polars DataFrame")

        if "equity" not in equity_curve.columns:
            raise ValueError("equity_curve must have 'equity' column")

        if equity_curve.height == 0:
            raise ValueError("equity_curve cannot be empty")

    async def _calculate_returns(self, equity_curve: pl.DataFrame) -> List[Decimal]:
        """Calculate daily returns."""
        equity_values = [Decimal(str(v)) for v in equity_curve["equity"].to_list()]
        returns = []

        for i in range(1, len(equity_values)):
            if equity_values[i-1] > Decimal("0"):
                ret = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
                returns.append(ret)

        return returns

    async def _calculate_var(self, returns: List[Decimal]) -> Dict[str, Decimal]:
        """Calculate Value at Risk (VaR).

        Args:
            returns: List of returns

        Returns:
            VaR metrics at different confidence levels
        """
        if not returns:
            return {
                "var_95": Decimal("0"),
                "var_99": Decimal("0"),
                "cvar_95": Decimal("0"),
                "cvar_99": Decimal("0")
            }

        sorted_returns = sorted(returns)
        n = len(sorted_returns)

        # VaR at 95% confidence
        var_95_idx = int(n * (1 - 0.95))
        var_95 = abs(sorted_returns[var_95_idx]) if var_95_idx < n else Decimal("0")

        # VaR at 99% confidence
        var_99_idx = int(n * (1 - 0.99))
        var_99 = abs(sorted_returns[var_99_idx]) if var_99_idx < n else Decimal("0")

        # CVaR (Expected Shortfall) at 95%
        tail_95 = sorted_returns[:var_95_idx+1] if var_95_idx >= 0 else []
        if tail_95:
            cvar_95 = abs(sum(tail_95) / Decimal(str(len(tail_95))))
        else:
            cvar_95 = Decimal("0")

        # CVaR at 99%
        tail_99 = sorted_returns[:var_99_idx+1] if var_99_idx >= 0 else []
        if tail_99:
            cvar_99 = abs(sum(tail_99) / Decimal(str(len(tail_99))))
        else:
            cvar_99 = Decimal("0")

        return {
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "cvar_99": cvar_99
        }

    async def _calculate_drawdown_metrics(
        self,
        equity_curve: pl.DataFrame
    ) -> Dict[str, Any]:
        """Calculate comprehensive drawdown metrics.

        Args:
            equity_curve: Equity curve DataFrame

        Returns:
            Drawdown analysis metrics
        """
        equity_values = [Decimal(str(v)) for v in equity_curve["equity"].to_list()]

        if not equity_values:
            return {
                "max_drawdown": Decimal("0"),
                "max_drawdown_duration": 0,
                "avg_drawdown": Decimal("0"),
                "drawdown_count": 0
            }

        peak = equity_values[0]
        max_dd = Decimal("0")
        current_dd_duration = 0
        max_dd_duration = 0

        drawdowns = []
        in_drawdown = False
        dd_start_value = Decimal("0")

        for equity in equity_values:
            if equity > peak:
                # New peak
                if in_drawdown:
                    # End of drawdown
                    in_drawdown = False

                peak = equity
                current_dd_duration = 0
            else:
                # In drawdown
                if not in_drawdown:
                    in_drawdown = True
                    dd_start_value = peak

                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)

                if peak > Decimal("0"):
                    dd = (peak - equity) / peak
                    max_dd = max(max_dd, dd)

                    if equity < dd_start_value:
                        drawdowns.append(dd)

        # Calculate average drawdown
        if drawdowns:
            avg_dd = sum(drawdowns) / Decimal(str(len(drawdowns)))
        else:
            avg_dd = Decimal("0")

        return {
            "max_drawdown": max_dd,
            "max_drawdown_duration": max_dd_duration,
            "avg_drawdown": avg_dd,
            "drawdown_count": len(drawdowns),
            "current_drawdown": (peak - equity_values[-1]) / peak if peak > Decimal("0") else Decimal("0")
        }

    async def _calculate_volatility_metrics(
        self,
        returns: List[Decimal]
    ) -> Dict[str, Decimal]:
        """Calculate volatility metrics.

        Args:
            returns: List of returns

        Returns:
            Volatility metrics
        """
        if not returns or len(returns) < 2:
            return {
                "daily_volatility": Decimal("0"),
                "annual_volatility": Decimal("0"),
                "downside_volatility": Decimal("0")
            }

        # Daily volatility (standard deviation)
        mean = sum(returns) / Decimal(str(len(returns)))
        variance = sum((r - mean) ** 2 for r in returns) / Decimal(str(len(returns) - 1))
        daily_vol = variance.sqrt() if variance > Decimal("0") else Decimal("0")

        # Annualized volatility
        annual_vol = daily_vol * Decimal(str(self.trading_days_per_year)).sqrt()

        # Downside volatility (semi-deviation)
        downside_returns = [r for r in returns if r < Decimal("0")]
        if downside_returns:
            downside_variance = sum(r ** 2 for r in downside_returns) / Decimal(str(len(downside_returns)))
            downside_vol = downside_variance.sqrt() if downside_variance > Decimal("0") else Decimal("0")
        else:
            downside_vol = Decimal("0")

        return {
            "daily_volatility": daily_vol,
            "annual_volatility": annual_vol,
            "downside_volatility": downside_vol
        }

    async def _calculate_risk_adjusted_metrics(
        self,
        returns: List[Decimal]
    ) -> Dict[str, Decimal]:
        """Calculate risk-adjusted performance metrics.

        Args:
            returns: List of returns

        Returns:
            Risk-adjusted metrics
        """
        if not returns or len(returns) < 2:
            return {
                "sharpe_ratio": Decimal("0"),
                "sortino_ratio": Decimal("0"),
                "information_ratio": Decimal("0")
            }

        # Calculate mean and std dev
        mean_return = sum(returns) / Decimal(str(len(returns)))
        variance = sum((r - mean_return) ** 2 for r in returns) / Decimal(str(len(returns) - 1))
        std_dev = variance.sqrt() if variance > Decimal("0") else Decimal("0")

        # Daily risk-free rate
        daily_rf = self.risk_free_rate / Decimal(str(self.trading_days_per_year))

        # Sharpe Ratio
        if std_dev > Decimal("0"):
            sharpe = ((mean_return - daily_rf) / std_dev) * Decimal(str(self.trading_days_per_year)).sqrt()
        else:
            sharpe = Decimal("0")

        # Sortino Ratio
        downside_returns = [r for r in returns if r < Decimal("0")]
        if downside_returns:
            downside_var = sum(r ** 2 for r in downside_returns) / Decimal(str(len(downside_returns)))
            downside_dev = downside_var.sqrt() if downside_var > Decimal("0") else Decimal("0")

            if downside_dev > Decimal("0"):
                sortino = ((mean_return - daily_rf) / downside_dev) * Decimal(str(self.trading_days_per_year)).sqrt()
            else:
                sortino = Decimal("0")
        else:
            sortino = sharpe

        # Information Ratio (simplified - assumes benchmark return is risk-free rate)
        if std_dev > Decimal("0"):
            info_ratio = (mean_return - daily_rf) / std_dev
        else:
            info_ratio = Decimal("0")

        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "information_ratio": info_ratio
        }

    async def _analyze_trade_risk(self, trades: pl.DataFrame) -> Dict[str, Any]:
        """Analyze risk from trade perspective.

        Args:
            trades: Trades DataFrame

        Returns:
            Trade risk metrics
        """
        if "pnl" not in trades.columns:
            return {"trade_risk": "unavailable"}

        pnl_values = [Decimal(str(v)) for v in trades["pnl"].to_list()]

        # Worst trade
        worst_trade = min(pnl_values) if pnl_values else Decimal("0")

        # Average loss
        losses = [p for p in pnl_values if p < Decimal("0")]
        avg_loss = sum(losses) / Decimal(str(len(losses))) if losses else Decimal("0")

        # Consecutive losses
        max_consecutive_losses = 0
        current_consecutive = 0

        for pnl in pnl_values:
            if pnl < Decimal("0"):
                current_consecutive += 1
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive)
            else:
                current_consecutive = 0

        return {
            "worst_trade": worst_trade,
            "avg_loss": avg_loss,
            "max_consecutive_losses": max_consecutive_losses,
            "loss_std_dev": self._calculate_loss_std_dev(losses)
        }

    def _calculate_loss_std_dev(self, losses: List[Decimal]) -> Decimal:
        """Calculate standard deviation of losses.

        Args:
            losses: List of loss values

        Returns:
            Standard deviation
        """
        if not losses or len(losses) < 2:
            return Decimal("0")

        mean_loss = sum(losses) / Decimal(str(len(losses)))
        variance = sum((l - mean_loss) ** 2 for l in losses) / Decimal(str(len(losses) - 1))

        return variance.sqrt() if variance > Decimal("0") else Decimal("0")
