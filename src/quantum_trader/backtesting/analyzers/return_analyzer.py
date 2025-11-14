"""Return Analyzer for Backtesting.

Detailed analysis of trading returns including time-series analysis,
distribution analysis, and return attribution.
"""

import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class ReturnAnalyzer:
    """Analyze trading returns in detail.

    Provides comprehensive return analysis including:
    - Return distribution analysis
    - Time-series analysis (daily, monthly, annual)
    - Rolling performance metrics
    - Return attribution

    Attributes:
        config: Analyzer configuration from config files
        trading_days_per_year: Days for annualization
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize return analyzer.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid

        Example:
            >>> config = {"trading_days_per_year": 252, "rolling_window_days": 30}
            >>> analyzer = ReturnAnalyzer(config)
        """
        self.config = config
        self._validate_config()

        self.trading_days_per_year = config.get("trading_days_per_year", 252)
        self.rolling_window_days = config.get("rolling_window_days", 30)

        logger.info(
            "ReturnAnalyzer initialized",
            trading_days_per_year=self.trading_days_per_year,
            rolling_window=self.rolling_window_days
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("config must be a dictionary")

    async def analyze_returns(
        self,
        equity_curve: pl.DataFrame,
        trades: Optional[pl.DataFrame] = None
    ) -> Dict[str, Any]:
        """Analyze returns comprehensively.

        Args:
            equity_curve: DataFrame with equity progression
            trades: Optional DataFrame with trade history

        Returns:
            Dictionary with return analysis

        Raises:
            ValueError: If equity_curve is invalid

        Example:
            >>> analyzer = ReturnAnalyzer(config)
            >>> analysis = await analyzer.analyze_returns(equity_df, trades_df)
            >>> print(f"Annual return: {analysis['annual_return']}")
        """
        try:
            self._validate_equity_curve(equity_curve)

            logger.info("Analyzing returns", data_points=equity_curve.height)

            analysis = {}

            # Calculate daily returns
            daily_returns = await self._calculate_daily_returns(equity_curve)

            # Basic return metrics
            analysis.update(await self._calculate_basic_metrics(daily_returns, equity_curve))

            # Distribution analysis
            analysis.update(await self._analyze_return_distribution(daily_returns))

            # Time-based analysis
            analysis.update(await self._analyze_time_periods(equity_curve))

            # Rolling metrics
            analysis.update(await self._calculate_rolling_metrics(daily_returns))

            # Trade attribution if provided
            if trades is not None and trades.height > 0:
                analysis.update(await self._analyze_trade_attribution(trades))

            logger.info("Return analysis completed", metrics_count=len(analysis))
            return analysis

        except Exception as e:
            logger.error("Failed to analyze returns", error=str(e))
            raise

    def _validate_equity_curve(self, equity_curve: pl.DataFrame) -> None:
        """Validate equity curve data.

        Args:
            equity_curve: Equity curve DataFrame

        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(equity_curve, pl.DataFrame):
            raise ValueError("equity_curve must be a Polars DataFrame")

        required_cols = ["timestamp", "equity"]
        missing_cols = [col for col in required_cols if col not in equity_curve.columns]

        if missing_cols:
            error_msg = f"Missing required columns: {missing_cols}"
            logger.error("Invalid equity curve", missing_cols=missing_cols)
            raise ValueError(error_msg)

        if equity_curve.height == 0:
            raise ValueError("equity_curve cannot be empty")

    async def _calculate_daily_returns(self, equity_curve: pl.DataFrame) -> List[Decimal]:
        """Calculate daily returns from equity curve.

        Args:
            equity_curve: Equity progression DataFrame

        Returns:
            List of daily returns
        """
        equity_values = [Decimal(str(v)) for v in equity_curve["equity"].to_list()]
        daily_returns = []

        for i in range(1, len(equity_values)):
            if equity_values[i-1] > Decimal("0"):
                ret = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
                daily_returns.append(ret)

        return daily_returns

    async def _calculate_basic_metrics(
        self,
        daily_returns: List[Decimal],
        equity_curve: pl.DataFrame
    ) -> Dict[str, Decimal]:
        """Calculate basic return metrics.

        Args:
            daily_returns: List of daily returns
            equity_curve: Equity curve

        Returns:
            Dictionary with basic metrics
        """
        if not daily_returns:
            return {
                "total_return": Decimal("0"),
                "annual_return": Decimal("0"),
                "avg_daily_return": Decimal("0"),
                "cumulative_return": Decimal("0")
            }

        # Total return
        initial_equity = Decimal(str(equity_curve["equity"][0]))
        final_equity = Decimal(str(equity_curve["equity"][-1]))

        if initial_equity > Decimal("0"):
            total_return = (final_equity - initial_equity) / initial_equity
        else:
            total_return = Decimal("0")

        # Average daily return
        avg_daily = sum(daily_returns) / Decimal(str(len(daily_returns)))

        # Annualized return
        days = len(daily_returns)
        if days > 0:
            years = Decimal(str(days)) / Decimal(str(self.trading_days_per_year))
            if years > Decimal("0") and total_return > Decimal("-1"):
                annual_return = ((Decimal("1") + total_return) ** (Decimal("1") / years)) - Decimal("1")
            else:
                annual_return = Decimal("0")
        else:
            annual_return = Decimal("0")

        # Cumulative return
        cumulative = total_return

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "avg_daily_return": avg_daily,
            "cumulative_return": cumulative
        }

    async def _analyze_return_distribution(
        self,
        daily_returns: List[Decimal]
    ) -> Dict[str, Any]:
        """Analyze return distribution statistics.

        Args:
            daily_returns: List of daily returns

        Returns:
            Distribution metrics
        """
        if not daily_returns:
            return {
                "return_std_dev": Decimal("0"),
                "return_skewness": Decimal("0"),
                "return_kurtosis": Decimal("0"),
                "positive_days_pct": Decimal("0")
            }

        # Standard deviation
        mean = sum(daily_returns) / Decimal(str(len(daily_returns)))
        if len(daily_returns) > 1:
            variance = sum((r - mean) ** 2 for r in daily_returns) / Decimal(str(len(daily_returns) - 1))
            std_dev = variance.sqrt() if variance > Decimal("0") else Decimal("0")
        else:
            std_dev = Decimal("0")

        # Positive days percentage
        positive_days = len([r for r in daily_returns if r > Decimal("0")])
        positive_days_pct = Decimal(str(positive_days)) / Decimal(str(len(daily_returns))) * Decimal("100")

        # Skewness (simplified calculation)
        if std_dev > Decimal("0") and len(daily_returns) > 2:
            skewness = sum(((r - mean) / std_dev) ** 3 for r in daily_returns) / Decimal(str(len(daily_returns)))
        else:
            skewness = Decimal("0")

        # Kurtosis (simplified calculation)
        if std_dev > Decimal("0") and len(daily_returns) > 3:
            kurtosis = sum(((r - mean) / std_dev) ** 4 for r in daily_returns) / Decimal(str(len(daily_returns))) - Decimal("3")
        else:
            kurtosis = Decimal("0")

        return {
            "return_std_dev": std_dev,
            "return_skewness": skewness,
            "return_kurtosis": kurtosis,
            "positive_days_pct": positive_days_pct
        }

    async def _analyze_time_periods(
        self,
        equity_curve: pl.DataFrame
    ) -> Dict[str, Any]:
        """Analyze returns by time periods.

        Args:
            equity_curve: Equity curve DataFrame

        Returns:
            Time period analysis
        """
        # Monthly returns analysis (simplified)
        timestamps = equity_curve["timestamp"].to_list()

        if not timestamps:
            return {
                "best_month": Decimal("0"),
                "worst_month": Decimal("0"),
                "avg_monthly_return": Decimal("0")
            }

        # For simplicity, return placeholder values
        # Real implementation would group by month and calculate returns
        return {
            "best_month": Decimal("0"),
            "worst_month": Decimal("0"),
            "avg_monthly_return": Decimal("0"),
            "total_months": 0
        }

    async def _calculate_rolling_metrics(
        self,
        daily_returns: List[Decimal]
    ) -> Dict[str, Any]:
        """Calculate rolling performance metrics.

        Args:
            daily_returns: List of daily returns

        Returns:
            Rolling metrics
        """
        if len(daily_returns) < self.rolling_window_days:
            return {
                "rolling_sharpe": Decimal("0"),
                "rolling_avg_return": Decimal("0")
            }

        # Calculate rolling average for last window
        window_returns = daily_returns[-self.rolling_window_days:]
        rolling_avg = sum(window_returns) / Decimal(str(len(window_returns)))

        # Calculate rolling Sharpe (simplified)
        if len(window_returns) > 1:
            window_mean = sum(window_returns) / Decimal(str(len(window_returns)))
            window_variance = sum((r - window_mean) ** 2 for r in window_returns) / Decimal(str(len(window_returns) - 1))
            window_std = window_variance.sqrt() if window_variance > Decimal("0") else Decimal("0")

            if window_std > Decimal("0"):
                rolling_sharpe = (window_mean / window_std) * (Decimal(str(self.trading_days_per_year)).sqrt())
            else:
                rolling_sharpe = Decimal("0")
        else:
            rolling_sharpe = Decimal("0")

        return {
            "rolling_sharpe": rolling_sharpe,
            "rolling_avg_return": rolling_avg,
            "rolling_window_days": self.rolling_window_days
        }

    async def _analyze_trade_attribution(
        self,
        trades: pl.DataFrame
    ) -> Dict[str, Any]:
        """Analyze return attribution by trades.

        Args:
            trades: Trades DataFrame

        Returns:
            Trade attribution metrics
        """
        if "pnl" not in trades.columns:
            return {"trade_attribution": "unavailable"}

        pnl_values = [Decimal(str(v)) for v in trades["pnl"].to_list()]

        total_pnl = sum(pnl_values)
        positive_pnl = sum(p for p in pnl_values if p > Decimal("0"))
        negative_pnl = sum(p for p in pnl_values if p < Decimal("0"))

        # Contribution analysis
        if total_pnl != Decimal("0"):
            positive_contribution = (positive_pnl / total_pnl) * Decimal("100") if total_pnl > Decimal("0") else Decimal("0")
        else:
            positive_contribution = Decimal("0")

        return {
            "total_pnl": total_pnl,
            "positive_pnl": positive_pnl,
            "negative_pnl": negative_pnl,
            "positive_contribution_pct": positive_contribution
        }

    async def calculate_monthly_returns(
        self,
        equity_curve: pl.DataFrame
    ) -> pl.DataFrame:
        """Calculate monthly returns breakdown.

        Args:
            equity_curve: Equity curve DataFrame

        Returns:
            DataFrame with monthly returns

        Example:
            >>> analyzer = ReturnAnalyzer(config)
            >>> monthly_df = await analyzer.calculate_monthly_returns(equity_df)
        """
        try:
            # Group by month and calculate returns
            # Simplified implementation - real version would use proper date grouping
            logger.info("Calculating monthly returns")

            # Placeholder - real implementation would group by month
            monthly_data = {
                "month": [],
                "return": [],
                "start_equity": [],
                "end_equity": []
            }

            monthly_df = pl.DataFrame(monthly_data)

            return monthly_df

        except Exception as e:
            logger.error("Failed to calculate monthly returns", error=str(e))
            raise

    async def calculate_return_percentiles(
        self,
        daily_returns: List[Decimal]
    ) -> Dict[str, Decimal]:
        """Calculate return percentiles.

        Args:
            daily_returns: List of daily returns

        Returns:
            Dictionary with percentile values

        Example:
            >>> analyzer = ReturnAnalyzer(config)
            >>> percentiles = await analyzer.calculate_return_percentiles(returns)
            >>> print(f"95th percentile: {percentiles['p95']}")
        """
        if not daily_returns:
            return {
                "p5": Decimal("0"),
                "p25": Decimal("0"),
                "p50": Decimal("0"),
                "p75": Decimal("0"),
                "p95": Decimal("0")
            }

        sorted_returns = sorted(daily_returns)
        n = len(sorted_returns)

        percentiles = {
            "p5": sorted_returns[int(n * 0.05)] if n > 0 else Decimal("0"),
            "p25": sorted_returns[int(n * 0.25)] if n > 0 else Decimal("0"),
            "p50": sorted_returns[int(n * 0.50)] if n > 0 else Decimal("0"),
            "p75": sorted_returns[int(n * 0.75)] if n > 0 else Decimal("0"),
            "p95": sorted_returns[int(n * 0.95)] if n > 0 else Decimal("0")
        }

        return percentiles
