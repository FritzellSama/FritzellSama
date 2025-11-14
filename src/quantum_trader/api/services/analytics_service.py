"""Analytics service for trading performance and metrics.

This module provides comprehensive analytics capabilities for trading performance,
risk metrics, portfolio analysis, and real-time monitoring.

Example:
    ```python
    from quantum_trader.api.services.analytics_service import AnalyticsService

    analytics = AnalyticsService(config)
    await analytics.initialize()

    # Get performance metrics
    metrics = await analytics.get_performance_metrics(
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )

    # Calculate Sharpe ratio
    sharpe = await analytics.calculate_sharpe_ratio(returns_df)
    ```
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Trading performance metrics.

    Attributes:
        total_pnl: Total profit and loss
        total_return_percent: Total return percentage
        sharpe_ratio: Risk-adjusted return metric
        max_drawdown_percent: Maximum drawdown percentage
        win_rate: Percentage of winning trades
        total_trades: Total number of trades
        avg_trade_pnl: Average PnL per trade
        avg_win: Average winning trade amount
        avg_loss: Average losing trade amount
        profit_factor: Ratio of gross profit to gross loss
        start_date: Start date of analysis period
        end_date: End date of analysis period
    """

    total_pnl: Decimal
    total_return_percent: Decimal
    sharpe_ratio: Decimal
    max_drawdown_percent: Decimal
    win_rate: Decimal
    total_trades: int
    avg_trade_pnl: Decimal
    avg_win: Decimal
    avg_loss: Decimal
    profit_factor: Decimal
    start_date: datetime
    end_date: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskMetrics:
    """Risk analysis metrics.

    Attributes:
        value_at_risk: Value at Risk (VaR) at specified confidence level
        conditional_var: Conditional VaR (Expected Shortfall)
        volatility: Portfolio volatility (annualized)
        beta: Portfolio beta relative to market
        current_drawdown_percent: Current drawdown from peak
        max_leverage: Maximum leverage used
        avg_leverage: Average leverage
        correlation_to_btc: Correlation to Bitcoin
        timestamp: Timestamp of calculation
    """

    value_at_risk: Decimal
    conditional_var: Decimal
    volatility: Decimal
    beta: Decimal
    current_drawdown_percent: Decimal
    max_leverage: Decimal
    avg_leverage: Decimal
    correlation_to_btc: Decimal
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnalyticsService:
    """Service for trading analytics and performance metrics.

    This service provides comprehensive analytics including:
    - Performance metrics (PnL, Sharpe ratio, win rate, etc.)
    - Risk metrics (VaR, drawdown, volatility, etc.)
    - Portfolio analysis
    - Trade statistics
    - Real-time monitoring

    All calculations use Decimal for precision and Polars for performance.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize analytics service.

        Args:
            config: Configuration dictionary with analytics settings

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        # Extract configuration
        self.risk_free_rate = Decimal(str(config.get("risk_free_rate", "0.02")))
        self.var_confidence = Decimal(str(config.get("var_confidence", "0.95")))
        self.trading_days_per_year = int(config.get("trading_days_per_year", 365))
        self.metrics_cache_seconds = int(config.get("metrics_cache_seconds", 60))

        # Cache for performance optimization
        self._metrics_cache: Dict[str, Tuple[datetime, Any]] = {}
        self._lock = asyncio.Lock()

        logger.info(
            "analytics_service_initialized",
            risk_free_rate=str(self.risk_free_rate),
            var_confidence=str(self.var_confidence)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = []  # All parameters have defaults
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        # Validate ranges
        if "var_confidence" in self.config:
            var_conf = Decimal(str(self.config["var_confidence"]))
            if not Decimal("0") < var_conf < Decimal("1"):
                raise ValueError("var_confidence must be between 0 and 1")

    async def initialize(self) -> None:
        """Initialize service resources."""
        logger.info("analytics_service_starting")
        # Initialize any async resources here
        logger.info("analytics_service_started")

    async def shutdown(self) -> None:
        """Cleanup service resources."""
        logger.info("analytics_service_shutting_down")
        async with self._lock:
            self._metrics_cache.clear()
        logger.info("analytics_service_shutdown_complete")

    async def get_performance_metrics(
        self,
        start_date: datetime,
        end_date: datetime,
        strategy: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics.

        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            strategy: Optional strategy filter
            symbol: Optional symbol filter

        Returns:
            PerformanceMetrics object with calculated metrics

        Raises:
            ValueError: If date range is invalid
        """
        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")

        cache_key = f"perf_{start_date.isoformat()}_{end_date.isoformat()}_{strategy}_{symbol}"

        # Check cache
        cached = await self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            # Fetch trade data (this would normally come from database)
            trades_df = await self._fetch_trades(start_date, end_date, strategy, symbol)

            if trades_df.is_empty():
                logger.warning("no_trades_found", start_date=start_date, end_date=end_date)
                return self._empty_performance_metrics(start_date, end_date)

            # Calculate metrics
            total_pnl = self._calculate_total_pnl(trades_df)
            total_return_percent = self._calculate_total_return(trades_df)
            sharpe_ratio = await self.calculate_sharpe_ratio(trades_df)
            max_drawdown = await self.calculate_max_drawdown(trades_df)
            win_rate = self._calculate_win_rate(trades_df)
            total_trades = trades_df.height
            avg_trade_pnl = total_pnl / Decimal(str(total_trades)) if total_trades > 0 else Decimal("0")
            avg_win, avg_loss = self._calculate_avg_win_loss(trades_df)
            profit_factor = self._calculate_profit_factor(trades_df)

            metrics = PerformanceMetrics(
                total_pnl=total_pnl,
                total_return_percent=total_return_percent,
                sharpe_ratio=sharpe_ratio,
                max_drawdown_percent=max_drawdown,
                win_rate=win_rate,
                total_trades=total_trades,
                avg_trade_pnl=avg_trade_pnl,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                start_date=start_date,
                end_date=end_date,
                metadata={"strategy": strategy, "symbol": symbol}
            )

            # Cache result
            await self._add_to_cache(cache_key, metrics)

            logger.info(
                "performance_metrics_calculated",
                total_pnl=str(total_pnl),
                sharpe_ratio=str(sharpe_ratio),
                win_rate=str(win_rate)
            )

            return metrics

        except Exception as e:
            logger.error("performance_metrics_calculation_failed", error=str(e))
            raise

    async def calculate_sharpe_ratio(
        self,
        returns_df: pl.DataFrame,
        risk_free_rate: Optional[Decimal] = None
    ) -> Decimal:
        """Calculate Sharpe ratio from returns.

        Args:
            returns_df: Polars DataFrame with 'returns' column
            risk_free_rate: Optional risk-free rate override

        Returns:
            Sharpe ratio (annualized)

        Raises:
            ValueError: If returns_df is invalid
        """
        if returns_df.is_empty():
            return Decimal("0")

        if "returns" not in returns_df.columns:
            raise ValueError("returns_df must have 'returns' column")

        try:
            rfr = risk_free_rate if risk_free_rate is not None else self.risk_free_rate

            # Calculate mean and std using Polars (convert to Decimal after)
            mean_return = Decimal(str(returns_df["returns"].mean() or 0))
            std_return = Decimal(str(returns_df["returns"].std() or 0))

            if std_return == Decimal("0"):
                return Decimal("0")

            # Annualize
            annual_mean = mean_return * Decimal(str(self.trading_days_per_year))
            annual_std = std_return * Decimal(str(self.trading_days_per_year)).sqrt()

            sharpe = (annual_mean - rfr) / annual_std

            return sharpe

        except Exception as e:
            logger.error("sharpe_calculation_failed", error=str(e))
            raise

    async def calculate_max_drawdown(self, equity_df: pl.DataFrame) -> Decimal:
        """Calculate maximum drawdown from equity curve.

        Args:
            equity_df: Polars DataFrame with 'equity' column

        Returns:
            Maximum drawdown as percentage (positive value)

        Raises:
            ValueError: If equity_df is invalid
        """
        if equity_df.is_empty():
            return Decimal("0")

        if "equity" not in equity_df.columns:
            raise ValueError("equity_df must have 'equity' column")

        try:
            # Calculate running maximum
            equity_series = equity_df["equity"]
            running_max = equity_series.cum_max()

            # Calculate drawdown
            drawdown_df = equity_df.with_columns([
                ((equity_series - running_max) / running_max * pl.lit(100)).alias("drawdown_pct")
            ])

            # Get maximum drawdown (most negative value)
            max_dd = Decimal(str(abs(drawdown_df["drawdown_pct"].min() or 0)))

            return max_dd

        except Exception as e:
            logger.error("max_drawdown_calculation_failed", error=str(e))
            raise

    async def get_risk_metrics(
        self,
        portfolio_df: pl.DataFrame,
        market_df: Optional[pl.DataFrame] = None
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics.

        Args:
            portfolio_df: Portfolio returns data
            market_df: Optional market benchmark returns

        Returns:
            RiskMetrics object

        Raises:
            ValueError: If data is invalid
        """
        if portfolio_df.is_empty():
            raise ValueError("portfolio_df cannot be empty")

        try:
            # Calculate VaR and CVaR
            var = await self.calculate_value_at_risk(portfolio_df, self.var_confidence)
            cvar = await self.calculate_conditional_var(portfolio_df, self.var_confidence)

            # Calculate volatility
            volatility = self._calculate_volatility(portfolio_df)

            # Calculate beta (if market data provided)
            beta = Decimal("1.0")
            if market_df is not None and not market_df.is_empty():
                beta = await self._calculate_beta(portfolio_df, market_df)

            # Calculate drawdown
            current_dd = await self.calculate_max_drawdown(portfolio_df)

            # Leverage metrics (would come from position data)
            max_leverage = Decimal("1.0")
            avg_leverage = Decimal("1.0")

            # Correlation to BTC (would calculate from actual BTC data)
            correlation_to_btc = Decimal("0.5")

            metrics = RiskMetrics(
                value_at_risk=var,
                conditional_var=cvar,
                volatility=volatility,
                beta=beta,
                current_drawdown_percent=current_dd,
                max_leverage=max_leverage,
                avg_leverage=avg_leverage,
                correlation_to_btc=correlation_to_btc,
                timestamp=datetime.utcnow(),
                metadata={}
            )

            logger.info(
                "risk_metrics_calculated",
                var=str(var),
                volatility=str(volatility),
                max_drawdown=str(current_dd)
            )

            return metrics

        except Exception as e:
            logger.error("risk_metrics_calculation_failed", error=str(e))
            raise

    async def calculate_value_at_risk(
        self,
        returns_df: pl.DataFrame,
        confidence: Decimal
    ) -> Decimal:
        """Calculate Value at Risk (VaR).

        Args:
            returns_df: DataFrame with returns
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            VaR value (positive represents potential loss)
        """
        if returns_df.is_empty():
            return Decimal("0")

        try:
            # Calculate VaR using historical simulation
            quantile = Decimal("1") - confidence
            var_value = Decimal(str(abs(
                returns_df["returns"].quantile(float(quantile)) or 0
            )))

            return var_value

        except Exception as e:
            logger.error("var_calculation_failed", error=str(e))
            raise

    async def calculate_conditional_var(
        self,
        returns_df: pl.DataFrame,
        confidence: Decimal
    ) -> Decimal:
        """Calculate Conditional VaR (Expected Shortfall).

        Args:
            returns_df: DataFrame with returns
            confidence: Confidence level

        Returns:
            CVaR value (positive represents expected loss beyond VaR)
        """
        if returns_df.is_empty():
            return Decimal("0")

        try:
            # Calculate VaR threshold
            var_threshold = await self.calculate_value_at_risk(returns_df, confidence)

            # Calculate mean of returns below VaR threshold
            tail_returns = returns_df.filter(
                pl.col("returns") <= -float(var_threshold)
            )

            if tail_returns.is_empty():
                return var_threshold

            cvar = Decimal(str(abs(tail_returns["returns"].mean() or 0)))

            return cvar

        except Exception as e:
            logger.error("cvar_calculation_failed", error=str(e))
            raise

    def _calculate_volatility(self, returns_df: pl.DataFrame) -> Decimal:
        """Calculate annualized volatility."""
        if returns_df.is_empty():
            return Decimal("0")

        std_dev = Decimal(str(returns_df["returns"].std() or 0))
        annual_vol = std_dev * Decimal(str(self.trading_days_per_year)).sqrt()

        return annual_vol

    async def _calculate_beta(
        self,
        portfolio_df: pl.DataFrame,
        market_df: pl.DataFrame
    ) -> Decimal:
        """Calculate portfolio beta relative to market."""
        try:
            # Join dataframes on timestamp
            joined = portfolio_df.join(market_df, on="timestamp", how="inner")

            if joined.is_empty():
                return Decimal("1.0")

            # Calculate covariance and variance
            portfolio_returns = joined["returns"]
            market_returns = joined["returns_right"]

            covariance = Decimal(str(
                ((portfolio_returns - portfolio_returns.mean()) *
                 (market_returns - market_returns.mean())).mean() or 0
            ))

            market_variance = Decimal(str(market_returns.var() or 1))

            if market_variance == Decimal("0"):
                return Decimal("1.0")

            beta = covariance / market_variance

            return beta

        except Exception as e:
            logger.error("beta_calculation_failed", error=str(e))
            return Decimal("1.0")

    def _calculate_total_pnl(self, trades_df: pl.DataFrame) -> Decimal:
        """Calculate total PnL from trades."""
        if trades_df.is_empty() or "pnl" not in trades_df.columns:
            return Decimal("0")

        total = Decimal(str(trades_df["pnl"].sum() or 0))
        return total

    def _calculate_total_return(self, trades_df: pl.DataFrame) -> Decimal:
        """Calculate total return percentage."""
        if trades_df.is_empty():
            return Decimal("0")

        # This would normally be calculated from initial capital
        # For now, use a simplified calculation
        initial_capital = Decimal("100000")  # Would come from config
        total_pnl = self._calculate_total_pnl(trades_df)

        return (total_pnl / initial_capital) * Decimal("100")

    def _calculate_win_rate(self, trades_df: pl.DataFrame) -> Decimal:
        """Calculate win rate (percentage of winning trades)."""
        if trades_df.is_empty() or "pnl" not in trades_df.columns:
            return Decimal("0")

        total_trades = trades_df.height
        winning_trades = trades_df.filter(pl.col("pnl") > 0).height

        if total_trades == 0:
            return Decimal("0")

        win_rate = (Decimal(str(winning_trades)) / Decimal(str(total_trades))) * Decimal("100")
        return win_rate

    def _calculate_avg_win_loss(self, trades_df: pl.DataFrame) -> Tuple[Decimal, Decimal]:
        """Calculate average win and average loss."""
        if trades_df.is_empty() or "pnl" not in trades_df.columns:
            return Decimal("0"), Decimal("0")

        winning_trades = trades_df.filter(pl.col("pnl") > 0)
        losing_trades = trades_df.filter(pl.col("pnl") < 0)

        avg_win = Decimal(str(winning_trades["pnl"].mean() or 0)) if not winning_trades.is_empty() else Decimal("0")
        avg_loss = Decimal(str(abs(losing_trades["pnl"].mean() or 0))) if not losing_trades.is_empty() else Decimal("0")

        return avg_win, avg_loss

    def _calculate_profit_factor(self, trades_df: pl.DataFrame) -> Decimal:
        """Calculate profit factor (gross profit / gross loss)."""
        if trades_df.is_empty() or "pnl" not in trades_df.columns:
            return Decimal("0")

        gross_profit = Decimal(str(trades_df.filter(pl.col("pnl") > 0)["pnl"].sum() or 0))
        gross_loss = Decimal(str(abs(trades_df.filter(pl.col("pnl") < 0)["pnl"].sum() or 1)))

        if gross_loss == Decimal("0"):
            return Decimal("0") if gross_profit == Decimal("0") else Decimal("999.99")

        profit_factor = gross_profit / gross_loss
        return profit_factor

    async def _fetch_trades(
        self,
        start_date: datetime,
        end_date: datetime,
        strategy: Optional[str],
        symbol: Optional[str]
    ) -> pl.DataFrame:
        """Fetch trade data from database.

        This is a placeholder that would normally query the database.
        Returns sample data for demonstration.
        """
        # In production, this would query TimescaleDB
        # For now, return empty DataFrame with correct schema
        schema = {
            "trade_id": pl.Utf8,
            "timestamp": pl.Datetime,
            "symbol": pl.Utf8,
            "strategy": pl.Utf8,
            "pnl": pl.Float64,
            "returns": pl.Float64,
            "equity": pl.Float64,
        }

        return pl.DataFrame(schema=schema)

    def _empty_performance_metrics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> PerformanceMetrics:
        """Return empty performance metrics."""
        return PerformanceMetrics(
            total_pnl=Decimal("0"),
            total_return_percent=Decimal("0"),
            sharpe_ratio=Decimal("0"),
            max_drawdown_percent=Decimal("0"),
            win_rate=Decimal("0"),
            total_trades=0,
            avg_trade_pnl=Decimal("0"),
            avg_win=Decimal("0"),
            avg_loss=Decimal("0"),
            profit_factor=Decimal("0"),
            start_date=start_date,
            end_date=end_date,
            metadata={}
        )

    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        async with self._lock:
            if key in self._metrics_cache:
                cached_time, cached_value = self._metrics_cache[key]
                if (datetime.utcnow() - cached_time).total_seconds() < self.metrics_cache_seconds:
                    return cached_value
                else:
                    del self._metrics_cache[key]
        return None

    async def _add_to_cache(self, key: str, value: Any) -> None:
        """Add value to cache with timestamp."""
        async with self._lock:
            self._metrics_cache[key] = (datetime.utcnow(), value)
