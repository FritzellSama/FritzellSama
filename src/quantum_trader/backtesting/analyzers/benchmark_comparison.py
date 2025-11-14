"""
Benchmark Comparison Analyzer - Compare strategy performance against benchmarks.

This module provides tools to compare trading strategy performance against
market benchmarks (e.g., buy-and-hold, market indices) with comprehensive metrics.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import os

from structlog import get_logger
import polars as pl

logger = get_logger(__name__)


class BenchmarkComparisonAnalyzer:
    """Analyzes strategy performance relative to benchmark strategies.

    Attributes:
        config: Analyzer configuration from environment
        strategy_results: Strategy performance data
        benchmark_results: Benchmark performance data
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize benchmark comparison analyzer.

        Args:
            config: Optional configuration override

        Raises:
            ValueError: If configuration invalid
        """
        self.config: Dict[str, Any] = config or self._load_config()
        self.strategy_results: Optional[pl.DataFrame] = None
        self.benchmark_results: Optional[pl.DataFrame] = None

        logger.info("BenchmarkComparisonAnalyzer initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables.

        Returns:
            Configuration dictionary
        """
        try:
            config = {
                'risk_free_rate': Decimal(os.getenv('RISK_FREE_RATE', '0.02')),
                'trading_days_per_year': int(os.getenv('TRADING_DAYS_PER_YEAR', '365')),
                'benchmark_symbol': os.getenv('BENCHMARK_SYMBOL', 'BTC/USDT'),
                'comparison_metrics': os.getenv(
                    'COMPARISON_METRICS',
                    'sharpe,sortino,max_drawdown,total_return,volatility'
                ).split(','),
            }

            logger.debug("Benchmark analyzer config loaded", config=config)
            return config

        except Exception as e:
            logger.error("Failed to load config", error=str(e))
            raise ValueError(f"Configuration load failed: {e}")

    async def compare_with_buy_hold(
        self,
        strategy_data: pl.DataFrame,
        market_data: pl.DataFrame,
        initial_capital: Decimal
    ) -> Dict[str, Any]:
        """Compare strategy performance with buy-and-hold benchmark.

        Args:
            strategy_data: Strategy performance timeseries
            market_data: Market price timeseries
            initial_capital: Starting capital

        Returns:
            Comparison metrics dictionary

        Example:
            >>> result = await analyzer.compare_with_buy_hold(
            ...     strategy_df, market_df, Decimal('100000')
            ... )
            >>> result['alpha']
            Decimal('0.05')
        """
        try:
            logger.info("Comparing with buy-and-hold", initial_capital=str(initial_capital))

            # Calculate buy-and-hold returns
            buy_hold_returns = await self._calculate_buy_hold_returns(
                market_data,
                initial_capital
            )

            # Calculate strategy returns
            strategy_returns = await self._calculate_strategy_returns(
                strategy_data
            )

            # Calculate comparative metrics
            comparison = await self._calculate_comparison_metrics(
                strategy_returns,
                buy_hold_returns
            )

            logger.info(
                "Buy-and-hold comparison complete",
                alpha=str(comparison.get('alpha', 'N/A')),
                beta=str(comparison.get('beta', 'N/A'))
            )

            return comparison

        except Exception as e:
            logger.error("Buy-and-hold comparison failed", error=str(e))
            raise

    async def compare_with_custom_benchmark(
        self,
        strategy_data: pl.DataFrame,
        benchmark_data: pl.DataFrame
    ) -> Dict[str, Any]:
        """Compare strategy with custom benchmark strategy.

        Args:
            strategy_data: Strategy performance data
            benchmark_data: Benchmark performance data

        Returns:
            Comparison metrics

        Raises:
            ValueError: If data validation fails
        """
        try:
            logger.info("Comparing with custom benchmark")

            # Validate data alignment
            await self._validate_data_alignment(strategy_data, benchmark_data)

            # Calculate returns
            strategy_returns = await self._calculate_strategy_returns(strategy_data)
            benchmark_returns = await self._calculate_strategy_returns(benchmark_data)

            # Calculate metrics
            comparison = await self._calculate_comparison_metrics(
                strategy_returns,
                benchmark_returns
            )

            logger.info("Custom benchmark comparison complete")

            return comparison

        except Exception as e:
            logger.error("Custom benchmark comparison failed", error=str(e))
            raise

    async def calculate_alpha_beta(
        self,
        strategy_returns: pl.DataFrame,
        benchmark_returns: pl.DataFrame
    ) -> Tuple[Decimal, Decimal]:
        """Calculate alpha and beta coefficients.

        Args:
            strategy_returns: Strategy return timeseries
            benchmark_returns: Benchmark return timeseries

        Returns:
            Tuple of (alpha, beta)

        Example:
            >>> alpha, beta = await analyzer.calculate_alpha_beta(strat_df, bench_df)
            >>> print(f"Alpha: {alpha}, Beta: {beta}")
            Alpha: 0.03, Beta: 1.2
        """
        try:
            logger.debug("Calculating alpha and beta")

            # Align data
            aligned = strategy_returns.join(
                benchmark_returns,
                on='timestamp',
                how='inner'
            )

            if aligned.height == 0:
                raise ValueError("No overlapping data for alpha/beta calculation")

            # Extract return columns
            strat_returns = aligned.select('strategy_return').to_series()
            bench_returns = aligned.select('benchmark_return').to_series()

            # Convert to Decimal for calculation
            strat_vals = [Decimal(str(x)) for x in strat_returns]
            bench_vals = [Decimal(str(x)) for x in bench_returns]

            # Calculate covariance and variance
            n = len(strat_vals)
            if n < 2:
                raise ValueError("Insufficient data points for calculation")

            strat_mean = sum(strat_vals) / Decimal(n)
            bench_mean = sum(bench_vals) / Decimal(n)

            covariance = sum(
                (s - strat_mean) * (b - bench_mean)
                for s, b in zip(strat_vals, bench_vals)
            ) / Decimal(n - 1)

            variance = sum(
                (b - bench_mean) ** 2
                for b in bench_vals
            ) / Decimal(n - 1)

            # Calculate beta
            if variance == Decimal('0'):
                beta = Decimal('0')
            else:
                beta = covariance / variance

            # Calculate alpha
            alpha = strat_mean - (beta * bench_mean)

            logger.debug("Alpha/Beta calculated", alpha=str(alpha), beta=str(beta))

            return alpha, beta

        except Exception as e:
            logger.error("Alpha/Beta calculation failed", error=str(e))
            raise

    async def calculate_information_ratio(
        self,
        strategy_returns: pl.DataFrame,
        benchmark_returns: pl.DataFrame
    ) -> Decimal:
        """Calculate information ratio (IR).

        IR = (Strategy Return - Benchmark Return) / Tracking Error

        Args:
            strategy_returns: Strategy return timeseries
            benchmark_returns: Benchmark return timeseries

        Returns:
            Information ratio

        Example:
            >>> ir = await analyzer.calculate_information_ratio(strat_df, bench_df)
            >>> ir
            Decimal('1.5')
        """
        try:
            logger.debug("Calculating information ratio")

            # Align data
            aligned = strategy_returns.join(
                benchmark_returns,
                on='timestamp',
                how='inner'
            )

            if aligned.height == 0:
                raise ValueError("No overlapping data for IR calculation")

            # Calculate excess returns
            strat_rets = [Decimal(str(x)) for x in aligned.select('strategy_return').to_series()]
            bench_rets = [Decimal(str(x)) for x in aligned.select('benchmark_return').to_series()]

            excess_returns = [s - b for s, b in zip(strat_rets, bench_rets)]

            # Calculate mean excess return
            mean_excess = sum(excess_returns) / Decimal(len(excess_returns))

            # Calculate tracking error (std dev of excess returns)
            variance = sum(
                (er - mean_excess) ** 2
                for er in excess_returns
            ) / Decimal(len(excess_returns) - 1)

            tracking_error = variance.sqrt()

            # Calculate information ratio
            if tracking_error == Decimal('0'):
                information_ratio = Decimal('0')
            else:
                information_ratio = mean_excess / tracking_error

            logger.debug("Information ratio calculated", ir=str(information_ratio))

            return information_ratio

        except Exception as e:
            logger.error("Information ratio calculation failed", error=str(e))
            raise

    async def calculate_win_loss_ratio(
        self,
        strategy_data: pl.DataFrame,
        benchmark_data: pl.DataFrame
    ) -> Decimal:
        """Calculate win/loss ratio compared to benchmark.

        Args:
            strategy_data: Strategy trades data
            benchmark_data: Benchmark performance data

        Returns:
            Win/loss ratio

        Example:
            >>> wl_ratio = await analyzer.calculate_win_loss_ratio(strat_df, bench_df)
            >>> wl_ratio
            Decimal('1.8')
        """
        try:
            logger.debug("Calculating win/loss ratio vs benchmark")

            # Calculate periods where strategy outperforms
            aligned = strategy_data.join(
                benchmark_data,
                on='timestamp',
                how='inner'
            )

            wins = Decimal('0')
            losses = Decimal('0')

            for row in aligned.iter_rows(named=True):
                strat_return = Decimal(str(row.get('strategy_return', 0)))
                bench_return = Decimal(str(row.get('benchmark_return', 0)))

                if strat_return > bench_return:
                    wins += Decimal('1')
                elif strat_return < bench_return:
                    losses += Decimal('1')

            # Calculate ratio
            if losses == Decimal('0'):
                win_loss_ratio = wins if wins > Decimal('0') else Decimal('0')
            else:
                win_loss_ratio = wins / losses

            logger.debug("Win/loss ratio calculated", ratio=str(win_loss_ratio))

            return win_loss_ratio

        except Exception as e:
            logger.error("Win/loss ratio calculation failed", error=str(e))
            raise

    async def _calculate_buy_hold_returns(
        self,
        market_data: pl.DataFrame,
        initial_capital: Decimal
    ) -> pl.DataFrame:
        """Calculate buy-and-hold returns from market data.

        Args:
            market_data: Market price timeseries
            initial_capital: Starting capital

        Returns:
            DataFrame with buy-and-hold returns
        """
        try:
            logger.debug("Calculating buy-and-hold returns")

            if market_data.height == 0:
                raise ValueError("Empty market data")

            # Get first and calculate position size
            first_price = Decimal(str(market_data['close'][0]))
            position_size = initial_capital / first_price

            # Calculate portfolio value over time
            returns_data = []

            for row in market_data.iter_rows(named=True):
                price = Decimal(str(row['close']))
                timestamp = row['timestamp']
                portfolio_value = position_size * price

                if len(returns_data) > 0:
                    prev_value = returns_data[-1]['portfolio_value']
                    period_return = (portfolio_value - prev_value) / prev_value
                else:
                    period_return = Decimal('0')

                returns_data.append({
                    'timestamp': timestamp,
                    'portfolio_value': portfolio_value,
                    'benchmark_return': period_return
                })

            returns_df = pl.DataFrame(returns_data)

            logger.debug("Buy-and-hold returns calculated", periods=len(returns_data))

            return returns_df

        except Exception as e:
            logger.error("Failed to calculate buy-and-hold returns", error=str(e))
            raise

    async def _calculate_strategy_returns(
        self,
        strategy_data: pl.DataFrame
    ) -> pl.DataFrame:
        """Extract/calculate strategy returns from data.

        Args:
            strategy_data: Strategy performance data

        Returns:
            DataFrame with strategy returns
        """
        try:
            logger.debug("Calculating strategy returns")

            if 'strategy_return' in strategy_data.columns:
                return strategy_data

            # Calculate returns from portfolio values
            if 'portfolio_value' not in strategy_data.columns:
                raise ValueError("Missing required columns for return calculation")

            returns_data = []

            for i, row in enumerate(strategy_data.iter_rows(named=True)):
                if i == 0:
                    period_return = Decimal('0')
                else:
                    curr_value = Decimal(str(row['portfolio_value']))
                    prev_value = Decimal(str(strategy_data['portfolio_value'][i - 1]))
                    period_return = (curr_value - prev_value) / prev_value

                returns_data.append({
                    'timestamp': row['timestamp'],
                    'portfolio_value': row['portfolio_value'],
                    'strategy_return': period_return
                })

            returns_df = pl.DataFrame(returns_data)

            logger.debug("Strategy returns calculated")

            return returns_df

        except Exception as e:
            logger.error("Failed to calculate strategy returns", error=str(e))
            raise

    async def _calculate_comparison_metrics(
        self,
        strategy_returns: pl.DataFrame,
        benchmark_returns: pl.DataFrame
    ) -> Dict[str, Any]:
        """Calculate comprehensive comparison metrics.

        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns

        Returns:
            Dictionary of comparison metrics
        """
        try:
            logger.debug("Calculating comparison metrics")

            # Calculate alpha and beta
            alpha, beta = await self.calculate_alpha_beta(
                strategy_returns,
                benchmark_returns
            )

            # Calculate information ratio
            information_ratio = await self.calculate_information_ratio(
                strategy_returns,
                benchmark_returns
            )

            # Calculate win/loss ratio
            win_loss_ratio = await self.calculate_win_loss_ratio(
                strategy_returns,
                benchmark_returns
            )

            # Calculate total returns
            strat_total_return = self._calculate_total_return(strategy_returns)
            bench_total_return = self._calculate_total_return(benchmark_returns)

            excess_return = strat_total_return - bench_total_return

            metrics = {
                'alpha': alpha,
                'beta': beta,
                'information_ratio': information_ratio,
                'win_loss_ratio': win_loss_ratio,
                'strategy_total_return': strat_total_return,
                'benchmark_total_return': bench_total_return,
                'excess_return': excess_return
            }

            logger.debug("Comparison metrics calculated", metrics=metrics)

            return metrics

        except Exception as e:
            logger.error("Failed to calculate comparison metrics", error=str(e))
            raise

    def _calculate_total_return(self, returns_df: pl.DataFrame) -> Decimal:
        """Calculate total cumulative return.

        Args:
            returns_df: Returns timeseries

        Returns:
            Total return
        """
        try:
            if returns_df.height == 0:
                return Decimal('0')

            first_value = Decimal(str(returns_df['portfolio_value'][0]))
            last_value = Decimal(str(returns_df['portfolio_value'][-1]))

            if first_value == Decimal('0'):
                return Decimal('0')

            total_return = (last_value - first_value) / first_value

            return total_return

        except Exception as e:
            logger.error("Failed to calculate total return", error=str(e))
            raise

    async def _validate_data_alignment(
        self,
        strategy_data: pl.DataFrame,
        benchmark_data: pl.DataFrame
    ) -> None:
        """Validate that strategy and benchmark data are properly aligned.

        Args:
            strategy_data: Strategy data
            benchmark_data: Benchmark data

        Raises:
            ValueError: If data alignment invalid
        """
        try:
            if strategy_data.height == 0 or benchmark_data.height == 0:
                raise ValueError("Empty data provided")

            if 'timestamp' not in strategy_data.columns or 'timestamp' not in benchmark_data.columns:
                raise ValueError("Missing timestamp column")

            logger.debug("Data alignment validated")

        except Exception as e:
            logger.error("Data validation failed", error=str(e))
            raise
