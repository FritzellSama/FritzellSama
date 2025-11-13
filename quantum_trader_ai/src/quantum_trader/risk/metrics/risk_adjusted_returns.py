"""
Risk-Adjusted Returns Calculator
CRITICAL: Calculate Sharpe, Sortino, Calmar ratios and other risk-adjusted metrics
"""

import logging
import numpy as np
import polars as pl
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional, Tuple
from datetime import datetime

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class RiskAdjustedReturnsCalculator:
    """Calculate risk-adjusted performance metrics"""

    def __init__(self):
        """Initialize risk-adjusted returns calculator with config"""
        self.config = get_config()
        self.target_sharpe = self.config.get_decimal("risk", "risk_metrics.target_sharpe_ratio")

        # Default risk-free rate (2% annual)
        self.risk_free_rate = Decimal("0.02")

        # Trading days per year
        self.trading_days = 252

        logger.info(
            f"RiskAdjustedReturnsCalculator initialized: target_sharpe={self.target_sharpe}, "
            f"risk_free_rate={self.risk_free_rate:.2%}"
        )

    async def calculate_sharpe_ratio(
        self,
        returns: pl.DataFrame,
        risk_free_rate: Optional[Decimal] = None,
        annualize: bool = True
    ) -> Decimal:
        """
        Calculate Sharpe ratio

        Args:
            returns: DataFrame with columns ['timestamp', 'return']
            risk_free_rate: Annual risk-free rate (default: 2%)
            annualize: Whether to annualize the result

        Returns:
            Sharpe ratio as Decimal
        """
        try:
            if returns.is_empty():
                logger.error("Cannot calculate Sharpe ratio: empty returns")
                raise ValueError("Returns DataFrame cannot be empty")

            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate

            # Extract returns array
            returns_array = returns.select("return").to_numpy().flatten()

            if len(returns_array) < 2:
                logger.warning("Insufficient data for Sharpe ratio")
                return Decimal("0")

            # Calculate daily risk-free rate
            daily_rf = float(risk_free_rate) / self.trading_days

            # Calculate excess returns
            excess_returns = returns_array - daily_rf

            # Calculate mean and std of excess returns
            mean_excess = np.mean(excess_returns)
            std_excess = np.std(excess_returns, ddof=1)

            if std_excess < 1e-10:
                logger.warning("Standard deviation too low for Sharpe ratio")
                return Decimal("0")

            # Calculate Sharpe ratio
            sharpe = mean_excess / std_excess

            # Annualize if requested
            if annualize:
                sharpe = sharpe * np.sqrt(self.trading_days)

            sharpe_decimal = Decimal(str(sharpe)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

            logger.info(f"Sharpe ratio: {sharpe_decimal:.4f} (target: {self.target_sharpe:.4f})")

            return sharpe_decimal

        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}", exc_info=True)
            raise

    async def calculate_sortino_ratio(
        self,
        returns: pl.DataFrame,
        risk_free_rate: Optional[Decimal] = None,
        annualize: bool = True
    ) -> Decimal:
        """
        Calculate Sortino ratio (uses downside deviation instead of total volatility)

        Args:
            returns: DataFrame with columns ['timestamp', 'return']
            risk_free_rate: Annual risk-free rate (default: 2%)
            annualize: Whether to annualize the result

        Returns:
            Sortino ratio as Decimal
        """
        try:
            if returns.is_empty():
                logger.error("Cannot calculate Sortino ratio: empty returns")
                raise ValueError("Returns DataFrame cannot be empty")

            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate

            # Extract returns array
            returns_array = returns.select("return").to_numpy().flatten()

            if len(returns_array) < 2:
                logger.warning("Insufficient data for Sortino ratio")
                return Decimal("0")

            # Calculate daily risk-free rate
            daily_rf = float(risk_free_rate) / self.trading_days

            # Calculate excess returns
            excess_returns = returns_array - daily_rf

            # Calculate mean excess return
            mean_excess = np.mean(excess_returns)

            # Calculate downside deviation (only negative returns)
            downside_returns = excess_returns[excess_returns < 0]

            if len(downside_returns) == 0:
                # No downside - perfect performance
                logger.info("No downside returns - infinite Sortino ratio, returning 100.0")
                return Decimal("100.0")

            downside_std = np.std(downside_returns, ddof=1)

            if downside_std < 1e-10:
                logger.warning("Downside deviation too low for Sortino ratio")
                return Decimal("100.0")

            # Calculate Sortino ratio
            sortino = mean_excess / downside_std

            # Annualize if requested
            if annualize:
                sortino = sortino * np.sqrt(self.trading_days)

            sortino_decimal = Decimal(str(sortino)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

            logger.info(
                f"Sortino ratio: {sortino_decimal:.4f} "
                f"(downside_periods={len(downside_returns)}/{len(returns_array)})"
            )

            return sortino_decimal

        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}", exc_info=True)
            raise

    async def calculate_calmar_ratio(
        self,
        returns: pl.DataFrame,
        portfolio_values: pl.DataFrame,
        window_days: Optional[int] = None
    ) -> Decimal:
        """
        Calculate Calmar ratio (return / max drawdown)

        Args:
            returns: DataFrame with columns ['timestamp', 'return']
            portfolio_values: DataFrame with columns ['timestamp', 'value']
            window_days: Optional window for calculation (default: use all data)

        Returns:
            Calmar ratio as Decimal
        """
        try:
            if returns.is_empty() or portfolio_values.is_empty():
                logger.error("Cannot calculate Calmar ratio: empty data")
                raise ValueError("Returns and portfolio values cannot be empty")

            # Apply window if specified
            if window_days:
                cutoff_date = datetime.now().timestamp() - (window_days * 86400)
                returns = returns.filter(pl.col("timestamp") >= cutoff_date)
                portfolio_values = portfolio_values.filter(pl.col("timestamp") >= cutoff_date)

            # Calculate annualized return
            returns_array = returns.select("return").to_numpy().flatten()

            if len(returns_array) < 2:
                logger.warning("Insufficient data for Calmar ratio")
                return Decimal("0")

            mean_return = np.mean(returns_array)
            annualized_return = mean_return * self.trading_days

            # Calculate max drawdown
            values = portfolio_values.select("value").to_numpy().flatten()

            if len(values) < 2:
                logger.warning("Insufficient portfolio values for max drawdown")
                return Decimal("0")

            running_max = np.maximum.accumulate(values)
            drawdowns = (values - running_max) / running_max
            max_drawdown = abs(np.min(drawdowns))

            if max_drawdown < 1e-10:
                # No drawdown - excellent performance
                logger.info("No drawdown - infinite Calmar ratio, returning 100.0")
                return Decimal("100.0")

            # Calculate Calmar ratio
            calmar = annualized_return / max_drawdown

            calmar_decimal = Decimal(str(calmar)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

            logger.info(
                f"Calmar ratio: {calmar_decimal:.4f} "
                f"(return={annualized_return:.4%}, max_dd={max_drawdown:.4%})"
            )

            return calmar_decimal

        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {e}", exc_info=True)
            raise

    async def calculate_all_ratios(
        self,
        returns: pl.DataFrame,
        portfolio_values: pl.DataFrame,
        risk_free_rate: Optional[Decimal] = None
    ) -> Dict[str, Decimal]:
        """
        Calculate all risk-adjusted return ratios

        Args:
            returns: DataFrame with columns ['timestamp', 'return']
            portfolio_values: DataFrame with columns ['timestamp', 'value']
            risk_free_rate: Annual risk-free rate (default: 2%)

        Returns:
            Dictionary with all ratios:
            {
                'sharpe_ratio': Decimal,
                'sortino_ratio': Decimal,
                'calmar_ratio': Decimal,
                'information_ratio': Decimal,
                'omega_ratio': Decimal
            }
        """
        try:
            if returns.is_empty():
                logger.error("Cannot calculate ratios: empty returns")
                raise ValueError("Returns DataFrame cannot be empty")

            # Calculate standard ratios
            sharpe = await self.calculate_sharpe_ratio(returns, risk_free_rate)
            sortino = await self.calculate_sortino_ratio(returns, risk_free_rate)
            calmar = await self.calculate_calmar_ratio(returns, portfolio_values)

            # Calculate information ratio (Sharpe against benchmark)
            # For now, using risk-free rate as benchmark
            information_ratio = sharpe  # Simplified version

            # Calculate Omega ratio
            omega = await self._calculate_omega_ratio(returns, risk_free_rate)

            ratios = {
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'calmar_ratio': calmar,
                'information_ratio': information_ratio,
                'omega_ratio': omega
            }

            logger.info(
                f"Risk-adjusted returns: Sharpe={sharpe:.4f}, "
                f"Sortino={sortino:.4f}, Calmar={calmar:.4f}, Omega={omega:.4f}"
            )

            return ratios

        except Exception as e:
            logger.error(f"Error calculating all ratios: {e}", exc_info=True)
            raise

    async def _calculate_omega_ratio(
        self,
        returns: pl.DataFrame,
        risk_free_rate: Optional[Decimal] = None,
        threshold: Optional[float] = None
    ) -> Decimal:
        """
        Calculate Omega ratio (probability-weighted ratio of gains vs losses)

        Args:
            returns: DataFrame with columns ['timestamp', 'return']
            risk_free_rate: Annual risk-free rate (default: 2%)
            threshold: Return threshold (default: risk-free rate)

        Returns:
            Omega ratio as Decimal
        """
        try:
            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate

            # Extract returns array
            returns_array = returns.select("return").to_numpy().flatten()

            if len(returns_array) < 2:
                return Decimal("0")

            # Set threshold (default to risk-free rate)
            if threshold is None:
                threshold = float(risk_free_rate) / self.trading_days

            # Calculate excess returns above/below threshold
            excess_above = np.maximum(returns_array - threshold, 0)
            excess_below = np.maximum(threshold - returns_array, 0)

            sum_above = np.sum(excess_above)
            sum_below = np.sum(excess_below)

            if sum_below < 1e-10:
                # All returns above threshold
                return Decimal("100.0")

            omega = sum_above / sum_below

            omega_decimal = Decimal(str(omega)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

            logger.debug(f"Omega ratio: {omega_decimal:.4f}")

            return omega_decimal

        except Exception as e:
            logger.error(f"Error calculating Omega ratio: {e}", exc_info=True)
            return Decimal("0")

    async def calculate_rolling_sharpe(
        self,
        returns: pl.DataFrame,
        window_days: int = 252,
        risk_free_rate: Optional[Decimal] = None
    ) -> pl.DataFrame:
        """
        Calculate rolling Sharpe ratio

        Args:
            returns: DataFrame with columns ['timestamp', 'return']
            window_days: Rolling window size in days
            risk_free_rate: Annual risk-free rate (default: 2%)

        Returns:
            DataFrame with columns ['timestamp', 'rolling_sharpe']
        """
        try:
            if returns.is_empty():
                logger.error("Cannot calculate rolling Sharpe: empty returns")
                raise ValueError("Returns DataFrame cannot be empty")

            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate

            daily_rf = float(risk_free_rate) / self.trading_days

            # Sort by timestamp
            returns_sorted = returns.sort("timestamp")

            # Calculate excess returns
            returns_with_excess = returns_sorted.with_columns(
                (pl.col("return") - daily_rf).alias("excess_return")
            )

            # Calculate rolling mean and std of excess returns
            rolling_sharpe_df = returns_with_excess.with_columns([
                pl.col("excess_return").rolling_mean(window_size=window_days).alias("mean_excess"),
                pl.col("excess_return").rolling_std(window_size=window_days).alias("std_excess")
            ])

            # Calculate Sharpe ratio (annualized)
            rolling_sharpe_df = rolling_sharpe_df.with_columns(
                ((pl.col("mean_excess") / pl.col("std_excess")) * np.sqrt(self.trading_days)).alias("rolling_sharpe")
            )

            # Select only timestamp and rolling_sharpe
            result = rolling_sharpe_df.select(["timestamp", "rolling_sharpe"]).drop_nulls()

            logger.info(
                f"Calculated rolling Sharpe ratio: window={window_days}d, "
                f"samples={len(result)}"
            )

            return result

        except Exception as e:
            logger.error(f"Error calculating rolling Sharpe: {e}", exc_info=True)
            raise

    async def compare_to_benchmark(
        self,
        strategy_returns: pl.DataFrame,
        benchmark_returns: pl.DataFrame,
        risk_free_rate: Optional[Decimal] = None
    ) -> Dict[str, Decimal]:
        """
        Compare strategy performance to benchmark

        Args:
            strategy_returns: DataFrame with columns ['timestamp', 'return']
            benchmark_returns: DataFrame with columns ['timestamp', 'return']
            risk_free_rate: Annual risk-free rate (default: 2%)

        Returns:
            Dictionary with comparison metrics:
            {
                'strategy_sharpe': Decimal,
                'benchmark_sharpe': Decimal,
                'excess_sharpe': Decimal,
                'tracking_error': Decimal,
                'information_ratio': Decimal,
                'alpha': Decimal,
                'beta': Decimal
            }
        """
        try:
            if strategy_returns.is_empty() or benchmark_returns.is_empty():
                logger.error("Cannot compare: empty returns")
                raise ValueError("Strategy and benchmark returns cannot be empty")

            # Calculate Sharpe ratios
            strategy_sharpe = await self.calculate_sharpe_ratio(strategy_returns, risk_free_rate)
            benchmark_sharpe = await self.calculate_sharpe_ratio(benchmark_returns, risk_free_rate)

            excess_sharpe = strategy_sharpe - benchmark_sharpe

            # Merge returns for additional calculations
            merged = strategy_returns.join(
                benchmark_returns.rename({"return": "benchmark_return"}),
                on="timestamp",
                how="inner"
            )

            if merged.is_empty():
                logger.error("No overlapping timestamps for comparison")
                raise ValueError("No matching timestamps between strategy and benchmark")

            strategy_ret = merged.select("return").to_numpy().flatten()
            benchmark_ret = merged.select("benchmark_return").to_numpy().flatten()

            # Calculate tracking error (std of excess returns)
            excess_returns = strategy_ret - benchmark_ret
            tracking_error = float(np.std(excess_returns, ddof=1)) * np.sqrt(self.trading_days)

            # Information ratio = excess return / tracking error
            mean_excess = float(np.mean(excess_returns)) * self.trading_days
            information_ratio = mean_excess / tracking_error if tracking_error > 1e-10 else 0.0

            # Calculate beta and alpha
            covariance = np.cov(strategy_ret, benchmark_ret)[0, 1]
            benchmark_variance = np.var(benchmark_ret, ddof=1)
            beta = covariance / benchmark_variance if benchmark_variance > 1e-10 else 1.0

            # Alpha = Strategy return - (Risk-free + Beta * (Benchmark - Risk-free))
            daily_rf = float(risk_free_rate or self.risk_free_rate) / self.trading_days
            strategy_mean = np.mean(strategy_ret)
            benchmark_mean = np.mean(benchmark_ret)
            alpha = (strategy_mean - daily_rf) - beta * (benchmark_mean - daily_rf)
            annualized_alpha = alpha * self.trading_days

            result = {
                'strategy_sharpe': strategy_sharpe,
                'benchmark_sharpe': benchmark_sharpe,
                'excess_sharpe': excess_sharpe,
                'tracking_error': Decimal(str(tracking_error)).quantize(Decimal("0.000001")),
                'information_ratio': Decimal(str(information_ratio)).quantize(Decimal("0.0001")),
                'alpha': Decimal(str(annualized_alpha)).quantize(Decimal("0.000001")),
                'beta': Decimal(str(beta)).quantize(Decimal("0.0001"))
            }

            logger.info(
                f"Benchmark comparison: strategy_sharpe={strategy_sharpe:.4f}, "
                f"benchmark_sharpe={benchmark_sharpe:.4f}, alpha={annualized_alpha:.6f}, "
                f"beta={beta:.4f}, IR={information_ratio:.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error comparing to benchmark: {e}", exc_info=True)
            raise
