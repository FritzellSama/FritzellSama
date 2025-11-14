"""
Quantum Trader AI - Sharpe Ratio Calculator
Production-grade Sharpe ratio calculations with confidence intervals

CRITICAL CONSTRAINTS:
- All numeric values use Decimal, NEVER float
- All data operations use polars DataFrame
- All external calls wrapped in try/except with retry logic
- Complete type hints everywhere
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml
import polars as pl
import numpy as np
from scipy import stats


logger = logging.getLogger(__name__)


class SharpeRatioCalculator:
    """
    Sharpe ratio calculation system

    Features:
    - Rolling Sharpe calculation
    - Annualized Sharpe
    - Ex-ante vs ex-post Sharpe
    - Confidence intervals
    """

    def __init__(
        self,
        config_path: Path = Path("/home/user/FritzellSama/config/bot/risk.yaml"),
        env_config_path: Path = Path("/home/user/FritzellSama/config/environments/production.yaml")
    ) -> None:
        """Initialize Sharpe ratio calculator with configuration"""
        self.config = self._load_config(config_path)
        self.env_config = self._load_config(env_config_path)

        # Risk model parameters
        risk_model = self.config.get("risk_model", {})
        self.min_sharpe_ratio = Decimal(str(risk_model.get("min_sharpe_ratio", 0.5)))
        self.var_lookback_days = int(risk_model.get("var_lookback_days", 252))

        # Sharpe calculation parameters
        self.risk_free_rate = Decimal('0.04')  # 4% annual risk-free rate (US Treasury)
        self.trading_days_per_year = Decimal('252')
        self.confidence_level = Decimal('0.95')  # 95% confidence interval
        self.min_periods = 30  # Minimum data points for calculation

        # Rolling window sizes
        self.rolling_windows = {
            'short': 30,    # 1 month
            'medium': 90,   # 3 months
            'long': 252     # 1 year
        }

        # Retry configuration
        self.retry_attempts = 3
        self.retry_delay_ms = 1000

        logger.info("SharpeRatioCalculator initialized")

    def _load_config(self, config_path: Path) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}

    def calculate_sharpe_ratio(
        self,
        returns: pl.Series,
        risk_free_rate: Optional[Decimal] = None,
        annualize: bool = True
    ) -> Decimal:
        """
        Calculate Sharpe ratio from returns series

        Sharpe Ratio = (Mean Return - Risk Free Rate) / Std Dev of Returns

        Args:
            returns: Series of returns (typically daily)
            risk_free_rate: Annual risk-free rate (uses default if None)
            annualize: Whether to annualize the result

        Returns:
            Sharpe ratio as Decimal
        """
        try:
            if len(returns) < self.min_periods:
                logger.warning(f"Insufficient data for Sharpe calculation: {len(returns)} < {self.min_periods}")
                return Decimal('0')

            # Use provided risk-free rate or default
            rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate

            # Convert to numpy for calculation
            returns_array = returns.to_numpy()

            # Calculate mean return
            mean_return = Decimal(str(np.mean(returns_array)))

            # Calculate standard deviation
            std_dev = Decimal(str(np.std(returns_array, ddof=1)))  # Sample std dev

            if std_dev == Decimal('0'):
                logger.warning("Zero standard deviation in returns")
                return Decimal('0')

            # Daily risk-free rate
            daily_rf = rf_rate / self.trading_days_per_year

            # Calculate Sharpe ratio
            sharpe = (mean_return - daily_rf) / std_dev

            # Annualize if requested
            if annualize:
                sharpe = sharpe * self.trading_days_per_year.sqrt()

            return sharpe

        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return Decimal('0')

    def calculate_rolling_sharpe(
        self,
        returns_df: pl.DataFrame,
        window: int,
        risk_free_rate: Optional[Decimal] = None
    ) -> pl.DataFrame:
        """
        Calculate rolling Sharpe ratio

        Args:
            returns_df: DataFrame with 'timestamp' and 'returns' columns
            window: Rolling window size in periods
            risk_free_rate: Annual risk-free rate

        Returns:
            DataFrame with rolling Sharpe ratios
        """
        try:
            if len(returns_df) < window:
                logger.warning(f"Insufficient data for rolling Sharpe: {len(returns_df)} < {window}")
                return pl.DataFrame()

            rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
            daily_rf = rf_rate / self.trading_days_per_year

            # Calculate rolling mean and std
            rolling_mean = returns_df.select([
                pl.col("timestamp"),
                pl.col("returns").rolling_mean(window_size=window).alias("rolling_mean")
            ])

            rolling_std = returns_df.select([
                pl.col("returns").rolling_std(window_size=window).alias("rolling_std")
            ])

            # Combine
            rolling_df = rolling_mean.with_columns(
                rolling_std.select("rolling_std")
            )

            # Calculate Sharpe ratio
            sharpe_values = []
            for row in rolling_df.iter_rows(named=True):
                if row["rolling_std"] is not None and row["rolling_std"] > 0:
                    mean_ret = Decimal(str(row["rolling_mean"]))
                    std_ret = Decimal(str(row["rolling_std"]))
                    sharpe = ((mean_ret - daily_rf) / std_ret) * self.trading_days_per_year.sqrt()
                    sharpe_values.append(float(sharpe))
                else:
                    sharpe_values.append(None)

            # Add Sharpe column
            result_df = rolling_df.with_columns(
                pl.Series("sharpe_ratio", sharpe_values)
            )

            return result_df

        except Exception as e:
            logger.error(f"Error calculating rolling Sharpe: {e}")
            return pl.DataFrame()

    def calculate_sharpe_confidence_interval(
        self,
        returns: pl.Series,
        confidence_level: Optional[Decimal] = None
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """
        Calculate Sharpe ratio with confidence intervals

        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Tuple of (sharpe_ratio, lower_bound, upper_bound)
        """
        try:
            if len(returns) < self.min_periods:
                return Decimal('0'), Decimal('0'), Decimal('0')

            conf_level = confidence_level if confidence_level is not None else self.confidence_level

            # Calculate Sharpe ratio
            sharpe = self.calculate_sharpe_ratio(returns, annualize=True)

            # Calculate standard error of Sharpe ratio
            n = len(returns)
            returns_array = returns.to_numpy()

            # Standard error approximation: SE(SR) ≈ sqrt((1 + SR²/2) / n)
            se_sharpe = Decimal(str(np.sqrt((1 + float(sharpe)**2 / 2) / n)))

            # Calculate confidence interval using t-distribution
            t_stat = Decimal(str(stats.t.ppf((1 + float(conf_level)) / 2, n - 1)))

            lower_bound = sharpe - (t_stat * se_sharpe)
            upper_bound = sharpe + (t_stat * se_sharpe)

            return sharpe, lower_bound, upper_bound

        except Exception as e:
            logger.error(f"Error calculating Sharpe confidence interval: {e}")
            return Decimal('0'), Decimal('0'), Decimal('0')

    def calculate_information_ratio(
        self,
        returns: pl.Series,
        benchmark_returns: pl.Series
    ) -> Decimal:
        """
        Calculate Information Ratio (similar to Sharpe but vs benchmark)

        IR = Mean(Active Return) / Std(Active Return)
        Where Active Return = Portfolio Return - Benchmark Return

        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns

        Returns:
            Information ratio
        """
        try:
            if len(returns) != len(benchmark_returns):
                logger.error("Returns and benchmark length mismatch")
                return Decimal('0')

            if len(returns) < self.min_periods:
                return Decimal('0')

            # Calculate active returns (excess returns)
            returns_array = returns.to_numpy()
            benchmark_array = benchmark_returns.to_numpy()
            active_returns = returns_array - benchmark_array

            # Calculate mean and std of active returns
            mean_active = Decimal(str(np.mean(active_returns)))
            std_active = Decimal(str(np.std(active_returns, ddof=1)))

            if std_active == Decimal('0'):
                return Decimal('0')

            # Information ratio
            ir = (mean_active / std_active) * self.trading_days_per_year.sqrt()

            return ir

        except Exception as e:
            logger.error(f"Error calculating Information Ratio: {e}")
            return Decimal('0')

    def calculate_sortino_ratio(
        self,
        returns: pl.Series,
        target_return: Optional[Decimal] = None,
        risk_free_rate: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate Sortino ratio (Sharpe using downside deviation only)

        Sortino Ratio = (Mean Return - Target) / Downside Deviation

        Args:
            returns: Series of returns
            target_return: Target return (uses risk-free rate if None)
            risk_free_rate: Annual risk-free rate

        Returns:
            Sortino ratio
        """
        try:
            if len(returns) < self.min_periods:
                return Decimal('0')

            rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
            daily_rf = rf_rate / self.trading_days_per_year

            target = target_return if target_return is not None else daily_rf

            # Calculate mean return
            returns_array = returns.to_numpy()
            mean_return = Decimal(str(np.mean(returns_array)))

            # Calculate downside deviation (only negative returns)
            downside_returns = returns_array[returns_array < float(target)]

            if len(downside_returns) == 0:
                # No downside - very good!
                return Decimal('999')  # Very high Sortino

            downside_dev = Decimal(str(np.std(downside_returns, ddof=1)))

            if downside_dev == Decimal('0'):
                return Decimal('0')

            # Calculate Sortino ratio
            sortino = ((mean_return - target) / downside_dev) * self.trading_days_per_year.sqrt()

            return sortino

        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return Decimal('0')

    def calculate_calmar_ratio(
        self,
        returns: pl.Series,
        max_drawdown: Decimal
    ) -> Decimal:
        """
        Calculate Calmar ratio

        Calmar Ratio = Annualized Return / Maximum Drawdown

        Args:
            returns: Series of returns
            max_drawdown: Maximum drawdown (as positive percentage)

        Returns:
            Calmar ratio
        """
        try:
            if len(returns) < self.min_periods or max_drawdown == Decimal('0'):
                return Decimal('0')

            # Calculate annualized return
            returns_array = returns.to_numpy()
            mean_return = Decimal(str(np.mean(returns_array)))
            annualized_return = mean_return * self.trading_days_per_year

            # Calmar ratio
            calmar = annualized_return / max_drawdown

            return calmar

        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {e}")
            return Decimal('0')

    def calculate_ex_ante_sharpe(
        self,
        expected_return: Decimal,
        expected_volatility: Decimal,
        risk_free_rate: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate ex-ante (forward-looking) Sharpe ratio

        Args:
            expected_return: Expected annual return
            expected_volatility: Expected annual volatility
            risk_free_rate: Annual risk-free rate

        Returns:
            Ex-ante Sharpe ratio
        """
        try:
            rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate

            if expected_volatility == Decimal('0'):
                return Decimal('0')

            # Ex-ante Sharpe
            sharpe = (expected_return - rf_rate) / expected_volatility

            return sharpe

        except Exception as e:
            logger.error(f"Error calculating ex-ante Sharpe: {e}")
            return Decimal('0')

    def calculate_multi_period_sharpe(
        self,
        returns_df: pl.DataFrame
    ) -> Dict[str, Decimal]:
        """
        Calculate Sharpe ratio for multiple time periods

        Args:
            returns_df: DataFrame with returns

        Returns:
            Dictionary with Sharpe ratios for different periods
        """
        try:
            results = {}

            for period_name, window in self.rolling_windows.items():
                if len(returns_df) >= window:
                    # Get last N returns
                    recent_returns = returns_df.tail(window).select("returns").to_series()

                    # Calculate Sharpe
                    sharpe = self.calculate_sharpe_ratio(recent_returns, annualize=True)
                    results[f"sharpe_{period_name}"] = sharpe
                else:
                    results[f"sharpe_{period_name}"] = Decimal('0')

            return results

        except Exception as e:
            logger.error(f"Error calculating multi-period Sharpe: {e}")
            return {}

    def calculate_risk_adjusted_metrics(
        self,
        returns: pl.Series,
        benchmark_returns: Optional[pl.Series] = None,
        max_drawdown: Optional[Decimal] = None
    ) -> Dict[str, Decimal]:
        """
        Calculate comprehensive risk-adjusted performance metrics

        Args:
            returns: Portfolio returns
            benchmark_returns: Optional benchmark returns
            max_drawdown: Optional max drawdown value

        Returns:
            Dictionary of risk-adjusted metrics
        """
        try:
            metrics = {}

            # Sharpe ratio
            sharpe, sharpe_lower, sharpe_upper = self.calculate_sharpe_confidence_interval(returns)
            metrics['sharpe_ratio'] = sharpe
            metrics['sharpe_ci_lower'] = sharpe_lower
            metrics['sharpe_ci_upper'] = sharpe_upper

            # Sortino ratio
            metrics['sortino_ratio'] = self.calculate_sortino_ratio(returns)

            # Information ratio (if benchmark provided)
            if benchmark_returns is not None and len(benchmark_returns) == len(returns):
                metrics['information_ratio'] = self.calculate_information_ratio(returns, benchmark_returns)

            # Calmar ratio (if max drawdown provided)
            if max_drawdown is not None:
                metrics['calmar_ratio'] = self.calculate_calmar_ratio(returns, max_drawdown)

            # Basic statistics
            returns_array = returns.to_numpy()
            metrics['mean_return'] = Decimal(str(np.mean(returns_array)))
            metrics['volatility'] = Decimal(str(np.std(returns_array, ddof=1)))
            metrics['skewness'] = Decimal(str(stats.skew(returns_array)))
            metrics['kurtosis'] = Decimal(str(stats.kurtosis(returns_array)))

            return metrics

        except Exception as e:
            logger.error(f"Error calculating risk-adjusted metrics: {e}")
            return {}

    def evaluate_sharpe_significance(
        self,
        returns: pl.Series,
        min_sharpe: Optional[Decimal] = None
    ) -> Dict[str, any]:
        """
        Evaluate if Sharpe ratio is statistically significant

        Args:
            returns: Series of returns
            min_sharpe: Minimum acceptable Sharpe (uses config default if None)

        Returns:
            Dictionary with evaluation results
        """
        try:
            min_sharpe_threshold = min_sharpe if min_sharpe is not None else self.min_sharpe_ratio

            # Calculate Sharpe with confidence interval
            sharpe, lower_ci, upper_ci = self.calculate_sharpe_confidence_interval(returns)

            # Check if lower CI is above zero (statistically significant)
            is_significant = lower_ci > Decimal('0')

            # Check if meets minimum threshold
            meets_threshold = sharpe >= min_sharpe_threshold

            # Check if lower CI meets threshold
            ci_meets_threshold = lower_ci >= min_sharpe_threshold

            evaluation = {
                'sharpe_ratio': sharpe,
                'confidence_interval': (lower_ci, upper_ci),
                'is_significant': is_significant,
                'meets_threshold': meets_threshold,
                'ci_meets_threshold': ci_meets_threshold,
                'threshold': min_sharpe_threshold,
                'sample_size': len(returns),
                'recommendation': self._get_sharpe_recommendation(
                    sharpe,
                    is_significant,
                    meets_threshold
                )
            }

            return evaluation

        except Exception as e:
            logger.error(f"Error evaluating Sharpe significance: {e}")
            return {}

    def _get_sharpe_recommendation(
        self,
        sharpe: Decimal,
        is_significant: bool,
        meets_threshold: bool
    ) -> str:
        """Get recommendation based on Sharpe analysis"""
        if sharpe < Decimal('0'):
            return "POOR: Negative Sharpe ratio - strategy losing to risk-free rate"
        elif not is_significant:
            return "UNCERTAIN: Sharpe not statistically significant - need more data"
        elif not meets_threshold:
            return "BELOW_TARGET: Sharpe significant but below minimum threshold"
        elif sharpe < Decimal('1'):
            return "ACCEPTABLE: Sharpe ratio acceptable but not exceptional"
        elif sharpe < Decimal('2'):
            return "GOOD: Strong risk-adjusted returns"
        else:
            return "EXCELLENT: Exceptional risk-adjusted returns"

    def generate_sharpe_report(
        self,
        returns_df: pl.DataFrame,
        benchmark_returns: Optional[pl.Series] = None
    ) -> Dict:
        """
        Generate comprehensive Sharpe ratio report

        Args:
            returns_df: DataFrame with timestamp and returns columns
            benchmark_returns: Optional benchmark returns

        Returns:
            Comprehensive report dictionary
        """
        try:
            if len(returns_df) == 0:
                return {"error": "No data provided"}

            returns = returns_df.select("returns").to_series()

            # Calculate main metrics
            sharpe, lower_ci, upper_ci = self.calculate_sharpe_confidence_interval(returns)

            # Multi-period analysis
            multi_period = self.calculate_multi_period_sharpe(returns_df)

            # Risk-adjusted metrics
            risk_metrics = self.calculate_risk_adjusted_metrics(returns, benchmark_returns)

            # Significance evaluation
            evaluation = self.evaluate_sharpe_significance(returns)

            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "sample_size": len(returns),
                "sharpe_ratio": {
                    "value": float(sharpe),
                    "confidence_interval": [float(lower_ci), float(upper_ci)],
                    "confidence_level": float(self.confidence_level)
                },
                "multi_period_sharpe": {k: float(v) for k, v in multi_period.items()},
                "risk_adjusted_metrics": {k: float(v) for k, v in risk_metrics.items()},
                "evaluation": {
                    "is_significant": evaluation.get('is_significant'),
                    "meets_threshold": evaluation.get('meets_threshold'),
                    "recommendation": evaluation.get('recommendation')
                }
            }

            return report

        except Exception as e:
            logger.error(f"Error generating Sharpe report: {e}")
            return {"error": str(e)}
