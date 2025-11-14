"""
Risk-Adjusted Return Metrics for Quantum Trader AI

Production-grade risk-adjusted performance metrics:
- Sharpe ratio (return per unit of total risk)
- Sortino ratio (return per unit of downside risk)
- Calmar ratio (return per unit of max drawdown)
- Information ratio (excess return per unit of tracking error)
- Treynor ratio (return per unit of systematic risk)
- Omega ratio
- Jensen's alpha

CRITICAL: All numeric values use Decimal, NEVER float
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Optional
import logging

import polars as pl
import yaml
import numpy as np

from quantum_trader.models import RiskMetrics


logger = logging.getLogger(__name__)


@dataclass
class RiskAdjustedReturnsConfig:
    """Configuration for risk-adjusted returns calculations"""
    risk_free_rate_annual: Decimal
    trading_days_per_year: int
    min_periods_required: int
    confidence_level: Decimal
    use_geometric_mean: bool


@dataclass
class PerformanceMetrics:
    """Complete set of risk-adjusted performance metrics"""
    total_return: Decimal
    annualized_return: Decimal
    annualized_volatility: Decimal
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    calmar_ratio: Decimal
    information_ratio: Decimal
    treynor_ratio: Decimal
    omega_ratio: Decimal
    jensens_alpha: Decimal
    max_drawdown: Decimal
    downside_deviation: Decimal
    tracking_error: Optional[Decimal]
    beta: Optional[Decimal]
    num_periods: int
    timestamp: datetime


class RiskAdjustedReturnsCalculator:
    """
    Calculator for risk-adjusted return metrics

    Provides comprehensive performance evaluation metrics that account for risk:
    - Sharpe ratio for total risk
    - Sortino ratio for downside risk
    - Calmar ratio for drawdown risk
    - Information ratio vs benchmark
    - Treynor ratio for systematic risk
    """

    def __init__(
        self,
        config_path: str = "/home/user/FritzellSama/config/bot/risk.yaml",
        env_config_path: str = "/home/user/FritzellSama/config/environments/production.yaml"
    ):
        """
        Initialize risk-adjusted returns calculator

        Args:
            config_path: Path to risk configuration file
            env_config_path: Path to environment configuration file
        """
        self.config = self._load_config(config_path, env_config_path)

        logger.info("RiskAdjustedReturnsCalculator initialized")

    def _load_config(self, config_path: str, env_config_path: str) -> RiskAdjustedReturnsConfig:
        """
        Load configuration from YAML files

        Args:
            config_path: Path to risk config
            env_config_path: Path to environment config

        Returns:
            Loaded configuration
        """
        try:
            with open(config_path, 'r') as f:
                risk_config = yaml.safe_load(f)

            with open(env_config_path, 'r') as f:
                env_config = yaml.safe_load(f)

            risk_model = risk_config.get('risk_model', {})

            return RiskAdjustedReturnsConfig(
                risk_free_rate_annual=Decimal('0.04'),  # 4% annual risk-free rate
                trading_days_per_year=252,
                min_periods_required=30,
                confidence_level=Decimal(str(risk_model.get('var_confidence_level', 0.95))),
                use_geometric_mean=True
            )

        except Exception as e:
            logger.error("Failed to load configuration: %s", e)
            raise

    def calculate_sharpe_ratio(
        self,
        returns_df: pl.DataFrame,
        risk_free_rate: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate Sharpe ratio: (return - risk_free) / volatility

        Measures excess return per unit of total risk

        Args:
            returns_df: DataFrame with column 'return' (periodic returns)
            risk_free_rate: Risk-free rate (uses config default if None)

        Returns:
            Sharpe ratio
        """
        if not isinstance(returns_df, pl.DataFrame):
            raise TypeError(f"returns_df must be polars DataFrame")

        if returns_df.height < self.config.min_periods_required:
            logger.warning(
                "Insufficient data for Sharpe ratio: %d periods (need %d)",
                returns_df.height, self.config.min_periods_required
            )
            return Decimal('0')

        if risk_free_rate is None:
            risk_free_rate = self.config.risk_free_rate_annual

        # Convert annual risk-free rate to period rate
        periods_per_year = self.config.trading_days_per_year
        risk_free_period = risk_free_rate / Decimal(str(periods_per_year))

        # Calculate mean return and volatility
        returns = [Decimal(str(r)) for r in returns_df['return'].to_list()]

        mean_return = sum(returns) / Decimal(str(len(returns)))

        # Calculate standard deviation
        variance = sum((r - mean_return) ** 2 for r in returns) / Decimal(str(len(returns)))
        volatility = Decimal(str(np.sqrt(float(variance))))

        if volatility == Decimal('0'):
            return Decimal('0')

        # Sharpe ratio
        excess_return = mean_return - risk_free_period
        sharpe = excess_return / volatility

        # Annualize
        sharpe_annual = sharpe * Decimal(str(np.sqrt(periods_per_year)))

        logger.info("Sharpe ratio calculated: %s", sharpe_annual)

        return sharpe_annual

    def calculate_sortino_ratio(
        self,
        returns_df: pl.DataFrame,
        risk_free_rate: Optional[Decimal] = None,
        target_return: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate Sortino ratio: (return - target) / downside_deviation

        Measures excess return per unit of downside risk

        Args:
            returns_df: DataFrame with column 'return'
            risk_free_rate: Risk-free rate (uses config default if None)
            target_return: Target return threshold (uses risk_free if None)

        Returns:
            Sortino ratio
        """
        if not isinstance(returns_df, pl.DataFrame):
            raise TypeError(f"returns_df must be polars DataFrame")

        if returns_df.height < self.config.min_periods_required:
            logger.warning("Insufficient data for Sortino ratio")
            return Decimal('0')

        if risk_free_rate is None:
            risk_free_rate = self.config.risk_free_rate_annual

        if target_return is None:
            # Use risk-free rate as target
            periods_per_year = self.config.trading_days_per_year
            target_return = risk_free_rate / Decimal(str(periods_per_year))

        # Calculate mean return
        returns = [Decimal(str(r)) for r in returns_df['return'].to_list()]
        mean_return = sum(returns) / Decimal(str(len(returns)))

        # Calculate downside deviation (only negative deviations from target)
        downside_returns = [min(r - target_return, Decimal('0')) for r in returns]
        downside_variance = sum(r ** 2 for r in downside_returns) / Decimal(str(len(returns)))
        downside_deviation = Decimal(str(np.sqrt(float(downside_variance))))

        if downside_deviation == Decimal('0'):
            return Decimal('0')

        # Sortino ratio
        excess_return = mean_return - target_return
        sortino = excess_return / downside_deviation

        # Annualize
        periods_per_year = self.config.trading_days_per_year
        sortino_annual = sortino * Decimal(str(np.sqrt(periods_per_year)))

        logger.info("Sortino ratio calculated: %s", sortino_annual)

        return sortino_annual

    def calculate_calmar_ratio(
        self,
        returns_df: pl.DataFrame,
        max_drawdown: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate Calmar ratio: annualized_return / max_drawdown

        Measures return per unit of maximum drawdown

        Args:
            returns_df: DataFrame with column 'return'
            max_drawdown: Maximum drawdown percentage (calculates if None)

        Returns:
            Calmar ratio
        """
        if not isinstance(returns_df, pl.DataFrame):
            raise TypeError(f"returns_df must be polars DataFrame")

        if returns_df.height < self.config.min_periods_required:
            logger.warning("Insufficient data for Calmar ratio")
            return Decimal('0')

        # Calculate annualized return
        returns = [Decimal(str(r)) for r in returns_df['return'].to_list()]
        mean_return = sum(returns) / Decimal(str(len(returns)))
        annualized_return = mean_return * Decimal(str(self.config.trading_days_per_year))

        # Calculate max drawdown if not provided
        if max_drawdown is None:
            max_drawdown = self._calculate_max_drawdown(returns_df)

        if max_drawdown == Decimal('0'):
            return Decimal('0')

        # Calmar ratio
        calmar = (annualized_return * Decimal('100')) / max_drawdown  # Convert to percentage

        logger.info("Calmar ratio calculated: %s", calmar)

        return calmar

    def calculate_information_ratio(
        self,
        portfolio_returns_df: pl.DataFrame,
        benchmark_returns_df: pl.DataFrame
    ) -> Decimal:
        """
        Calculate Information ratio: excess_return / tracking_error

        Measures excess return vs benchmark per unit of tracking error

        Args:
            portfolio_returns_df: DataFrame with portfolio returns
            benchmark_returns_df: DataFrame with benchmark returns

        Returns:
            Information ratio
        """
        if not isinstance(portfolio_returns_df, pl.DataFrame):
            raise TypeError(f"portfolio_returns_df must be polars DataFrame")

        if not isinstance(benchmark_returns_df, pl.DataFrame):
            raise TypeError(f"benchmark_returns_df must be polars DataFrame")

        # Join returns
        df = portfolio_returns_df.join(
            benchmark_returns_df,
            on='timestamp',
            how='inner',
            suffix='_benchmark'
        )

        if df.height < self.config.min_periods_required:
            logger.warning("Insufficient data for Information ratio")
            return Decimal('0')

        # Calculate excess returns
        excess_returns = []
        for row in df.iter_rows(named=True):
            portfolio_return = Decimal(str(row['return']))
            benchmark_return = Decimal(str(row['return_benchmark']))
            excess = portfolio_return - benchmark_return
            excess_returns.append(excess)

        # Mean excess return
        mean_excess = sum(excess_returns) / Decimal(str(len(excess_returns)))

        # Tracking error (std of excess returns)
        variance = sum((er - mean_excess) ** 2 for er in excess_returns) / Decimal(str(len(excess_returns)))
        tracking_error = Decimal(str(np.sqrt(float(variance))))

        if tracking_error == Decimal('0'):
            return Decimal('0')

        # Information ratio
        info_ratio = mean_excess / tracking_error

        # Annualize
        periods_per_year = self.config.trading_days_per_year
        info_ratio_annual = info_ratio * Decimal(str(np.sqrt(periods_per_year)))

        logger.info("Information ratio calculated: %s", info_ratio_annual)

        return info_ratio_annual

    def calculate_treynor_ratio(
        self,
        portfolio_returns_df: pl.DataFrame,
        benchmark_returns_df: pl.DataFrame,
        beta: Optional[Decimal] = None,
        risk_free_rate: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate Treynor ratio: (return - risk_free) / beta

        Measures excess return per unit of systematic risk

        Args:
            portfolio_returns_df: DataFrame with portfolio returns
            benchmark_returns_df: DataFrame with benchmark returns
            beta: Portfolio beta (calculates if None)
            risk_free_rate: Risk-free rate

        Returns:
            Treynor ratio
        """
        if not isinstance(portfolio_returns_df, pl.DataFrame):
            raise TypeError(f"portfolio_returns_df must be polars DataFrame")

        if portfolio_returns_df.height < self.config.min_periods_required:
            logger.warning("Insufficient data for Treynor ratio")
            return Decimal('0')

        if risk_free_rate is None:
            risk_free_rate = self.config.risk_free_rate_annual

        # Calculate portfolio return
        returns = [Decimal(str(r)) for r in portfolio_returns_df['return'].to_list()]
        mean_return = sum(returns) / Decimal(str(len(returns)))
        annualized_return = mean_return * Decimal(str(self.config.trading_days_per_year))

        # Calculate beta if not provided
        if beta is None:
            beta = self._calculate_beta(portfolio_returns_df, benchmark_returns_df)

        if beta == Decimal('0'):
            return Decimal('0')

        # Treynor ratio
        excess_return = annualized_return - risk_free_rate
        treynor = excess_return / beta

        logger.info("Treynor ratio calculated: %s", treynor)

        return treynor

    def calculate_omega_ratio(
        self,
        returns_df: pl.DataFrame,
        threshold: Decimal = Decimal('0')
    ) -> Decimal:
        """
        Calculate Omega ratio: probability-weighted gains / probability-weighted losses

        Ratio of gains to losses relative to threshold

        Args:
            returns_df: DataFrame with returns
            threshold: Return threshold (default 0)

        Returns:
            Omega ratio
        """
        if not isinstance(returns_df, pl.DataFrame):
            raise TypeError(f"returns_df must be polars DataFrame")

        if returns_df.height < self.config.min_periods_required:
            logger.warning("Insufficient data for Omega ratio")
            return Decimal('0')

        returns = [Decimal(str(r)) for r in returns_df['return'].to_list()]

        # Calculate gains and losses relative to threshold
        gains = sum(max(r - threshold, Decimal('0')) for r in returns)
        losses = sum(abs(min(r - threshold, Decimal('0'))) for r in returns)

        if losses == Decimal('0'):
            return Decimal('0') if gains == Decimal('0') else Decimal('999999')

        omega = gains / losses

        logger.info("Omega ratio calculated: %s", omega)

        return omega

    def calculate_jensens_alpha(
        self,
        portfolio_returns_df: pl.DataFrame,
        benchmark_returns_df: pl.DataFrame,
        beta: Optional[Decimal] = None,
        risk_free_rate: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate Jensen's alpha: portfolio_return - [risk_free + beta * (benchmark_return - risk_free)]

        Excess return above CAPM expected return

        Args:
            portfolio_returns_df: DataFrame with portfolio returns
            benchmark_returns_df: DataFrame with benchmark returns
            beta: Portfolio beta (calculates if None)
            risk_free_rate: Risk-free rate

        Returns:
            Jensen's alpha
        """
        if not isinstance(portfolio_returns_df, pl.DataFrame):
            raise TypeError(f"portfolio_returns_df must be polars DataFrame")

        if risk_free_rate is None:
            risk_free_rate = self.config.risk_free_rate_annual

        # Calculate returns
        portfolio_returns = [Decimal(str(r)) for r in portfolio_returns_df['return'].to_list()]
        mean_portfolio_return = sum(portfolio_returns) / Decimal(str(len(portfolio_returns)))
        annualized_portfolio_return = mean_portfolio_return * Decimal(str(self.config.trading_days_per_year))

        benchmark_returns = [Decimal(str(r)) for r in benchmark_returns_df['return'].to_list()]
        mean_benchmark_return = sum(benchmark_returns) / Decimal(str(len(benchmark_returns)))
        annualized_benchmark_return = mean_benchmark_return * Decimal(str(self.config.trading_days_per_year))

        # Calculate beta if not provided
        if beta is None:
            beta = self._calculate_beta(portfolio_returns_df, benchmark_returns_df)

        # Jensen's alpha
        expected_return = risk_free_rate + beta * (annualized_benchmark_return - risk_free_rate)
        alpha = annualized_portfolio_return - expected_return

        logger.info("Jensen's alpha calculated: %s%%", alpha * Decimal('100'))

        return alpha

    def _calculate_beta(
        self,
        portfolio_returns_df: pl.DataFrame,
        benchmark_returns_df: pl.DataFrame
    ) -> Decimal:
        """
        Calculate portfolio beta vs benchmark

        Args:
            portfolio_returns_df: Portfolio returns
            benchmark_returns_df: Benchmark returns

        Returns:
            Beta
        """
        # Join returns
        df = portfolio_returns_df.join(
            benchmark_returns_df,
            on='timestamp',
            how='inner',
            suffix='_benchmark'
        )

        if df.height < 2:
            return Decimal('1.0')

        # Convert to numpy arrays
        portfolio_returns = df['return'].to_numpy()
        benchmark_returns = df['return_benchmark'].to_numpy()

        # Calculate covariance and variance
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)

        if benchmark_variance == 0:
            return Decimal('1.0')

        beta = Decimal(str(covariance / benchmark_variance))

        return beta

    def _calculate_max_drawdown(self, returns_df: pl.DataFrame) -> Decimal:
        """
        Calculate maximum drawdown from returns

        Args:
            returns_df: DataFrame with returns

        Returns:
            Maximum drawdown percentage
        """
        returns = [Decimal(str(r)) for r in returns_df['return'].to_list()]

        # Calculate cumulative returns
        cumulative = Decimal('1')
        cumulative_returns = []

        for r in returns:
            cumulative = cumulative * (Decimal('1') + r)
            cumulative_returns.append(cumulative)

        if not cumulative_returns:
            return Decimal('0')

        # Calculate drawdowns
        max_drawdown = Decimal('0')
        peak = cumulative_returns[0]

        for value in cumulative_returns:
            if value > peak:
                peak = value

            if peak > Decimal('0'):
                drawdown = ((peak - value) / peak) * Decimal('100')
                max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def calculate_comprehensive_metrics(
        self,
        portfolio_returns_df: pl.DataFrame,
        benchmark_returns_df: Optional[pl.DataFrame] = None
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive risk-adjusted performance metrics

        Args:
            portfolio_returns_df: Portfolio returns DataFrame
            benchmark_returns_df: Optional benchmark returns

        Returns:
            Complete performance metrics
        """
        if not isinstance(portfolio_returns_df, pl.DataFrame):
            raise TypeError(f"portfolio_returns_df must be polars DataFrame")

        # Calculate basic metrics
        returns = [Decimal(str(r)) for r in portfolio_returns_df['return'].to_list()]

        if not returns:
            raise ValueError("No returns data provided")

        mean_return = sum(returns) / Decimal(str(len(returns)))
        annualized_return = mean_return * Decimal(str(self.config.trading_days_per_year))

        # Volatility
        variance = sum((r - mean_return) ** 2 for r in returns) / Decimal(str(len(returns)))
        volatility = Decimal(str(np.sqrt(float(variance))))
        annualized_volatility = volatility * Decimal(str(np.sqrt(self.config.trading_days_per_year)))

        # Total return
        cumulative = Decimal('1')
        for r in returns:
            cumulative = cumulative * (Decimal('1') + r)
        total_return = (cumulative - Decimal('1')) * Decimal('100')

        # Risk-adjusted metrics
        sharpe = self.calculate_sharpe_ratio(portfolio_returns_df)
        sortino = self.calculate_sortino_ratio(portfolio_returns_df)

        max_drawdown = self._calculate_max_drawdown(portfolio_returns_df)
        calmar = self.calculate_calmar_ratio(portfolio_returns_df, max_drawdown)

        omega = self.calculate_omega_ratio(portfolio_returns_df)

        # Downside deviation
        risk_free_period = self.config.risk_free_rate_annual / Decimal(str(self.config.trading_days_per_year))
        downside_returns = [min(r - risk_free_period, Decimal('0')) for r in returns]
        downside_variance = sum(r ** 2 for r in downside_returns) / Decimal(str(len(returns)))
        downside_deviation = Decimal(str(np.sqrt(float(downside_variance))))

        # Benchmark-based metrics
        if benchmark_returns_df is not None:
            beta = self._calculate_beta(portfolio_returns_df, benchmark_returns_df)
            information_ratio = self.calculate_information_ratio(portfolio_returns_df, benchmark_returns_df)
            treynor = self.calculate_treynor_ratio(portfolio_returns_df, benchmark_returns_df, beta)
            jensens_alpha = self.calculate_jensens_alpha(portfolio_returns_df, benchmark_returns_df, beta)

            # Tracking error
            df = portfolio_returns_df.join(benchmark_returns_df, on='timestamp', how='inner', suffix='_benchmark')
            excess_returns = []
            for row in df.iter_rows(named=True):
                excess = Decimal(str(row['return'])) - Decimal(str(row['return_benchmark']))
                excess_returns.append(excess)

            te_variance = sum((er - sum(excess_returns) / Decimal(str(len(excess_returns)))) ** 2
                            for er in excess_returns) / Decimal(str(len(excess_returns)))
            tracking_error = Decimal(str(np.sqrt(float(te_variance))))
        else:
            beta = None
            information_ratio = Decimal('0')
            treynor = Decimal('0')
            jensens_alpha = Decimal('0')
            tracking_error = None

        metrics = PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return * Decimal('100'),  # Convert to percentage
            annualized_volatility=annualized_volatility * Decimal('100'),
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=information_ratio,
            treynor_ratio=treynor,
            omega_ratio=omega,
            jensens_alpha=jensens_alpha * Decimal('100'),
            max_drawdown=max_drawdown,
            downside_deviation=downside_deviation * Decimal('100'),
            tracking_error=tracking_error * Decimal('100') if tracking_error else None,
            beta=beta,
            num_periods=len(returns),
            timestamp=datetime.now()
        )

        logger.info(
            "Comprehensive metrics calculated: Sharpe=%s, Sortino=%s, Calmar=%s",
            sharpe, sortino, calmar
        )

        return metrics
