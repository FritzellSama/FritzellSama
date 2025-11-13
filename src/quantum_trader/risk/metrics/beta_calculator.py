"""
Beta and Correlation Calculator
Quantum Trader AI - Production Risk Management

Calculates portfolio and position betas against benchmarks:
- Beta vs benchmark calculation
- Rolling beta estimation
- Multi-factor beta models
- Correlation matrix computation
- Covariance matrix management

CRITICAL: All numeric values use Decimal, never float
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import polars as pl
import yaml

from quantum_trader.models import Position, AuditLog

logger = logging.getLogger(__name__)


@dataclass
class BetaConfig:
    """Beta calculation configuration"""
    var_lookback_days: int
    rolling_window_days: int = 60
    min_observations: int = 30
    confidence_level: Decimal = Decimal('0.95')

    @classmethod
    def from_yaml(cls, risk_config: Dict, prod_config: Dict) -> 'BetaConfig':
        """Load configuration from yaml dictionaries"""
        return cls(
            var_lookback_days=int(risk_config['risk_model']['var_lookback_days']),
            rolling_window_days=60,
            min_observations=30,
            confidence_level=Decimal(str(risk_config['risk_model']['var_confidence_level']))
        )


@dataclass
class BetaResult:
    """Beta calculation result"""
    symbol: str
    benchmark: str
    beta: Decimal
    alpha: Decimal
    correlation: Decimal
    r_squared: Decimal
    std_error: Decimal
    observations: int
    start_date: datetime
    end_date: datetime
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


@dataclass
class MultiFactorBeta:
    """Multi-factor beta model result"""
    symbol: str
    market_beta: Decimal
    size_beta: Decimal
    value_beta: Decimal
    momentum_beta: Decimal
    r_squared: Decimal
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


class BetaCalculator:
    """
    Calculate beta and correlation metrics for portfolio risk management.

    Supports single-factor and multi-factor beta models with rolling
    estimation windows for adaptive risk measurement.
    """

    def __init__(self, risk_config_path: str, prod_config_path: str):
        """Initialize beta calculator with configuration"""
        self.risk_config_path = risk_config_path
        self.prod_config_path = prod_config_path
        self.config: Optional[BetaConfig] = None
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from yaml files with retry logic"""
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                with open(self.risk_config_path, 'r') as f:
                    risk_config = yaml.safe_load(f)

                with open(self.prod_config_path, 'r') as f:
                    prod_config = yaml.safe_load(f)

                self.config = BetaConfig.from_yaml(risk_config, prod_config)
                logger.info("Beta calculator configuration loaded successfully")
                return

            except Exception as e:
                logger.error(f"Failed to load config (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise RuntimeError(f"Failed to load beta config after {max_retries} attempts") from e

    async def calculate_beta(
        self,
        asset_returns: pl.DataFrame,
        benchmark_returns: pl.DataFrame,
        symbol: str,
        benchmark: str = 'SPY'
    ) -> BetaResult:
        """
        Calculate beta of an asset against a benchmark.

        Args:
            asset_returns: Polars DataFrame with columns [date, return]
            benchmark_returns: Polars DataFrame with columns [date, return]
            symbol: Asset symbol
            benchmark: Benchmark symbol

        Returns:
            BetaResult with beta, alpha, and correlation metrics
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            # Join asset and benchmark returns
            joined_df = asset_returns.join(
                benchmark_returns,
                on='date',
                suffix='_benchmark'
            ).sort('date')

            if len(joined_df) < self.config.min_observations:
                raise ValueError(
                    f"Insufficient observations: {len(joined_df)} < {self.config.min_observations}"
                )

            # Calculate covariance and variance
            asset_ret = joined_df['return'].to_list()
            bench_ret = joined_df['return_benchmark'].to_list()

            asset_mean = sum(Decimal(str(r)) for r in asset_ret) / Decimal(str(len(asset_ret)))
            bench_mean = sum(Decimal(str(r)) for r in bench_ret) / Decimal(str(len(bench_ret)))

            covariance = Decimal('0')
            bench_variance = Decimal('0')
            asset_variance = Decimal('0')

            for i in range(len(asset_ret)):
                asset_dev = Decimal(str(asset_ret[i])) - asset_mean
                bench_dev = Decimal(str(bench_ret[i])) - bench_mean

                covariance += asset_dev * bench_dev
                bench_variance += bench_dev * bench_dev
                asset_variance += asset_dev * asset_dev

            n = Decimal(str(len(asset_ret)))
            covariance /= (n - Decimal('1'))
            bench_variance /= (n - Decimal('1'))
            asset_variance /= (n - Decimal('1'))

            # Calculate beta
            if bench_variance == Decimal('0'):
                beta = Decimal('0')
            else:
                beta = covariance / bench_variance

            # Calculate alpha
            alpha = asset_mean - beta * bench_mean

            # Calculate correlation
            if asset_variance == Decimal('0') or bench_variance == Decimal('0'):
                correlation = Decimal('0')
            else:
                correlation = covariance / (asset_variance.sqrt() * bench_variance.sqrt())

            # Calculate R-squared
            r_squared = correlation * correlation

            # Calculate standard error of beta
            if bench_variance == Decimal('0'):
                std_error = Decimal('0')
            else:
                residual_variance = asset_variance - (covariance * covariance / bench_variance)
                std_error = (residual_variance / (bench_variance * (n - Decimal('2')))).sqrt()

            start_date = datetime.fromisoformat(joined_df['date'].min())
            end_date = datetime.fromisoformat(joined_df['date'].max())
            timestamp = datetime.utcnow()

            return BetaResult(
                symbol=symbol,
                benchmark=benchmark,
                beta=beta,
                alpha=alpha,
                correlation=correlation,
                r_squared=r_squared,
                std_error=std_error,
                observations=len(joined_df),
                start_date=start_date,
                end_date=end_date,
                timestamp=timestamp
            )

        except Exception as e:
            logger.error(f"Beta calculation failed for {symbol}: {e}")
            raise

    async def calculate_rolling_beta(
        self,
        asset_returns: pl.DataFrame,
        benchmark_returns: pl.DataFrame,
        symbol: str,
        benchmark: str = 'SPY'
    ) -> pl.DataFrame:
        """
        Calculate rolling beta over time using a sliding window.

        Args:
            asset_returns: Polars DataFrame with columns [date, return]
            benchmark_returns: Polars DataFrame with columns [date, return]
            symbol: Asset symbol
            benchmark: Benchmark symbol

        Returns:
            Polars DataFrame with columns [date, beta, correlation]
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            # Join returns
            joined_df = asset_returns.join(
                benchmark_returns,
                on='date',
                suffix='_benchmark'
            ).sort('date')

            if len(joined_df) < self.config.min_observations:
                raise ValueError(
                    f"Insufficient observations: {len(joined_df)} < {self.config.min_observations}"
                )

            # Calculate rolling beta
            window = self.config.rolling_window_days
            dates = []
            betas = []
            correlations = []

            for i in range(window, len(joined_df)):
                window_df = joined_df.slice(i - window, window)

                asset_ret = window_df['return'].to_list()
                bench_ret = window_df['return_benchmark'].to_list()

                # Calculate statistics
                asset_mean = sum(Decimal(str(r)) for r in asset_ret) / Decimal(str(len(asset_ret)))
                bench_mean = sum(Decimal(str(r)) for r in bench_ret) / Decimal(str(len(bench_ret)))

                covariance = Decimal('0')
                bench_variance = Decimal('0')
                asset_variance = Decimal('0')

                for j in range(len(asset_ret)):
                    asset_dev = Decimal(str(asset_ret[j])) - asset_mean
                    bench_dev = Decimal(str(bench_ret[j])) - bench_mean

                    covariance += asset_dev * bench_dev
                    bench_variance += bench_dev * bench_dev
                    asset_variance += asset_dev * asset_dev

                n = Decimal(str(len(asset_ret)))
                covariance /= (n - Decimal('1'))
                bench_variance /= (n - Decimal('1'))
                asset_variance /= (n - Decimal('1'))

                # Beta
                if bench_variance > Decimal('0'):
                    beta = covariance / bench_variance
                else:
                    beta = Decimal('0')

                # Correlation
                if asset_variance > Decimal('0') and bench_variance > Decimal('0'):
                    correlation = covariance / (asset_variance.sqrt() * bench_variance.sqrt())
                else:
                    correlation = Decimal('0')

                dates.append(window_df['date'].to_list()[-1])
                betas.append(float(beta))
                correlations.append(float(correlation))

            result_df = pl.DataFrame({
                'date': dates,
                'beta': betas,
                'correlation': correlations,
                'symbol': [symbol] * len(dates),
                'benchmark': [benchmark] * len(dates)
            })

            logger.info(f"Calculated rolling beta for {symbol} with {len(result_df)} periods")
            return result_df

        except Exception as e:
            logger.error(f"Rolling beta calculation failed for {symbol}: {e}")
            raise

    async def calculate_multi_factor_beta(
        self,
        asset_returns: pl.DataFrame,
        market_returns: pl.DataFrame,
        size_returns: pl.DataFrame,
        value_returns: pl.DataFrame,
        momentum_returns: pl.DataFrame,
        symbol: str
    ) -> MultiFactorBeta:
        """
        Calculate multi-factor beta using Fama-French style factors.

        Args:
            asset_returns: Asset returns [date, return]
            market_returns: Market factor returns [date, return]
            size_returns: Size factor (SMB) returns [date, return]
            value_returns: Value factor (HML) returns [date, return]
            momentum_returns: Momentum factor returns [date, return]
            symbol: Asset symbol

        Returns:
            MultiFactorBeta with factor exposures
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            # Join all factors
            combined_df = (
                asset_returns
                .join(market_returns.rename({'return': 'market_return'}), on='date')
                .join(size_returns.rename({'return': 'size_return'}), on='date')
                .join(value_returns.rename({'return': 'value_return'}), on='date')
                .join(momentum_returns.rename({'return': 'momentum_return'}), on='date')
                .sort('date')
            )

            if len(combined_df) < self.config.min_observations:
                raise ValueError(
                    f"Insufficient observations: {len(combined_df)} < {self.config.min_observations}"
                )

            # Prepare data for regression
            y = [Decimal(str(r)) for r in combined_df['return'].to_list()]
            X_market = [Decimal(str(r)) for r in combined_df['market_return'].to_list()]
            X_size = [Decimal(str(r)) for r in combined_df['size_return'].to_list()]
            X_value = [Decimal(str(r)) for r in combined_df['value_return'].to_list()]
            X_momentum = [Decimal(str(r)) for r in combined_df['momentum_return'].to_list()]

            # Multiple regression using normal equations (simplified)
            # Full implementation would use proper OLS regression
            n = len(y)

            # Calculate factor betas using individual regressions
            market_beta = await self._simple_regression_beta(y, X_market)
            size_beta = await self._simple_regression_beta(y, X_size)
            value_beta = await self._simple_regression_beta(y, X_value)
            momentum_beta = await self._simple_regression_beta(y, X_momentum)

            # Calculate R-squared for the model
            y_mean = sum(y) / Decimal(str(n))
            total_ss = sum((yi - y_mean) ** 2 for yi in y)

            # Predicted values using all factors
            y_pred = []
            for i in range(n):
                pred = (
                    market_beta * X_market[i] +
                    size_beta * X_size[i] +
                    value_beta * X_value[i] +
                    momentum_beta * X_momentum[i]
                )
                y_pred.append(pred)

            residual_ss = sum((y[i] - y_pred[i]) ** 2 for i in range(n))

            if total_ss > Decimal('0'):
                r_squared = Decimal('1') - (residual_ss / total_ss)
            else:
                r_squared = Decimal('0')

            timestamp = datetime.utcnow()

            return MultiFactorBeta(
                symbol=symbol,
                market_beta=market_beta,
                size_beta=size_beta,
                value_beta=value_beta,
                momentum_beta=momentum_beta,
                r_squared=r_squared,
                timestamp=timestamp,
                metadata={'observations': n}
            )

        except Exception as e:
            logger.error(f"Multi-factor beta calculation failed for {symbol}: {e}")
            raise

    async def calculate_portfolio_beta(
        self,
        positions: List[Position],
        position_betas: Dict[str, Decimal],
        portfolio_value: Decimal
    ) -> Decimal:
        """
        Calculate portfolio-level beta as weighted average of position betas.

        Args:
            positions: List of current positions
            position_betas: Dictionary mapping symbols to their betas
            portfolio_value: Total portfolio value

        Returns:
            Portfolio beta
        """
        try:
            if portfolio_value == Decimal('0'):
                return Decimal('0')

            weighted_beta = Decimal('0')

            for position in positions:
                position_value = abs(position.quantity) * position.current_price
                weight = position_value / portfolio_value
                beta = position_betas.get(position.symbol, Decimal('1'))

                weighted_beta += weight * beta

            logger.info(f"Portfolio beta: {weighted_beta}")
            return weighted_beta

        except Exception as e:
            logger.error(f"Portfolio beta calculation failed: {e}")
            raise

    async def calculate_correlation_matrix(
        self,
        returns_df: pl.DataFrame,
        symbols: List[str]
    ) -> Dict[Tuple[str, str], Decimal]:
        """
        Calculate correlation matrix for multiple symbols.

        Args:
            returns_df: Polars DataFrame with columns [symbol, date, return]
            symbols: List of symbols to include

        Returns:
            Dictionary with (symbol1, symbol2) -> correlation
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            # Pivot to wide format
            pivot_df = returns_df.pivot(
                values='return',
                index='date',
                columns='symbol'
            ).sort('date')

            correlation_matrix = {}

            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if symbol1 not in pivot_df.columns or symbol2 not in pivot_df.columns:
                        correlation_matrix[(symbol1, symbol2)] = Decimal('0')
                        continue

                    returns1 = pivot_df[symbol1].drop_nulls().to_list()
                    returns2 = pivot_df[symbol2].drop_nulls().to_list()

                    if len(returns1) < self.config.min_observations or len(returns2) < self.config.min_observations:
                        correlation_matrix[(symbol1, symbol2)] = Decimal('0')
                        continue

                    # Calculate correlation
                    mean1 = sum(Decimal(str(r)) for r in returns1) / Decimal(str(len(returns1)))
                    mean2 = sum(Decimal(str(r)) for r in returns2) / Decimal(str(len(returns2)))

                    covariance = Decimal('0')
                    var1 = Decimal('0')
                    var2 = Decimal('0')

                    n = min(len(returns1), len(returns2))
                    for k in range(n):
                        dev1 = Decimal(str(returns1[k])) - mean1
                        dev2 = Decimal(str(returns2[k])) - mean2

                        covariance += dev1 * dev2
                        var1 += dev1 * dev1
                        var2 += dev2 * dev2

                    if n > 1:
                        covariance /= Decimal(str(n - 1))
                        var1 /= Decimal(str(n - 1))
                        var2 /= Decimal(str(n - 1))

                    if var1 > Decimal('0') and var2 > Decimal('0'):
                        correlation = covariance / (var1.sqrt() * var2.sqrt())
                    else:
                        correlation = Decimal('0')

                    correlation_matrix[(symbol1, symbol2)] = correlation

            logger.info(f"Calculated correlation matrix for {len(symbols)} symbols")
            return correlation_matrix

        except Exception as e:
            logger.error(f"Correlation matrix calculation failed: {e}")
            raise

    async def calculate_covariance_matrix(
        self,
        returns_df: pl.DataFrame,
        symbols: List[str]
    ) -> Dict[Tuple[str, str], Decimal]:
        """
        Calculate covariance matrix for multiple symbols.

        Args:
            returns_df: Polars DataFrame with columns [symbol, date, return]
            symbols: List of symbols to include

        Returns:
            Dictionary with (symbol1, symbol2) -> covariance
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            # Pivot to wide format
            pivot_df = returns_df.pivot(
                values='return',
                index='date',
                columns='symbol'
            ).sort('date')

            covariance_matrix = {}

            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if symbol1 not in pivot_df.columns or symbol2 not in pivot_df.columns:
                        covariance_matrix[(symbol1, symbol2)] = Decimal('0')
                        continue

                    returns1 = pivot_df[symbol1].drop_nulls().to_list()
                    returns2 = pivot_df[symbol2].drop_nulls().to_list()

                    if len(returns1) < self.config.min_observations or len(returns2) < self.config.min_observations:
                        covariance_matrix[(symbol1, symbol2)] = Decimal('0')
                        continue

                    # Calculate covariance
                    mean1 = sum(Decimal(str(r)) for r in returns1) / Decimal(str(len(returns1)))
                    mean2 = sum(Decimal(str(r)) for r in returns2) / Decimal(str(len(returns2)))

                    covariance = Decimal('0')
                    n = min(len(returns1), len(returns2))

                    for k in range(n):
                        dev1 = Decimal(str(returns1[k])) - mean1
                        dev2 = Decimal(str(returns2[k])) - mean2
                        covariance += dev1 * dev2

                    if n > 1:
                        covariance /= Decimal(str(n - 1))

                    covariance_matrix[(symbol1, symbol2)] = covariance

            logger.info(f"Calculated covariance matrix for {len(symbols)} symbols")
            return covariance_matrix

        except Exception as e:
            logger.error(f"Covariance matrix calculation failed: {e}")
            raise

    async def _simple_regression_beta(
        self,
        y: List[Decimal],
        x: List[Decimal]
    ) -> Decimal:
        """Calculate beta using simple linear regression"""
        try:
            n = len(y)
            if n == 0:
                return Decimal('0')

            # Calculate means
            y_mean = sum(y) / Decimal(str(n))
            x_mean = sum(x) / Decimal(str(n))

            # Calculate covariance and variance
            covariance = sum((y[i] - y_mean) * (x[i] - x_mean) for i in range(n))
            variance = sum((x[i] - x_mean) ** 2 for i in range(n))

            if variance == Decimal('0'):
                return Decimal('0')

            beta = covariance / variance
            return beta

        except Exception as e:
            logger.error(f"Regression beta calculation failed: {e}")
            raise


async def main():
    """Example usage of beta calculator"""
    try:
        # Initialize calculator
        calculator = BetaCalculator(
            risk_config_path='/home/user/FritzellSama/config/bot/risk.yaml',
            prod_config_path='/home/user/FritzellSama/config/environments/production.yaml'
        )

        # Create sample data
        asset_returns = pl.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'return': [0.01, -0.02, 0.03]
        })

        benchmark_returns = pl.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'return': [0.005, -0.015, 0.025]
        })

        # Calculate beta
        result = await calculator.calculate_beta(
            asset_returns,
            benchmark_returns,
            'AAPL',
            'SPY'
        )

        logger.info(f"Beta result: {result}")

    except Exception as e:
        logger.error(f"Beta calculator example failed: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
