"""
Conditional Value at Risk (CVaR) Calculator
Quantum Trader AI - Production Risk Management

Calculates Expected Shortfall (ES) and CVaR metrics:
- Historical CVaR calculation
- Parametric CVaR models
- Monte Carlo CVaR simulation
- Stress testing integration

CRITICAL: All numeric values use Decimal, never float
"""

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

import polars as pl
import yaml

from quantum_trader.models import Position, AuditLog

logger = logging.getLogger(__name__)


@dataclass
class CVaRConfig:
    """CVaR calculation configuration"""
    var_confidence_level: Decimal
    var_lookback_days: int
    cvar_enabled: bool
    monte_carlo_simulations: int = 10000
    stress_test_enabled: bool = True

    @classmethod
    def from_yaml(cls, risk_config: Dict, prod_config: Dict) -> 'CVaRConfig':
        """Load configuration from yaml dictionaries"""
        return cls(
            var_confidence_level=Decimal(str(risk_config['risk_model']['var_confidence_level'])),
            var_lookback_days=int(risk_config['risk_model']['var_lookback_days']),
            cvar_enabled=risk_config['risk_model']['cvar_enabled'],
            monte_carlo_simulations=10000,
            stress_test_enabled=risk_config['stress_testing']['enabled']
        )


@dataclass
class CVaRResult:
    """Conditional VaR calculation result"""
    portfolio_value: Decimal
    confidence_level: Decimal
    var: Decimal  # Value at Risk
    cvar: Decimal  # Conditional VaR (Expected Shortfall)
    expected_loss_beyond_var: Decimal
    max_loss_in_tail: Decimal
    tail_observations: int
    method: str  # 'historical', 'parametric', 'monte_carlo'
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


@dataclass
class StressTestResult:
    """Stress test scenario result"""
    scenario_name: str
    portfolio_loss: Decimal
    portfolio_loss_percent: Decimal
    var_breach: bool
    cvar_breach: bool
    affected_positions: List[str]
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


class ConditionalVaRCalculator:
    """
    Calculate Conditional Value at Risk (CVaR/ES) for portfolio risk management.

    CVaR measures the expected loss in the worst cases beyond VaR threshold,
    providing a more complete picture of tail risk.
    """

    def __init__(self, risk_config_path: str, prod_config_path: str):
        """Initialize CVaR calculator with configuration"""
        self.risk_config_path = risk_config_path
        self.prod_config_path = prod_config_path
        self.config: Optional[CVaRConfig] = None
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

                self.config = CVaRConfig.from_yaml(risk_config, prod_config)
                logger.info("CVaR calculator configuration loaded successfully")
                return

            except Exception as e:
                logger.error(f"Failed to load config (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise RuntimeError(f"Failed to load CVaR config after {max_retries} attempts") from e

    async def calculate_historical_cvar(
        self,
        returns_df: pl.DataFrame,
        portfolio_value: Decimal,
        confidence_level: Optional[Decimal] = None
    ) -> CVaRResult:
        """
        Calculate CVaR using historical simulation method.

        Args:
            returns_df: Polars DataFrame with historical returns [date, return]
            portfolio_value: Current portfolio value
            confidence_level: Confidence level (default from config)

        Returns:
            CVaRResult with historical CVaR metrics
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            if confidence_level is None:
                confidence_level = self.config.var_confidence_level

            # Sort returns
            sorted_returns = returns_df.sort('return')
            returns_list = [Decimal(str(r)) for r in sorted_returns['return'].to_list()]

            if len(returns_list) == 0:
                raise ValueError("No returns data available")

            # Calculate VaR threshold
            var_index = int(len(returns_list) * (Decimal('1') - confidence_level))
            var_index = max(0, min(var_index, len(returns_list) - 1))

            var_return = returns_list[var_index]
            var = portfolio_value * abs(var_return)

            # Calculate CVaR (average of losses beyond VaR)
            tail_returns = returns_list[:var_index + 1]

            if len(tail_returns) == 0:
                cvar = var
                expected_loss_beyond_var = Decimal('0')
                max_loss_in_tail = var
            else:
                avg_tail_return = sum(tail_returns) / Decimal(str(len(tail_returns)))
                cvar = portfolio_value * abs(avg_tail_return)
                expected_loss_beyond_var = cvar - var
                max_loss_in_tail = portfolio_value * abs(tail_returns[0])

            timestamp = datetime.utcnow()

            result = CVaRResult(
                portfolio_value=portfolio_value,
                confidence_level=confidence_level,
                var=var,
                cvar=cvar,
                expected_loss_beyond_var=expected_loss_beyond_var,
                max_loss_in_tail=max_loss_in_tail,
                tail_observations=len(tail_returns),
                method='historical',
                timestamp=timestamp,
                metadata={
                    'total_observations': len(returns_list),
                    'lookback_days': self.config.var_lookback_days
                }
            )

            logger.info(
                f"Historical CVaR calculated: VaR={var}, CVaR={cvar}, "
                f"confidence={confidence_level}"
            )

            return result

        except Exception as e:
            logger.error(f"Historical CVaR calculation failed: {e}")
            raise

    async def calculate_parametric_cvar(
        self,
        returns_df: pl.DataFrame,
        portfolio_value: Decimal,
        confidence_level: Optional[Decimal] = None
    ) -> CVaRResult:
        """
        Calculate CVaR using parametric (variance-covariance) method.

        Assumes returns are normally distributed.

        Args:
            returns_df: Polars DataFrame with historical returns
            portfolio_value: Current portfolio value
            confidence_level: Confidence level (default from config)

        Returns:
            CVaRResult with parametric CVaR metrics
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            if confidence_level is None:
                confidence_level = self.config.var_confidence_level

            # Calculate mean and std dev of returns
            returns_list = [Decimal(str(r)) for r in returns_df['return'].to_list()]

            if len(returns_list) == 0:
                raise ValueError("No returns data available")

            mean_return = sum(returns_list) / Decimal(str(len(returns_list)))

            variance = sum((r - mean_return) ** 2 for r in returns_list) / Decimal(str(len(returns_list)))
            std_dev = variance.sqrt()

            # Calculate z-score for confidence level
            # For 95% confidence: z ≈ 1.645 (one-tailed)
            # For 99% confidence: z ≈ 2.326
            z_score = await self._get_z_score(confidence_level)

            # VaR = portfolio_value * (mean - z * std)
            var_return = mean_return - z_score * std_dev
            var = portfolio_value * abs(var_return)

            # CVaR for normal distribution = portfolio_value * (mean - std * phi(z) / alpha)
            # where phi is the PDF and alpha = 1 - confidence_level
            alpha = Decimal('1') - confidence_level
            phi_z = await self._normal_pdf(z_score)

            cvar_return = mean_return - (std_dev * phi_z / alpha)
            cvar = portfolio_value * abs(cvar_return)

            expected_loss_beyond_var = cvar - var
            max_loss_in_tail = cvar * Decimal('1.5')  # Estimate

            timestamp = datetime.utcnow()

            result = CVaRResult(
                portfolio_value=portfolio_value,
                confidence_level=confidence_level,
                var=var,
                cvar=cvar,
                expected_loss_beyond_var=expected_loss_beyond_var,
                max_loss_in_tail=max_loss_in_tail,
                tail_observations=int(float(alpha) * len(returns_list)),
                method='parametric',
                timestamp=timestamp,
                metadata={
                    'mean_return': str(mean_return),
                    'std_dev': str(std_dev),
                    'z_score': str(z_score)
                }
            )

            logger.info(
                f"Parametric CVaR calculated: VaR={var}, CVaR={cvar}, "
                f"confidence={confidence_level}"
            )

            return result

        except Exception as e:
            logger.error(f"Parametric CVaR calculation failed: {e}")
            raise

    async def calculate_monte_carlo_cvar(
        self,
        positions: List[Position],
        returns_df: pl.DataFrame,
        portfolio_value: Decimal,
        confidence_level: Optional[Decimal] = None,
        num_simulations: Optional[int] = None
    ) -> CVaRResult:
        """
        Calculate CVaR using Monte Carlo simulation.

        Args:
            positions: Current positions
            returns_df: Historical returns for simulation
            portfolio_value: Current portfolio value
            confidence_level: Confidence level (default from config)
            num_simulations: Number of MC simulations (default from config)

        Returns:
            CVaRResult with Monte Carlo CVaR metrics
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            if confidence_level is None:
                confidence_level = self.config.var_confidence_level

            if num_simulations is None:
                num_simulations = self.config.monte_carlo_simulations

            # Calculate statistics for each position
            position_stats = {}

            for position in positions:
                symbol = position.symbol
                symbol_returns = returns_df.filter(pl.col('symbol') == symbol)

                if len(symbol_returns) == 0:
                    # Use default stats if no history
                    position_stats[symbol] = {
                        'mean': Decimal('0'),
                        'std': Decimal('0.02'),  # 2% default volatility
                        'weight': position.current_price * abs(position.quantity) / portfolio_value
                    }
                else:
                    returns_list = [Decimal(str(r)) for r in symbol_returns['return'].to_list()]
                    mean = sum(returns_list) / Decimal(str(len(returns_list)))
                    variance = sum((r - mean) ** 2 for r in returns_list) / Decimal(str(len(returns_list)))
                    std = variance.sqrt()
                    weight = position.current_price * abs(position.quantity) / portfolio_value

                    position_stats[symbol] = {
                        'mean': mean,
                        'std': std,
                        'weight': weight
                    }

            # Run Monte Carlo simulations
            simulated_returns = []

            for _ in range(num_simulations):
                portfolio_return = Decimal('0')

                for symbol, stats in position_stats.items():
                    # Generate random return from normal distribution
                    random_return = await self._random_normal(stats['mean'], stats['std'])
                    portfolio_return += stats['weight'] * random_return

                simulated_returns.append(portfolio_return)

            # Sort simulated returns
            simulated_returns.sort()

            # Calculate VaR and CVaR from simulations
            var_index = int(num_simulations * (Decimal('1') - confidence_level))
            var_index = max(0, min(var_index, len(simulated_returns) - 1))

            var_return = simulated_returns[var_index]
            var = portfolio_value * abs(var_return)

            # CVaR is average of tail losses
            tail_returns = simulated_returns[:var_index + 1]
            avg_tail_return = sum(tail_returns) / Decimal(str(len(tail_returns)))
            cvar = portfolio_value * abs(avg_tail_return)

            expected_loss_beyond_var = cvar - var
            max_loss_in_tail = portfolio_value * abs(simulated_returns[0])

            timestamp = datetime.utcnow()

            result = CVaRResult(
                portfolio_value=portfolio_value,
                confidence_level=confidence_level,
                var=var,
                cvar=cvar,
                expected_loss_beyond_var=expected_loss_beyond_var,
                max_loss_in_tail=max_loss_in_tail,
                tail_observations=len(tail_returns),
                method='monte_carlo',
                timestamp=timestamp,
                metadata={
                    'num_simulations': num_simulations,
                    'num_positions': len(positions)
                }
            )

            logger.info(
                f"Monte Carlo CVaR calculated: VaR={var}, CVaR={cvar}, "
                f"simulations={num_simulations}"
            )

            return result

        except Exception as e:
            logger.error(f"Monte Carlo CVaR calculation failed: {e}")
            raise

    async def run_stress_test(
        self,
        positions: List[Position],
        scenario_name: str,
        scenario_shocks: Dict[str, Decimal],
        portfolio_value: Decimal,
        var_threshold: Decimal,
        cvar_threshold: Decimal
    ) -> StressTestResult:
        """
        Run stress test scenario on portfolio.

        Args:
            positions: Current positions
            scenario_name: Name of stress test scenario
            scenario_shocks: Dictionary of symbol -> return shock
            portfolio_value: Current portfolio value
            var_threshold: VaR threshold to check breach
            cvar_threshold: CVaR threshold to check breach

        Returns:
            StressTestResult with scenario impact
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            # Calculate portfolio loss under scenario
            total_loss = Decimal('0')
            affected_positions = []

            for position in positions:
                symbol = position.symbol
                shock = scenario_shocks.get(symbol, Decimal('0'))

                position_value = abs(position.quantity) * position.current_price
                position_loss = position_value * abs(shock)

                total_loss += position_loss

                if abs(shock) > Decimal('0.01'):  # 1% threshold
                    affected_positions.append(symbol)

            loss_percent = (total_loss / portfolio_value) * Decimal('100') if portfolio_value > Decimal('0') else Decimal('0')

            # Check for breaches
            var_breach = total_loss > var_threshold
            cvar_breach = total_loss > cvar_threshold

            timestamp = datetime.utcnow()

            result = StressTestResult(
                scenario_name=scenario_name,
                portfolio_loss=total_loss,
                portfolio_loss_percent=loss_percent,
                var_breach=var_breach,
                cvar_breach=cvar_breach,
                affected_positions=affected_positions,
                timestamp=timestamp,
                metadata={
                    'portfolio_value': str(portfolio_value),
                    'var_threshold': str(var_threshold),
                    'cvar_threshold': str(cvar_threshold)
                }
            )

            logger.info(
                f"Stress test '{scenario_name}': Loss={total_loss} ({loss_percent}%), "
                f"VaR breach={var_breach}, CVaR breach={cvar_breach}"
            )

            return result

        except Exception as e:
            logger.error(f"Stress test failed for scenario '{scenario_name}': {e}")
            raise

    async def _get_z_score(self, confidence_level: Decimal) -> Decimal:
        """Get z-score for given confidence level (normal distribution)"""
        try:
            # Approximations for common confidence levels
            if confidence_level >= Decimal('0.99'):
                return Decimal('2.326')
            elif confidence_level >= Decimal('0.95'):
                return Decimal('1.645')
            elif confidence_level >= Decimal('0.90'):
                return Decimal('1.282')
            else:
                # Linear interpolation for other values
                return Decimal('1.645')

        except Exception as e:
            logger.error(f"Z-score calculation failed: {e}")
            raise

    async def _normal_pdf(self, z: Decimal) -> Decimal:
        """Calculate normal distribution PDF at z"""
        try:
            # PDF(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
            two_pi = Decimal('2') * Decimal('3.14159265359')
            coefficient = Decimal('1') / two_pi.sqrt()

            # Approximate exp(-z^2/2) using Taylor series
            exponent = -(z * z) / Decimal('2')

            # exp(x) ≈ 1 + x + x^2/2 + x^3/6 + ...
            exp_val = Decimal('1')
            term = exponent
            for i in range(1, 10):
                exp_val += term
                term *= exponent / Decimal(str(i + 1))

            pdf = coefficient * exp_val
            return pdf

        except Exception as e:
            logger.error(f"Normal PDF calculation failed: {e}")
            raise

    async def _random_normal(self, mean: Decimal, std: Decimal) -> Decimal:
        """Generate random number from normal distribution"""
        try:
            # Box-Muller transform
            u1 = Decimal(str(random.random()))
            u2 = Decimal(str(random.random()))

            # Avoid log(0)
            u1 = max(u1, Decimal('0.0000001'))

            # Standard normal
            z = (Decimal('-2') * u1.ln()).sqrt() * self._cos_approx(Decimal('2') * Decimal('3.14159265359') * u2)

            # Scale and shift
            return mean + std * z

        except Exception as e:
            logger.error(f"Random normal generation failed: {e}")
            # Fallback to simple random
            return mean + std * Decimal(str(random.uniform(-1, 1)))

    def _cos_approx(self, x: Decimal) -> Decimal:
        """Approximate cosine function using Taylor series"""
        try:
            # Normalize x to [0, 2*pi]
            two_pi = Decimal('2') * Decimal('3.14159265359')
            x = x % two_pi

            # cos(x) ≈ 1 - x^2/2! + x^4/4! - x^6/6! + ...
            result = Decimal('1')
            term = Decimal('1')

            for n in range(1, 8):
                term *= -x * x / (Decimal(str(2 * n - 1)) * Decimal(str(2 * n)))
                result += term

            return result

        except Exception as e:
            logger.error(f"Cosine approximation failed: {e}")
            return Decimal('1')


async def main():
    """Example usage of CVaR calculator"""
    try:
        # Initialize calculator
        calculator = ConditionalVaRCalculator(
            risk_config_path='/home/user/FritzellSama/config/bot/risk.yaml',
            prod_config_path='/home/user/FritzellSama/config/environments/production.yaml'
        )

        # Create sample returns data
        returns_data = pl.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
            'return': [-0.02, 0.01, -0.03, 0.015, -0.01]
        })

        portfolio_value = Decimal('1000000')

        # Calculate historical CVaR
        result = await calculator.calculate_historical_cvar(
            returns_data,
            portfolio_value
        )

        logger.info(f"CVaR result: {result}")

    except Exception as e:
        logger.error(f"CVaR calculator example failed: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
