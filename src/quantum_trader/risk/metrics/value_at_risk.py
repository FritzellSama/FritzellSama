"""
Quantum Trader AI - Value at Risk (VaR) Calculator
Production-grade VaR calculations with multiple methodologies

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


class ValueAtRiskCalculator:
    """
    Value at Risk (VaR) calculation system

    Features:
    - Historical VaR
    - Parametric VaR (variance-covariance)
    - Monte Carlo VaR
    - Backtesting VaR accuracy
    """

    def __init__(
        self,
        config_path: Path = Path("/home/user/FritzellSama/config/bot/risk.yaml"),
        env_config_path: Path = Path("/home/user/FritzellSama/config/environments/production.yaml")
    ) -> None:
        """Initialize VaR calculator with configuration"""
        self.config = self._load_config(config_path)
        self.env_config = self._load_config(env_config_path)

        # Risk model parameters
        risk_model = self.config.get("risk_model", {})
        self.var_confidence_level = Decimal(str(risk_model.get("var_confidence_level", 0.95)))
        self.var_lookback_days = int(risk_model.get("var_lookback_days", 252))
        self.cvar_enabled = risk_model.get("cvar_enabled", True)

        # VaR calculation parameters
        self.confidence_levels = [
            Decimal('0.90'),  # 90% confidence
            Decimal('0.95'),  # 95% confidence
            Decimal('0.99')   # 99% confidence
        ]

        # Monte Carlo parameters
        self.mc_simulations = 10000
        self.mc_time_horizon = 1  # 1 day

        # Backtesting parameters
        self.backtest_window = 250  # Days for backtesting
        self.backtest_threshold = Decimal('0.05')  # 5% violation rate acceptable

        # Retry configuration
        self.retry_attempts = 3
        self.retry_delay_ms = 1000

        logger.info("ValueAtRiskCalculator initialized")

    def _load_config(self, config_path: Path) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}

    def calculate_historical_var(
        self,
        returns: pl.Series,
        confidence_level: Optional[Decimal] = None,
        portfolio_value: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate Historical VaR using empirical distribution

        Historical VaR uses actual historical returns to estimate VaR.
        Most straightforward method, no distributional assumptions.

        Args:
            returns: Series of historical returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            portfolio_value: Portfolio value to scale VaR to dollar amount

        Returns:
            VaR value (as percentage or dollar amount)
        """
        try:
            if len(returns) == 0:
                logger.warning("No returns data for Historical VaR")
                return Decimal('0')

            conf_level = confidence_level if confidence_level is not None else self.var_confidence_level

            # Sort returns in ascending order
            returns_array = returns.to_numpy()
            sorted_returns = np.sort(returns_array)

            # Find percentile corresponding to (1 - confidence level)
            percentile = 1 - float(conf_level)
            index = int(percentile * len(sorted_returns))

            # Get VaR as the percentile value (negative of the loss)
            var_value = Decimal(str(abs(sorted_returns[index])))

            # Scale to portfolio value if provided
            if portfolio_value is not None:
                var_value = var_value * portfolio_value

            logger.info(f"Historical VaR at {conf_level*100}% confidence: {var_value}")
            return var_value

        except Exception as e:
            logger.error(f"Error calculating Historical VaR: {e}")
            return Decimal('0')

    def calculate_parametric_var(
        self,
        returns: pl.Series,
        confidence_level: Optional[Decimal] = None,
        portfolio_value: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate Parametric VaR using variance-covariance method

        Assumes returns are normally distributed.
        VaR = μ - (z * σ)

        Args:
            returns: Series of historical returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            portfolio_value: Portfolio value to scale VaR

        Returns:
            VaR value
        """
        try:
            if len(returns) == 0:
                logger.warning("No returns data for Parametric VaR")
                return Decimal('0')

            conf_level = confidence_level if confidence_level is not None else self.var_confidence_level

            # Calculate mean and standard deviation
            returns_array = returns.to_numpy()
            mean_return = Decimal(str(np.mean(returns_array)))
            std_return = Decimal(str(np.std(returns_array, ddof=1)))

            # Get z-score for confidence level
            z_score = Decimal(str(abs(stats.norm.ppf(1 - float(conf_level)))))

            # Calculate VaR: we want the loss, so it's negative return
            # VaR = -(mean - z*std) = z*std - mean
            var_percentage = z_score * std_return - mean_return

            # Scale to portfolio value if provided
            if portfolio_value is not None:
                var_value = var_percentage * portfolio_value
            else:
                var_value = var_percentage

            logger.info(f"Parametric VaR at {conf_level*100}% confidence: {var_value}")
            return var_value

        except Exception as e:
            logger.error(f"Error calculating Parametric VaR: {e}")
            return Decimal('0')

    def calculate_monte_carlo_var(
        self,
        returns: pl.Series,
        confidence_level: Optional[Decimal] = None,
        portfolio_value: Optional[Decimal] = None,
        num_simulations: Optional[int] = None
    ) -> Decimal:
        """
        Calculate Monte Carlo VaR using simulation

        Simulates future returns based on historical distribution.

        Args:
            returns: Series of historical returns
            confidence_level: Confidence level
            portfolio_value: Portfolio value to scale VaR
            num_simulations: Number of Monte Carlo simulations

        Returns:
            VaR value
        """
        try:
            if len(returns) == 0:
                logger.warning("No returns data for Monte Carlo VaR")
                return Decimal('0')

            conf_level = confidence_level if confidence_level is not None else self.var_confidence_level
            num_sims = num_simulations if num_simulations is not None else self.mc_simulations

            # Calculate mean and std from historical data
            returns_array = returns.to_numpy()
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array, ddof=1)

            # Generate random simulations
            np.random.seed(42)  # For reproducibility
            simulated_returns = np.random.normal(mean_return, std_return, num_sims)

            # Sort simulations
            sorted_sims = np.sort(simulated_returns)

            # Find VaR at confidence level
            percentile = 1 - float(conf_level)
            index = int(percentile * num_sims)
            var_percentage = Decimal(str(abs(sorted_sims[index])))

            # Scale to portfolio value if provided
            if portfolio_value is not None:
                var_value = var_percentage * portfolio_value
            else:
                var_value = var_percentage

            logger.info(f"Monte Carlo VaR at {conf_level*100}% confidence: {var_value}")
            return var_value

        except Exception as e:
            logger.error(f"Error calculating Monte Carlo VaR: {e}")
            return Decimal('0')

    def calculate_conditional_var(
        self,
        returns: pl.Series,
        confidence_level: Optional[Decimal] = None,
        portfolio_value: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate Conditional VaR (CVaR) / Expected Shortfall

        CVaR is the expected loss given that the loss exceeds VaR.
        More conservative than VaR.

        Args:
            returns: Series of historical returns
            confidence_level: Confidence level
            portfolio_value: Portfolio value to scale CVaR

        Returns:
            CVaR value
        """
        try:
            if len(returns) == 0:
                logger.warning("No returns data for CVaR")
                return Decimal('0')

            conf_level = confidence_level if confidence_level is not None else self.var_confidence_level

            # Sort returns
            returns_array = returns.to_numpy()
            sorted_returns = np.sort(returns_array)

            # Find VaR threshold
            percentile = 1 - float(conf_level)
            var_index = int(percentile * len(sorted_returns))

            # CVaR is the average of returns worse than VaR
            cvar_returns = sorted_returns[:var_index]

            if len(cvar_returns) > 0:
                cvar_percentage = Decimal(str(abs(np.mean(cvar_returns))))
            else:
                # Fallback to VaR if no tail data
                cvar_percentage = Decimal(str(abs(sorted_returns[var_index])))

            # Scale to portfolio value if provided
            if portfolio_value is not None:
                cvar_value = cvar_percentage * portfolio_value
            else:
                cvar_value = cvar_percentage

            logger.info(f"CVaR at {conf_level*100}% confidence: {cvar_value}")
            return cvar_value

        except Exception as e:
            logger.error(f"Error calculating CVaR: {e}")
            return Decimal('0')

    def calculate_all_var_methods(
        self,
        returns: pl.Series,
        confidence_level: Optional[Decimal] = None,
        portfolio_value: Optional[Decimal] = None
    ) -> Dict[str, Decimal]:
        """
        Calculate VaR using all methods for comparison

        Args:
            returns: Series of historical returns
            confidence_level: Confidence level
            portfolio_value: Portfolio value

        Returns:
            Dictionary with all VaR methods
        """
        try:
            conf_level = confidence_level if confidence_level is not None else self.var_confidence_level

            results = {
                'historical_var': self.calculate_historical_var(returns, conf_level, portfolio_value),
                'parametric_var': self.calculate_parametric_var(returns, conf_level, portfolio_value),
                'monte_carlo_var': self.calculate_monte_carlo_var(returns, conf_level, portfolio_value),
                'conditional_var': self.calculate_conditional_var(returns, conf_level, portfolio_value)
            }

            return results

        except Exception as e:
            logger.error(f"Error calculating all VaR methods: {e}")
            return {}

    def calculate_multi_confidence_var(
        self,
        returns: pl.Series,
        portfolio_value: Optional[Decimal] = None,
        method: str = 'historical'
    ) -> Dict[str, Decimal]:
        """
        Calculate VaR at multiple confidence levels

        Args:
            returns: Series of historical returns
            portfolio_value: Portfolio value
            method: VaR method ('historical', 'parametric', 'monte_carlo')

        Returns:
            Dictionary with VaR at different confidence levels
        """
        try:
            results = {}

            for conf_level in self.confidence_levels:
                key = f"var_{int(conf_level * 100)}"

                if method == 'historical':
                    var_value = self.calculate_historical_var(returns, conf_level, portfolio_value)
                elif method == 'parametric':
                    var_value = self.calculate_parametric_var(returns, conf_level, portfolio_value)
                elif method == 'monte_carlo':
                    var_value = self.calculate_monte_carlo_var(returns, conf_level, portfolio_value)
                else:
                    logger.warning(f"Unknown VaR method: {method}")
                    var_value = Decimal('0')

                results[key] = var_value

            return results

        except Exception as e:
            logger.error(f"Error calculating multi-confidence VaR: {e}")
            return {}

    def backtest_var(
        self,
        returns_df: pl.DataFrame,
        var_estimates: pl.DataFrame,
        confidence_level: Optional[Decimal] = None
    ) -> Dict[str, any]:
        """
        Backtest VaR accuracy

        Checks if actual losses exceed VaR predictions at the expected rate.
        For 95% VaR, we expect ~5% of days to exceed VaR.

        Args:
            returns_df: DataFrame with actual returns
            var_estimates: DataFrame with VaR estimates
            confidence_level: Confidence level used for VaR

        Returns:
            Backtesting results dictionary
        """
        try:
            if len(returns_df) == 0 or len(var_estimates) == 0:
                logger.warning("Insufficient data for VaR backtesting")
                return {}

            conf_level = confidence_level if confidence_level is not None else self.var_confidence_level
            expected_violation_rate = 1 - float(conf_level)

            # Align dataframes
            merged_df = returns_df.join(
                var_estimates,
                on="timestamp",
                how="inner"
            )

            if len(merged_df) == 0:
                logger.warning("No matching data for backtesting")
                return {}

            # Count violations (actual loss > VaR estimate)
            violations = 0
            total_days = len(merged_df)

            for row in merged_df.iter_rows(named=True):
                actual_loss = abs(row["returns"]) if row["returns"] < 0 else 0
                var_estimate = row["var_estimate"]

                if actual_loss > float(var_estimate):
                    violations += 1

            # Calculate violation rate
            actual_violation_rate = violations / total_days if total_days > 0 else 0

            # Kupiec test (likelihood ratio test for VaR accuracy)
            kupiec_stat = self._calculate_kupiec_statistic(
                violations,
                total_days,
                expected_violation_rate
            )

            # Critical value for 95% confidence (chi-squared with 1 df)
            critical_value = 3.841

            # VaR model is acceptable if test statistic < critical value
            is_acceptable = kupiec_stat < critical_value

            results = {
                'total_days': total_days,
                'violations': violations,
                'expected_violation_rate': expected_violation_rate,
                'actual_violation_rate': actual_violation_rate,
                'kupiec_statistic': kupiec_stat,
                'critical_value': critical_value,
                'is_acceptable': is_acceptable,
                'recommendation': 'PASS' if is_acceptable else 'FAIL'
            }

            logger.info(f"VaR backtesting: {violations}/{total_days} violations ({actual_violation_rate:.2%})")
            return results

        except Exception as e:
            logger.error(f"Error backtesting VaR: {e}")
            return {}

    def _calculate_kupiec_statistic(
        self,
        violations: int,
        total_days: int,
        expected_rate: float
    ) -> float:
        """
        Calculate Kupiec likelihood ratio test statistic

        LR = -2 * ln[(1-p)^(T-N) * p^N / (1-π)^(T-N) * π^N]

        Where:
        - T = total days
        - N = violations
        - p = expected violation rate
        - π = actual violation rate
        """
        try:
            if violations == 0 or violations == total_days:
                return 0.0

            actual_rate = violations / total_days

            # Calculate likelihood ratio
            likelihood_expected = ((1 - expected_rate) ** (total_days - violations)) * (expected_rate ** violations)
            likelihood_actual = ((1 - actual_rate) ** (total_days - violations)) * (actual_rate ** violations)

            if likelihood_actual == 0:
                return float('inf')

            lr_statistic = -2 * np.log(likelihood_expected / likelihood_actual)

            return float(lr_statistic)

        except Exception as e:
            logger.error(f"Error calculating Kupiec statistic: {e}")
            return 0.0

    def calculate_incremental_var(
        self,
        portfolio_returns: pl.Series,
        position_returns: pl.Series,
        position_weight: Decimal,
        confidence_level: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate Incremental VaR

        Measures how much VaR changes by adding/removing a position.

        Args:
            portfolio_returns: Full portfolio returns
            position_returns: Returns of the specific position
            position_weight: Weight of position in portfolio
            confidence_level: Confidence level

        Returns:
            Incremental VaR
        """
        try:
            conf_level = confidence_level if confidence_level is not None else self.var_confidence_level

            # Calculate VaR of full portfolio
            full_var = self.calculate_parametric_var(portfolio_returns, conf_level)

            # Calculate portfolio without this position
            # Approximate by scaling: (1 - weight) factor
            adjusted_weight = Decimal('1') - position_weight

            if adjusted_weight > 0:
                # Scale returns
                adjusted_returns_array = portfolio_returns.to_numpy() * float(adjusted_weight)
                adjusted_returns = pl.Series(adjusted_returns_array)

                reduced_var = self.calculate_parametric_var(adjusted_returns, conf_level)

                # Incremental VaR
                incremental_var = full_var - reduced_var
            else:
                incremental_var = full_var

            return incremental_var

        except Exception as e:
            logger.error(f"Error calculating Incremental VaR: {e}")
            return Decimal('0')

    def calculate_marginal_var(
        self,
        returns: pl.Series,
        weights: List[Decimal],
        covariance_matrix: np.ndarray,
        confidence_level: Optional[Decimal] = None
    ) -> List[Decimal]:
        """
        Calculate Marginal VaR for each position

        Marginal VaR = ∂VaR/∂w_i (partial derivative of VaR with respect to position weight)

        Args:
            returns: Portfolio returns
            weights: Position weights
            covariance_matrix: Covariance matrix of positions
            confidence_level: Confidence level

        Returns:
            List of marginal VaR values for each position
        """
        try:
            conf_level = confidence_level if confidence_level is not None else self.var_confidence_level

            # Get z-score
            z_score = abs(stats.norm.ppf(1 - float(conf_level)))

            # Calculate portfolio volatility
            weights_array = np.array([float(w) for w in weights])
            portfolio_var = weights_array @ covariance_matrix @ weights_array
            portfolio_vol = np.sqrt(portfolio_var)

            # Marginal VaR for each position
            marginal_vars = []
            for i in range(len(weights)):
                # ∂σ_p/∂w_i = (Σ * w)_i / σ_p
                marginal_vol = (covariance_matrix @ weights_array)[i] / portfolio_vol
                marginal_var = Decimal(str(z_score * marginal_vol))
                marginal_vars.append(marginal_var)

            return marginal_vars

        except Exception as e:
            logger.error(f"Error calculating Marginal VaR: {e}")
            return [Decimal('0')] * len(weights)

    def generate_var_report(
        self,
        returns: pl.Series,
        portfolio_value: Decimal,
        confidence_level: Optional[Decimal] = None
    ) -> Dict:
        """
        Generate comprehensive VaR report

        Args:
            returns: Historical returns
            portfolio_value: Current portfolio value
            confidence_level: Confidence level

        Returns:
            Comprehensive VaR report
        """
        try:
            conf_level = confidence_level if confidence_level is not None else self.var_confidence_level

            # Calculate all VaR methods
            all_vars = self.calculate_all_var_methods(returns, conf_level, portfolio_value)

            # Calculate multi-confidence VaR
            multi_conf_var = self.calculate_multi_confidence_var(
                returns,
                portfolio_value,
                method='historical'
            )

            # Calculate basic statistics
            returns_array = returns.to_numpy()
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array, ddof=1)
            min_return = np.min(returns_array)
            max_return = np.max(returns_array)

            report = {
                'timestamp': datetime.utcnow().isoformat(),
                'portfolio_value': float(portfolio_value),
                'confidence_level': float(conf_level),
                'sample_size': len(returns),
                'var_estimates': {
                    'historical': float(all_vars.get('historical_var', 0)),
                    'parametric': float(all_vars.get('parametric_var', 0)),
                    'monte_carlo': float(all_vars.get('monte_carlo_var', 0)),
                    'conditional_var': float(all_vars.get('conditional_var', 0))
                },
                'multi_confidence_var': {k: float(v) for k, v in multi_conf_var.items()},
                'return_statistics': {
                    'mean': float(mean_return),
                    'std_dev': float(std_return),
                    'min': float(min_return),
                    'max': float(max_return)
                },
                'interpretation': self._interpret_var_results(all_vars, portfolio_value)
            }

            return report

        except Exception as e:
            logger.error(f"Error generating VaR report: {e}")
            return {'error': str(e)}

    def _interpret_var_results(
        self,
        var_results: Dict[str, Decimal],
        portfolio_value: Decimal
    ) -> Dict[str, str]:
        """Interpret VaR results and provide recommendations"""
        try:
            # Get average VaR across methods
            var_values = [v for v in var_results.values() if v > 0]
            if not var_values:
                return {'status': 'INSUFFICIENT_DATA', 'message': 'No valid VaR calculations'}

            avg_var = sum(var_values) / len(var_values)
            var_percentage = (avg_var / portfolio_value * Decimal('100')) if portfolio_value > 0 else Decimal('0')

            # Interpret severity
            if var_percentage < Decimal('2'):
                severity = 'LOW'
                message = 'Portfolio risk is low'
            elif var_percentage < Decimal('5'):
                severity = 'MODERATE'
                message = 'Portfolio risk is moderate'
            elif var_percentage < Decimal('10'):
                severity = 'HIGH'
                message = 'Portfolio risk is high - consider risk reduction'
            else:
                severity = 'CRITICAL'
                message = 'Portfolio risk is critical - immediate action recommended'

            return {
                'status': severity,
                'message': message,
                'var_percentage': f"{float(var_percentage):.2f}%"
            }

        except Exception as e:
            logger.error(f"Error interpreting VaR results: {e}")
            return {'status': 'ERROR', 'message': str(e)}
