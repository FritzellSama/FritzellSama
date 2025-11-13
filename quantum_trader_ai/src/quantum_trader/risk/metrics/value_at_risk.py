"""
Value at Risk (VaR) Calculator
CRITICAL: Calculate VaR using historical, parametric, and Monte Carlo methods
"""

import logging
import numpy as np
import polars as pl
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional, Tuple
from datetime import datetime
from scipy import stats

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class ValueAtRiskCalculator:
    """Calculate Value at Risk using multiple methodologies"""

    def __init__(self):
        """Initialize VaR calculator with config"""
        self.config = get_config()

        # Load VaR parameters
        self.var_confidence = self.config.get_decimal("risk", "risk_metrics.var_confidence")
        self.var_horizon_days = self.config.get_int("risk", "risk_metrics.var_horizon_days")
        self.var_limit_usd = self.config.get_decimal("risk", "loss_limits.var_limit_usd")

        logger.info(
            f"ValueAtRiskCalculator initialized: confidence={self.var_confidence:.2%}, "
            f"horizon={self.var_horizon_days}d, limit=${self.var_limit_usd:,.0f}"
        )

    async def calculate_var_historical(
        self,
        returns: pl.DataFrame,
        portfolio_value: Decimal,
        confidence_level: Optional[Decimal] = None,
        horizon_days: Optional[int] = None
    ) -> Decimal:
        """
        Calculate Historical VaR (non-parametric method)

        Args:
            returns: DataFrame with columns ['timestamp', 'return']
            portfolio_value: Current portfolio value
            confidence_level: Confidence level (default: from config)
            horizon_days: Time horizon in days (default: from config)

        Returns:
            VaR as Decimal (positive number representing potential loss)
        """
        try:
            if returns.is_empty():
                logger.error("Cannot calculate historical VaR: empty returns")
                raise ValueError("Returns DataFrame cannot be empty")

            if confidence_level is None:
                confidence_level = self.var_confidence

            if horizon_days is None:
                horizon_days = self.var_horizon_days

            # Extract returns array
            returns_array = returns.select("return").to_numpy().flatten()

            if len(returns_array) < 10:
                logger.warning("Insufficient data for historical VaR")
                return Decimal("0")

            # Scale returns to horizon if needed
            if horizon_days > 1:
                # Square root of time scaling
                returns_array = returns_array * np.sqrt(horizon_days)

            # Calculate percentile
            percentile = (1 - float(confidence_level)) * 100
            var_return = np.percentile(returns_array, percentile)

            # Convert to dollar amount (negative return = positive VaR)
            var_usd = abs(var_return * float(portfolio_value))

            var_decimal = Decimal(str(var_usd)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            logger.info(
                f"Historical VaR ({confidence_level:.0%}, {horizon_days}d): "
                f"${var_decimal:,.2f} ({abs(var_return):.4%} of portfolio)"
            )

            return var_decimal

        except Exception as e:
            logger.error(f"Error calculating historical VaR: {e}", exc_info=True)
            raise

    async def calculate_var_parametric(
        self,
        returns: pl.DataFrame,
        portfolio_value: Decimal,
        confidence_level: Optional[Decimal] = None,
        horizon_days: Optional[int] = None
    ) -> Decimal:
        """
        Calculate Parametric VaR (assumes normal distribution)

        Args:
            returns: DataFrame with columns ['timestamp', 'return']
            portfolio_value: Current portfolio value
            confidence_level: Confidence level (default: from config)
            horizon_days: Time horizon in days (default: from config)

        Returns:
            VaR as Decimal (positive number representing potential loss)
        """
        try:
            if returns.is_empty():
                logger.error("Cannot calculate parametric VaR: empty returns")
                raise ValueError("Returns DataFrame cannot be empty")

            if confidence_level is None:
                confidence_level = self.var_confidence

            if horizon_days is None:
                horizon_days = self.var_horizon_days

            # Extract returns array
            returns_array = returns.select("return").to_numpy().flatten()

            if len(returns_array) < 2:
                logger.warning("Insufficient data for parametric VaR")
                return Decimal("0")

            # Calculate mean and standard deviation
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array, ddof=1)

            # Calculate z-score for confidence level
            z_score = stats.norm.ppf(1 - float(confidence_level))

            # Calculate VaR return (mean + z_score * std)
            # Note: z_score is negative for losses
            var_return = mean_return + z_score * std_return

            # Scale to horizon if needed
            if horizon_days > 1:
                # Adjust for time horizon (square root of time rule)
                var_return = var_return * np.sqrt(horizon_days)

            # Convert to dollar amount
            var_usd = abs(var_return * float(portfolio_value))

            var_decimal = Decimal(str(var_usd)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            logger.info(
                f"Parametric VaR ({confidence_level:.0%}, {horizon_days}d): "
                f"${var_decimal:,.2f} ({abs(var_return):.4%} of portfolio)"
            )

            return var_decimal

        except Exception as e:
            logger.error(f"Error calculating parametric VaR: {e}", exc_info=True)
            raise

    async def calculate_var_monte_carlo(
        self,
        returns: pl.DataFrame,
        portfolio_value: Decimal,
        n_simulations: int = 10000,
        confidence_level: Optional[Decimal] = None,
        horizon_days: Optional[int] = None,
        random_seed: Optional[int] = None
    ) -> Decimal:
        """
        Calculate Monte Carlo VaR (simulation-based method)

        Args:
            returns: DataFrame with columns ['timestamp', 'return']
            portfolio_value: Current portfolio value
            n_simulations: Number of Monte Carlo simulations
            confidence_level: Confidence level (default: from config)
            horizon_days: Time horizon in days (default: from config)
            random_seed: Random seed for reproducibility

        Returns:
            VaR as Decimal (positive number representing potential loss)
        """
        try:
            if returns.is_empty():
                logger.error("Cannot calculate Monte Carlo VaR: empty returns")
                raise ValueError("Returns DataFrame cannot be empty")

            if confidence_level is None:
                confidence_level = self.var_confidence

            if horizon_days is None:
                horizon_days = self.var_horizon_days

            # Set random seed if provided
            if random_seed is not None:
                np.random.seed(random_seed)

            # Extract returns array
            returns_array = returns.select("return").to_numpy().flatten()

            if len(returns_array) < 10:
                logger.warning("Insufficient data for Monte Carlo VaR")
                return Decimal("0")

            # Calculate parameters from historical data
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array, ddof=1)

            # Generate random scenarios
            simulated_returns = np.random.normal(
                mean_return,
                std_return,
                (n_simulations, horizon_days)
            )

            # Calculate cumulative returns for each path
            # Using geometric returns: (1 + r1) * (1 + r2) * ... - 1
            cumulative_returns = np.prod(1 + simulated_returns, axis=1) - 1

            # Calculate percentile
            percentile = (1 - float(confidence_level)) * 100
            var_return = np.percentile(cumulative_returns, percentile)

            # Convert to dollar amount
            var_usd = abs(var_return * float(portfolio_value))

            var_decimal = Decimal(str(var_usd)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            logger.info(
                f"Monte Carlo VaR ({confidence_level:.0%}, {horizon_days}d, "
                f"{n_simulations} sims): ${var_decimal:,.2f} "
                f"({abs(var_return):.4%} of portfolio)"
            )

            return var_decimal

        except Exception as e:
            logger.error(f"Error calculating Monte Carlo VaR: {e}", exc_info=True)
            raise

    async def calculate_all_var_methods(
        self,
        returns: pl.DataFrame,
        portfolio_value: Decimal,
        confidence_level: Optional[Decimal] = None,
        horizon_days: Optional[int] = None
    ) -> Dict[str, Decimal]:
        """
        Calculate VaR using all three methods for comparison

        Args:
            returns: DataFrame with columns ['timestamp', 'return']
            portfolio_value: Current portfolio value
            confidence_level: Confidence level (default: from config)
            horizon_days: Time horizon in days (default: from config)

        Returns:
            Dictionary with VaR from each method
        """
        try:
            logger.info("Calculating VaR using all methods")

            # Calculate using all methods
            historical_var = await self.calculate_var_historical(
                returns, portfolio_value, confidence_level, horizon_days
            )

            parametric_var = await self.calculate_var_parametric(
                returns, portfolio_value, confidence_level, horizon_days
            )

            monte_carlo_var = await self.calculate_var_monte_carlo(
                returns, portfolio_value, n_simulations=10000,
                confidence_level=confidence_level, horizon_days=horizon_days
            )

            # Calculate average VaR
            avg_var = (historical_var + parametric_var + monte_carlo_var) / Decimal("3")

            result = {
                'historical_var': historical_var,
                'parametric_var': parametric_var,
                'monte_carlo_var': monte_carlo_var,
                'average_var': avg_var.quantize(Decimal("0.01")),
                'min_var': min(historical_var, parametric_var, monte_carlo_var),
                'max_var': max(historical_var, parametric_var, monte_carlo_var)
            }

            logger.info(
                f"VaR comparison: historical=${historical_var:,.2f}, "
                f"parametric=${parametric_var:,.2f}, "
                f"monte_carlo=${monte_carlo_var:,.2f}, "
                f"average=${avg_var:,.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error calculating all VaR methods: {e}", exc_info=True)
            raise

    async def calculate_conditional_var(
        self,
        returns: pl.DataFrame,
        portfolio_value: Decimal,
        confidence_level: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate Conditional VaR (CVaR / Expected Shortfall)
        Average loss beyond the VaR threshold

        Args:
            returns: DataFrame with columns ['timestamp', 'return']
            portfolio_value: Current portfolio value
            confidence_level: Confidence level (default: from config)

        Returns:
            CVaR as Decimal (positive number representing potential loss)
        """
        try:
            if returns.is_empty():
                logger.error("Cannot calculate CVaR: empty returns")
                raise ValueError("Returns DataFrame cannot be empty")

            if confidence_level is None:
                confidence_level = self.var_confidence

            # Extract returns array
            returns_array = returns.select("return").to_numpy().flatten()

            if len(returns_array) < 10:
                logger.warning("Insufficient data for CVaR")
                return Decimal("0")

            # Calculate VaR threshold
            percentile = (1 - float(confidence_level)) * 100
            var_threshold = np.percentile(returns_array, percentile)

            # Calculate average of losses beyond VaR
            losses_beyond_var = returns_array[returns_array <= var_threshold]

            if len(losses_beyond_var) == 0:
                # No losses beyond VaR - use VaR itself
                cvar_return = var_threshold
            else:
                cvar_return = np.mean(losses_beyond_var)

            # Convert to dollar amount
            cvar_usd = abs(cvar_return * float(portfolio_value))

            cvar_decimal = Decimal(str(cvar_usd)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            logger.info(
                f"Conditional VaR ({confidence_level:.0%}): ${cvar_decimal:,.2f} "
                f"({abs(cvar_return):.4%} of portfolio)"
            )

            return cvar_decimal

        except Exception as e:
            logger.error(f"Error calculating CVaR: {e}", exc_info=True)
            raise

    async def check_var_breach(
        self,
        current_var: Decimal
    ) -> Tuple[bool, str]:
        """
        Check if VaR exceeds configured limit

        Args:
            current_var: Current VaR value

        Returns:
            Tuple of (is_breached, severity_level)
        """
        try:
            if current_var <= Decimal("0"):
                return False, "safe"

            # Calculate utilization
            utilization = current_var / self.var_limit_usd

            if utilization >= Decimal("1.0"):
                logger.critical(
                    f"CRITICAL: VaR ${current_var:,.2f} exceeds limit "
                    f"${self.var_limit_usd:,.2f}"
                )
                return True, "halt"

            elif utilization >= Decimal("0.9"):
                logger.error(
                    f"ERROR: VaR ${current_var:,.2f} approaching limit "
                    f"${self.var_limit_usd:,.2f} ({utilization:.1%} utilized)"
                )
                return True, "critical"

            elif utilization >= Decimal("0.75"):
                logger.warning(
                    f"WARNING: VaR ${current_var:,.2f} at {utilization:.1%} of limit"
                )
                return False, "warning"

            else:
                return False, "safe"

        except Exception as e:
            logger.error(f"Error checking VaR breach: {e}", exc_info=True)
            raise

    async def calculate_marginal_var(
        self,
        returns: pl.DataFrame,
        portfolio_value: Decimal,
        position_weight: Decimal,
        confidence_level: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate Marginal VaR (change in VaR from adding a position)

        Args:
            returns: DataFrame with columns ['timestamp', 'return']
            portfolio_value: Current portfolio value
            position_weight: Weight of position to add (as decimal)
            confidence_level: Confidence level (default: from config)

        Returns:
            Marginal VaR as Decimal
        """
        try:
            # Calculate current VaR
            current_var = await self.calculate_var_parametric(
                returns, portfolio_value, confidence_level
            )

            # Calculate VaR with increased position
            adjusted_portfolio_value = portfolio_value * (Decimal("1") + position_weight)
            adjusted_var = await self.calculate_var_parametric(
                returns, adjusted_portfolio_value, confidence_level
            )

            # Marginal VaR is the difference
            marginal_var = adjusted_var - current_var

            logger.info(
                f"Marginal VaR for {position_weight:.2%} increase: "
                f"${marginal_var:,.2f}"
            )

            return marginal_var

        except Exception as e:
            logger.error(f"Error calculating marginal VaR: {e}", exc_info=True)
            raise

    async def calculate_component_var(
        self,
        position_returns: Dict[str, pl.DataFrame],
        position_weights: Dict[str, Decimal],
        portfolio_value: Decimal,
        confidence_level: Optional[Decimal] = None
    ) -> Dict[str, Decimal]:
        """
        Calculate Component VaR (VaR contribution of each position)

        Args:
            position_returns: Dictionary mapping symbol to returns DataFrame
            position_weights: Dictionary mapping symbol to weight
            portfolio_value: Current portfolio value
            confidence_level: Confidence level (default: from config)

        Returns:
            Dictionary mapping symbol to component VaR
        """
        try:
            component_vars = {}

            for symbol in position_weights.keys():
                if symbol not in position_returns:
                    logger.warning(f"No returns data for {symbol}, skipping")
                    continue

                weight = position_weights[symbol]
                returns = position_returns[symbol]

                # Calculate VaR for this position
                position_value = weight * portfolio_value
                position_var = await self.calculate_var_parametric(
                    returns, position_value, confidence_level
                )

                component_vars[symbol] = position_var

            logger.info(
                f"Calculated component VaR for {len(component_vars)} positions"
            )

            return component_vars

        except Exception as e:
            logger.error(f"Error calculating component VaR: {e}", exc_info=True)
            raise
