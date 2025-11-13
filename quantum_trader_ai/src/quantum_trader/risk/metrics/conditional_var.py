"""
Conditional Value at Risk (CVaR / Expected Shortfall)
CRITICAL: Calculate CVaR, expected shortfall, and stress testing for tail risk
"""

import logging
import numpy as np
import polars as pl
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from scipy import stats

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class ConditionalVaRCalculator:
    """Calculate CVaR (Conditional Value at Risk) and Expected Shortfall"""

    def __init__(self):
        """Initialize CVaR calculator with config"""
        self.config = get_config()
        self.var_confidence = self.config.get_decimal("risk", "risk_metrics.var_confidence")
        self.cvar_limit = self.config.get_decimal("risk", "loss_limits.cvar_limit_usd")
        self.var_limit = self.config.get_decimal("risk", "loss_limits.var_limit_usd")
        self.stress_scenarios = self.config.get_int("risk", "risk_metrics.stress_test_scenarios", 10)

        logger.info(
            f"ConditionalVaRCalculator initialized: confidence={self.var_confidence}, "
            f"cvar_limit=${self.cvar_limit}, stress_scenarios={self.stress_scenarios}"
        )

    async def calculate_cvar(
        self,
        returns: pl.DataFrame,
        portfolio_value: Decimal,
        confidence_level: Optional[Decimal] = None,
        method: str = "historical"
    ) -> Dict[str, Decimal]:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall

        Args:
            returns: DataFrame with columns ['timestamp', 'return'] (portfolio returns)
            portfolio_value: Total portfolio value in USD
            confidence_level: Confidence level (default from config)
            method: Calculation method ('historical', 'parametric', 'monte_carlo')

        Returns:
            Dictionary with CVaR metrics:
            {
                'cvar': Decimal (dollar amount),
                'cvar_pct': Decimal (percentage),
                'var': Decimal (VaR for comparison),
                'var_pct': Decimal,
                'confidence_level': Decimal,
                'method': str
            }
        """
        try:
            if returns.is_empty():
                logger.error("Cannot calculate CVaR: empty returns DataFrame")
                raise ValueError("Returns DataFrame is empty")

            if confidence_level is None:
                confidence_level = self.var_confidence

            returns_array = returns.select("return").to_numpy().flatten()

            if len(returns_array) < 30:
                logger.warning(f"Limited data for CVaR calculation: {len(returns_array)} observations")

            # Calculate based on method
            if method == "historical":
                cvar_pct, var_pct = self._calculate_cvar_historical(returns_array, confidence_level)
            elif method == "parametric":
                cvar_pct, var_pct = self._calculate_cvar_parametric(returns_array, confidence_level)
            elif method == "monte_carlo":
                cvar_pct, var_pct = await self._calculate_cvar_monte_carlo(returns_array, confidence_level)
            else:
                raise ValueError(f"Unknown CVaR calculation method: {method}")

            # Convert to dollar amounts
            cvar_dollar = (abs(cvar_pct) * portfolio_value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            var_dollar = (abs(var_pct) * portfolio_value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            result = {
                "cvar": cvar_dollar,
                "cvar_pct": cvar_pct,
                "var": var_dollar,
                "var_pct": var_pct,
                "confidence_level": confidence_level,
                "method": method,
                "portfolio_value": portfolio_value,
                "exceeds_limit": cvar_dollar > self.cvar_limit,
                "limit": self.cvar_limit
            }

            if result["exceeds_limit"]:
                logger.warning(
                    f"CVaR ${cvar_dollar:,.2f} exceeds limit ${self.cvar_limit:,.2f} "
                    f"({confidence_level*100:.1f}% confidence)"
                )
            else:
                logger.info(
                    f"CVaR calculated: ${cvar_dollar:,.2f} ({cvar_pct*100:.2f}%) "
                    f"at {confidence_level*100:.1f}% confidence using {method} method"
                )

            return result

        except Exception as e:
            logger.error(f"Error calculating CVaR: {e}", exc_info=True)
            raise

    async def calculate_expected_shortfall(
        self,
        returns: pl.DataFrame,
        portfolio_value: Decimal,
        threshold_pct: Optional[Decimal] = None
    ) -> Dict[str, Decimal]:
        """
        Calculate Expected Shortfall (average loss beyond a threshold)

        Args:
            returns: DataFrame with columns ['timestamp', 'return']
            portfolio_value: Total portfolio value in USD
            threshold_pct: Loss threshold as percentage (default: VaR at confidence level)

        Returns:
            Dictionary with expected shortfall metrics
        """
        try:
            if returns.is_empty():
                logger.error("Cannot calculate expected shortfall: empty returns DataFrame")
                raise ValueError("Returns DataFrame is empty")

            returns_array = returns.select("return").to_numpy().flatten()

            # If no threshold provided, use VaR at confidence level
            if threshold_pct is None:
                var_threshold_idx = int((1.0 - float(self.var_confidence)) * len(returns_array))
                sorted_returns = np.sort(returns_array)
                threshold_pct = Decimal(str(sorted_returns[var_threshold_idx]))

            threshold_float = float(threshold_pct)

            # Calculate expected shortfall: average of returns below threshold
            tail_returns = returns_array[returns_array <= threshold_float]

            if len(tail_returns) == 0:
                logger.warning("No returns below threshold for expected shortfall calculation")
                expected_shortfall_pct = threshold_pct
            else:
                expected_shortfall_pct = Decimal(str(np.mean(tail_returns))).quantize(
                    Decimal("0.000001"), rounding=ROUND_HALF_UP
                )

            expected_shortfall_dollar = (
                abs(expected_shortfall_pct) * portfolio_value
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            result = {
                "expected_shortfall": expected_shortfall_dollar,
                "expected_shortfall_pct": expected_shortfall_pct,
                "threshold_pct": threshold_pct,
                "tail_observations": len(tail_returns),
                "total_observations": len(returns_array),
                "tail_probability": Decimal(str(len(tail_returns) / len(returns_array))).quantize(
                    Decimal("0.0001")
                ),
                "portfolio_value": portfolio_value
            }

            logger.info(
                f"Expected shortfall: ${expected_shortfall_dollar:,.2f} "
                f"({expected_shortfall_pct*100:.2f}%) with {len(tail_returns)} tail observations"
            )

            return result

        except Exception as e:
            logger.error(f"Error calculating expected shortfall: {e}", exc_info=True)
            raise

    async def stress_test(
        self,
        portfolio_positions: pl.DataFrame,
        scenarios: Optional[List[Dict[str, Decimal]]] = None
    ) -> pl.DataFrame:
        """
        Perform stress testing on portfolio under various scenarios

        Args:
            portfolio_positions: DataFrame with ['asset', 'position_size', 'current_price']
            scenarios: Optional list of scenario dicts with asset price shocks
                      Format: [{'asset': shock_pct, ...}, ...]
                      If None, generates default scenarios

        Returns:
            Polars DataFrame with stress test results:
            ['scenario_id', 'scenario_name', 'portfolio_value_change', 'pct_change']
        """
        try:
            if portfolio_positions.is_empty():
                logger.error("Cannot stress test: empty portfolio")
                raise ValueError("Portfolio positions DataFrame is empty")

            # Generate scenarios if not provided
            if scenarios is None:
                scenarios = self._generate_stress_scenarios()

            # Calculate current portfolio value
            current_value = self._calculate_portfolio_value(portfolio_positions)

            results = []

            for idx, scenario in enumerate(scenarios):
                scenario_name = scenario.get("name", f"Scenario_{idx+1}")
                shocks = scenario.get("shocks", {})

                # Apply shocks to positions
                stressed_positions = portfolio_positions.clone()

                for asset, shock_pct in shocks.items():
                    # Apply price shock
                    mask = stressed_positions["asset"] == asset
                    if mask.sum() > 0:
                        shock_multiplier = 1.0 + float(shock_pct)
                        stressed_positions = stressed_positions.with_columns(
                            pl.when(pl.col("asset") == asset)
                            .then(pl.col("current_price") * shock_multiplier)
                            .otherwise(pl.col("current_price"))
                            .alias("current_price")
                        )

                # Calculate stressed portfolio value
                stressed_value = self._calculate_portfolio_value(stressed_positions)

                # Calculate change
                value_change = (stressed_value - current_value).quantize(Decimal("0.01"))
                pct_change = ((stressed_value / current_value - Decimal("1")) * Decimal("100")).quantize(
                    Decimal("0.01")
                ) if current_value > 0 else Decimal("0")

                results.append({
                    "scenario_id": idx + 1,
                    "scenario_name": scenario_name,
                    "portfolio_value_change": value_change,
                    "pct_change": pct_change,
                    "current_value": current_value,
                    "stressed_value": stressed_value
                })

            results_df = pl.DataFrame(results)

            # Find worst case
            worst_scenario = results_df.sort("pct_change").head(1)

            logger.info(
                f"Stress test completed: {len(scenarios)} scenarios, "
                f"worst case: {worst_scenario['scenario_name'][0]} "
                f"({worst_scenario['pct_change'][0]:.2f}%)"
            )

            return results_df

        except Exception as e:
            logger.error(f"Error performing stress test: {e}", exc_info=True)
            raise

    def _calculate_cvar_historical(
        self,
        returns: np.ndarray,
        confidence_level: Decimal
    ) -> Tuple[Decimal, Decimal]:
        """Calculate CVaR using historical simulation"""
        alpha = 1.0 - float(confidence_level)

        # Sort returns
        sorted_returns = np.sort(returns)

        # VaR: alpha quantile
        var_idx = int(alpha * len(sorted_returns))
        var_pct = float(sorted_returns[var_idx]) if var_idx < len(sorted_returns) else float(sorted_returns[0])

        # CVaR: average of returns beyond VaR
        tail_returns = sorted_returns[:var_idx+1]
        cvar_pct = float(np.mean(tail_returns)) if len(tail_returns) > 0 else var_pct

        return (
            Decimal(str(cvar_pct)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP),
            Decimal(str(var_pct)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
        )

    def _calculate_cvar_parametric(
        self,
        returns: np.ndarray,
        confidence_level: Decimal
    ) -> Tuple[Decimal, Decimal]:
        """Calculate CVaR using parametric (normal distribution) approach"""
        alpha = 1.0 - float(confidence_level)

        # Calculate mean and std dev
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)

        # VaR using normal distribution
        z_score = stats.norm.ppf(alpha)
        var_pct = mean + z_score * std

        # CVaR for normal distribution: mean + std * phi(z) / alpha
        # where phi is the standard normal PDF
        phi_z = stats.norm.pdf(z_score)
        cvar_pct = mean - std * phi_z / alpha

        return (
            Decimal(str(cvar_pct)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP),
            Decimal(str(var_pct)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
        )

    async def _calculate_cvar_monte_carlo(
        self,
        returns: np.ndarray,
        confidence_level: Decimal,
        n_simulations: int = 10000
    ) -> Tuple[Decimal, Decimal]:
        """Calculate CVaR using Monte Carlo simulation"""
        alpha = 1.0 - float(confidence_level)

        # Estimate distribution parameters
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)

        # Generate simulated returns
        simulated_returns = np.random.normal(mean, std, n_simulations)

        # Calculate CVaR from simulated returns
        sorted_returns = np.sort(simulated_returns)

        var_idx = int(alpha * n_simulations)
        var_pct = float(sorted_returns[var_idx])

        tail_returns = sorted_returns[:var_idx+1]
        cvar_pct = float(np.mean(tail_returns))

        return (
            Decimal(str(cvar_pct)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP),
            Decimal(str(var_pct)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
        )

    def _generate_stress_scenarios(self) -> List[Dict]:
        """Generate default stress test scenarios"""
        scenarios = [
            {
                "name": "Market Crash (-30%)",
                "shocks": {"BTC": Decimal("-0.30"), "ETH": Decimal("-0.35"), "SOL": Decimal("-0.40")}
            },
            {
                "name": "Flash Crash (-20%)",
                "shocks": {"BTC": Decimal("-0.20"), "ETH": Decimal("-0.20"), "SOL": Decimal("-0.20")}
            },
            {
                "name": "Bitcoin Dominance",
                "shocks": {"BTC": Decimal("0.15"), "ETH": Decimal("-0.10"), "SOL": Decimal("-0.15")}
            },
            {
                "name": "Altcoin Rally",
                "shocks": {"BTC": Decimal("0.05"), "ETH": Decimal("0.25"), "SOL": Decimal("0.30")}
            },
            {
                "name": "Moderate Decline (-10%)",
                "shocks": {"BTC": Decimal("-0.10"), "ETH": Decimal("-0.10"), "SOL": Decimal("-0.10")}
            },
            {
                "name": "High Volatility Spike",
                "shocks": {"BTC": Decimal("-0.15"), "ETH": Decimal("0.10"), "SOL": Decimal("-0.20")}
            },
            {
                "name": "Liquidity Crisis (-25%)",
                "shocks": {"BTC": Decimal("-0.25"), "ETH": Decimal("-0.28"), "SOL": Decimal("-0.32")}
            },
            {
                "name": "Strong Bull Market (+20%)",
                "shocks": {"BTC": Decimal("0.20"), "ETH": Decimal("0.25"), "SOL": Decimal("0.30")}
            },
            {
                "name": "Stablecoin Crisis",
                "shocks": {"BTC": Decimal("-0.12"), "ETH": Decimal("-0.15"), "SOL": Decimal("-0.18")}
            },
            {
                "name": "Regulatory Shock (-15%)",
                "shocks": {"BTC": Decimal("-0.15"), "ETH": Decimal("-0.18"), "SOL": Decimal("-0.20")}
            }
        ]

        return scenarios[:self.stress_scenarios]

    def _calculate_portfolio_value(self, positions: pl.DataFrame) -> Decimal:
        """Calculate total portfolio value from positions"""
        if positions.is_empty():
            return Decimal("0")

        # Calculate value: position_size * current_price
        values_df = positions.with_columns(
            (pl.col("position_size") * pl.col("current_price")).alias("value")
        )

        total_value = values_df.select("value").sum().item()

        return Decimal(str(total_value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
