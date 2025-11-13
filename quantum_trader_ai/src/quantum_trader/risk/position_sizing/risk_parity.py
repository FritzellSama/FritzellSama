"""
Risk Parity Position Sizing
CRITICAL: Equal risk contribution position sizing strategy
"""

import logging
import numpy as np
import polars as pl
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class RiskParitySizer:
    """Risk parity position sizing - allocate based on equal risk contribution"""

    def __init__(self):
        """Initialize risk parity sizer with config"""
        self.config = get_config()

        # Load configuration
        self.max_position_size_pct = self.config.get_decimal("risk", "position_sizing.max_position_size_pct")
        self.min_position_size_usd = self.config.get_decimal("risk", "position_sizing.min_position_size_usd")

        logger.info(
            f"RiskParitySizer initialized: max_size_pct={self.max_position_size_pct:.2%}"
        )

    async def calculate_risk_parity_weights(
        self,
        volatilities: Dict[str, Decimal],
        correlations: Optional[pl.DataFrame] = None,
        target_volatility: Optional[Decimal] = None
    ) -> Dict[str, Decimal]:
        """
        Calculate risk parity weights for portfolio

        Args:
            volatilities: Dictionary mapping symbol to volatility
            correlations: Optional correlation matrix (if None, assumes uncorrelated)
            target_volatility: Optional target portfolio volatility

        Returns:
            Dictionary mapping symbol to weight (as Decimal)
        """
        try:
            if not volatilities:
                logger.error("Cannot calculate risk parity: empty volatilities")
                raise ValueError("Volatilities dictionary cannot be empty")

            symbols = list(volatilities.keys())
            n_assets = len(symbols)

            logger.info(f"Calculating risk parity weights for {n_assets} assets")

            # Convert volatilities to numpy array
            vols = np.array([float(volatilities[s]) for s in symbols])

            if correlations is None:
                # Simple inverse volatility weighting (assumes uncorrelated)
                inv_vols = 1.0 / vols
                weights_array = inv_vols / np.sum(inv_vols)

                logger.info("Using inverse volatility weighting (no correlation data)")

            else:
                # Full risk parity optimization with correlation matrix
                corr_matrix = self._extract_correlation_matrix(correlations, symbols)
                weights_array = await self._optimize_risk_parity(vols, corr_matrix)

                logger.info("Using optimized risk parity with correlations")

            # Convert to dictionary with Decimal
            weights = {}
            for i, symbol in enumerate(symbols):
                weight = Decimal(str(weights_array[i])).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

                # Apply constraints
                weight = max(Decimal("0"), min(weight, self.max_position_size_pct))

                weights[symbol] = weight

            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {
                    symbol: (weight / total_weight).quantize(Decimal("0.000001"))
                    for symbol, weight in weights.items()
                }

            logger.info(
                f"Risk parity weights calculated: "
                f"mean={np.mean(list(float(w) for w in weights.values())):.4f}, "
                f"min={min(weights.values()):.4f}, max={max(weights.values()):.4f}"
            )

            return weights

        except Exception as e:
            logger.error(f"Error calculating risk parity weights: {e}", exc_info=True)
            raise

    async def get_risk_contribution(
        self,
        weights: Dict[str, Decimal],
        volatilities: Dict[str, Decimal],
        correlations: Optional[pl.DataFrame] = None
    ) -> Dict[str, Decimal]:
        """
        Calculate risk contribution of each asset

        Args:
            weights: Portfolio weights
            volatilities: Asset volatilities
            correlations: Optional correlation matrix

        Returns:
            Dictionary mapping symbol to risk contribution percentage
        """
        try:
            if not weights or not volatilities:
                logger.error("Cannot calculate risk contribution: empty inputs")
                raise ValueError("Weights and volatilities cannot be empty")

            symbols = list(weights.keys())
            n_assets = len(symbols)

            # Convert to numpy arrays
            w = np.array([float(weights.get(s, Decimal("0"))) for s in symbols])
            vols = np.array([float(volatilities.get(s, Decimal("0"))) for s in symbols])

            if correlations is None:
                # Simple case: uncorrelated assets
                # Risk contribution = weight * volatility / portfolio_volatility
                portfolio_variance = np.sum((w * vols) ** 2)
                portfolio_vol = np.sqrt(portfolio_variance)

                if portfolio_vol < 1e-10:
                    logger.warning("Portfolio volatility too low")
                    # Equal risk contribution as fallback
                    risk_contrib = {s: Decimal(str(1.0 / n_assets)) for s in symbols}
                else:
                    risk_contributions = (w * vols ** 2) / portfolio_vol
                    risk_contrib = {
                        symbol: Decimal(str(rc)).quantize(Decimal("0.000001"))
                        for symbol, rc in zip(symbols, risk_contributions)
                    }

            else:
                # With correlations
                corr_matrix = self._extract_correlation_matrix(correlations, symbols)

                # Covariance matrix = diag(vols) @ corr @ diag(vols)
                cov_matrix = np.outer(vols, vols) * corr_matrix

                # Portfolio variance = w^T @ cov @ w
                portfolio_variance = w @ cov_matrix @ w
                portfolio_vol = np.sqrt(portfolio_variance)

                if portfolio_vol < 1e-10:
                    logger.warning("Portfolio volatility too low")
                    risk_contrib = {s: Decimal(str(1.0 / n_assets)) for s in symbols}
                else:
                    # Marginal contribution to risk (MCR) = (cov @ w) / portfolio_vol
                    mcr = (cov_matrix @ w) / portfolio_vol

                    # Risk contribution = weight * MCR
                    risk_contributions = w * mcr

                    # Normalize to percentages
                    total_rc = np.sum(risk_contributions)
                    if total_rc > 0:
                        risk_contributions = risk_contributions / total_rc

                    risk_contrib = {
                        symbol: Decimal(str(rc)).quantize(Decimal("0.000001"))
                        for symbol, rc in zip(symbols, risk_contributions)
                    }

            logger.info(
                f"Risk contributions calculated: "
                f"mean={np.mean([float(rc) for rc in risk_contrib.values()]):.4f}"
            )

            return risk_contrib

        except Exception as e:
            logger.error(f"Error calculating risk contribution: {e}", exc_info=True)
            raise

    async def optimize_weights(
        self,
        volatilities: Dict[str, Decimal],
        correlations: Optional[pl.DataFrame] = None,
        target_return: Optional[Decimal] = None,
        risk_aversion: Decimal = Decimal("1.0")
    ) -> Dict[str, Decimal]:
        """
        Optimize portfolio weights with risk parity constraint

        Args:
            volatilities: Asset volatilities
            correlations: Optional correlation matrix
            target_return: Optional target return constraint
            risk_aversion: Risk aversion parameter (higher = more risk averse)

        Returns:
            Dictionary mapping symbol to optimized weight
        """
        try:
            if not volatilities:
                logger.error("Cannot optimize weights: empty volatilities")
                raise ValueError("Volatilities dictionary cannot be empty")

            # Start with risk parity weights
            rp_weights = await self.calculate_risk_parity_weights(
                volatilities,
                correlations
            )

            # If no additional constraints, return risk parity weights
            if target_return is None:
                return rp_weights

            # Otherwise, adjust weights based on risk aversion and target return
            # This is a simplified adjustment - full optimization would require expected returns
            adjusted_weights = {}
            for symbol, weight in rp_weights.items():
                # Apply risk aversion adjustment (placeholder logic)
                adjusted_weight = weight / risk_aversion
                adjusted_weights[symbol] = adjusted_weight

            # Normalize
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                adjusted_weights = {
                    symbol: (weight / total_weight).quantize(Decimal("0.000001"))
                    for symbol, weight in adjusted_weights.items()
                }

            logger.info("Optimized weights with risk parity constraint")

            return adjusted_weights

        except Exception as e:
            logger.error(f"Error optimizing weights: {e}", exc_info=True)
            raise

    async def _optimize_risk_parity(
        self,
        volatilities: np.ndarray,
        correlation_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Optimize for equal risk contribution using scipy

        Args:
            volatilities: Array of asset volatilities
            correlation_matrix: Correlation matrix

        Returns:
            Array of optimal weights
        """
        try:
            n_assets = len(volatilities)

            # Covariance matrix
            cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix

            # Objective function: minimize difference in risk contributions
            def objective(weights):
                portfolio_variance = weights @ cov_matrix @ weights
                portfolio_vol = np.sqrt(portfolio_variance)

                if portfolio_vol < 1e-10:
                    return 1e10  # Large penalty for zero volatility

                # Marginal contribution to risk
                mcr = (cov_matrix @ weights) / portfolio_vol

                # Risk contributions
                risk_contributions = weights * mcr

                # Target: equal risk contribution (1/n for each asset)
                target_rc = portfolio_vol / n_assets

                # Sum of squared deviations from target
                return np.sum((risk_contributions - target_rc) ** 2)

            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
            ]

            # Bounds: each weight between 0 and max_position_size_pct
            bounds = [(0.0, float(self.max_position_size_pct)) for _ in range(n_assets)]

            # Initial guess: equal weights
            initial_weights = np.ones(n_assets) / n_assets

            # Optimize
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )

            if not result.success:
                logger.warning(
                    f"Optimization did not converge: {result.message}, "
                    f"falling back to inverse volatility weighting"
                )
                # Fallback to simple inverse volatility
                inv_vols = 1.0 / volatilities
                return inv_vols / np.sum(inv_vols)

            logger.debug(f"Risk parity optimization converged: {result.message}")

            return result.x

        except Exception as e:
            logger.error(f"Error in risk parity optimization: {e}", exc_info=True)
            # Fallback to inverse volatility
            inv_vols = 1.0 / volatilities
            return inv_vols / np.sum(inv_vols)

    def _extract_correlation_matrix(
        self,
        correlations: pl.DataFrame,
        symbols: List[str]
    ) -> np.ndarray:
        """
        Extract correlation matrix for specified symbols

        Args:
            correlations: Polars DataFrame with correlation data
            symbols: List of symbols in order

        Returns:
            Numpy array correlation matrix
        """
        try:
            n = len(symbols)
            corr_matrix = np.eye(n)  # Start with identity matrix

            # If DataFrame format is symbol1, symbol2, correlation
            if 'symbol1' in correlations.columns and 'symbol2' in correlations.columns:
                for i, sym1 in enumerate(symbols):
                    for j, sym2 in enumerate(symbols):
                        if i == j:
                            corr_matrix[i, j] = 1.0
                        else:
                            # Find correlation
                            corr_row = correlations.filter(
                                ((pl.col('symbol1') == sym1) & (pl.col('symbol2') == sym2)) |
                                ((pl.col('symbol1') == sym2) & (pl.col('symbol2') == sym1))
                            )

                            if not corr_row.is_empty():
                                corr_value = float(corr_row.select('correlation').to_numpy()[0, 0])
                                corr_matrix[i, j] = corr_value
                                corr_matrix[j, i] = corr_value

            logger.debug(f"Extracted {n}x{n} correlation matrix")

            return corr_matrix

        except Exception as e:
            logger.error(f"Error extracting correlation matrix: {e}", exc_info=True)
            # Return identity matrix as fallback
            return np.eye(len(symbols))

    async def calculate_position_sizes(
        self,
        weights: Dict[str, Decimal],
        portfolio_value: Decimal,
        prices: Dict[str, Decimal]
    ) -> Dict[str, Decimal]:
        """
        Convert weights to position sizes

        Args:
            weights: Portfolio weights
            portfolio_value: Total portfolio value
            prices: Current prices

        Returns:
            Dictionary mapping symbol to quantity
        """
        try:
            positions = {}

            for symbol, weight in weights.items():
                price = prices.get(symbol, Decimal("0"))

                if price <= Decimal("0"):
                    logger.warning(f"Invalid price for {symbol}, skipping")
                    continue

                # Calculate dollar allocation
                allocation = weight * portfolio_value

                # Check minimum size
                if allocation < self.min_position_size_usd:
                    logger.debug(
                        f"Allocation for {symbol} (${allocation}) below minimum "
                        f"${self.min_position_size_usd}, skipping"
                    )
                    continue

                # Calculate quantity
                quantity = allocation / price

                positions[symbol] = quantity.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)

            logger.info(
                f"Calculated position sizes for {len(positions)} assets "
                f"from {len(weights)} weights"
            )

            return positions

        except Exception as e:
            logger.error(f"Error calculating position sizes: {e}", exc_info=True)
            raise
