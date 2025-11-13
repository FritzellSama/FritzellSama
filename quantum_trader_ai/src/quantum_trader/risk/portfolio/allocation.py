"""
Portfolio Allocation Optimization
CRITICAL: Mean-variance optimization, efficient frontier, risk parity, and portfolio rebalancing
"""

import logging
import numpy as np
import polars as pl
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from scipy.optimize import minimize
from scipy.linalg import sqrtm

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class AllocationOptimizer:
    """Portfolio allocation optimization using modern portfolio theory"""

    def __init__(self):
        """Initialize allocation optimizer with config"""
        self.config = get_config()
        self.max_position_pct = self.config.get_decimal("risk", "position_sizing.max_position_size_pct")
        self.min_assets = self.config.get_int("risk", "portfolio.diversification_min_assets", 10)
        self.rebalance_threshold = self.config.get_decimal("risk", "portfolio.rebalance_threshold_pct")
        self.target_sharpe = self.config.get_decimal("risk", "risk_metrics.target_sharpe_ratio")

        logger.info(
            f"AllocationOptimizer initialized: max_position={self.max_position_pct}, "
            f"min_assets={self.min_assets}, rebalance_threshold={self.rebalance_threshold}"
        )

    async def optimize_allocation(
        self,
        returns_df: pl.DataFrame,
        method: str = "mean_variance",
        constraints: Optional[Dict[str, Decimal]] = None
    ) -> Dict[str, Decimal]:
        """
        Optimize portfolio allocation using specified method

        Args:
            returns_df: Polars DataFrame with columns ['timestamp', 'asset', 'return']
            method: Optimization method ('mean_variance', 'risk_parity', 'max_sharpe', 'min_variance')
            constraints: Optional constraints dict with asset-level limits

        Returns:
            Dictionary mapping asset symbols to allocation weights (as Decimals)
        """
        try:
            if returns_df.is_empty():
                logger.error("Cannot optimize allocation: empty returns DataFrame")
                raise ValueError("Returns DataFrame is empty")

            # Convert to pivot table for optimization (assets as columns)
            returns_pivot = self._prepare_returns_matrix(returns_df)

            if returns_pivot.shape[1] < self.min_assets:
                logger.warning(
                    f"Asset count ({returns_pivot.shape[1]}) below minimum ({self.min_assets}). "
                    "Proceeding with available assets."
                )

            # Calculate expected returns and covariance matrix
            expected_returns = np.array([returns_pivot[:, i].mean() for i in range(returns_pivot.shape[1])])
            cov_matrix = np.cov(returns_pivot.T)

            assets = returns_df.select("asset").unique().to_series().to_list()

            # Apply optimization method
            if method == "mean_variance":
                weights = await self._optimize_mean_variance(expected_returns, cov_matrix, assets, constraints)
            elif method == "risk_parity":
                weights = await self._optimize_risk_parity(cov_matrix, assets, constraints)
            elif method == "max_sharpe":
                weights = await self._optimize_max_sharpe(expected_returns, cov_matrix, assets, constraints)
            elif method == "min_variance":
                weights = await self._optimize_min_variance(cov_matrix, assets, constraints)
            else:
                raise ValueError(f"Unknown optimization method: {method}")

            # Convert to Decimal and validate
            allocation = {
                asset: Decimal(str(weight)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
                for asset, weight in zip(assets, weights)
            }

            # Validate allocation sums to 1.0
            total_weight = sum(allocation.values())
            if abs(total_weight - Decimal("1.0")) > Decimal("0.0001"):
                logger.error(f"Allocation weights sum to {total_weight}, not 1.0")
                raise ValueError("Allocation weights must sum to 1.0")

            logger.info(
                f"Optimized allocation using {method}: {len(allocation)} assets, "
                f"max_weight={max(allocation.values()):.4f}, min_weight={min(allocation.values()):.4f}"
            )

            return allocation

        except Exception as e:
            logger.error(f"Error optimizing allocation: {e}", exc_info=True)
            raise

    async def _optimize_mean_variance(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        assets: List[str],
        constraints: Optional[Dict[str, Decimal]]
    ) -> np.ndarray:
        """Mean-variance optimization (maximize Sharpe ratio)"""
        n_assets = len(assets)

        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            # Negative Sharpe ratio (we minimize)
            return -portfolio_return / (portfolio_std + 1e-10)

        # Constraints: weights sum to 1
        constraints_list = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Bounds: 0 <= weight <= max_position_pct (or custom constraint)
        bounds = []
        max_pct_float = float(self.max_position_pct)

        for asset in assets:
            if constraints and asset in constraints:
                max_weight = float(constraints[asset])
            else:
                max_weight = max_pct_float
            bounds.append((0.0, max_weight))

        # Initial guess: equal weights
        initial_weights = np.array([1.0 / n_assets] * n_assets)

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_list,
            options={"maxiter": 1000}
        )

        if not result.success:
            logger.warning(f"Mean-variance optimization did not fully converge: {result.message}")

        return result.x

    async def _optimize_risk_parity(
        self,
        cov_matrix: np.ndarray,
        assets: List[str],
        constraints: Optional[Dict[str, Decimal]]
    ) -> np.ndarray:
        """Risk parity optimization (equal risk contribution)"""
        n_assets = len(assets)

        def risk_contribution(weights, cov_matrix):
            """Calculate risk contribution of each asset"""
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / (portfolio_vol + 1e-10)
            return risk_contrib

        def objective(weights):
            """Minimize variance of risk contributions"""
            rc = risk_contribution(weights, cov_matrix)
            target_rc = np.ones(n_assets) / n_assets
            return np.sum((rc - target_rc) ** 2)

        # Constraints
        constraints_list = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Bounds
        bounds = []
        max_pct_float = float(self.max_position_pct)

        for asset in assets:
            if constraints and asset in constraints:
                max_weight = float(constraints[asset])
            else:
                max_weight = max_pct_float
            bounds.append((0.0001, max_weight))  # Small minimum to avoid division by zero

        # Initial guess: equal weights
        initial_weights = np.array([1.0 / n_assets] * n_assets)

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_list,
            options={"maxiter": 1000}
        )

        if not result.success:
            logger.warning(f"Risk parity optimization did not fully converge: {result.message}")

        return result.x

    async def _optimize_max_sharpe(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        assets: List[str],
        constraints: Optional[Dict[str, Decimal]]
    ) -> np.ndarray:
        """Maximize Sharpe ratio (same as mean-variance)"""
        return await self._optimize_mean_variance(expected_returns, cov_matrix, assets, constraints)

    async def _optimize_min_variance(
        self,
        cov_matrix: np.ndarray,
        assets: List[str],
        constraints: Optional[Dict[str, Decimal]]
    ) -> np.ndarray:
        """Minimum variance portfolio"""
        n_assets = len(assets)

        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        # Constraints
        constraints_list = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Bounds
        bounds = []
        max_pct_float = float(self.max_position_pct)

        for asset in assets:
            if constraints and asset in constraints:
                max_weight = float(constraints[asset])
            else:
                max_weight = max_pct_float
            bounds.append((0.0, max_weight))

        # Initial guess: equal weights
        initial_weights = np.array([1.0 / n_assets] * n_assets)

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_list,
            options={"maxiter": 1000}
        )

        if not result.success:
            logger.warning(f"Min variance optimization did not fully converge: {result.message}")

        return result.x

    async def calculate_efficient_frontier(
        self,
        returns_df: pl.DataFrame,
        num_points: int = 50
    ) -> pl.DataFrame:
        """
        Calculate the efficient frontier (risk-return tradeoff)

        Args:
            returns_df: Polars DataFrame with columns ['timestamp', 'asset', 'return']
            num_points: Number of points on the frontier

        Returns:
            Polars DataFrame with columns ['expected_return', 'volatility', 'sharpe_ratio', 'weights']
        """
        try:
            if returns_df.is_empty():
                logger.error("Cannot calculate efficient frontier: empty returns DataFrame")
                raise ValueError("Returns DataFrame is empty")

            # Prepare returns matrix
            returns_pivot = self._prepare_returns_matrix(returns_df)

            # Calculate statistics
            expected_returns = np.array([returns_pivot[:, i].mean() for i in range(returns_pivot.shape[1])])
            cov_matrix = np.cov(returns_pivot.T)

            assets = returns_df.select("asset").unique().to_series().to_list()
            n_assets = len(assets)

            # Calculate min and max return portfolios
            min_return = expected_returns.min()
            max_return = expected_returns.max()

            # Generate target returns
            target_returns = np.linspace(min_return, max_return, num_points)

            frontier_points = []

            for target_return in target_returns:
                try:
                    # Optimize for minimum variance given target return
                    def objective(weights):
                        return np.dot(weights.T, np.dot(cov_matrix, weights))

                    constraints_list = [
                        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                        {"type": "eq", "fun": lambda w: np.dot(w, expected_returns) - target_return}
                    ]

                    bounds = [(0.0, float(self.max_position_pct))] * n_assets
                    initial_weights = np.array([1.0 / n_assets] * n_assets)

                    result = minimize(
                        objective,
                        initial_weights,
                        method="SLSQP",
                        bounds=bounds,
                        constraints=constraints_list,
                        options={"maxiter": 1000}
                    )

                    if result.success:
                        weights = result.x
                        portfolio_return = np.dot(weights, expected_returns)
                        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                        sharpe = portfolio_return / (portfolio_vol + 1e-10) if portfolio_vol > 0 else 0.0

                        frontier_points.append({
                            "expected_return": Decimal(str(portfolio_return)).quantize(Decimal("0.000001")),
                            "volatility": Decimal(str(portfolio_vol)).quantize(Decimal("0.000001")),
                            "sharpe_ratio": Decimal(str(sharpe)).quantize(Decimal("0.000001")),
                            "weights": str({asset: float(w) for asset, w in zip(assets, weights)})
                        })

                except Exception as e:
                    logger.debug(f"Could not optimize for target return {target_return}: {e}")
                    continue

            if not frontier_points:
                logger.error("Failed to generate any efficient frontier points")
                raise ValueError("Could not calculate efficient frontier")

            frontier_df = pl.DataFrame(frontier_points)

            logger.info(
                f"Calculated efficient frontier: {len(frontier_points)} points, "
                f"return range [{frontier_df['expected_return'].min():.6f}, {frontier_df['expected_return'].max():.6f}]"
            )

            return frontier_df

        except Exception as e:
            logger.error(f"Error calculating efficient frontier: {e}", exc_info=True)
            raise

    async def rebalance_portfolio(
        self,
        current_allocation: Dict[str, Decimal],
        target_allocation: Dict[str, Decimal],
        portfolio_value: Decimal
    ) -> Dict[str, Dict[str, Decimal]]:
        """
        Calculate rebalancing trades to move from current to target allocation

        Args:
            current_allocation: Current portfolio weights (asset -> weight)
            target_allocation: Target portfolio weights (asset -> weight)
            portfolio_value: Total portfolio value in USD

        Returns:
            Dictionary with 'trades' (asset -> {'action': 'buy'/'sell', 'amount': Decimal})
            and 'should_rebalance' (bool)
        """
        try:
            # Validate inputs
            if not current_allocation or not target_allocation:
                logger.error("Cannot rebalance: empty allocation dictionaries")
                raise ValueError("Allocation dictionaries cannot be empty")

            # Get all assets (union of current and target)
            all_assets = set(current_allocation.keys()) | set(target_allocation.keys())

            # Calculate drift for each asset
            max_drift = Decimal("0")
            drifts = {}

            for asset in all_assets:
                current_weight = current_allocation.get(asset, Decimal("0"))
                target_weight = target_allocation.get(asset, Decimal("0"))
                drift = abs(target_weight - current_weight)
                drifts[asset] = drift
                max_drift = max(max_drift, drift)

            # Check if rebalancing is needed
            should_rebalance = max_drift > self.rebalance_threshold

            trades = {}

            if should_rebalance:
                # Calculate trades
                for asset in all_assets:
                    current_weight = current_allocation.get(asset, Decimal("0"))
                    target_weight = target_allocation.get(asset, Decimal("0"))

                    weight_change = target_weight - current_weight
                    dollar_change = (weight_change * portfolio_value).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )

                    if abs(dollar_change) > Decimal("1.0"):  # Ignore tiny trades
                        action = "buy" if dollar_change > 0 else "sell"
                        trades[asset] = {
                            "action": action,
                            "amount": abs(dollar_change),
                            "current_weight": current_weight,
                            "target_weight": target_weight,
                            "drift": drifts[asset]
                        }

                logger.info(
                    f"Rebalancing recommended: max_drift={max_drift:.4f} "
                    f"(threshold={self.rebalance_threshold}), {len(trades)} trades"
                )
            else:
                logger.info(
                    f"No rebalancing needed: max_drift={max_drift:.4f} "
                    f"< threshold={self.rebalance_threshold}"
                )

            return {
                "should_rebalance": should_rebalance,
                "trades": trades,
                "max_drift": max_drift,
                "portfolio_value": portfolio_value
            }

        except Exception as e:
            logger.error(f"Error calculating rebalancing trades: {e}", exc_info=True)
            raise

    def _prepare_returns_matrix(self, returns_df: pl.DataFrame) -> np.ndarray:
        """
        Convert long-format returns DataFrame to pivot matrix (time x assets)

        Args:
            returns_df: DataFrame with columns ['timestamp', 'asset', 'return']

        Returns:
            Numpy array of returns (rows=time, columns=assets)
        """
        try:
            # Pivot to wide format
            pivot_df = returns_df.pivot(
                values="return",
                index="timestamp",
                columns="asset"
            )

            # Convert to numpy (excluding timestamp column)
            returns_matrix = pivot_df.select(pl.exclude("timestamp")).to_numpy()

            # Fill NaN with 0 (missing returns)
            returns_matrix = np.nan_to_num(returns_matrix, nan=0.0)

            return returns_matrix

        except Exception as e:
            logger.error(f"Error preparing returns matrix: {e}", exc_info=True)
            raise
