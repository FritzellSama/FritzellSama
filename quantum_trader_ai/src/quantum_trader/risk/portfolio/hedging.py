"""
Portfolio Hedging Strategies
CRITICAL: Calculate hedge ratios and manage portfolio hedging positions
"""

import logging
import numpy as np
import polars as pl
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class PortfolioHedger:
    """Manage portfolio hedging strategies and positions"""

    def __init__(self):
        """Initialize portfolio hedger with config"""
        self.config = get_config()
        self.hedge_threshold = self.config.get_decimal("risk", "portfolio.hedge_threshold_pct")
        self.max_leverage = self.config.get_decimal("risk", "position_limits.max_leverage")

        self.active_hedges: Dict[str, Dict] = {}

        logger.info(
            f"PortfolioHedger initialized: hedge_threshold={self.hedge_threshold}, "
            f"max_leverage={self.max_leverage}"
        )

    async def calculate_hedge_ratio(
        self,
        portfolio_returns: pl.DataFrame,
        hedge_asset_returns: pl.DataFrame,
        method: str = "minimum_variance"
    ) -> Dict[str, Decimal]:
        """
        Calculate optimal hedge ratio using various methods

        Args:
            portfolio_returns: DataFrame with columns ['timestamp', 'return']
            hedge_asset_returns: DataFrame with columns ['timestamp', 'return']
            method: Hedge calculation method ('minimum_variance', 'beta', 'correlation')

        Returns:
            Dictionary with hedge ratio and metrics:
            {
                'hedge_ratio': Decimal,
                'effectiveness': Decimal (R-squared),
                'method': str,
                'correlation': Decimal
            }
        """
        try:
            if portfolio_returns.is_empty() or hedge_asset_returns.is_empty():
                logger.error("Cannot calculate hedge ratio: empty returns data")
                raise ValueError("Returns DataFrames cannot be empty")

            # Merge returns on timestamp
            merged_df = portfolio_returns.join(
                hedge_asset_returns.rename({"return": "hedge_return"}),
                on="timestamp",
                how="inner"
            )

            if merged_df.is_empty():
                logger.error("No overlapping timestamps for hedge calculation")
                raise ValueError("No matching timestamps between portfolio and hedge asset")

            # Extract returns as numpy arrays
            portfolio_ret = merged_df.select("return").to_numpy().flatten()
            hedge_ret = merged_df.select("hedge_return").to_numpy().flatten()

            # Calculate based on method
            if method == "minimum_variance":
                hedge_ratio = self._calculate_minimum_variance_hedge(portfolio_ret, hedge_ret)
            elif method == "beta":
                hedge_ratio = self._calculate_beta_hedge(portfolio_ret, hedge_ret)
            elif method == "correlation":
                hedge_ratio = self._calculate_correlation_hedge(portfolio_ret, hedge_ret)
            else:
                raise ValueError(f"Unknown hedge calculation method: {method}")

            # Calculate hedge effectiveness (R-squared)
            correlation = np.corrcoef(portfolio_ret, hedge_ret)[0, 1]
            r_squared = correlation ** 2

            # Calculate correlation
            correlation_decimal = Decimal(str(correlation)).quantize(Decimal("0.0001"))

            result = {
                "hedge_ratio": hedge_ratio,
                "effectiveness": Decimal(str(r_squared)).quantize(Decimal("0.0001")),
                "correlation": correlation_decimal,
                "method": method,
                "observations": len(portfolio_ret),
                "timestamp": datetime.now().isoformat()
            }

            logger.info(
                f"Hedge ratio calculated: {hedge_ratio:.4f} ({method}), "
                f"effectiveness={r_squared:.4f}, correlation={correlation:.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error calculating hedge ratio: {e}", exc_info=True)
            raise

    async def create_hedge_position(
        self,
        portfolio_value: Decimal,
        hedge_ratio: Decimal,
        hedge_asset: str,
        hedge_price: Decimal,
        exposure_to_hedge: Decimal
    ) -> Dict[str, Decimal]:
        """
        Create a hedge position

        Args:
            portfolio_value: Total portfolio value
            hedge_ratio: Optimal hedge ratio
            hedge_asset: Asset to use for hedging
            hedge_price: Current price of hedge asset
            exposure_to_hedge: Dollar amount of exposure to hedge

        Returns:
            Dictionary with hedge position details:
            {
                'hedge_size_usd': Decimal,
                'hedge_size_units': Decimal,
                'hedge_ratio': Decimal,
                'exposure_hedged': Decimal,
                'hedge_pct': Decimal
            }
        """
        try:
            # Calculate hedge size in USD
            hedge_size_usd = (exposure_to_hedge * hedge_ratio).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

            # Calculate hedge size in units
            hedge_size_units = (hedge_size_usd / hedge_price).quantize(
                Decimal("0.00000001"), rounding=ROUND_HALF_UP
            )

            # Calculate percentage of portfolio
            hedge_pct = ((hedge_size_usd / portfolio_value) * Decimal("100")).quantize(
                Decimal("0.01")
            ) if portfolio_value > 0 else Decimal("0")

            # Store active hedge
            hedge_id = f"{hedge_asset}_{datetime.now().timestamp()}"
            self.active_hedges[hedge_id] = {
                "hedge_asset": hedge_asset,
                "hedge_size_usd": hedge_size_usd,
                "hedge_size_units": hedge_size_units,
                "hedge_ratio": hedge_ratio,
                "entry_price": hedge_price,
                "exposure_hedged": exposure_to_hedge,
                "timestamp": datetime.now().isoformat()
            }

            result = {
                "hedge_id": hedge_id,
                "hedge_asset": hedge_asset,
                "hedge_size_usd": hedge_size_usd,
                "hedge_size_units": hedge_size_units,
                "hedge_ratio": hedge_ratio,
                "exposure_hedged": exposure_to_hedge,
                "hedge_pct": hedge_pct,
                "entry_price": hedge_price,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(
                f"Hedge position created: {hedge_asset}, ${hedge_size_usd:,.2f} "
                f"({hedge_size_units:.8f} units), ratio={hedge_ratio:.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error creating hedge position: {e}", exc_info=True)
            raise

    async def monitor_hedge_effectiveness(
        self,
        hedge_id: str,
        current_portfolio_value: Decimal,
        current_hedge_price: Decimal,
        portfolio_returns: pl.DataFrame,
        hedge_returns: pl.DataFrame
    ) -> Dict[str, any]:
        """
        Monitor effectiveness of an active hedge

        Args:
            hedge_id: Hedge identifier
            current_portfolio_value: Current portfolio value
            current_hedge_price: Current price of hedge asset
            portfolio_returns: Recent portfolio returns
            hedge_returns: Recent hedge asset returns

        Returns:
            Dictionary with hedge monitoring results:
            {
                'is_effective': bool,
                'current_effectiveness': Decimal,
                'hedge_pnl': Decimal,
                'portfolio_change': Decimal,
                'needs_rebalancing': bool,
                'recommendation': str
            }
        """
        try:
            if hedge_id not in self.active_hedges:
                logger.error(f"Hedge not found: {hedge_id}")
                raise ValueError(f"Unknown hedge ID: {hedge_id}")

            hedge = self.active_hedges[hedge_id]

            # Calculate hedge P&L
            entry_price = hedge["entry_price"]
            hedge_units = hedge["hedge_size_units"]
            hedge_pnl = (hedge_units * (current_hedge_price - entry_price)).quantize(
                Decimal("0.01")
            )

            # Recalculate current hedge ratio
            current_hedge_ratio_result = await self.calculate_hedge_ratio(
                portfolio_returns, hedge_returns, method="minimum_variance"
            )
            current_hedge_ratio = current_hedge_ratio_result["hedge_ratio"]
            current_effectiveness = current_hedge_ratio_result["effectiveness"]

            # Compare with original hedge ratio
            original_hedge_ratio = hedge["hedge_ratio"]
            hedge_drift = abs(current_hedge_ratio - original_hedge_ratio)

            # Check if rebalancing is needed (drift > 20%)
            needs_rebalancing = hedge_drift > (original_hedge_ratio * Decimal("0.2"))

            # Assess effectiveness (R² > 0.5 is considered effective)
            is_effective = current_effectiveness > Decimal("0.5")

            # Calculate portfolio change
            exposure_hedged = hedge["exposure_hedged"]
            portfolio_change = current_portfolio_value - exposure_hedged

            # Determine recommendation
            recommendation = self._get_hedge_recommendation(
                is_effective, needs_rebalancing, hedge_drift, current_effectiveness
            )

            result = {
                "hedge_id": hedge_id,
                "hedge_asset": hedge["hedge_asset"],
                "is_effective": is_effective,
                "current_effectiveness": current_effectiveness,
                "original_hedge_ratio": original_hedge_ratio,
                "current_hedge_ratio": current_hedge_ratio,
                "hedge_drift": hedge_drift.quantize(Decimal("0.0001")),
                "hedge_pnl": hedge_pnl,
                "hedge_pnl_pct": ((hedge_pnl / hedge["hedge_size_usd"]) * Decimal("100")).quantize(
                    Decimal("0.01")
                ) if hedge["hedge_size_usd"] > 0 else Decimal("0"),
                "portfolio_change": portfolio_change,
                "needs_rebalancing": needs_rebalancing,
                "recommendation": recommendation,
                "timestamp": datetime.now().isoformat()
            }

            if not is_effective or needs_rebalancing:
                logger.warning(
                    f"Hedge {hedge_id}: effective={is_effective}, "
                    f"effectiveness={current_effectiveness:.4f}, "
                    f"needs_rebalancing={needs_rebalancing}"
                )
            else:
                logger.info(
                    f"Hedge {hedge_id}: effective, effectiveness={current_effectiveness:.4f}, "
                    f"P&L=${hedge_pnl:,.2f}"
                )

            return result

        except Exception as e:
            logger.error(f"Error monitoring hedge effectiveness: {e}", exc_info=True)
            raise

    async def calculate_dynamic_hedge_ratio(
        self,
        portfolio_returns: pl.DataFrame,
        hedge_asset_returns: pl.DataFrame,
        lookback_windows: List[int] = [20, 60, 120]
    ) -> Dict[str, Decimal]:
        """
        Calculate dynamic hedge ratio using multiple lookback periods

        Args:
            portfolio_returns: Portfolio returns DataFrame
            hedge_asset_returns: Hedge asset returns DataFrame
            lookback_windows: List of lookback periods in days

        Returns:
            Dictionary with dynamic hedge ratios for each window
        """
        try:
            results = {}

            for window in lookback_windows:
                # Get returns for lookback window
                cutoff_timestamp = datetime.now().timestamp() - (window * 86400)

                window_portfolio_returns = portfolio_returns.filter(
                    pl.col("timestamp") >= cutoff_timestamp
                )
                window_hedge_returns = hedge_asset_returns.filter(
                    pl.col("timestamp") >= cutoff_timestamp
                )

                if len(window_portfolio_returns) < 10:
                    logger.warning(f"Insufficient data for {window}-day window, skipping")
                    continue

                # Calculate hedge ratio
                hedge_result = await self.calculate_hedge_ratio(
                    window_portfolio_returns,
                    window_hedge_returns,
                    method="minimum_variance"
                )

                results[f"hedge_ratio_{window}d"] = hedge_result["hedge_ratio"]
                results[f"effectiveness_{window}d"] = hedge_result["effectiveness"]

            # Calculate weighted average (favor shorter windows)
            if results:
                weights = [Decimal("0.5"), Decimal("0.3"), Decimal("0.2")][:len(lookback_windows)]
                hedge_ratios = [
                    results.get(f"hedge_ratio_{w}d", Decimal("0"))
                    for w in lookback_windows
                ]

                weighted_hedge_ratio = sum(
                    ratio * weight for ratio, weight in zip(hedge_ratios, weights)
                ).quantize(Decimal("0.0001"))

                results["weighted_hedge_ratio"] = weighted_hedge_ratio
            else:
                results["weighted_hedge_ratio"] = Decimal("1.0")

            results["timestamp"] = datetime.now().isoformat()

            logger.info(
                f"Dynamic hedge ratios: {len(results)} windows, "
                f"weighted={results.get('weighted_hedge_ratio', 0):.4f}"
            )

            return results

        except Exception as e:
            logger.error(f"Error calculating dynamic hedge ratio: {e}", exc_info=True)
            raise

    def close_hedge(self, hedge_id: str) -> bool:
        """Close an active hedge position"""
        try:
            if hedge_id not in self.active_hedges:
                logger.error(f"Cannot close hedge: {hedge_id} not found")
                return False

            hedge = self.active_hedges[hedge_id]
            del self.active_hedges[hedge_id]

            logger.info(
                f"Hedge closed: {hedge_id}, asset={hedge['hedge_asset']}, "
                f"size=${hedge['hedge_size_usd']:,.2f}"
            )

            return True

        except Exception as e:
            logger.error(f"Error closing hedge: {e}", exc_info=True)
            return False

    def get_active_hedges(self) -> Dict[str, Dict]:
        """Get all active hedge positions"""
        return self.active_hedges.copy()

    def _calculate_minimum_variance_hedge(
        self,
        portfolio_returns: np.ndarray,
        hedge_returns: np.ndarray
    ) -> Decimal:
        """Calculate minimum variance hedge ratio: Cov(P,H) / Var(H)"""
        covariance = np.cov(portfolio_returns, hedge_returns)[0, 1]
        hedge_variance = np.var(hedge_returns, ddof=1)

        if hedge_variance < 1e-10:
            logger.warning("Hedge variance too low, using ratio of 1.0")
            return Decimal("1.0")

        hedge_ratio = covariance / hedge_variance

        return Decimal(str(hedge_ratio)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    def _calculate_beta_hedge(
        self,
        portfolio_returns: np.ndarray,
        hedge_returns: np.ndarray
    ) -> Decimal:
        """Calculate beta-based hedge ratio: same as minimum variance"""
        return self._calculate_minimum_variance_hedge(portfolio_returns, hedge_returns)

    def _calculate_correlation_hedge(
        self,
        portfolio_returns: np.ndarray,
        hedge_returns: np.ndarray
    ) -> Decimal:
        """
        Calculate correlation-based hedge ratio:
        hedge_ratio = correlation * (σ_portfolio / σ_hedge)
        """
        correlation = np.corrcoef(portfolio_returns, hedge_returns)[0, 1]
        portfolio_std = np.std(portfolio_returns, ddof=1)
        hedge_std = np.std(hedge_returns, ddof=1)

        if hedge_std < 1e-10:
            logger.warning("Hedge std dev too low, using ratio of 1.0")
            return Decimal("1.0")

        hedge_ratio = correlation * (portfolio_std / hedge_std)

        return Decimal(str(hedge_ratio)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    def _get_hedge_recommendation(
        self,
        is_effective: bool,
        needs_rebalancing: bool,
        hedge_drift: Decimal,
        effectiveness: Decimal
    ) -> str:
        """Get hedge recommendation based on monitoring results"""
        if not is_effective:
            if effectiveness < Decimal("0.3"):
                return "CLOSE_HEDGE - Ineffective hedge (R² < 0.3)"
            else:
                return "REBALANCE_HEDGE - Low effectiveness"

        if needs_rebalancing:
            return "REBALANCE_HEDGE - Significant drift detected"

        if hedge_drift > Decimal("0.1"):
            return "MONITOR_CLOSELY - Moderate drift detected"

        return "MAINTAIN_HEDGE - Effective and stable"
