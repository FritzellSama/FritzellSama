"""
Portfolio Diversification Metrics
CRITICAL: Calculate diversification metrics including Herfindahl index and concentration risk
"""

import logging
import numpy as np
import polars as pl
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class DiversificationAnalyzer:
    """Analyze portfolio diversification and concentration risk"""

    def __init__(self):
        """Initialize diversification analyzer with config"""
        self.config = get_config()
        self.max_concentration = self.config.get_decimal("risk", "position_limits.max_concentration_pct")
        self.min_assets = self.config.get_int("risk", "portfolio.diversification_min_assets", 10)
        self.max_position_pct = self.config.get_decimal("risk", "position_sizing.max_position_size_pct")

        logger.info(
            f"DiversificationAnalyzer initialized: max_concentration={self.max_concentration}, "
            f"min_assets={self.min_assets}, max_position_pct={self.max_position_pct}"
        )

    async def calculate_herfindahl_index(
        self,
        portfolio_weights: Dict[str, Decimal]
    ) -> Dict[str, Decimal]:
        """
        Calculate Herfindahl-Hirschman Index (HHI) for portfolio concentration

        Args:
            portfolio_weights: Dictionary mapping asset to portfolio weight (0-1)

        Returns:
            Dictionary with diversification metrics:
            {
                'hhi': Decimal (0-1, lower is more diversified),
                'effective_n': Decimal (effective number of assets),
                'diversification_ratio': Decimal (1/HHI),
                'is_diversified': bool
            }
        """
        try:
            if not portfolio_weights:
                logger.error("Cannot calculate HHI: empty portfolio weights")
                raise ValueError("Portfolio weights cannot be empty")

            # Validate weights sum to 1.0
            total_weight = sum(portfolio_weights.values())
            if abs(total_weight - Decimal("1.0")) > Decimal("0.0001"):
                logger.warning(f"Portfolio weights sum to {total_weight}, normalizing to 1.0")
                portfolio_weights = {
                    asset: weight / total_weight
                    for asset, weight in portfolio_weights.items()
                }

            # Calculate HHI: sum of squared weights
            hhi = sum(weight ** 2 for weight in portfolio_weights.values())

            # Effective number of assets (inverse of HHI)
            effective_n = Decimal("1") / hhi if hhi > 0 else Decimal("0")

            # Diversification ratio (same as effective_n)
            diversification_ratio = effective_n

            # Check if portfolio is adequately diversified
            is_diversified = (
                len(portfolio_weights) >= self.min_assets and
                hhi < (Decimal("1") / Decimal(str(self.min_assets)))
            )

            result = {
                "hhi": hhi.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP),
                "effective_n": effective_n.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                "diversification_ratio": diversification_ratio.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                "is_diversified": is_diversified,
                "num_assets": len(portfolio_weights),
                "min_required_assets": self.min_assets,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(
                f"HHI calculated: {result['hhi']:.6f}, effective_n={result['effective_n']:.2f}, "
                f"diversified={is_diversified}"
            )

            return result

        except Exception as e:
            logger.error(f"Error calculating Herfindahl index: {e}", exc_info=True)
            raise

    async def get_concentration_risk(
        self,
        portfolio_weights: Dict[str, Decimal],
        asset_correlations: Optional[Dict[Tuple[str, str], Decimal]] = None
    ) -> Dict[str, any]:
        """
        Assess portfolio concentration risk

        Args:
            portfolio_weights: Dictionary mapping asset to portfolio weight
            asset_correlations: Optional dictionary of pairwise correlations

        Returns:
            Dictionary with concentration risk analysis:
            {
                'concentration_risk': str ('low', 'medium', 'high', 'critical'),
                'top_positions': List[Tuple[str, Decimal]],
                'top_5_concentration': Decimal,
                'top_10_concentration': Decimal,
                'violations': List[str],
                'correlated_concentration': Optional[Decimal]
            }
        """
        try:
            if not portfolio_weights:
                logger.error("Cannot assess concentration risk: empty portfolio weights")
                raise ValueError("Portfolio weights cannot be empty")

            # Sort positions by weight (descending)
            sorted_positions = sorted(
                portfolio_weights.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Calculate top N concentrations
            top_5_concentration = sum(weight for _, weight in sorted_positions[:5])
            top_10_concentration = sum(weight for _, weight in sorted_positions[:10])

            # Check for position limit violations
            violations = []
            for asset, weight in portfolio_weights.items():
                if weight > self.max_concentration:
                    violations.append(
                        f"{asset}: {weight*100:.2f}% exceeds limit {self.max_concentration*100:.2f}%"
                    )

            # Calculate correlated concentration (if correlations provided)
            correlated_concentration = None
            if asset_correlations:
                correlated_concentration = self._calculate_correlated_concentration(
                    portfolio_weights, asset_correlations
                )

            # Determine overall concentration risk level
            risk_level = self._assess_concentration_risk_level(
                top_5_concentration,
                top_10_concentration,
                len(violations),
                correlated_concentration
            )

            result = {
                "concentration_risk": risk_level,
                "top_positions": sorted_positions[:10],
                "top_5_concentration": top_5_concentration.quantize(Decimal("0.0001")),
                "top_10_concentration": top_10_concentration.quantize(Decimal("0.0001")),
                "violations": violations,
                "num_violations": len(violations),
                "correlated_concentration": correlated_concentration,
                "num_assets": len(portfolio_weights),
                "timestamp": datetime.now().isoformat()
            }

            if violations:
                logger.warning(
                    f"Concentration risk: {risk_level} - {len(violations)} violations detected"
                )
            else:
                logger.info(
                    f"Concentration risk: {risk_level} - top 5: {top_5_concentration*100:.2f}%"
                )

            return result

        except Exception as e:
            logger.error(f"Error assessing concentration risk: {e}", exc_info=True)
            raise

    async def optimize_diversification(
        self,
        current_weights: Dict[str, Decimal],
        available_assets: List[str],
        target_hhi: Optional[Decimal] = None
    ) -> Dict[str, Decimal]:
        """
        Optimize portfolio to improve diversification

        Args:
            current_weights: Current portfolio weights
            available_assets: List of all available assets for diversification
            target_hhi: Target HHI value (default: 1/min_assets)

        Returns:
            Dictionary with optimized portfolio weights
        """
        try:
            if not available_assets:
                logger.error("Cannot optimize diversification: no available assets")
                raise ValueError("Available assets list cannot be empty")

            # Set target HHI
            if target_hhi is None:
                target_hhi = Decimal("1") / Decimal(str(self.min_assets))

            # Calculate current HHI
            current_hhi_result = await self.calculate_herfindahl_index(current_weights)
            current_hhi = current_hhi_result["hhi"]

            if current_hhi <= target_hhi:
                logger.info(f"Portfolio already meets diversification target: HHI={current_hhi:.6f}")
                return current_weights

            # Identify underweight or missing assets
            current_assets = set(current_weights.keys())
            missing_assets = set(available_assets) - current_assets

            # Strategy: reduce overweight positions and add missing assets
            optimized_weights = {}

            # Cap existing positions at max_position_pct
            reduced_total = Decimal("0")
            for asset, weight in current_weights.items():
                if weight > self.max_position_pct:
                    optimized_weights[asset] = self.max_position_pct
                    logger.info(f"Reducing {asset} from {weight:.4f} to {self.max_position_pct:.4f}")
                else:
                    optimized_weights[asset] = weight
                reduced_total += optimized_weights[asset]

            # Distribute remaining weight to new assets
            remaining_weight = Decimal("1.0") - reduced_total

            if remaining_weight > 0 and missing_assets:
                weight_per_new_asset = remaining_weight / Decimal(str(len(missing_assets)))

                for asset in missing_assets:
                    optimized_weights[asset] = weight_per_new_asset

                logger.info(
                    f"Added {len(missing_assets)} new assets with weight {weight_per_new_asset:.4f} each"
                )

            # Normalize weights to ensure they sum to 1.0
            total_weight = sum(optimized_weights.values())
            if abs(total_weight - Decimal("1.0")) > Decimal("0.0001"):
                optimized_weights = {
                    asset: (weight / total_weight).quantize(Decimal("0.000001"))
                    for asset, weight in optimized_weights.items()
                }

            # Calculate optimized HHI
            optimized_hhi_result = await self.calculate_herfindahl_index(optimized_weights)
            optimized_hhi = optimized_hhi_result["hhi"]

            logger.info(
                f"Diversification optimization: HHI {current_hhi:.6f} -> {optimized_hhi:.6f}, "
                f"assets {len(current_weights)} -> {len(optimized_weights)}"
            )

            return optimized_weights

        except Exception as e:
            logger.error(f"Error optimizing diversification: {e}", exc_info=True)
            raise

    async def calculate_diversification_benefit(
        self,
        asset_returns: pl.DataFrame,
        portfolio_weights: Dict[str, Decimal]
    ) -> Dict[str, Decimal]:
        """
        Calculate diversification benefit (reduction in risk vs. weighted average)

        Args:
            asset_returns: DataFrame with columns ['timestamp', 'asset', 'return']
            portfolio_weights: Portfolio allocation weights

        Returns:
            Dictionary with diversification benefit metrics
        """
        try:
            if asset_returns.is_empty():
                logger.error("Cannot calculate diversification benefit: empty returns")
                raise ValueError("Asset returns DataFrame is empty")

            # Calculate portfolio returns
            portfolio_returns = []

            timestamps = asset_returns.select("timestamp").unique().sort("timestamp")

            for ts in timestamps.to_series():
                period_returns = asset_returns.filter(pl.col("timestamp") == ts)

                portfolio_return = Decimal("0")
                for asset, weight in portfolio_weights.items():
                    asset_return = period_returns.filter(pl.col("asset") == asset).select("return")
                    if not asset_return.is_empty():
                        portfolio_return += weight * Decimal(str(asset_return.item()))

                portfolio_returns.append(float(portfolio_return))

            # Portfolio volatility
            portfolio_vol = Decimal(str(np.std(portfolio_returns, ddof=1)))

            # Calculate weighted average of individual volatilities
            weighted_avg_vol = Decimal("0")

            for asset, weight in portfolio_weights.items():
                asset_ret = asset_returns.filter(pl.col("asset") == asset).select("return").to_numpy().flatten()
                if len(asset_ret) > 1:
                    asset_vol = Decimal(str(np.std(asset_ret, ddof=1)))
                    weighted_avg_vol += weight * asset_vol

            # Diversification benefit
            diversification_benefit = (
                (weighted_avg_vol - portfolio_vol) / weighted_avg_vol * Decimal("100")
            ) if weighted_avg_vol > 0 else Decimal("0")

            result = {
                "portfolio_volatility": portfolio_vol.quantize(Decimal("0.000001")),
                "weighted_avg_volatility": weighted_avg_vol.quantize(Decimal("0.000001")),
                "diversification_benefit_pct": diversification_benefit.quantize(Decimal("0.01")),
                "risk_reduction": (weighted_avg_vol - portfolio_vol).quantize(Decimal("0.000001")),
                "timestamp": datetime.now().isoformat()
            }

            logger.info(
                f"Diversification benefit: {diversification_benefit:.2f}% risk reduction "
                f"(portfolio vol: {portfolio_vol:.6f}, weighted avg: {weighted_avg_vol:.6f})"
            )

            return result

        except Exception as e:
            logger.error(f"Error calculating diversification benefit: {e}", exc_info=True)
            raise

    def _calculate_correlated_concentration(
        self,
        portfolio_weights: Dict[str, Decimal],
        asset_correlations: Dict[Tuple[str, str], Decimal]
    ) -> Decimal:
        """
        Calculate concentration risk considering correlations

        Highly correlated positions increase concentration risk
        """
        try:
            assets = list(portfolio_weights.keys())
            n = len(assets)

            if n < 2:
                return Decimal("0")

            # Calculate effective concentration with correlation adjustment
            correlation_adjusted_concentration = Decimal("0")

            for i, asset_i in enumerate(assets):
                weight_i = portfolio_weights[asset_i]

                for j, asset_j in enumerate(assets):
                    if i == j:
                        continue

                    weight_j = portfolio_weights[asset_j]

                    # Get correlation (default to 0 if not found)
                    corr = asset_correlations.get((asset_i, asset_j), Decimal("0"))
                    if corr == Decimal("0"):
                        corr = asset_correlations.get((asset_j, asset_i), Decimal("0"))

                    # Add correlation-weighted concentration
                    correlation_adjusted_concentration += weight_i * weight_j * abs(corr)

            return correlation_adjusted_concentration.quantize(Decimal("0.000001"))

        except Exception as e:
            logger.error(f"Error calculating correlated concentration: {e}", exc_info=True)
            return Decimal("0")

    def _assess_concentration_risk_level(
        self,
        top_5_concentration: Decimal,
        top_10_concentration: Decimal,
        num_violations: int,
        correlated_concentration: Optional[Decimal]
    ) -> str:
        """Assess overall concentration risk level"""
        # Critical: severe violations or extreme concentration
        if num_violations > 0 or top_5_concentration > Decimal("0.7"):
            return "critical"

        # High: high concentration in top positions
        if top_5_concentration > Decimal("0.5") or top_10_concentration > Decimal("0.8"):
            return "high"

        # Medium: moderate concentration
        if top_5_concentration > Decimal("0.35") or (
            correlated_concentration and correlated_concentration > Decimal("0.3")
        ):
            return "medium"

        # Low: well diversified
        return "low"
