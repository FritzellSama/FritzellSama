"""
Volatility-Based Position Sizing
CRITICAL: Position sizing based on volatility targeting and regime detection
"""

import logging
import numpy as np
import polars as pl
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class VolatilityBasedSizer:
    """Volatility-based position sizing with regime detection"""

    def __init__(self):
        """Initialize volatility-based sizer with config"""
        self.config = get_config()

        # Load configuration
        self.max_position_size_pct = self.config.get_decimal("risk", "position_sizing.max_position_size_pct")
        self.min_position_size_usd = self.config.get_decimal("risk", "position_sizing.min_position_size_usd")
        self.max_position_size_usd = self.config.get_decimal("risk", "position_limits.max_position_size_usd")

        # Default target volatility (annualized)
        self.target_volatility = Decimal("0.15")  # 15% annual volatility

        logger.info(
            f"VolatilityBasedSizer initialized: target_vol={self.target_volatility:.2%}, "
            f"max_size_pct={self.max_position_size_pct:.2%}"
        )

    async def calculate_position_size(
        self,
        symbol: str,
        portfolio_value: Decimal,
        asset_volatility: Decimal,
        target_volatility: Optional[Decimal] = None,
        current_price: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate position size based on volatility targeting

        Args:
            symbol: Asset symbol
            portfolio_value: Current portfolio value
            asset_volatility: Annualized volatility of asset
            target_volatility: Target portfolio volatility (default: 15%)
            current_price: Current asset price (for quantity calculation)

        Returns:
            Position size as percentage of portfolio (as Decimal)
        """
        try:
            if portfolio_value <= Decimal("0"):
                logger.error("Cannot calculate position size: invalid portfolio value")
                raise ValueError("Portfolio value must be positive")

            if asset_volatility <= Decimal("0"):
                logger.warning(f"Invalid volatility for {symbol}, using default 2%")
                asset_volatility = Decimal("0.02")

            if target_volatility is None:
                target_volatility = self.target_volatility

            # Position size = target_volatility / asset_volatility
            # This ensures position contributes target volatility to portfolio
            position_size_pct = target_volatility / asset_volatility

            # Apply constraints
            position_size_pct = max(Decimal("0"), min(position_size_pct, self.max_position_size_pct))

            # Check minimum dollar amount
            position_value = position_size_pct * portfolio_value
            if position_value < self.min_position_size_usd:
                logger.debug(
                    f"Position size ${position_value:,.2f} below minimum, "
                    f"adjusting to ${self.min_position_size_usd}"
                )
                position_size_pct = self.min_position_size_usd / portfolio_value

            # Check maximum dollar amount
            if position_value > self.max_position_size_usd:
                logger.warning(
                    f"Position size ${position_value:,.2f} exceeds maximum, "
                    f"capping at ${self.max_position_size_usd}"
                )
                position_size_pct = self.max_position_size_usd / portfolio_value

            position_size_pct = position_size_pct.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

            logger.info(
                f"Position size for {symbol}: {position_size_pct:.4%} "
                f"(${position_value:,.2f}), vol={asset_volatility:.2%}"
            )

            return position_size_pct

        except Exception as e:
            logger.error(f"Error calculating position size: {e}", exc_info=True)
            raise

    async def estimate_volatility(
        self,
        returns: pl.DataFrame,
        window_days: int = 20,
        method: str = "ewma"
    ) -> Decimal:
        """
        Estimate asset volatility

        Args:
            returns: DataFrame with columns ['timestamp', 'return']
            window_days: Rolling window size for estimation
            method: Estimation method ('simple', 'ewma', 'garch')

        Returns:
            Annualized volatility as Decimal
        """
        try:
            if returns.is_empty():
                logger.error("Cannot estimate volatility: empty returns")
                raise ValueError("Returns DataFrame cannot be empty")

            # Extract returns array
            returns_array = returns.select("return").to_numpy().flatten()

            if len(returns_array) < 2:
                logger.warning("Insufficient data for volatility estimation")
                return Decimal("0.02")  # Default 2%

            if method == "simple":
                # Simple standard deviation
                if len(returns_array) < window_days:
                    volatility = np.std(returns_array, ddof=1)
                else:
                    # Use most recent window_days
                    volatility = np.std(returns_array[-window_days:], ddof=1)

            elif method == "ewma":
                # Exponentially Weighted Moving Average
                # More weight on recent observations
                lambda_param = 0.94  # RiskMetrics standard

                # Initialize with first squared return
                ewma_var = returns_array[0] ** 2

                # Update EWMA variance
                for ret in returns_array[1:]:
                    ewma_var = lambda_param * ewma_var + (1 - lambda_param) * ret ** 2

                volatility = np.sqrt(ewma_var)

            elif method == "garch":
                # Simplified GARCH(1,1) estimation
                # Full GARCH would require arch package
                logger.warning("Full GARCH not implemented, using EWMA")
                return await self.estimate_volatility(returns, window_days, "ewma")

            else:
                raise ValueError(f"Unknown volatility estimation method: {method}")

            # Annualize (assuming 252 trading days)
            annualized_volatility = volatility * np.sqrt(252)

            volatility_decimal = Decimal(str(annualized_volatility)).quantize(
                Decimal("0.000001"), rounding=ROUND_HALF_UP
            )

            logger.info(
                f"Estimated volatility ({method}, {window_days}d): {volatility_decimal:.4%}"
            )

            return volatility_decimal

        except Exception as e:
            logger.error(f"Error estimating volatility: {e}", exc_info=True)
            raise

    async def adjust_for_regime(
        self,
        position_size: Decimal,
        current_volatility: Decimal,
        historical_volatility: Decimal,
        regime_threshold: Decimal = Decimal("1.5")
    ) -> Tuple[Decimal, str]:
        """
        Adjust position size based on volatility regime

        Args:
            position_size: Base position size
            current_volatility: Current volatility estimate
            historical_volatility: Long-term average volatility
            regime_threshold: Multiplier threshold for regime change

        Returns:
            Tuple of (adjusted_position_size, regime_label)
        """
        try:
            if historical_volatility <= Decimal("0"):
                logger.warning("Invalid historical volatility, no regime adjustment")
                return position_size, "unknown"

            # Calculate volatility ratio
            vol_ratio = current_volatility / historical_volatility

            # Determine regime and adjustment
            if vol_ratio >= regime_threshold:
                # High volatility regime - reduce position size
                regime = "high_volatility"
                adjustment_factor = Decimal("1") / vol_ratio
                adjusted_size = position_size * adjustment_factor

                logger.info(
                    f"High volatility regime detected: vol_ratio={vol_ratio:.2f}, "
                    f"reducing position by {(1 - adjustment_factor) * Decimal('100'):.1f}%"
                )

            elif vol_ratio <= Decimal("1") / regime_threshold:
                # Low volatility regime - can increase position size moderately
                regime = "low_volatility"
                adjustment_factor = min(Decimal("1.2"), Decimal("1") / vol_ratio)
                adjusted_size = position_size * adjustment_factor

                logger.info(
                    f"Low volatility regime detected: vol_ratio={vol_ratio:.2f}, "
                    f"increasing position by {(adjustment_factor - Decimal('1')) * Decimal('100'):.1f}%"
                )

            else:
                # Normal regime - no adjustment
                regime = "normal"
                adjusted_size = position_size

            # Apply maximum constraint
            adjusted_size = min(adjusted_size, self.max_position_size_pct)

            adjusted_size = adjusted_size.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

            logger.debug(
                f"Regime-adjusted position: {position_size:.4%} -> {adjusted_size:.4%} "
                f"(regime: {regime})"
            )

            return adjusted_size, regime

        except Exception as e:
            logger.error(f"Error adjusting for regime: {e}", exc_info=True)
            raise

    async def calculate_portfolio_volatility(
        self,
        position_volatilities: Dict[str, Decimal],
        position_weights: Dict[str, Decimal],
        correlations: Optional[pl.DataFrame] = None
    ) -> Decimal:
        """
        Calculate portfolio volatility from position volatilities and weights

        Args:
            position_volatilities: Dictionary mapping symbol to volatility
            position_weights: Dictionary mapping symbol to weight
            correlations: Optional correlation matrix

        Returns:
            Portfolio volatility as Decimal
        """
        try:
            if not position_volatilities or not position_weights:
                logger.error("Cannot calculate portfolio volatility: empty inputs")
                raise ValueError("Inputs cannot be empty")

            symbols = list(position_weights.keys())
            n_assets = len(symbols)

            # Convert to numpy arrays
            weights = np.array([float(position_weights.get(s, Decimal("0"))) for s in symbols])
            vols = np.array([float(position_volatilities.get(s, Decimal("0"))) for s in symbols])

            if correlations is None:
                # Assume uncorrelated - portfolio vol = sqrt(sum(w_i^2 * vol_i^2))
                portfolio_variance = np.sum((weights * vols) ** 2)

                logger.info("Calculating portfolio volatility (assuming uncorrelated)")

            else:
                # With correlations - full covariance matrix
                # Cov = diag(vols) @ Corr @ diag(vols)
                corr_matrix = self._extract_correlation_matrix(correlations, symbols)
                cov_matrix = np.outer(vols, vols) * corr_matrix

                # Portfolio variance = w^T @ Cov @ w
                portfolio_variance = weights @ cov_matrix @ weights

                logger.info("Calculating portfolio volatility (with correlations)")

            portfolio_vol = np.sqrt(portfolio_variance)

            portfolio_vol_decimal = Decimal(str(portfolio_vol)).quantize(
                Decimal("0.000001"), rounding=ROUND_HALF_UP
            )

            logger.info(f"Portfolio volatility: {portfolio_vol_decimal:.4%}")

            return portfolio_vol_decimal

        except Exception as e:
            logger.error(f"Error calculating portfolio volatility: {e}", exc_info=True)
            raise

    async def calculate_volatility_scaled_sizes(
        self,
        symbols: list,
        volatilities: Dict[str, Decimal],
        portfolio_value: Decimal,
        target_portfolio_vol: Optional[Decimal] = None
    ) -> Dict[str, Decimal]:
        """
        Calculate volatility-scaled position sizes for multiple assets

        Args:
            symbols: List of asset symbols
            volatilities: Dictionary mapping symbol to volatility
            portfolio_value: Current portfolio value
            target_portfolio_vol: Target portfolio volatility

        Returns:
            Dictionary mapping symbol to position size percentage
        """
        try:
            if not symbols or not volatilities:
                logger.error("Cannot calculate scaled sizes: empty inputs")
                raise ValueError("Symbols and volatilities cannot be empty")

            if target_portfolio_vol is None:
                target_portfolio_vol = self.target_volatility

            position_sizes = {}

            # Calculate inverse volatility weights
            inv_vols = {
                symbol: Decimal("1") / volatilities.get(symbol, Decimal("0.02"))
                for symbol in symbols
                if volatilities.get(symbol, Decimal("0")) > Decimal("0")
            }

            # Normalize to sum to target portfolio volatility
            total_inv_vol = sum(inv_vols.values())

            if total_inv_vol > Decimal("0"):
                for symbol, inv_vol in inv_vols.items():
                    # Weight proportional to inverse volatility
                    weight = inv_vol / total_inv_vol

                    # Scale to achieve target portfolio volatility
                    # This is simplified - assumes equal correlation
                    position_size = weight * target_portfolio_vol / volatilities[symbol]

                    # Apply constraints
                    position_size = max(Decimal("0"), min(position_size, self.max_position_size_pct))

                    position_sizes[symbol] = position_size.quantize(Decimal("0.000001"))

            # Normalize to ensure total doesn't exceed 100%
            total_size = sum(position_sizes.values())
            if total_size > Decimal("1"):
                position_sizes = {
                    symbol: (size / total_size).quantize(Decimal("0.000001"))
                    for symbol, size in position_sizes.items()
                }

            logger.info(
                f"Calculated volatility-scaled sizes for {len(position_sizes)} assets: "
                f"total={sum(position_sizes.values()):.2%}"
            )

            return position_sizes

        except Exception as e:
            logger.error(f"Error calculating volatility-scaled sizes: {e}", exc_info=True)
            raise

    def _extract_correlation_matrix(
        self,
        correlations: pl.DataFrame,
        symbols: list
    ) -> np.ndarray:
        """Extract correlation matrix from DataFrame"""
        try:
            n = len(symbols)
            corr_matrix = np.eye(n)

            if 'symbol1' in correlations.columns and 'symbol2' in correlations.columns:
                for i, sym1 in enumerate(symbols):
                    for j, sym2 in enumerate(symbols):
                        if i == j:
                            corr_matrix[i, j] = 1.0
                        else:
                            corr_row = correlations.filter(
                                ((pl.col('symbol1') == sym1) & (pl.col('symbol2') == sym2)) |
                                ((pl.col('symbol1') == sym2) & (pl.col('symbol2') == sym1))
                            )

                            if not corr_row.is_empty():
                                corr_value = float(corr_row.select('correlation').to_numpy()[0, 0])
                                corr_matrix[i, j] = corr_value
                                corr_matrix[j, i] = corr_value

            return corr_matrix

        except Exception as e:
            logger.error(f"Error extracting correlation matrix: {e}", exc_info=True)
            return np.eye(len(symbols))
