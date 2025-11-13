"""
Beta Calculator
CRITICAL: Calculate beta, alpha, and systematic risk against market benchmark
"""

import logging
import numpy as np
import polars as pl
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Tuple, Optional
from datetime import datetime

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class BetaCalculator:
    """Calculate beta and alpha metrics for portfolio analysis"""

    def __init__(self):
        """Initialize beta calculator with config"""
        self.config = get_config()
        self.var_confidence = self.config.get_decimal("risk", "risk_metrics.var_confidence")
        self.target_sharpe = self.config.get_decimal("risk", "risk_metrics.target_sharpe_ratio")

        logger.info(
            f"BetaCalculator initialized: var_confidence={self.var_confidence}, "
            f"target_sharpe={self.target_sharpe}"
        )

    async def calculate_beta(
        self,
        asset_returns: pl.DataFrame,
        market_returns: pl.DataFrame,
        window_days: Optional[int] = None
    ) -> Dict[str, Decimal]:
        """
        Calculate beta for each asset against market benchmark

        Args:
            asset_returns: DataFrame with columns ['timestamp', 'asset', 'return']
            market_returns: DataFrame with columns ['timestamp', 'return']
            window_days: Optional rolling window in days (None = all history)

        Returns:
            Dictionary mapping asset symbols to beta values (as Decimals)
        """
        try:
            if asset_returns.is_empty() or market_returns.is_empty():
                logger.error("Cannot calculate beta: empty returns DataFrame")
                raise ValueError("Returns DataFrames cannot be empty")

            # Merge asset and market returns on timestamp
            merged_df = asset_returns.join(
                market_returns.rename({"return": "market_return"}),
                on="timestamp",
                how="inner"
            )

            if merged_df.is_empty():
                logger.error("No overlapping timestamps between asset and market returns")
                raise ValueError("No matching timestamps for beta calculation")

            # Apply rolling window if specified
            if window_days:
                cutoff_date = datetime.now().timestamp() - (window_days * 86400)
                merged_df = merged_df.filter(pl.col("timestamp") >= cutoff_date)

            # Calculate beta for each asset
            betas = {}
            assets = merged_df.select("asset").unique().to_series().to_list()

            for asset in assets:
                asset_data = merged_df.filter(pl.col("asset") == asset)

                if len(asset_data) < 2:
                    logger.warning(f"Insufficient data for beta calculation: {asset}")
                    betas[asset] = Decimal("1.0")  # Default to market beta
                    continue

                # Extract returns as numpy arrays
                asset_ret = asset_data.select("return").to_numpy().flatten()
                market_ret = asset_data.select("market_return").to_numpy().flatten()

                # Calculate covariance and variance
                covariance = np.cov(asset_ret, market_ret)[0, 1]
                market_variance = np.var(market_ret, ddof=1)

                if market_variance < 1e-10:
                    logger.warning(f"Market variance too low for {asset}, using beta=1.0")
                    beta = 1.0
                else:
                    beta = covariance / market_variance

                betas[asset] = Decimal(str(beta)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

                logger.debug(f"Beta for {asset}: {betas[asset]}")

            logger.info(
                f"Calculated beta for {len(betas)} assets: "
                f"mean={np.mean([float(b) for b in betas.values()]):.4f}, "
                f"range=[{min(betas.values()):.4f}, {max(betas.values()):.4f}]"
            )

            return betas

        except Exception as e:
            logger.error(f"Error calculating beta: {e}", exc_info=True)
            raise

    async def calculate_alpha(
        self,
        asset_returns: pl.DataFrame,
        market_returns: pl.DataFrame,
        risk_free_rate: Decimal = Decimal("0.02"),
        window_days: Optional[int] = None
    ) -> Dict[str, Decimal]:
        """
        Calculate Jensen's alpha for each asset

        Args:
            asset_returns: DataFrame with columns ['timestamp', 'asset', 'return']
            market_returns: DataFrame with columns ['timestamp', 'return']
            risk_free_rate: Annual risk-free rate (default 2%)
            window_days: Optional rolling window in days

        Returns:
            Dictionary mapping asset symbols to alpha values (as Decimals)
        """
        try:
            if asset_returns.is_empty() or market_returns.is_empty():
                logger.error("Cannot calculate alpha: empty returns DataFrame")
                raise ValueError("Returns DataFrames cannot be empty")

            # Calculate beta first
            betas = await self.calculate_beta(asset_returns, market_returns, window_days)

            # Merge returns
            merged_df = asset_returns.join(
                market_returns.rename({"return": "market_return"}),
                on="timestamp",
                how="inner"
            )

            if window_days:
                cutoff_date = datetime.now().timestamp() - (window_days * 86400)
                merged_df = merged_df.filter(pl.col("timestamp") >= cutoff_date)

            # Calculate daily risk-free rate (assuming 252 trading days)
            daily_rf = float(risk_free_rate) / 252.0

            # Calculate alpha for each asset
            alphas = {}
            assets = merged_df.select("asset").unique().to_series().to_list()

            for asset in assets:
                asset_data = merged_df.filter(pl.col("asset") == asset)

                if len(asset_data) < 2:
                    logger.warning(f"Insufficient data for alpha calculation: {asset}")
                    alphas[asset] = Decimal("0.0")
                    continue

                # Extract returns
                asset_ret = asset_data.select("return").to_numpy().flatten()
                market_ret = asset_data.select("market_return").to_numpy().flatten()

                # Calculate average returns
                avg_asset_ret = np.mean(asset_ret)
                avg_market_ret = np.mean(market_ret)

                # Alpha = (Asset Return - Risk Free) - Beta * (Market Return - Risk Free)
                beta = float(betas[asset])
                alpha = (avg_asset_ret - daily_rf) - beta * (avg_market_ret - daily_rf)

                # Annualize alpha
                annualized_alpha = alpha * 252.0

                alphas[asset] = Decimal(str(annualized_alpha)).quantize(
                    Decimal("0.000001"), rounding=ROUND_HALF_UP
                )

                logger.debug(f"Alpha for {asset}: {alphas[asset]}")

            logger.info(
                f"Calculated alpha for {len(alphas)} assets: "
                f"mean={np.mean([float(a) for a in alphas.values()]):.6f}"
            )

            return alphas

        except Exception as e:
            logger.error(f"Error calculating alpha: {e}", exc_info=True)
            raise

    async def get_systematic_risk(
        self,
        asset_returns: pl.DataFrame,
        market_returns: pl.DataFrame,
        window_days: Optional[int] = None
    ) -> Dict[str, Dict[str, Decimal]]:
        """
        Decompose total risk into systematic and idiosyncratic components

        Args:
            asset_returns: DataFrame with columns ['timestamp', 'asset', 'return']
            market_returns: DataFrame with columns ['timestamp', 'return']
            window_days: Optional rolling window in days

        Returns:
            Dictionary with risk decomposition for each asset:
            {
                'asset': {
                    'total_risk': Decimal (std dev),
                    'systematic_risk': Decimal (beta * market std),
                    'idiosyncratic_risk': Decimal (residual std),
                    'systematic_pct': Decimal (% of total variance),
                    'r_squared': Decimal (correlation^2)
                }
            }
        """
        try:
            if asset_returns.is_empty() or market_returns.is_empty():
                logger.error("Cannot calculate systematic risk: empty returns DataFrame")
                raise ValueError("Returns DataFrames cannot be empty")

            # Calculate beta
            betas = await self.calculate_beta(asset_returns, market_returns, window_days)

            # Merge returns
            merged_df = asset_returns.join(
                market_returns.rename({"return": "market_return"}),
                on="timestamp",
                how="inner"
            )

            if window_days:
                cutoff_date = datetime.now().timestamp() - (window_days * 86400)
                merged_df = merged_df.filter(pl.col("timestamp") >= cutoff_date)

            # Calculate market volatility
            market_std = float(merged_df.select("market_return").to_numpy().std(ddof=1))

            # Decompose risk for each asset
            risk_decomposition = {}
            assets = merged_df.select("asset").unique().to_series().to_list()

            for asset in assets:
                asset_data = merged_df.filter(pl.col("asset") == asset)

                if len(asset_data) < 2:
                    logger.warning(f"Insufficient data for risk decomposition: {asset}")
                    continue

                # Extract returns
                asset_ret = asset_data.select("return").to_numpy().flatten()
                market_ret = asset_data.select("market_return").to_numpy().flatten()

                # Total risk (standard deviation)
                total_std = np.std(asset_ret, ddof=1)

                # Systematic risk (beta * market std dev)
                beta = float(betas[asset])
                systematic_std = abs(beta) * market_std

                # Correlation and R-squared
                correlation = np.corrcoef(asset_ret, market_ret)[0, 1]
                r_squared = correlation ** 2

                # Idiosyncratic risk (residual std dev)
                # sqrt(total_variance - systematic_variance)
                total_variance = total_std ** 2
                systematic_variance = systematic_std ** 2
                idiosyncratic_variance = max(0.0, total_variance - systematic_variance)
                idiosyncratic_std = np.sqrt(idiosyncratic_variance)

                # Percentage of variance explained by market
                systematic_pct = (systematic_variance / total_variance * 100.0) if total_variance > 0 else 0.0

                risk_decomposition[asset] = {
                    "total_risk": Decimal(str(total_std)).quantize(Decimal("0.000001")),
                    "systematic_risk": Decimal(str(systematic_std)).quantize(Decimal("0.000001")),
                    "idiosyncratic_risk": Decimal(str(idiosyncratic_std)).quantize(Decimal("0.000001")),
                    "systematic_pct": Decimal(str(systematic_pct)).quantize(Decimal("0.01")),
                    "r_squared": Decimal(str(r_squared)).quantize(Decimal("0.0001")),
                    "beta": betas[asset]
                }

                logger.debug(
                    f"Risk decomposition for {asset}: "
                    f"total={total_std:.6f}, systematic={systematic_std:.6f}, "
                    f"idiosyncratic={idiosyncratic_std:.6f}, R²={r_squared:.4f}"
                )

            logger.info(
                f"Calculated systematic risk decomposition for {len(risk_decomposition)} assets"
            )

            return risk_decomposition

        except Exception as e:
            logger.error(f"Error calculating systematic risk: {e}", exc_info=True)
            raise

    async def calculate_portfolio_beta(
        self,
        portfolio_weights: Dict[str, Decimal],
        asset_betas: Dict[str, Decimal]
    ) -> Decimal:
        """
        Calculate weighted portfolio beta

        Args:
            portfolio_weights: Asset allocation weights
            asset_betas: Beta for each asset

        Returns:
            Portfolio beta as Decimal
        """
        try:
            if not portfolio_weights or not asset_betas:
                logger.error("Cannot calculate portfolio beta: empty inputs")
                raise ValueError("Portfolio weights and betas cannot be empty")

            portfolio_beta = Decimal("0")

            for asset, weight in portfolio_weights.items():
                if asset in asset_betas:
                    portfolio_beta += weight * asset_betas[asset]
                else:
                    logger.warning(f"Beta not found for {asset}, using 1.0")
                    portfolio_beta += weight * Decimal("1.0")

            portfolio_beta = portfolio_beta.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

            logger.info(f"Calculated portfolio beta: {portfolio_beta}")

            return portfolio_beta

        except Exception as e:
            logger.error(f"Error calculating portfolio beta: {e}", exc_info=True)
            raise
