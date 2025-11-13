"""
Correlation Matrix Calculator
CRITICAL: Calculate and monitor correlation matrices for portfolio risk analysis
"""

import logging
import numpy as np
import polars as pl
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class CorrelationMatrixCalculator:
    """Calculate and analyze correlation matrices for portfolio assets"""

    def __init__(self):
        """Initialize correlation matrix calculator with config"""
        self.config = get_config()
        self.max_correlation = self.config.get_decimal("risk", "risk_metrics.max_correlation")
        self.update_interval = self.config.get_int(
            "risk", "portfolio.correlation_update_interval_hours", 1
        )

        self.last_update_time = None
        self.cached_matrix = None
        self.cached_assets = None

        logger.info(
            f"CorrelationMatrixCalculator initialized: max_correlation={self.max_correlation}, "
            f"update_interval={self.update_interval}h"
        )

    async def calculate_correlation_matrix(
        self,
        returns_df: pl.DataFrame,
        method: str = "pearson",
        min_periods: int = 20
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate correlation matrix from returns data

        Args:
            returns_df: DataFrame with columns ['timestamp', 'asset', 'return']
            method: Correlation method ('pearson', 'spearman', 'kendall')
            min_periods: Minimum number of observations required

        Returns:
            Tuple of (correlation_matrix as numpy array, list of asset symbols)
        """
        try:
            if returns_df.is_empty():
                logger.error("Cannot calculate correlation matrix: empty returns DataFrame")
                raise ValueError("Returns DataFrame is empty")

            # Pivot to wide format (timestamps x assets)
            pivot_df = returns_df.pivot(
                values="return",
                index="timestamp",
                columns="asset"
            ).sort("timestamp")

            # Get asset list (excluding timestamp column)
            assets = [col for col in pivot_df.columns if col != "timestamp"]

            if len(assets) < 2:
                logger.warning("Need at least 2 assets for correlation calculation")
                return np.eye(len(assets)), assets

            # Convert to numpy array (excluding timestamp)
            returns_matrix = pivot_df.select(assets).to_numpy()

            # Check minimum periods
            valid_rows = np.sum(~np.isnan(returns_matrix), axis=0)
            if np.any(valid_rows < min_periods):
                logger.warning(
                    f"Some assets have fewer than {min_periods} observations: {valid_rows.tolist()}"
                )

            # Calculate correlation matrix based on method
            if method == "pearson":
                # Use numpy's corrcoef (handles NaN with pairwise deletion)
                corr_matrix = np.corrcoef(returns_matrix.T)
            elif method == "spearman":
                # Rank-based correlation
                corr_matrix = self._spearman_correlation(returns_matrix)
            elif method == "kendall":
                # Kendall's tau
                corr_matrix = self._kendall_correlation(returns_matrix)
            else:
                raise ValueError(f"Unknown correlation method: {method}")

            # Replace NaN with 0 (uncorrelated)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

            # Ensure diagonal is 1.0
            np.fill_diagonal(corr_matrix, 1.0)

            # Cache results
            self.cached_matrix = corr_matrix
            self.cached_assets = assets
            self.last_update_time = datetime.now()

            logger.info(
                f"Calculated {method} correlation matrix: {len(assets)} assets, "
                f"avg_correlation={self._calculate_average_correlation(corr_matrix):.4f}"
            )

            return corr_matrix, assets

        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}", exc_info=True)
            raise

    async def detect_correlation_changes(
        self,
        current_matrix: np.ndarray,
        previous_matrix: np.ndarray,
        asset_list: List[str],
        threshold: Decimal = Decimal("0.15")
    ) -> Dict[str, any]:
        """
        Detect significant changes in correlation structure

        Args:
            current_matrix: Current correlation matrix
            previous_matrix: Previous correlation matrix
            asset_list: List of asset symbols
            threshold: Threshold for significant change detection

        Returns:
            Dictionary with correlation change analysis:
            {
                'significant_changes': bool,
                'num_changes': int,
                'changed_pairs': List[Tuple[str, str, Decimal, Decimal]],
                'avg_change': Decimal,
                'max_change': Decimal
            }
        """
        try:
            if current_matrix.shape != previous_matrix.shape:
                logger.error("Correlation matrices have different shapes")
                raise ValueError("Matrix shape mismatch")

            # Calculate difference matrix (excluding diagonal)
            diff_matrix = np.abs(current_matrix - previous_matrix)
            np.fill_diagonal(diff_matrix, 0.0)

            # Find significant changes
            n = len(asset_list)
            changed_pairs = []

            for i in range(n):
                for j in range(i + 1, n):
                    change = Decimal(str(diff_matrix[i, j])).quantize(Decimal("0.0001"))

                    if change >= threshold:
                        current_corr = Decimal(str(current_matrix[i, j])).quantize(Decimal("0.0001"))
                        previous_corr = Decimal(str(previous_matrix[i, j])).quantize(Decimal("0.0001"))

                        changed_pairs.append((
                            asset_list[i],
                            asset_list[j],
                            previous_corr,
                            current_corr,
                            change
                        ))

            # Sort by magnitude of change
            changed_pairs.sort(key=lambda x: x[4], reverse=True)

            # Calculate statistics
            avg_change = Decimal(str(np.mean(diff_matrix))).quantize(Decimal("0.0001"))
            max_change = Decimal(str(np.max(diff_matrix))).quantize(Decimal("0.0001"))

            significant_changes = len(changed_pairs) > 0

            result = {
                "significant_changes": significant_changes,
                "num_changes": len(changed_pairs),
                "changed_pairs": changed_pairs[:20],  # Top 20 changes
                "avg_change": avg_change,
                "max_change": max_change,
                "threshold": threshold,
                "timestamp": datetime.now().isoformat()
            }

            if significant_changes:
                logger.warning(
                    f"Significant correlation changes detected: {len(changed_pairs)} pairs changed, "
                    f"max_change={max_change:.4f}"
                )
            else:
                logger.debug(f"No significant correlation changes: max_change={max_change:.4f}")

            return result

        except Exception as e:
            logger.error(f"Error detecting correlation changes: {e}", exc_info=True)
            raise

    async def get_correlated_assets(
        self,
        correlation_matrix: np.ndarray,
        asset_list: List[str],
        target_asset: str,
        min_correlation: Decimal = Decimal("0.5")
    ) -> List[Tuple[str, Decimal]]:
        """
        Get assets correlated with target asset

        Args:
            correlation_matrix: Correlation matrix
            asset_list: List of asset symbols
            target_asset: Asset to find correlations for
            min_correlation: Minimum absolute correlation threshold

        Returns:
            List of (asset, correlation) tuples, sorted by correlation magnitude
        """
        try:
            if target_asset not in asset_list:
                logger.error(f"Target asset '{target_asset}' not found in asset list")
                raise ValueError(f"Asset not found: {target_asset}")

            target_idx = asset_list.index(target_asset)

            # Get correlations for target asset
            correlations = []

            for idx, asset in enumerate(asset_list):
                if idx == target_idx:
                    continue  # Skip self-correlation

                corr_value = Decimal(str(correlation_matrix[target_idx, idx])).quantize(Decimal("0.0001"))

                if abs(corr_value) >= min_correlation:
                    correlations.append((asset, corr_value))

            # Sort by absolute correlation (descending)
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)

            logger.info(
                f"Found {len(correlations)} assets correlated with {target_asset} "
                f"(threshold={min_correlation})"
            )

            return correlations

        except Exception as e:
            logger.error(f"Error getting correlated assets: {e}", exc_info=True)
            raise

    async def calculate_rolling_correlation(
        self,
        returns_df: pl.DataFrame,
        asset_pair: Tuple[str, str],
        window_days: int = 30
    ) -> pl.DataFrame:
        """
        Calculate rolling correlation for a pair of assets

        Args:
            returns_df: DataFrame with columns ['timestamp', 'asset', 'return']
            asset_pair: Tuple of two asset symbols
            window_days: Rolling window size in days

        Returns:
            Polars DataFrame with columns ['timestamp', 'correlation']
        """
        try:
            asset1, asset2 = asset_pair

            # Filter returns for the two assets
            asset1_df = returns_df.filter(pl.col("asset") == asset1).select(["timestamp", "return"]).rename(
                {"return": "return1"}
            )
            asset2_df = returns_df.filter(pl.col("asset") == asset2).select(["timestamp", "return"]).rename(
                {"return": "return2"}
            )

            # Merge on timestamp
            merged_df = asset1_df.join(asset2_df, on="timestamp", how="inner").sort("timestamp")

            if len(merged_df) < window_days:
                logger.warning(
                    f"Insufficient data for rolling correlation: {len(merged_df)} < {window_days}"
                )

            # Convert to numpy for rolling calculation
            timestamps = merged_df.select("timestamp").to_series().to_list()
            returns1 = merged_df.select("return1").to_numpy().flatten()
            returns2 = merged_df.select("return2").to_numpy().flatten()

            # Calculate rolling correlation
            rolling_correlations = []

            for i in range(window_days - 1, len(returns1)):
                window_ret1 = returns1[i - window_days + 1 : i + 1]
                window_ret2 = returns2[i - window_days + 1 : i + 1]

                if len(window_ret1) > 1:
                    corr = np.corrcoef(window_ret1, window_ret2)[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                else:
                    corr = 0.0

                rolling_correlations.append({
                    "timestamp": timestamps[i],
                    "correlation": Decimal(str(corr)).quantize(Decimal("0.0001"))
                })

            result_df = pl.DataFrame(rolling_correlations)

            logger.info(
                f"Calculated rolling correlation for {asset1}-{asset2}: "
                f"{len(result_df)} observations, window={window_days} days"
            )

            return result_df

        except Exception as e:
            logger.error(f"Error calculating rolling correlation: {e}", exc_info=True)
            raise

    def get_correlation_clusters(
        self,
        correlation_matrix: np.ndarray,
        asset_list: List[str],
        threshold: Decimal = Decimal("0.7")
    ) -> List[List[str]]:
        """
        Identify clusters of highly correlated assets using hierarchical clustering

        Args:
            correlation_matrix: Correlation matrix
            asset_list: List of asset symbols
            threshold: Correlation threshold for clustering

        Returns:
            List of asset clusters (each cluster is a list of assets)
        """
        try:
            n = len(asset_list)
            threshold_float = float(threshold)

            # Simple clustering: group assets with correlation > threshold
            visited = set()
            clusters = []

            for i in range(n):
                if i in visited:
                    continue

                cluster = [asset_list[i]]
                visited.add(i)

                for j in range(i + 1, n):
                    if j not in visited:
                        # Check if j is correlated with any asset in current cluster
                        max_corr_to_cluster = max(
                            abs(correlation_matrix[asset_list.index(asset), j])
                            for asset in cluster
                        )

                        if max_corr_to_cluster >= threshold_float:
                            cluster.append(asset_list[j])
                            visited.add(j)

                if len(cluster) > 1:  # Only include multi-asset clusters
                    clusters.append(cluster)

            logger.info(
                f"Identified {len(clusters)} correlation clusters "
                f"(threshold={threshold}): sizes={[len(c) for c in clusters]}"
            )

            return clusters

        except Exception as e:
            logger.error(f"Error identifying correlation clusters: {e}", exc_info=True)
            raise

    def _calculate_average_correlation(self, corr_matrix: np.ndarray) -> float:
        """Calculate average absolute correlation (excluding diagonal)"""
        n = corr_matrix.shape[0]
        if n <= 1:
            return 0.0

        upper_triangle = np.triu(np.abs(corr_matrix), k=1)
        num_elements = (n * (n - 1)) // 2

        if num_elements == 0:
            return 0.0

        return float(np.sum(upper_triangle) / num_elements)

    def _spearman_correlation(self, data: np.ndarray) -> np.ndarray:
        """Calculate Spearman rank correlation"""
        from scipy.stats import spearmanr

        n_assets = data.shape[1]
        corr_matrix = np.eye(n_assets)

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                # Get valid (non-NaN) paired observations
                mask = ~(np.isnan(data[:, i]) | np.isnan(data[:, j]))
                if np.sum(mask) < 2:
                    corr_matrix[i, j] = corr_matrix[j, i] = 0.0
                    continue

                corr, _ = spearmanr(data[mask, i], data[mask, j])
                if np.isnan(corr):
                    corr = 0.0

                corr_matrix[i, j] = corr_matrix[j, i] = corr

        return corr_matrix

    def _kendall_correlation(self, data: np.ndarray) -> np.ndarray:
        """Calculate Kendall's tau correlation"""
        from scipy.stats import kendalltau

        n_assets = data.shape[1]
        corr_matrix = np.eye(n_assets)

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                # Get valid (non-NaN) paired observations
                mask = ~(np.isnan(data[:, i]) | np.isnan(data[:, j]))
                if np.sum(mask) < 2:
                    corr_matrix[i, j] = corr_matrix[j, i] = 0.0
                    continue

                corr, _ = kendalltau(data[mask, i], data[mask, j])
                if np.isnan(corr):
                    corr = 0.0

                corr_matrix[i, j] = corr_matrix[j, i] = corr

        return corr_matrix

    def needs_update(self) -> bool:
        """Check if correlation matrix needs update based on update interval"""
        if self.last_update_time is None:
            return True

        update_delta = timedelta(hours=self.update_interval)
        return datetime.now() >= (self.last_update_time + update_delta)
