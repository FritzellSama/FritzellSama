"""Correlation features for multi-asset trading strategies.

This module provides comprehensive correlation analysis between trading pairs,
including Pearson, Spearman, Kendall correlations, and dynamic correlation tracking.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from scipy import stats
from scipy.cluster import hierarchy
from structlog import get_logger

logger = get_logger(__name__)


class CorrelationAnalyzer:
    """Analyzes correlations between multiple trading pairs.

    Provides various correlation metrics and analysis tools for
    understanding relationships between assets.

    Attributes:
        config: Configuration dictionary for correlation analysis
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize correlation analyzer.

        Args:
            config: Configuration dictionary with parameters:
                - correlation_types: List of correlation types to compute
                - min_periods: Minimum periods for correlation calculation
                - rolling_window: Window size for rolling correlations
                - significance_level: P-value threshold for significance tests

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.correlation_types = config.get(
            "correlation_types", ["pearson", "spearman", "kendall"]
        )
        self.min_periods = int(config.get("min_periods", 20))
        self.rolling_window = int(config.get("rolling_window", 30))
        self.significance_level = Decimal(str(config.get("significance_level", "0.05")))

        logger.info(
            "correlation_analyzer_initialized",
            correlation_types=self.correlation_types,
            min_periods=self.min_periods,
            rolling_window=self.rolling_window,
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError(f"Config must be a dictionary, got {type(self.config)}")

    def calculate_correlation_matrix(
        self,
        data: pl.DataFrame,
        symbols: List[str],
        price_column: str = "close",
        method: str = "pearson",
    ) -> pl.DataFrame:
        """Calculate correlation matrix for multiple symbols.

        Args:
            data: Polars DataFrame with price data
            symbols: List of symbols to analyze
            price_column: Column containing prices
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            Correlation matrix as Polars DataFrame

        Raises:
            ValueError: If calculation fails
        """
        try:
            logger.info(
                "calculating_correlation_matrix",
                num_symbols=len(symbols),
                method=method,
            )

            # Validate method
            if method not in ["pearson", "spearman", "kendall"]:
                raise ValueError(f"Invalid correlation method: {method}")

            # Extract prices for each symbol
            price_matrix = []
            for symbol in symbols:
                symbol_data = data.filter(pl.col("symbol") == symbol)
                if len(symbol_data) < self.min_periods:
                    raise ValueError(
                        f"Insufficient data for {symbol}: {len(symbol_data)} rows"
                    )
                prices = symbol_data[price_column].to_numpy()
                price_matrix.append(prices)

            price_matrix = np.array(price_matrix)

            # Calculate correlation matrix
            if method == "pearson":
                corr_matrix = np.corrcoef(price_matrix)
            elif method == "spearman":
                corr_matrix = self._spearman_correlation_matrix(price_matrix)
            elif method == "kendall":
                corr_matrix = self._kendall_correlation_matrix(price_matrix)

            # Convert to Polars DataFrame
            corr_dict = {symbols[i]: corr_matrix[i] for i in range(len(symbols))}
            corr_df = pl.DataFrame(corr_dict)
            corr_df = corr_df.insert_column(0, pl.Series("symbol", symbols))

            logger.info("correlation_matrix_calculated", shape=corr_matrix.shape)

            return corr_df

        except Exception as e:
            logger.error("correlation_matrix_calculation_failed", error=str(e))
            raise

    def _spearman_correlation_matrix(self, data: np.ndarray) -> np.ndarray:
        """Calculate Spearman correlation matrix.

        Args:
            data: 2D array where each row is a time series

        Returns:
            Correlation matrix
        """
        n = len(data)
        corr_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                corr, _ = stats.spearmanr(data[i], data[j])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        return corr_matrix

    def _kendall_correlation_matrix(self, data: np.ndarray) -> np.ndarray:
        """Calculate Kendall correlation matrix.

        Args:
            data: 2D array where each row is a time series

        Returns:
            Correlation matrix
        """
        n = len(data)
        corr_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                corr, _ = stats.kendalltau(data[i], data[j])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        return corr_matrix

    def calculate_rolling_correlation(
        self,
        data: pl.DataFrame,
        symbol1: str,
        symbol2: str,
        price_column: str = "close",
        window: Optional[int] = None,
        method: str = "pearson",
    ) -> pl.DataFrame:
        """Calculate rolling correlation between two symbols.

        Args:
            data: Polars DataFrame with price data
            symbol1: First symbol
            symbol2: Second symbol
            price_column: Price column name
            window: Rolling window size (None = use default)
            method: Correlation method

        Returns:
            DataFrame with rolling correlation values

        Raises:
            ValueError: If calculation fails
        """
        try:
            if window is None:
                window = self.rolling_window

            logger.info(
                "calculating_rolling_correlation",
                symbol1=symbol1,
                symbol2=symbol2,
                window=window,
                method=method,
            )

            # Extract prices
            prices1 = self._extract_prices(data, symbol1, price_column)
            prices2 = self._extract_prices(data, symbol2, price_column)

            # Calculate rolling correlation
            rolling_corr = []
            timestamps = []

            for i in range(window, len(prices1)):
                window_prices1 = prices1[i - window : i]
                window_prices2 = prices2[i - window : i]

                if method == "pearson":
                    corr, _ = stats.pearsonr(window_prices1, window_prices2)
                elif method == "spearman":
                    corr, _ = stats.spearmanr(window_prices1, window_prices2)
                elif method == "kendall":
                    corr, _ = stats.kendalltau(window_prices1, window_prices2)
                else:
                    raise ValueError(f"Invalid method: {method}")

                rolling_corr.append(Decimal(str(corr)))
                # Get timestamp from data
                symbol1_data = data.filter(pl.col("symbol") == symbol1)
                timestamps.append(symbol1_data["timestamp"][i])

            result_df = pl.DataFrame(
                {
                    "timestamp": timestamps,
                    "correlation": rolling_corr,
                    "symbol1": [symbol1] * len(rolling_corr),
                    "symbol2": [symbol2] * len(rolling_corr),
                }
            )

            logger.info(
                "rolling_correlation_calculated",
                rows=len(result_df),
                mean_corr=str(sum(rolling_corr) / Decimal(str(len(rolling_corr)))),
            )

            return result_df

        except Exception as e:
            logger.error("rolling_correlation_failed", error=str(e))
            raise

    def _extract_prices(
        self, data: pl.DataFrame, symbol: str, price_column: str
    ) -> np.ndarray:
        """Extract price series for a symbol.

        Args:
            data: DataFrame with price data
            symbol: Symbol identifier
            price_column: Price column name

        Returns:
            Numpy array of prices

        Raises:
            ValueError: If extraction fails
        """
        try:
            symbol_data = data.filter(pl.col("symbol") == symbol)
            if len(symbol_data) == 0:
                raise ValueError(f"No data found for symbol: {symbol}")

            prices = symbol_data[price_column].to_numpy()
            return prices.astype(float)

        except Exception as e:
            logger.error("price_extraction_failed", symbol=symbol, error=str(e))
            raise

    def calculate_dynamic_correlation(
        self,
        data: pl.DataFrame,
        symbol1: str,
        symbol2: str,
        price_column: str = "close",
        decay_factor: Optional[Decimal] = None,
    ) -> pl.DataFrame:
        """Calculate exponentially weighted dynamic correlation.

        Args:
            data: DataFrame with price data
            symbol1: First symbol
            symbol2: Second symbol
            price_column: Price column name
            decay_factor: Exponential decay factor (None = auto)

        Returns:
            DataFrame with dynamic correlation values

        Raises:
            ValueError: If calculation fails
        """
        try:
            if decay_factor is None:
                decay_factor = Decimal("0.94")  # Typical value for daily data

            logger.info(
                "calculating_dynamic_correlation",
                symbol1=symbol1,
                symbol2=symbol2,
                decay_factor=str(decay_factor),
            )

            # Extract prices
            prices1 = self._extract_prices(data, symbol1, price_column)
            prices2 = self._extract_prices(data, symbol2, price_column)

            # Calculate returns
            returns1 = np.diff(prices1) / prices1[:-1]
            returns2 = np.diff(prices2) / prices2[:-1]

            # Initialize EWMA variables
            mean1 = Decimal(str(returns1[0]))
            mean2 = Decimal(str(returns2[0]))
            var1 = Decimal("0.0")
            var2 = Decimal("0.0")
            covar = Decimal("0.0")

            dynamic_corr = []
            timestamps = []

            # Calculate dynamic correlation
            for i in range(len(returns1)):
                r1 = Decimal(str(returns1[i]))
                r2 = Decimal(str(returns2[i]))

                # Update EWMA means
                mean1 = decay_factor * mean1 + (Decimal("1.0") - decay_factor) * r1
                mean2 = decay_factor * mean2 + (Decimal("1.0") - decay_factor) * r2

                # Update EWMA variances and covariance
                var1 = decay_factor * var1 + (Decimal("1.0") - decay_factor) * (
                    r1 - mean1
                ) ** Decimal("2")
                var2 = decay_factor * var2 + (Decimal("1.0") - decay_factor) * (
                    r2 - mean2
                ) ** Decimal("2")
                covar = decay_factor * covar + (Decimal("1.0") - decay_factor) * (
                    r1 - mean1
                ) * (r2 - mean2)

                # Calculate correlation
                if var1 > Decimal("0.0") and var2 > Decimal("0.0"):
                    corr = covar / ((var1 * var2) ** Decimal("0.5"))
                else:
                    corr = Decimal("0.0")

                dynamic_corr.append(corr)
                # Get timestamp (offset by 1 due to returns)
                symbol1_data = data.filter(pl.col("symbol") == symbol1)
                timestamps.append(symbol1_data["timestamp"][i + 1])

            result_df = pl.DataFrame(
                {
                    "timestamp": timestamps,
                    "dynamic_correlation": dynamic_corr,
                    "symbol1": [symbol1] * len(dynamic_corr),
                    "symbol2": [symbol2] * len(dynamic_corr),
                }
            )

            logger.info("dynamic_correlation_calculated", rows=len(result_df))

            return result_df

        except Exception as e:
            logger.error("dynamic_correlation_failed", error=str(e))
            raise

    def find_highly_correlated_pairs(
        self,
        data: pl.DataFrame,
        symbols: List[str],
        threshold: Decimal,
        price_column: str = "close",
        method: str = "pearson",
    ) -> List[Dict[str, Any]]:
        """Find pairs with correlation above threshold.

        Args:
            data: DataFrame with price data
            symbols: List of symbols to analyze
            threshold: Minimum correlation threshold
            price_column: Price column name
            method: Correlation method

        Returns:
            List of highly correlated pairs

        Raises:
            ValueError: If search fails
        """
        try:
            logger.info(
                "finding_highly_correlated_pairs",
                num_symbols=len(symbols),
                threshold=str(threshold),
            )

            # Calculate correlation matrix
            corr_df = self.calculate_correlation_matrix(
                data, symbols, price_column, method
            )

            # Find pairs above threshold
            highly_correlated = []

            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i + 1 :], start=i + 1):
                    corr_value = Decimal(str(corr_df[symbol2][i]))

                    if abs(corr_value) >= threshold:
                        highly_correlated.append(
                            {
                                "symbol1": symbol1,
                                "symbol2": symbol2,
                                "correlation": corr_value,
                                "method": method,
                            }
                        )

            # Sort by absolute correlation
            highly_correlated.sort(
                key=lambda x: abs(x["correlation"]), reverse=True
            )

            logger.info(
                "highly_correlated_pairs_found",
                num_pairs=len(highly_correlated),
            )

            return highly_correlated

        except Exception as e:
            logger.error("find_pairs_failed", error=str(e))
            raise

    def calculate_correlation_stability(
        self,
        data: pl.DataFrame,
        symbol1: str,
        symbol2: str,
        price_column: str = "close",
        num_windows: int = 10,
    ) -> Dict[str, Any]:
        """Calculate stability of correlation over time.

        Args:
            data: DataFrame with price data
            symbol1: First symbol
            symbol2: Second symbol
            price_column: Price column name
            num_windows: Number of time windows to analyze

        Returns:
            Dictionary with stability metrics

        Raises:
            ValueError: If calculation fails
        """
        try:
            logger.info(
                "calculating_correlation_stability",
                symbol1=symbol1,
                symbol2=symbol2,
                num_windows=num_windows,
            )

            # Extract prices
            prices1 = self._extract_prices(data, symbol1, price_column)
            prices2 = self._extract_prices(data, symbol2, price_column)

            # Calculate correlations for different windows
            window_size = len(prices1) // num_windows
            correlations = []

            for i in range(num_windows):
                start_idx = i * window_size
                end_idx = start_idx + window_size

                if end_idx <= len(prices1):
                    window_prices1 = prices1[start_idx:end_idx]
                    window_prices2 = prices2[start_idx:end_idx]

                    corr, _ = stats.pearsonr(window_prices1, window_prices2)
                    correlations.append(Decimal(str(corr)))

            # Calculate stability metrics
            mean_corr = sum(correlations) / Decimal(str(len(correlations)))
            std_corr = Decimal(str(np.std([float(c) for c in correlations])))
            min_corr = min(correlations)
            max_corr = max(correlations)
            range_corr = max_corr - min_corr

            # Stability score (lower is more stable)
            stability_score = std_corr / (abs(mean_corr) + Decimal("0.01"))

            result = {
                "symbol1": symbol1,
                "symbol2": symbol2,
                "mean_correlation": mean_corr,
                "std_correlation": std_corr,
                "min_correlation": min_corr,
                "max_correlation": max_corr,
                "range": range_corr,
                "stability_score": stability_score,
                "num_windows": len(correlations),
            }

            logger.info(
                "correlation_stability_calculated",
                mean_corr=str(mean_corr),
                stability_score=str(stability_score),
            )

            return result

        except Exception as e:
            logger.error("stability_calculation_failed", error=str(e))
            raise

    def cluster_correlated_assets(
        self,
        data: pl.DataFrame,
        symbols: List[str],
        price_column: str = "close",
        num_clusters: int = 5,
    ) -> Dict[str, Any]:
        """Cluster assets based on correlation.

        Args:
            data: DataFrame with price data
            symbols: List of symbols to cluster
            price_column: Price column name
            num_clusters: Number of clusters to create

        Returns:
            Dictionary with clustering results

        Raises:
            ValueError: If clustering fails
        """
        try:
            logger.info(
                "clustering_assets",
                num_symbols=len(symbols),
                num_clusters=num_clusters,
            )

            # Calculate correlation matrix
            corr_df = self.calculate_correlation_matrix(data, symbols, price_column)

            # Convert to distance matrix (1 - correlation)
            corr_matrix = corr_df.drop("symbol").to_numpy()
            distance_matrix = 1 - np.abs(corr_matrix)

            # Perform hierarchical clustering
            condensed_dist = hierarchy.distance.squareform(distance_matrix)
            linkage_matrix = hierarchy.linkage(condensed_dist, method="ward")

            # Cut tree to form clusters
            cluster_labels = hierarchy.fcluster(
                linkage_matrix, num_clusters, criterion="maxclust"
            )

            # Organize results
            clusters = {}
            for symbol, label in zip(symbols, cluster_labels):
                cluster_id = int(label)
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(symbol)

            result = {
                "clusters": clusters,
                "num_clusters": len(clusters),
                "linkage_matrix": linkage_matrix.tolist(),
                "cluster_labels": cluster_labels.tolist(),
            }

            logger.info(
                "assets_clustered",
                num_clusters=len(clusters),
                cluster_sizes=[len(c) for c in clusters.values()],
            )

            return result

        except Exception as e:
            logger.error("clustering_failed", error=str(e))
            raise


__all__ = ["CorrelationAnalyzer"]
