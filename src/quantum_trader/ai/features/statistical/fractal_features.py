"""Fractal dimension and chaos theory features.

This module computes fractal dimension, Hurst exponent, and other chaos theory
metrics to characterize market complexity, self-similarity, and long-range
dependence. Critical for understanding market regime changes and persistence.
"""

import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import polars as pl
import numpy as np
from structlog import get_logger
from scipy import stats

logger = get_logger(__name__)


class FractalFeaturesError(Exception):
    """Base exception for fractal features computation."""
    pass


class FractalFeatures:
    """Compute fractal dimension and chaos theory features.

    Implements various methods for estimating fractal dimension, Hurst exponent,
    Lyapunov exponent, and other nonlinear dynamics measures that characterize
    the complexity and predictability of financial time series.

    Attributes:
        config: Configuration dictionary
        methods: List of fractal methods to apply
        window_sizes: Window sizes for rolling computations

    Example:
        >>> config = {
        ...     'methods': ['hurst', 'higuchi', 'detrended_fluctuation'],
        ...     'window_sizes': [50, 100, 200],
        ...     'hurst_lags': [2, 5, 10, 20],
        ...     'higuchi_kmax': 10
        ... }
        >>> fractal = FractalFeatures(config)
        >>> features = await fractal.compute(price_data)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize fractal features computer.

        Args:
            config: Configuration dictionary with keys:
                - methods: List of methods to use
                - window_sizes: Window sizes for rolling computations
                - hurst_lags: Lag values for Hurst exponent
                - higuchi_kmax: Maximum k for Higuchi fractal dimension
                - dfa_scales: Scales for DFA analysis
                - embedding_dim: Embedding dimension for chaos analysis
                - time_delay: Time delay for embedding

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.methods = self.config.get('methods', ['hurst', 'higuchi', 'detrended_fluctuation'])
        self.window_sizes = self.config.get('window_sizes', [50, 100, 200])
        self.hurst_lags = self.config.get('hurst_lags', [2, 5, 10, 20])
        self.higuchi_kmax = self.config.get('higuchi_kmax', 10)
        self.dfa_scales = self.config.get('dfa_scales', None)
        self.embedding_dim = self.config.get('embedding_dim', 3)
        self.time_delay = self.config.get('time_delay', 1)

        self._lock = asyncio.Lock()

        logger.info(
            "fractal_features_initialized",
            methods=self.methods,
            window_sizes=self.window_sizes
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if 'window_sizes' in self.config:
            sizes = self.config['window_sizes']
            if not all(s > 0 for s in sizes):
                raise ValueError("All window_sizes must be positive")

        if 'hurst_lags' in self.config:
            lags = self.config['hurst_lags']
            if not all(lag > 0 for lag in lags):
                raise ValueError("All hurst_lags must be positive")

        if 'higuchi_kmax' in self.config and self.config['higuchi_kmax'] < 2:
            raise ValueError("higuchi_kmax must be at least 2")

        if 'embedding_dim' in self.config and self.config['embedding_dim'] < 2:
            raise ValueError("embedding_dim must be at least 2")

    async def compute(
        self,
        data: pl.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """Compute fractal features from time series data.

        Args:
            data: Input DataFrame with time series
            columns: Columns to compute features for (default: ['close'])

        Returns:
            DataFrame with fractal features added

        Raises:
            FractalFeaturesError: If computation fails
            ValueError: If input data is invalid

        Example:
            >>> features_df = await fractal.compute(
            ...     price_df,
            ...     columns=['close', 'volume']
            ... )
        """
        try:
            if data.is_empty():
                raise ValueError("Input data cannot be empty")

            min_window = min(self.window_sizes)
            if data.height < min_window:
                logger.warning(
                    "insufficient_data_for_fractal",
                    data_length=data.height,
                    required=min_window
                )
                # Return data with null features
                return self._add_null_features(data, columns or ['close'])

            # Determine columns to process
            if columns is None:
                columns = ['close'] if 'close' in data.columns else [data.columns[0]]

            result = data.clone()

            # Compute fractal features for each column
            for col in columns:
                if col not in result.columns:
                    logger.warning(f"column_not_found", column=col)
                    continue

                # Apply each method
                if 'hurst' in self.methods:
                    result = await self._compute_hurst_exponent(result, col)

                if 'higuchi' in self.methods:
                    result = await self._compute_higuchi_dimension(result, col)

                if 'detrended_fluctuation' in self.methods:
                    result = await self._compute_dfa(result, col)

                if 'correlation_dimension' in self.methods:
                    result = await self._compute_correlation_dimension(result, col)

                if 'entropy' in self.methods:
                    result = await self._compute_sample_entropy(result, col)

            logger.info(
                "fractal_features_computed",
                row_count=result.height,
                input_columns=len(columns),
                feature_count=len(result.columns) - len(data.columns)
            )

            return result

        except Exception as e:
            logger.error("fractal_features_computation_failed", error=str(e))
            raise FractalFeaturesError(
                f"Failed to compute fractal features: {str(e)}"
            ) from e

    async def _compute_hurst_exponent(
        self,
        data: pl.DataFrame,
        column: str
    ) -> pl.DataFrame:
        """Compute Hurst exponent using R/S analysis.

        The Hurst exponent characterizes long-range dependence:
        - H < 0.5: Mean-reverting (anti-persistent)
        - H = 0.5: Random walk (no memory)
        - H > 0.5: Trending (persistent)

        Args:
            data: Input DataFrame
            column: Column to analyze

        Returns:
            DataFrame with Hurst features
        """
        result = data
        series = data[column].to_numpy().astype(np.float64)

        try:
            # Compute rolling Hurst exponent
            for window_size in self.window_sizes:
                hurst_values = []

                for i in range(len(series)):
                    if i < window_size - 1:
                        hurst_values.append(Decimal('0.5'))  # Neutral default
                    else:
                        window = series[i - window_size + 1:i + 1]
                        hurst = self._calculate_hurst_rs(window)
                        hurst_values.append(Decimal(str(hurst)))

                result = result.with_columns([
                    pl.Series(f'{column}_hurst_{window_size}', hurst_values)
                ])

            logger.debug(
                "hurst_exponent_computed",
                column=column,
                window_sizes=self.window_sizes
            )

        except Exception as e:
            logger.warning(
                "hurst_computation_failed",
                column=column,
                error=str(e)
            )

        return result

    def _calculate_hurst_rs(self, series: np.ndarray) -> float:
        """Calculate Hurst exponent using rescaled range (R/S) analysis.

        Args:
            series: Time series data

        Returns:
            Hurst exponent
        """
        try:
            n = len(series)
            if n < 20:
                return 0.5

            # Use logarithmically spaced lags
            max_lag = min(n // 4, max(self.hurst_lags))
            lags = np.unique(np.logspace(0.5, np.log10(max_lag), num=10, dtype=int))
            lags = lags[lags > 1]

            if len(lags) < 2:
                return 0.5

            rs_values = []

            for lag in lags:
                # Split series into chunks
                n_chunks = n // lag
                if n_chunks == 0:
                    continue

                rs_chunk = []

                for i in range(n_chunks):
                    chunk = series[i * lag:(i + 1) * lag]

                    # Mean-adjusted cumulative sum
                    mean_chunk = chunk.mean()
                    cumsum = np.cumsum(chunk - mean_chunk)

                    # Range
                    R = cumsum.max() - cumsum.min()

                    # Standard deviation
                    S = chunk.std()

                    if S > 0:
                        rs_chunk.append(R / S)

                if rs_chunk:
                    rs_values.append(np.mean(rs_chunk))

            if len(rs_values) < 2:
                return 0.5

            # Fit log(R/S) vs log(lag) to get Hurst exponent
            log_lags = np.log(lags[:len(rs_values)])
            log_rs = np.log(rs_values)

            # Linear regression
            slope, _, _, _, _ = stats.linregress(log_lags, log_rs)

            # Clip to valid range
            hurst = np.clip(slope, 0.0, 1.0)

            return float(hurst)

        except Exception as e:
            logger.debug("hurst_calculation_failed", error=str(e))
            return 0.5

    async def _compute_higuchi_dimension(
        self,
        data: pl.DataFrame,
        column: str
    ) -> pl.DataFrame:
        """Compute Higuchi fractal dimension.

        Measures the complexity and irregularity of the time series.
        Higher values indicate more complex, irregular patterns.

        Args:
            data: Input DataFrame
            column: Column to analyze

        Returns:
            DataFrame with Higuchi dimension features
        """
        result = data
        series = data[column].to_numpy().astype(np.float64)

        try:
            for window_size in self.window_sizes:
                fd_values = []

                for i in range(len(series)):
                    if i < window_size - 1:
                        fd_values.append(Decimal('1.0'))  # Default dimension
                    else:
                        window = series[i - window_size + 1:i + 1]
                        fd = self._calculate_higuchi_fd(window)
                        fd_values.append(Decimal(str(fd)))

                result = result.with_columns([
                    pl.Series(f'{column}_higuchi_fd_{window_size}', fd_values)
                ])

            logger.debug(
                "higuchi_dimension_computed",
                column=column,
                window_sizes=self.window_sizes
            )

        except Exception as e:
            logger.warning(
                "higuchi_computation_failed",
                column=column,
                error=str(e)
            )

        return result

    def _calculate_higuchi_fd(self, series: np.ndarray) -> float:
        """Calculate Higuchi fractal dimension.

        Args:
            series: Time series data

        Returns:
            Fractal dimension
        """
        try:
            n = len(series)
            if n < 10:
                return 1.0

            kmax = min(self.higuchi_kmax, n // 4)
            if kmax < 2:
                return 1.0

            k_values = range(1, kmax + 1)
            lk_values = []

            for k in k_values:
                lm_k = []

                for m in range(k):
                    # Construct subsequence
                    indices = range(m, n, k)
                    if len(indices) < 2:
                        continue

                    # Calculate length of curve
                    length = 0
                    for i in range(1, len(indices)):
                        length += abs(series[indices[i]] - series[indices[i - 1]])

                    # Normalize
                    length *= (n - 1) / (k * len(indices))

                    lm_k.append(length)

                if lm_k:
                    lk_values.append(np.mean(lm_k))

            if len(lk_values) < 2:
                return 1.0

            # Fit log(L(k)) vs log(1/k)
            log_k = np.log(list(k_values[:len(lk_values)]))
            log_lk = np.log(lk_values)

            # Remove invalid values
            valid_mask = np.isfinite(log_k) & np.isfinite(log_lk)
            if valid_mask.sum() < 2:
                return 1.0

            log_k = log_k[valid_mask]
            log_lk = log_lk[valid_mask]

            # Linear regression
            slope, _, _, _, _ = stats.linregress(log_k, log_lk)

            # Fractal dimension is negative of slope
            fd = -slope

            # Clip to reasonable range
            fd = np.clip(fd, 1.0, 2.0)

            return float(fd)

        except Exception as e:
            logger.debug("higuchi_calculation_failed", error=str(e))
            return 1.0

    async def _compute_dfa(
        self,
        data: pl.DataFrame,
        column: str
    ) -> pl.DataFrame:
        """Compute Detrended Fluctuation Analysis (DFA).

        DFA quantifies long-range correlations in time series.
        Similar interpretation to Hurst exponent.

        Args:
            data: Input DataFrame
            column: Column to analyze

        Returns:
            DataFrame with DFA features
        """
        result = data
        series = data[column].to_numpy().astype(np.float64)

        try:
            for window_size in self.window_sizes:
                dfa_values = []

                for i in range(len(series)):
                    if i < window_size - 1:
                        dfa_values.append(Decimal('0.5'))  # Neutral default
                    else:
                        window = series[i - window_size + 1:i + 1]
                        alpha = self._calculate_dfa(window)
                        dfa_values.append(Decimal(str(alpha)))

                result = result.with_columns([
                    pl.Series(f'{column}_dfa_{window_size}', dfa_values)
                ])

            logger.debug(
                "dfa_computed",
                column=column,
                window_sizes=self.window_sizes
            )

        except Exception as e:
            logger.warning(
                "dfa_computation_failed",
                column=column,
                error=str(e)
            )

        return result

    def _calculate_dfa(self, series: np.ndarray) -> float:
        """Calculate DFA scaling exponent.

        Args:
            series: Time series data

        Returns:
            DFA exponent (alpha)
        """
        try:
            n = len(series)
            if n < 20:
                return 0.5

            # Compute cumulative sum (profile)
            profile = np.cumsum(series - series.mean())

            # Define scales
            if self.dfa_scales is not None:
                scales = self.dfa_scales
            else:
                scales = np.unique(np.logspace(0.7, np.log10(n // 4), num=10, dtype=int))

            scales = scales[(scales >= 4) & (scales < n // 2)]

            if len(scales) < 2:
                return 0.5

            fluctuations = []

            for scale in scales:
                # Divide profile into segments
                n_segments = n // scale
                if n_segments == 0:
                    continue

                segment_fluct = []

                for i in range(n_segments):
                    segment = profile[i * scale:(i + 1) * scale]

                    # Fit linear trend
                    x = np.arange(len(segment))
                    coeffs = np.polyfit(x, segment, 1)
                    trend = np.polyval(coeffs, x)

                    # Calculate fluctuation
                    fluct = np.sqrt(np.mean((segment - trend) ** 2))
                    segment_fluct.append(fluct)

                if segment_fluct:
                    fluctuations.append(np.mean(segment_fluct))

            if len(fluctuations) < 2:
                return 0.5

            # Fit log(F) vs log(scale)
            log_scales = np.log(scales[:len(fluctuations)])
            log_fluctuations = np.log(fluctuations)

            # Remove invalid values
            valid_mask = np.isfinite(log_scales) & np.isfinite(log_fluctuations)
            if valid_mask.sum() < 2:
                return 0.5

            log_scales = log_scales[valid_mask]
            log_fluctuations = log_fluctuations[valid_mask]

            # Linear regression
            slope, _, _, _, _ = stats.linregress(log_scales, log_fluctuations)

            # Clip to valid range
            alpha = np.clip(slope, 0.0, 2.0)

            return float(alpha)

        except Exception as e:
            logger.debug("dfa_calculation_failed", error=str(e))
            return 0.5

    async def _compute_correlation_dimension(
        self,
        data: pl.DataFrame,
        column: str
    ) -> pl.DataFrame:
        """Compute correlation dimension (approximation).

        Measures the dimensionality of the attractor in phase space.
        Provides insights into the complexity of the underlying dynamics.

        Args:
            data: Input DataFrame
            column: Column to analyze

        Returns:
            DataFrame with correlation dimension features
        """
        result = data
        series = data[column].to_numpy().astype(np.float64)

        try:
            for window_size in self.window_sizes:
                if window_size < 50:  # Too small for reliable estimation
                    continue

                cd_values = []

                for i in range(len(series)):
                    if i < window_size - 1:
                        cd_values.append(Decimal('1.0'))
                    else:
                        window = series[i - window_size + 1:i + 1]
                        cd = self._calculate_correlation_dimension(window)
                        cd_values.append(Decimal(str(cd)))

                result = result.with_columns([
                    pl.Series(f'{column}_correlation_dim_{window_size}', cd_values)
                ])

            logger.debug("correlation_dimension_computed", column=column)

        except Exception as e:
            logger.warning(
                "correlation_dimension_computation_failed",
                column=column,
                error=str(e)
            )

        return result

    def _calculate_correlation_dimension(self, series: np.ndarray) -> float:
        """Calculate correlation dimension using Grassberger-Procaccia algorithm.

        Args:
            series: Time series data

        Returns:
            Correlation dimension
        """
        try:
            # Phase space embedding
            embedded = self._embed_time_series(series, self.embedding_dim, self.time_delay)

            n_points = len(embedded)
            if n_points < 10:
                return 1.0

            # Sample points for efficiency
            max_points = min(n_points, 1000)
            if n_points > max_points:
                indices = np.random.choice(n_points, max_points, replace=False)
                embedded = embedded[indices]
                n_points = max_points

            # Compute pairwise distances
            distances = []
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    dist = np.linalg.norm(embedded[i] - embedded[j])
                    distances.append(dist)

            distances = np.array(distances)
            if len(distances) == 0:
                return 1.0

            # Use a range of radii
            radii = np.percentile(distances, [10, 20, 30, 40, 50])

            # Count pairs within each radius
            correlations = []
            for r in radii:
                count = np.sum(distances <= r)
                correlation = count / len(distances)
                if correlation > 0:
                    correlations.append(correlation)

            if len(correlations) < 2:
                return 1.0

            # Fit log(C) vs log(r)
            log_radii = np.log(radii[:len(correlations)])
            log_correlations = np.log(correlations)

            valid_mask = np.isfinite(log_radii) & np.isfinite(log_correlations)
            if valid_mask.sum() < 2:
                return 1.0

            slope, _, _, _, _ = stats.linregress(
                log_radii[valid_mask],
                log_correlations[valid_mask]
            )

            # Correlation dimension is the slope
            cd = np.clip(slope, 0.0, float(self.embedding_dim))

            return float(cd)

        except Exception as e:
            logger.debug("correlation_dimension_calculation_failed", error=str(e))
            return 1.0

    async def _compute_sample_entropy(
        self,
        data: pl.DataFrame,
        column: str
    ) -> pl.DataFrame:
        """Compute sample entropy.

        Measures the complexity and regularity of time series.
        Higher values indicate more complexity/randomness.

        Args:
            data: Input DataFrame
            column: Column to analyze

        Returns:
            DataFrame with sample entropy features
        """
        result = data
        series = data[column].to_numpy().astype(np.float64)

        try:
            for window_size in self.window_sizes:
                entropy_values = []

                for i in range(len(series)):
                    if i < window_size - 1:
                        entropy_values.append(Decimal('0.0'))
                    else:
                        window = series[i - window_size + 1:i + 1]
                        entropy = self._calculate_sample_entropy(window)
                        entropy_values.append(Decimal(str(entropy)))

                result = result.with_columns([
                    pl.Series(f'{column}_sample_entropy_{window_size}', entropy_values)
                ])

            logger.debug("sample_entropy_computed", column=column)

        except Exception as e:
            logger.warning(
                "sample_entropy_computation_failed",
                column=column,
                error=str(e)
            )

        return result

    def _calculate_sample_entropy(
        self,
        series: np.ndarray,
        m: int = 2,
        r: Optional[float] = None
    ) -> float:
        """Calculate sample entropy.

        Args:
            series: Time series data
            m: Embedding dimension
            r: Tolerance (default: 0.2 * std)

        Returns:
            Sample entropy
        """
        try:
            n = len(series)
            if n < 10:
                return 0.0

            if r is None:
                r = 0.2 * np.std(series)

            # Count template matches
            def _count_matches(m_val):
                templates = np.array([series[i:i + m_val] for i in range(n - m_val + 1)])
                n_templates = len(templates)

                matches = 0
                for i in range(n_templates):
                    # Chebyshev distance
                    distances = np.max(np.abs(templates - templates[i]), axis=1)
                    matches += np.sum((distances <= r) & (distances > 0))

                return matches

            # Count matches for m and m+1
            matches_m = _count_matches(m)
            matches_m1 = _count_matches(m + 1)

            if matches_m == 0 or matches_m1 == 0:
                return 0.0

            # Sample entropy
            entropy = -np.log(matches_m1 / matches_m)

            return float(entropy)

        except Exception as e:
            logger.debug("sample_entropy_calculation_failed", error=str(e))
            return 0.0

    @staticmethod
    def _embed_time_series(
        series: np.ndarray,
        embedding_dim: int,
        time_delay: int
    ) -> np.ndarray:
        """Embed time series into phase space.

        Args:
            series: Time series data
            embedding_dim: Embedding dimension
            time_delay: Time delay

        Returns:
            Embedded time series
        """
        n = len(series)
        n_points = n - (embedding_dim - 1) * time_delay

        if n_points <= 0:
            return np.array([])

        embedded = np.zeros((n_points, embedding_dim))

        for i in range(n_points):
            for j in range(embedding_dim):
                embedded[i, j] = series[i + j * time_delay]

        return embedded

    def _add_null_features(
        self,
        data: pl.DataFrame,
        columns: List[str]
    ) -> pl.DataFrame:
        """Add null fractal features when data is insufficient.

        Args:
            data: Input DataFrame
            columns: Columns to add features for

        Returns:
            DataFrame with null features
        """
        result = data

        for col in columns:
            if 'hurst' in self.methods:
                for ws in self.window_sizes:
                    result = result.with_columns([
                        pl.lit(Decimal('0.5')).alias(f'{col}_hurst_{ws}')
                    ])

            if 'higuchi' in self.methods:
                for ws in self.window_sizes:
                    result = result.with_columns([
                        pl.lit(Decimal('1.0')).alias(f'{col}_higuchi_fd_{ws}')
                    ])

            if 'detrended_fluctuation' in self.methods:
                for ws in self.window_sizes:
                    result = result.with_columns([
                        pl.lit(Decimal('0.5')).alias(f'{col}_dfa_{ws}')
                    ])

            if 'correlation_dimension' in self.methods:
                for ws in self.window_sizes:
                    if ws >= 50:
                        result = result.with_columns([
                            pl.lit(Decimal('1.0')).alias(f'{col}_correlation_dim_{ws}')
                        ])

            if 'entropy' in self.methods:
                for ws in self.window_sizes:
                    result = result.with_columns([
                        pl.lit(Decimal('0.0')).alias(f'{col}_sample_entropy_{ws}')
                    ])

        return result
