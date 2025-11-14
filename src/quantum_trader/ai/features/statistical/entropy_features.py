"""Entropy-based features for market data analysis.

This module extracts information-theoretic features including Shannon entropy,
approximate entropy, sample entropy, and permutation entropy from trading data.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import polars as pl
from scipy.stats import entropy as scipy_entropy
from structlog import get_logger

logger = get_logger(__name__)


class EntropyFeatureExtractor:
    """Extract entropy-based features from market data.

    Uses information theory to measure market complexity, predictability,
    and regime changes through various entropy measures.

    Attributes:
        config: Configuration dictionary
        lookback_periods: List of lookback periods
        n_bins: Number of bins for histogram-based entropy
        embedding_dim: Embedding dimension for ApEn/SampEn
        tolerance: Tolerance parameter for ApEn/SampEn
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize entropy feature extractor.

        Args:
            config: Configuration dictionary containing:
                - lookback_periods: List of periods for calculations
                - n_bins: Number of bins for discretization
                - embedding_dim: Embedding dimension
                - tolerance: Tolerance for entropy calculations
                - permutation_order: Order for permutation entropy

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self._validate_config()

        self.lookback_periods = config.get('lookback_periods', [20, 60, 120])
        self.n_bins = config.get('n_bins', 10)
        self.embedding_dim = config.get('embedding_dim', 2)
        self.tolerance = Decimal(str(config.get('tolerance', '0.2')))
        self.permutation_order = config.get('permutation_order', 3)

        logger.info(
            "entropy_feature_extractor_initialized",
            lookback_periods=self.lookback_periods,
            n_bins=self.n_bins,
            embedding_dim=self.embedding_dim
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If config parameters are invalid
        """
        if 'lookback_periods' in self.config:
            periods = self.config['lookback_periods']
            if not isinstance(periods, list) or not all(p > 0 for p in periods):
                raise ValueError("lookback_periods must be list of positive integers")

        if 'n_bins' in self.config and self.config['n_bins'] <= 0:
            raise ValueError("n_bins must be positive")

        if 'embedding_dim' in self.config and self.config['embedding_dim'] <= 0:
            raise ValueError("embedding_dim must be positive")

    def extract_features(
        self,
        market_data: pl.DataFrame
    ) -> pl.DataFrame:
        """Extract entropy features from market data.

        Args:
            market_data: Polars DataFrame with columns:
                - timestamp: UTC timestamp
                - open: Opening price
                - high: High price
                - low: Low price
                - close: Closing price
                - volume: Trading volume

        Returns:
            DataFrame with entropy features

        Raises:
            ValueError: If input data is invalid
        """
        try:
            self._validate_market_data(market_data)

            # Calculate returns
            returns = self._calculate_returns(market_data)

            features = market_data.select(['timestamp'])

            # Extract entropy features for each period
            for period in self.lookback_periods:
                # Shannon entropy
                shannon_features = self._calculate_shannon_entropy(
                    returns, period
                )
                features = features.hstack(shannon_features)

                # Approximate entropy
                apen_features = self._calculate_approximate_entropy(
                    returns, period
                )
                features = features.hstack(apen_features)

                # Sample entropy
                sampen_features = self._calculate_sample_entropy(
                    returns, period
                )
                features = features.hstack(sampen_features)

                # Permutation entropy
                perm_entropy_features = self._calculate_permutation_entropy(
                    returns, period
                )
                features = features.hstack(perm_entropy_features)

                # Conditional entropy
                cond_entropy_features = self._calculate_conditional_entropy(
                    returns, period
                )
                features = features.hstack(cond_entropy_features)

                # Volume entropy
                vol_entropy_features = self._calculate_volume_entropy(
                    market_data, period
                )
                features = features.hstack(vol_entropy_features)

            logger.debug(
                "entropy_features_extracted",
                rows=features.height,
                features=features.width - 1
            )

            return features

        except Exception as e:
            logger.error("entropy_feature_extraction_failed", error=str(e))
            raise

    def _validate_market_data(self, df: pl.DataFrame) -> None:
        """Validate market data format.

        Args:
            df: Market data dataframe

        Raises:
            ValueError: If data format is invalid
        """
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            error_msg = f"Missing required columns: {missing_cols}"
            logger.error("market_data_validation_failed", error=error_msg)
            raise ValueError(error_msg)

        if df.height == 0:
            raise ValueError("Empty market data dataframe")

    def _calculate_returns(self, df: pl.DataFrame) -> np.ndarray:
        """Calculate log returns from price data.

        Args:
            df: Market data dataframe

        Returns:
            Array of log returns
        """
        close_prices = df['close'].to_numpy()
        returns = np.diff(np.log(close_prices))
        return returns

    def _calculate_shannon_entropy(
        self,
        returns: np.ndarray,
        period: int
    ) -> pl.DataFrame:
        """Calculate Shannon entropy of returns distribution.

        Args:
            returns: Returns array
            period: Lookback period

        Returns:
            DataFrame with Shannon entropy features
        """
        try:
            entropy_values = []

            for i in range(len(returns)):
                if i < period - 1:
                    entropy_values.append(None)
                else:
                    window = returns[i - period + 1:i + 1]
                    window_clean = window[~np.isnan(window)]

                    if len(window_clean) > 0:
                        # Create histogram
                        hist, _ = np.histogram(window_clean, bins=self.n_bins)
                        hist = hist + 1e-10  # Avoid log(0)

                        # Calculate Shannon entropy
                        prob = hist / hist.sum()
                        ent = float(scipy_entropy(prob))
                        entropy_values.append(ent)
                    else:
                        entropy_values.append(None)

            features_dict = {
                f'shannon_entropy_{period}': pl.Series(entropy_values)
            }

            # Normalized entropy (0 to 1)
            max_entropy = np.log(self.n_bins)
            normalized_values = [
                val / max_entropy if val is not None else None
                for val in entropy_values
            ]
            features_dict[f'shannon_entropy_norm_{period}'] = pl.Series(normalized_values)

            return pl.DataFrame(features_dict)

        except Exception as e:
            logger.error("shannon_entropy_calculation_failed", error=str(e), period=period)
            raise

    def _calculate_approximate_entropy(
        self,
        returns: np.ndarray,
        period: int
    ) -> pl.DataFrame:
        """Calculate approximate entropy (ApEn).

        Args:
            returns: Returns array
            period: Lookback period

        Returns:
            DataFrame with ApEn features
        """
        try:
            apen_values = []
            tolerance = float(self.tolerance)

            for i in range(len(returns)):
                if i < period - 1:
                    apen_values.append(None)
                else:
                    window = returns[i - period + 1:i + 1]
                    window_clean = window[~np.isnan(window)]

                    if len(window_clean) >= self.embedding_dim + 1:
                        apen = self._compute_approximate_entropy(
                            window_clean,
                            self.embedding_dim,
                            tolerance
                        )
                        apen_values.append(apen)
                    else:
                        apen_values.append(None)

            return pl.DataFrame({
                f'approximate_entropy_{period}': pl.Series(apen_values)
            })

        except Exception as e:
            logger.error("approximate_entropy_calculation_failed", error=str(e), period=period)
            raise

    def _compute_approximate_entropy(
        self,
        data: np.ndarray,
        m: int,
        r: float
    ) -> float:
        """Compute approximate entropy.

        Args:
            data: Time series data
            m: Embedding dimension
            r: Tolerance (as fraction of std dev)

        Returns:
            Approximate entropy value
        """
        def _maxdist(xi, xj):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])

        def _phi(m):
            N = len(data)
            patterns = np.array([data[i:i + m] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)

            for i in range(N - m + 1):
                template = patterns[i]
                for j in range(N - m + 1):
                    if _maxdist(template, patterns[j]) <= r * np.std(data):
                        C[i] += 1

            C = C / (N - m + 1)
            return np.sum(np.log(C + 1e-10)) / (N - m + 1)

        return float(_phi(m) - _phi(m + 1))

    def _calculate_sample_entropy(
        self,
        returns: np.ndarray,
        period: int
    ) -> pl.DataFrame:
        """Calculate sample entropy (SampEn).

        Args:
            returns: Returns array
            period: Lookback period

        Returns:
            DataFrame with SampEn features
        """
        try:
            sampen_values = []
            tolerance = float(self.tolerance)

            for i in range(len(returns)):
                if i < period - 1:
                    sampen_values.append(None)
                else:
                    window = returns[i - period + 1:i + 1]
                    window_clean = window[~np.isnan(window)]

                    if len(window_clean) >= self.embedding_dim + 1:
                        sampen = self._compute_sample_entropy(
                            window_clean,
                            self.embedding_dim,
                            tolerance
                        )
                        sampen_values.append(sampen)
                    else:
                        sampen_values.append(None)

            return pl.DataFrame({
                f'sample_entropy_{period}': pl.Series(sampen_values)
            })

        except Exception as e:
            logger.error("sample_entropy_calculation_failed", error=str(e), period=period)
            raise

    def _compute_sample_entropy(
        self,
        data: np.ndarray,
        m: int,
        r: float
    ) -> float:
        """Compute sample entropy.

        Args:
            data: Time series data
            m: Embedding dimension
            r: Tolerance (as fraction of std dev)

        Returns:
            Sample entropy value
        """
        N = len(data)
        B = 0.0
        A = 0.0

        tolerance = r * np.std(data)

        # Embedding dimension m
        for i in range(N - m):
            template_m = data[i:i + m]
            for j in range(i + 1, N - m):
                if np.max(np.abs(template_m - data[j:j + m])) <= tolerance:
                    B += 1

                    # Embedding dimension m+1
                    if np.max(np.abs(data[i:i + m + 1] - data[j:j + m + 1])) <= tolerance:
                        A += 1

        if B > 0:
            return float(-np.log(A / B))
        else:
            return float('inf')

    def _calculate_permutation_entropy(
        self,
        returns: np.ndarray,
        period: int
    ) -> pl.DataFrame:
        """Calculate permutation entropy.

        Args:
            returns: Returns array
            period: Lookback period

        Returns:
            DataFrame with permutation entropy features
        """
        try:
            perm_entropy_values = []
            order = self.permutation_order

            for i in range(len(returns)):
                if i < period - 1:
                    perm_entropy_values.append(None)
                else:
                    window = returns[i - period + 1:i + 1]
                    window_clean = window[~np.isnan(window)]

                    if len(window_clean) >= order:
                        perm_ent = self._compute_permutation_entropy(
                            window_clean,
                            order
                        )
                        perm_entropy_values.append(perm_ent)
                    else:
                        perm_entropy_values.append(None)

            # Normalized permutation entropy
            max_perm_entropy = np.log(np.math.factorial(order))
            normalized_values = [
                val / max_perm_entropy if val is not None and max_perm_entropy > 0 else None
                for val in perm_entropy_values
            ]

            return pl.DataFrame({
                f'permutation_entropy_{period}': pl.Series(perm_entropy_values),
                f'permutation_entropy_norm_{period}': pl.Series(normalized_values)
            })

        except Exception as e:
            logger.error("permutation_entropy_calculation_failed", error=str(e), period=period)
            raise

    def _compute_permutation_entropy(
        self,
        data: np.ndarray,
        order: int
    ) -> float:
        """Compute permutation entropy.

        Args:
            data: Time series data
            order: Permutation order

        Returns:
            Permutation entropy value
        """
        N = len(data)
        permutations = {}

        for i in range(N - order + 1):
            # Get sorting permutation
            segment = data[i:i + order]
            perm = tuple(np.argsort(segment))

            if perm in permutations:
                permutations[perm] += 1
            else:
                permutations[perm] = 1

        # Calculate entropy
        total = sum(permutations.values())
        probs = [count / total for count in permutations.values()]

        return float(scipy_entropy(probs))

    def _calculate_conditional_entropy(
        self,
        returns: np.ndarray,
        period: int
    ) -> pl.DataFrame:
        """Calculate conditional entropy H(X|Y).

        Args:
            returns: Returns array
            period: Lookback period

        Returns:
            DataFrame with conditional entropy features
        """
        try:
            cond_entropy_values = []

            for i in range(len(returns)):
                if i < period - 1:
                    cond_entropy_values.append(None)
                else:
                    window = returns[i - period + 1:i + 1]
                    window_clean = window[~np.isnan(window)]

                    if len(window_clean) > 2:
                        # Split into X (current) and Y (lagged)
                        X = window_clean[1:]
                        Y = window_clean[:-1]

                        # Discretize
                        X_binned = np.digitize(X, bins=np.linspace(X.min(), X.max(), self.n_bins))
                        Y_binned = np.digitize(Y, bins=np.linspace(Y.min(), Y.max(), self.n_bins))

                        # Calculate joint and marginal distributions
                        joint_hist = np.histogram2d(
                            X_binned,
                            Y_binned,
                            bins=self.n_bins
                        )[0]
                        joint_hist = joint_hist + 1e-10

                        # Conditional entropy H(X|Y) = H(X,Y) - H(Y)
                        joint_prob = joint_hist / joint_hist.sum()
                        marginal_Y = joint_prob.sum(axis=0)

                        h_xy = scipy_entropy(joint_prob.flatten())
                        h_y = scipy_entropy(marginal_Y)
                        h_x_given_y = h_xy - h_y

                        cond_entropy_values.append(float(h_x_given_y))
                    else:
                        cond_entropy_values.append(None)

            return pl.DataFrame({
                f'conditional_entropy_{period}': pl.Series(cond_entropy_values)
            })

        except Exception as e:
            logger.error("conditional_entropy_calculation_failed", error=str(e), period=period)
            raise

    def _calculate_volume_entropy(
        self,
        market_data: pl.DataFrame,
        period: int
    ) -> pl.DataFrame:
        """Calculate entropy of volume distribution.

        Args:
            market_data: Market data dataframe
            period: Lookback period

        Returns:
            DataFrame with volume entropy features
        """
        try:
            volume_array = market_data['volume'].to_numpy()
            vol_entropy_values = []

            for i in range(len(volume_array)):
                if i < period - 1:
                    vol_entropy_values.append(None)
                else:
                    window = volume_array[i - period + 1:i + 1]
                    window_clean = window[~np.isnan(window)]

                    if len(window_clean) > 0:
                        # Create histogram
                        hist, _ = np.histogram(window_clean, bins=self.n_bins)
                        hist = hist + 1e-10

                        # Calculate entropy
                        prob = hist / hist.sum()
                        ent = float(scipy_entropy(prob))
                        vol_entropy_values.append(ent)
                    else:
                        vol_entropy_values.append(None)

            return pl.DataFrame({
                f'volume_entropy_{period}': pl.Series(vol_entropy_values)
            })

        except Exception as e:
            logger.error("volume_entropy_calculation_failed", error=str(e), period=period)
            raise

    async def extract_features_async(
        self,
        market_data: pl.DataFrame
    ) -> pl.DataFrame:
        """Async wrapper for feature extraction.

        Args:
            market_data: Market data dataframe

        Returns:
            DataFrame with extracted features
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.extract_features,
            market_data
        )

    def calculate_mutual_information(
        self,
        series1: np.ndarray,
        series2: np.ndarray
    ) -> Decimal:
        """Calculate mutual information between two series.

        Args:
            series1: First time series
            series2: Second time series

        Returns:
            Mutual information value

        Raises:
            ValueError: If series have different lengths
        """
        if len(series1) != len(series2):
            raise ValueError("Series must have same length")

        # Discretize both series
        s1_binned = np.digitize(series1, bins=np.linspace(series1.min(), series1.max(), self.n_bins))
        s2_binned = np.digitize(series2, bins=np.linspace(series2.min(), series2.max(), self.n_bins))

        # Calculate joint distribution
        joint_hist = np.histogram2d(s1_binned, s2_binned, bins=self.n_bins)[0]
        joint_hist = joint_hist + 1e-10
        joint_prob = joint_hist / joint_hist.sum()

        # Calculate marginal distributions
        marginal_1 = joint_prob.sum(axis=1)
        marginal_2 = joint_prob.sum(axis=0)

        # Mutual information
        mi = 0.0
        for i in range(len(marginal_1)):
            for j in range(len(marginal_2)):
                if joint_prob[i, j] > 0:
                    mi += joint_prob[i, j] * np.log(
                        joint_prob[i, j] / (marginal_1[i] * marginal_2[j])
                    )

        return Decimal(str(mi))
