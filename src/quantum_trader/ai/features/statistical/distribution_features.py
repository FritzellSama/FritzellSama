"""Statistical distribution features for market data analysis.

Extracts features based on statistical distributions of prices, returns, and volumes
including skewness, kurtosis, and various distribution moments.
"""

import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import polars as pl
from scipy import stats
from structlog import get_logger

logger = get_logger(__name__)


class DistributionFeatureExtractor:
    """Extract statistical distribution features from market data.

    Analyzes the statistical properties of price and volume distributions
    to identify market regimes and trading opportunities.

    Attributes:
        config: Configuration dictionary
        lookback_periods: List of lookback periods for feature calculation
        return_type: Type of returns to calculate ('simple', 'log')
        normalize: Whether to normalize features
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize distribution feature extractor.

        Args:
            config: Configuration dictionary containing:
                - lookback_periods: List of periods for rolling calculations
                - return_type: 'simple' or 'log' returns
                - normalize: Whether to normalize features
                - min_observations: Minimum observations for calculation

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self._validate_config()

        self.lookback_periods = config.get('lookback_periods', [20, 60, 120])
        self.return_type = config.get('return_type', 'log')
        self.normalize = config.get('normalize', True)
        self.min_observations = config.get('min_observations', 30)

        logger.info(
            "distribution_feature_extractor_initialized",
            lookback_periods=self.lookback_periods,
            return_type=self.return_type
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required config parameters are missing or invalid
        """
        if 'lookback_periods' in self.config:
            periods = self.config['lookback_periods']
            if not isinstance(periods, list) or not all(p > 0 for p in periods):
                raise ValueError("lookback_periods must be list of positive integers")

        if 'return_type' in self.config:
            if self.config['return_type'] not in ['simple', 'log']:
                raise ValueError("return_type must be 'simple' or 'log'")

    def extract_features(
        self,
        market_data: pl.DataFrame
    ) -> pl.DataFrame:
        """Extract distribution features from market data.

        Args:
            market_data: Polars DataFrame with columns:
                - timestamp: UTC timestamp
                - open: Opening price
                - high: High price
                - low: Low price
                - close: Closing price
                - volume: Trading volume

        Returns:
            DataFrame with distribution features

        Raises:
            ValueError: If input data is invalid
        """
        try:
            self._validate_market_data(market_data)

            # Calculate returns
            returns_df = self._calculate_returns(market_data)

            features = market_data.select(['timestamp'])

            # Extract distribution moments
            for period in self.lookback_periods:
                moment_features = self._calculate_distribution_moments(
                    returns_df, period
                )
                features = features.hstack(moment_features)

                # Calculate higher moments
                higher_moment_features = self._calculate_higher_moments(
                    returns_df, period
                )
                features = features.hstack(higher_moment_features)

                # Calculate quantile features
                quantile_features = self._calculate_quantile_features(
                    returns_df, period
                )
                features = features.hstack(quantile_features)

                # Calculate distribution shape features
                shape_features = self._calculate_shape_features(
                    returns_df, period
                )
                features = features.hstack(shape_features)

                # Volume distribution features
                volume_features = self._calculate_volume_distribution_features(
                    market_data, period
                )
                features = features.hstack(volume_features)

            logger.debug(
                "distribution_features_extracted",
                rows=features.height,
                features=features.width - 1
            )

            return features

        except Exception as e:
            logger.error("distribution_feature_extraction_failed", error=str(e))
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

        if df.height < self.min_observations:
            raise ValueError(
                f"Need at least {self.min_observations} observations, "
                f"got {df.height}"
            )

    def _calculate_returns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate returns from price data.

        Args:
            df: Market data dataframe

        Returns:
            DataFrame with returns
        """
        if self.return_type == 'log':
            # Log returns: ln(P_t / P_{t-1})
            returns = df.select([
                'timestamp',
                (pl.col('close').log() - pl.col('close').shift(1).log()).alias('returns'),
                pl.col('volume')
            ])
        else:
            # Simple returns: (P_t - P_{t-1}) / P_{t-1}
            returns = df.select([
                'timestamp',
                ((pl.col('close') - pl.col('close').shift(1)) / pl.col('close').shift(1)).alias('returns'),
                pl.col('volume')
            ])

        return returns.drop_nulls()

    def _calculate_distribution_moments(
        self,
        returns_df: pl.DataFrame,
        period: int
    ) -> pl.DataFrame:
        """Calculate basic distribution moments.

        Args:
            returns_df: DataFrame with returns
            period: Lookback period

        Returns:
            DataFrame with moment features
        """
        try:
            features_dict = {}

            # Mean return
            features_dict[f'mean_return_{period}'] = (
                returns_df['returns'].rolling_mean(window_size=period)
            )

            # Standard deviation (volatility)
            features_dict[f'std_return_{period}'] = (
                returns_df['returns'].rolling_std(window_size=period)
            )

            # Variance
            features_dict[f'var_return_{period}'] = (
                returns_df['returns'].rolling_var(window_size=period)
            )

            # Coefficient of variation
            mean_returns = returns_df['returns'].rolling_mean(window_size=period)
            std_returns = returns_df['returns'].rolling_std(window_size=period)

            features_dict[f'cv_return_{period}'] = (
                std_returns / mean_returns.abs().fill_null(Decimal('1'))
            )

            return pl.DataFrame(features_dict)

        except Exception as e:
            logger.error("moment_calculation_failed", error=str(e), period=period)
            raise

    def _calculate_higher_moments(
        self,
        returns_df: pl.DataFrame,
        period: int
    ) -> pl.DataFrame:
        """Calculate higher-order distribution moments.

        Args:
            returns_df: DataFrame with returns
            period: Lookback period

        Returns:
            DataFrame with higher moment features
        """
        try:
            features_dict = {}

            # Convert to numpy for scipy calculations
            returns_array = returns_df['returns'].to_numpy()

            # Calculate rolling skewness
            skewness_values = []
            kurtosis_values = []

            for i in range(len(returns_array)):
                if i < period - 1:
                    skewness_values.append(None)
                    kurtosis_values.append(None)
                else:
                    window = returns_array[i - period + 1:i + 1]
                    window_clean = window[~np.isnan(window)]

                    if len(window_clean) >= self.min_observations:
                        skew = float(stats.skew(window_clean))
                        kurt = float(stats.kurtosis(window_clean))
                        skewness_values.append(skew)
                        kurtosis_values.append(kurt)
                    else:
                        skewness_values.append(None)
                        kurtosis_values.append(None)

            features_dict[f'skewness_{period}'] = pl.Series(skewness_values)
            features_dict[f'kurtosis_{period}'] = pl.Series(kurtosis_values)

            # Excess kurtosis (kurtosis - 3)
            kurt_series = pl.Series(kurtosis_values)
            features_dict[f'excess_kurtosis_{period}'] = kurt_series - Decimal('3')

            return pl.DataFrame(features_dict)

        except Exception as e:
            logger.error("higher_moment_calculation_failed", error=str(e), period=period)
            raise

    def _calculate_quantile_features(
        self,
        returns_df: pl.DataFrame,
        period: int
    ) -> pl.DataFrame:
        """Calculate quantile-based features.

        Args:
            returns_df: DataFrame with returns
            period: Lookback period

        Returns:
            DataFrame with quantile features
        """
        try:
            features_dict = {}

            returns_array = returns_df['returns'].to_numpy()
            quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

            for q in quantiles:
                q_values = []

                for i in range(len(returns_array)):
                    if i < period - 1:
                        q_values.append(None)
                    else:
                        window = returns_array[i - period + 1:i + 1]
                        window_clean = window[~np.isnan(window)]

                        if len(window_clean) > 0:
                            q_val = float(np.quantile(window_clean, q))
                            q_values.append(q_val)
                        else:
                            q_values.append(None)

                q_label = str(q).replace('.', 'p')
                features_dict[f'quantile_{q_label}_{period}'] = pl.Series(q_values)

            # Interquartile range
            q25_series = features_dict[f'quantile_0p25_{period}']
            q75_series = features_dict[f'quantile_0p75_{period}']
            features_dict[f'iqr_{period}'] = q75_series - q25_series

            # Range (max - min)
            max_values = []
            min_values = []

            for i in range(len(returns_array)):
                if i < period - 1:
                    max_values.append(None)
                    min_values.append(None)
                else:
                    window = returns_array[i - period + 1:i + 1]
                    window_clean = window[~np.isnan(window)]

                    if len(window_clean) > 0:
                        max_values.append(float(np.max(window_clean)))
                        min_values.append(float(np.min(window_clean)))
                    else:
                        max_values.append(None)
                        min_values.append(None)

            max_series = pl.Series(max_values)
            min_series = pl.Series(min_values)
            features_dict[f'range_{period}'] = max_series - min_series

            return pl.DataFrame(features_dict)

        except Exception as e:
            logger.error("quantile_calculation_failed", error=str(e), period=period)
            raise

    def _calculate_shape_features(
        self,
        returns_df: pl.DataFrame,
        period: int
    ) -> pl.DataFrame:
        """Calculate distribution shape features.

        Args:
            returns_df: DataFrame with returns
            period: Lookback period

        Returns:
            DataFrame with shape features
        """
        try:
            features_dict = {}

            returns_array = returns_df['returns'].to_numpy()

            # Jarque-Bera test statistic (normality test)
            jb_values = []
            jb_pvalues = []

            for i in range(len(returns_array)):
                if i < period - 1:
                    jb_values.append(None)
                    jb_pvalues.append(None)
                else:
                    window = returns_array[i - period + 1:i + 1]
                    window_clean = window[~np.isnan(window)]

                    if len(window_clean) >= self.min_observations:
                        try:
                            jb_stat, jb_p = stats.jarque_bera(window_clean)
                            jb_values.append(float(jb_stat))
                            jb_pvalues.append(float(jb_p))
                        except:
                            jb_values.append(None)
                            jb_pvalues.append(None)
                    else:
                        jb_values.append(None)
                        jb_pvalues.append(None)

            features_dict[f'jarque_bera_stat_{period}'] = pl.Series(jb_values)
            features_dict[f'jarque_bera_pval_{period}'] = pl.Series(jb_pvalues)

            # Positive/negative return ratios
            pos_ratio_values = []
            neg_ratio_values = []

            for i in range(len(returns_array)):
                if i < period - 1:
                    pos_ratio_values.append(None)
                    neg_ratio_values.append(None)
                else:
                    window = returns_array[i - period + 1:i + 1]
                    window_clean = window[~np.isnan(window)]

                    if len(window_clean) > 0:
                        pos_count = np.sum(window_clean > 0)
                        neg_count = np.sum(window_clean < 0)
                        total = len(window_clean)

                        pos_ratio_values.append(float(pos_count / total))
                        neg_ratio_values.append(float(neg_count / total))
                    else:
                        pos_ratio_values.append(None)
                        neg_ratio_values.append(None)

            features_dict[f'positive_ratio_{period}'] = pl.Series(pos_ratio_values)
            features_dict[f'negative_ratio_{period}'] = pl.Series(neg_ratio_values)

            return pl.DataFrame(features_dict)

        except Exception as e:
            logger.error("shape_feature_calculation_failed", error=str(e), period=period)
            raise

    def _calculate_volume_distribution_features(
        self,
        market_data: pl.DataFrame,
        period: int
    ) -> pl.DataFrame:
        """Calculate volume distribution features.

        Args:
            market_data: Market data dataframe
            period: Lookback period

        Returns:
            DataFrame with volume distribution features
        """
        try:
            features_dict = {}

            # Volume statistics
            features_dict[f'volume_mean_{period}'] = (
                market_data['volume'].rolling_mean(window_size=period)
            )

            features_dict[f'volume_std_{period}'] = (
                market_data['volume'].rolling_std(window_size=period)
            )

            # Volume coefficient of variation
            vol_mean = market_data['volume'].rolling_mean(window_size=period)
            vol_std = market_data['volume'].rolling_std(window_size=period)

            features_dict[f'volume_cv_{period}'] = (
                vol_std / vol_mean.fill_null(Decimal('1'))
            )

            # Volume percentile rank (current volume vs historical)
            volume_array = market_data['volume'].to_numpy()
            vol_percentile_values = []

            for i in range(len(volume_array)):
                if i < period - 1:
                    vol_percentile_values.append(None)
                else:
                    window = volume_array[i - period + 1:i + 1]
                    current_vol = volume_array[i]

                    # Calculate percentile rank
                    rank = np.sum(window <= current_vol) / len(window)
                    vol_percentile_values.append(float(rank))

            features_dict[f'volume_percentile_{period}'] = pl.Series(vol_percentile_values)

            return pl.DataFrame(features_dict)

        except Exception as e:
            logger.error(
                "volume_distribution_calculation_failed",
                error=str(e),
                period=period
            )
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

    def normalize_features(
        self,
        features_df: pl.DataFrame,
        method: str = 'zscore'
    ) -> pl.DataFrame:
        """Normalize features.

        Args:
            features_df: Features dataframe
            method: Normalization method ('zscore', 'minmax', 'robust')

        Returns:
            Normalized features dataframe

        Raises:
            ValueError: If method is invalid
        """
        try:
            if method not in ['zscore', 'minmax', 'robust']:
                raise ValueError(f"Invalid normalization method: {method}")

            # Keep timestamp column
            timestamp_col = features_df.select(['timestamp'])
            feature_cols = [c for c in features_df.columns if c != 'timestamp']

            if method == 'zscore':
                # Z-score normalization
                normalized = features_df.select([
                    ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std())
                    .alias(col)
                    for col in feature_cols
                ])

            elif method == 'minmax':
                # Min-max normalization to [0, 1]
                normalized = features_df.select([
                    ((pl.col(col) - pl.col(col).min()) /
                     (pl.col(col).max() - pl.col(col).min()))
                    .alias(col)
                    for col in feature_cols
                ])

            else:  # robust
                # Robust normalization using median and IQR
                normalized = features_df.select([
                    ((pl.col(col) - pl.col(col).median()) /
                     (pl.col(col).quantile(0.75) - pl.col(col).quantile(0.25)))
                    .alias(col)
                    for col in feature_cols
                ])

            return timestamp_col.hstack(normalized)

        except Exception as e:
            logger.error("feature_normalization_failed", error=str(e))
            raise
