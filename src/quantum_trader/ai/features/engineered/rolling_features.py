"""Rolling window feature engineering for time series data.

This module implements rolling window features like moving averages, volatility,
momentum indicators, and statistical features optimized for trading.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RollingFeatureConfig:
    """Configuration for a single rolling feature.

    Attributes:
        name: Feature name
        function: Function to apply ('mean', 'std', 'min', 'max', 'sum', 'skew', 'kurt')
        window: Window size
        column: Column to compute feature on
        min_periods: Minimum periods required
    """
    name: str
    function: str
    window: int
    column: str
    min_periods: Optional[int] = None


class RollingFeatureEngine:
    """Rolling window feature engineering for trading data.

    Computes rolling window features efficiently using Polars DataFrame
    operations with proper Decimal handling.

    Attributes:
        config: Configuration dictionary
        feature_configs: List of feature configurations

    Example:
        >>> config = {
        ...     'features': [
        ...         {'name': 'sma_20', 'function': 'mean', 'window': 20, 'column': 'close'},
        ...         {'name': 'vol_20', 'function': 'std', 'window': 20, 'column': 'returns'},
        ...         {'name': 'momentum_10', 'function': 'custom', 'window': 10, 'column': 'close'}
        ...     ],
        ...     'handle_missing': 'drop'
        ... }
        >>> engine = RollingFeatureEngine(config)
        >>> features_df = engine.compute_features(data)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize rolling feature engine.

        Args:
            config: Configuration with keys:
                - features: List of feature configurations
                - handle_missing: 'drop', 'forward_fill', or 'zero'
                - parallelize: Whether to parallelize computations
        """
        self.config = config
        self._validate_config()

        self.handle_missing = config.get('handle_missing', 'drop')
        self.parallelize = config.get('parallelize', False)

        # Parse feature configurations
        self.feature_configs: List[RollingFeatureConfig] = []
        for feat_conf in config['features']:
            self.feature_configs.append(
                RollingFeatureConfig(
                    name=feat_conf['name'],
                    function=feat_conf['function'],
                    window=feat_conf['window'],
                    column=feat_conf['column'],
                    min_periods=feat_conf.get('min_periods')
                )
            )

        logger.info(
            "initialized_rolling_feature_engine",
            n_features=len(self.feature_configs),
            handle_missing=self.handle_missing
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If config is invalid
        """
        if 'features' not in self.config:
            raise ValueError("Missing required config key: features")

        if not self.config['features']:
            raise ValueError("features list cannot be empty")

        for feat in self.config['features']:
            required = ['name', 'function', 'window', 'column']
            for key in required:
                if key not in feat:
                    raise ValueError(f"Feature config missing key: {key}")

            if feat['window'] <= 0:
                raise ValueError(f"Window must be positive: {feat['name']}")

    def compute_features(self, data: pl.DataFrame) -> pl.DataFrame:
        """Compute all rolling features.

        Args:
            data: Input data (Polars DataFrame)

        Returns:
            DataFrame with original data and new features

        Raises:
            ValueError: If required columns missing
        """
        try:
            logger.info("computing_rolling_features", n_rows=len(data))

            result = data.clone()

            for feat_config in self.feature_configs:
                # Check if column exists
                if feat_config.column not in result.columns:
                    raise ValueError(f"Column not found: {feat_config.column}")

                # Compute feature
                feature_series = self._compute_single_feature(
                    result,
                    feat_config
                )

                # Add to result
                result = result.with_columns(
                    feature_series.alias(feat_config.name)
                )

                logger.debug(
                    "computed_feature",
                    name=feat_config.name,
                    function=feat_config.function,
                    window=feat_config.window
                )

            # Handle missing values
            result = self._handle_missing_values(result)

            logger.info(
                "rolling_features_complete",
                n_rows=len(result),
                n_features=len(self.feature_configs)
            )

            return result

        except Exception as e:
            logger.error("failed_to_compute_features", error=str(e))
            raise

    def _compute_single_feature(
        self,
        data: pl.DataFrame,
        feat_config: RollingFeatureConfig
    ) -> pl.Series:
        """Compute single rolling feature.

        Args:
            data: Input data
            feat_config: Feature configuration

        Returns:
            Feature series
        """
        column = data[feat_config.column]
        window = feat_config.window
        min_periods = feat_config.min_periods or window

        if feat_config.function == 'mean':
            return column.rolling_mean(window, min_periods=min_periods)

        elif feat_config.function == 'std':
            return column.rolling_std(window, min_periods=min_periods)

        elif feat_config.function == 'min':
            return column.rolling_min(window, min_periods=min_periods)

        elif feat_config.function == 'max':
            return column.rolling_max(window, min_periods=min_periods)

        elif feat_config.function == 'sum':
            return column.rolling_sum(window, min_periods=min_periods)

        elif feat_config.function == 'var':
            return column.rolling_var(window, min_periods=min_periods)

        elif feat_config.function == 'median':
            return column.rolling_median(window, min_periods=min_periods)

        elif feat_config.function == 'skew':
            return self._rolling_skew(column, window, min_periods)

        elif feat_config.function == 'kurt':
            return self._rolling_kurtosis(column, window, min_periods)

        elif feat_config.function == 'momentum':
            return self._rolling_momentum(column, window)

        elif feat_config.function == 'roc':
            return self._rolling_rate_of_change(column, window)

        elif feat_config.function == 'zscore':
            return self._rolling_zscore(column, window, min_periods)

        else:
            raise ValueError(f"Unknown function: {feat_config.function}")

    def _rolling_skew(
        self,
        series: pl.Series,
        window: int,
        min_periods: int
    ) -> pl.Series:
        """Calculate rolling skewness."""
        # Convert to numpy for calculation
        values = series.to_numpy()
        result = np.full(len(values), np.nan)

        for i in range(len(values)):
            start = max(0, i - window + 1)
            window_data = values[start:i+1]

            if len(window_data) >= min_periods:
                # Calculate skewness
                mean = np.mean(window_data)
                std = np.std(window_data, ddof=1)

                if std > 1e-8:
                    skew = np.mean(((window_data - mean) / std) ** 3)
                    result[i] = skew

        return pl.Series(series.name, result)

    def _rolling_kurtosis(
        self,
        series: pl.Series,
        window: int,
        min_periods: int
    ) -> pl.Series:
        """Calculate rolling kurtosis."""
        values = series.to_numpy()
        result = np.full(len(values), np.nan)

        for i in range(len(values)):
            start = max(0, i - window + 1)
            window_data = values[start:i+1]

            if len(window_data) >= min_periods:
                mean = np.mean(window_data)
                std = np.std(window_data, ddof=1)

                if std > 1e-8:
                    # Excess kurtosis
                    kurt = np.mean(((window_data - mean) / std) ** 4) - 3
                    result[i] = kurt

        return pl.Series(series.name, result)

    def _rolling_momentum(
        self,
        series: pl.Series,
        window: int
    ) -> pl.Series:
        """Calculate rolling momentum (current - past)."""
        return series - series.shift(window)

    def _rolling_rate_of_change(
        self,
        series: pl.Series,
        window: int
    ) -> pl.Series:
        """Calculate rolling rate of change."""
        past_values = series.shift(window)
        return ((series - past_values) / past_values) * 100

    def _rolling_zscore(
        self,
        series: pl.Series,
        window: int,
        min_periods: int
    ) -> pl.Series:
        """Calculate rolling z-score."""
        rolling_mean = series.rolling_mean(window, min_periods=min_periods)
        rolling_std = series.rolling_std(window, min_periods=min_periods)

        zscore = (series - rolling_mean) / rolling_std

        return zscore

    def _handle_missing_values(self, data: pl.DataFrame) -> pl.DataFrame:
        """Handle missing values based on configuration.

        Args:
            data: DataFrame with potential missing values

        Returns:
            DataFrame with missing values handled
        """
        if self.handle_missing == 'drop':
            return data.drop_nulls()

        elif self.handle_missing == 'forward_fill':
            return data.fill_null(strategy='forward')

        elif self.handle_missing == 'zero':
            return data.fill_null(0)

        else:
            return data

    def compute_technical_indicators(
        self,
        data: pl.DataFrame,
        price_col: str = 'close'
    ) -> pl.DataFrame:
        """Compute common technical indicators.

        Args:
            data: Input data with OHLCV columns
            price_col: Price column to use

        Returns:
            DataFrame with technical indicators added
        """
        try:
            logger.info("computing_technical_indicators")

            result = data.clone()

            # Simple Moving Averages
            for window in [5, 10, 20, 50, 200]:
                result = result.with_columns(
                    result[price_col]
                    .rolling_mean(window)
                    .alias(f'sma_{window}')
                )

            # Exponential Moving Averages
            for window in [12, 26]:
                alpha = 2.0 / (window + 1)
                result = result.with_columns(
                    result[price_col]
                    .ewm_mean(alpha=alpha)
                    .alias(f'ema_{window}')
                )

            # MACD (if we have ema_12 and ema_26)
            if 'ema_12' in result.columns and 'ema_26' in result.columns:
                result = result.with_columns(
                    (result['ema_12'] - result['ema_26']).alias('macd')
                )

                # MACD signal line
                alpha_signal = 2.0 / 10  # 9-period EMA
                result = result.with_columns(
                    result['macd'].ewm_mean(alpha=alpha_signal).alias('macd_signal')
                )

                # MACD histogram
                result = result.with_columns(
                    (result['macd'] - result['macd_signal']).alias('macd_histogram')
                )

            # Bollinger Bands
            sma_20 = result[price_col].rolling_mean(20)
            std_20 = result[price_col].rolling_std(20)

            result = result.with_columns([
                (sma_20 + 2 * std_20).alias('bb_upper'),
                sma_20.alias('bb_middle'),
                (sma_20 - 2 * std_20).alias('bb_lower'),
                ((result[price_col] - sma_20) / std_20).alias('bb_position')
            ])

            # RSI (Relative Strength Index)
            result = self._compute_rsi(result, price_col, 14)

            # ATR (Average True Range)
            if all(col in result.columns for col in ['high', 'low', 'close']):
                result = self._compute_atr(result, 14)

            # Volume indicators (if volume exists)
            if 'volume' in result.columns:
                result = result.with_columns(
                    result['volume'].rolling_mean(20).alias('volume_sma_20')
                )

                result = result.with_columns(
                    (result['volume'] / result['volume_sma_20']).alias('volume_ratio')
                )

            logger.info("technical_indicators_complete", n_indicators=len(result.columns) - len(data.columns))

            return result

        except Exception as e:
            logger.error("failed_to_compute_technical_indicators", error=str(e))
            raise

    def _compute_rsi(
        self,
        data: pl.DataFrame,
        price_col: str,
        window: int = 14
    ) -> pl.DataFrame:
        """Compute Relative Strength Index.

        Args:
            data: Input data
            price_col: Price column
            window: RSI window (default 14)

        Returns:
            DataFrame with RSI added
        """
        # Calculate price changes
        delta = data[price_col].diff()

        # Separate gains and losses
        gains = delta.clip_min(0)
        losses = (-delta).clip_min(0)

        # Calculate average gains and losses
        avg_gains = gains.rolling_mean(window)
        avg_losses = losses.rolling_mean(window)

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return data.with_columns(rsi.alias(f'rsi_{window}'))

    def _compute_atr(
        self,
        data: pl.DataFrame,
        window: int = 14
    ) -> pl.DataFrame:
        """Compute Average True Range.

        Args:
            data: Input data with high, low, close
            window: ATR window

        Returns:
            DataFrame with ATR added
        """
        # True Range components
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift(1)).abs()
        low_close = (data['low'] - data['close'].shift(1)).abs()

        # True Range is max of the three
        tr = pl.max_horizontal(high_low, high_close, low_close)

        # ATR is rolling average of TR
        atr = tr.rolling_mean(window)

        return data.with_columns([
            tr.alias('true_range'),
            atr.alias(f'atr_{window}')
        ])

    def get_feature_names(self) -> List[str]:
        """Get list of computed feature names.

        Returns:
            List of feature names
        """
        return [conf.name for conf in self.feature_configs]

    def get_feature_info(self) -> List[Dict[str, Any]]:
        """Get information about all features.

        Returns:
            List of feature information dictionaries
        """
        return [
            {
                'name': conf.name,
                'function': conf.function,
                'window': conf.window,
                'column': conf.column,
                'min_periods': conf.min_periods
            }
            for conf in self.feature_configs
        ]
