"""Fear & Greed Index Feature Extraction for Trading.

This module implements production-ready feature extraction from market sentiment
indicators, including the Fear & Greed Index and related metrics.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class FearGreedFeatureExtractor:
    """Extract trading features from Fear & Greed Index and sentiment data.

    This extractor calculates features based on:
    - Fear & Greed Index values and momentum
    - Market volatility indicators
    - Volume and momentum metrics
    - Social sentiment signals
    - Put/Call ratios

    Attributes:
        config: Configuration dictionary
        lookback_periods: List of lookback windows for features
        smoothing_windows: Windows for moving averages
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Fear & Greed feature extractor.

        Args:
            config: Configuration with keys:
                - lookback_periods: List of lookback windows (e.g., [7, 14, 30])
                - smoothing_windows: List of smoothing windows (e.g., [3, 7])
                - volatility_window: Window for volatility calculation
                - momentum_periods: Periods for momentum features
                - normalization_method: 'zscore' or 'minmax'

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self._validate_config()

        self.lookback_periods: List[int] = config.get("lookback_periods", [7, 14, 30])
        self.smoothing_windows: List[int] = config.get("smoothing_windows", [3, 7])
        self.volatility_window: int = config.get("volatility_window", 20)
        self.momentum_periods: List[int] = config.get("momentum_periods", [1, 3, 7])
        self.normalization_method: str = config.get("normalization_method", "zscore")

        # Feature cache
        self._feature_cache: Dict[str, Any] = {}

        logger.info(
            "fear_greed_extractor_initialized",
            lookback_periods=self.lookback_periods,
            smoothing_windows=self.smoothing_windows
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required parameters missing or invalid
        """
        required_keys = ["lookback_periods", "volatility_window"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"{key} required in config")

        lookback = self.config["lookback_periods"]
        if not isinstance(lookback, list) or not all(isinstance(x, int) for x in lookback):
            raise ValueError("lookback_periods must be list of integers")

        if not all(x > 0 for x in lookback):
            raise ValueError("All lookback_periods must be positive")

    def extract_features(self, data: pl.DataFrame) -> pl.DataFrame:
        """Extract Fear & Greed features from market data.

        Args:
            data: Polars DataFrame with columns:
                - timestamp: UTC datetime
                - fear_greed_index: Fear & Greed value (0-100)
                - volatility_index: VIX or similar
                - volume: Trading volume
                - price: Asset price
                - put_call_ratio: Put/Call ratio (optional)

        Returns:
            DataFrame with original data plus extracted features

        Raises:
            ValueError: If data is invalid or missing required columns
        """
        try:
            self._validate_input_data(data)

            logger.debug("extracting_fear_greed_features", rows=len(data))

            # Create working copy
            result = data.clone()

            # Core Fear & Greed features
            result = self._add_fear_greed_features(result)

            # Momentum features
            result = self._add_momentum_features(result)

            # Volatility features
            result = self._add_volatility_features(result)

            # Extremes and regime features
            result = self._add_regime_features(result)

            # Statistical features
            result = self._add_statistical_features(result)

            # Normalize features if configured
            if self.normalization_method:
                result = self._normalize_features(result)

            logger.debug(
                "fear_greed_features_extracted",
                feature_count=len(result.columns) - len(data.columns)
            )

            return result

        except Exception as e:
            logger.error("feature_extraction_failed", error=str(e))
            raise

    def _validate_input_data(self, data: pl.DataFrame) -> None:
        """Validate input DataFrame.

        Args:
            data: Input DataFrame to validate

        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(data, pl.DataFrame):
            raise ValueError(f"data must be Polars DataFrame, got {type(data)}")

        required_columns = ["timestamp", "fear_greed_index", "volatility_index", "volume", "price"]
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if len(data) == 0:
            raise ValueError("DataFrame is empty")

        # Check for minimum data points
        min_required = max(self.lookback_periods) + max(self.smoothing_windows)
        if len(data) < min_required:
            raise ValueError(
                f"Insufficient data points: {len(data)} < {min_required}"
            )

    def _add_fear_greed_features(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add core Fear & Greed Index features.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with Fear & Greed features added
        """
        result = data

        # Current level categorization
        result = result.with_columns([
            # Normalize to 0-1 range
            (pl.col("fear_greed_index") / Decimal("100")).alias("fg_normalized"),

            # Binary regimes
            (pl.col("fear_greed_index") < Decimal("25")).alias("fg_extreme_fear"),
            (pl.col("fear_greed_index") > Decimal("75")).alias("fg_extreme_greed"),
            (
                (pl.col("fear_greed_index") >= Decimal("45")) &
                (pl.col("fear_greed_index") <= Decimal("55"))
            ).alias("fg_neutral"),
        ])

        # Distance from extremes
        result = result.with_columns([
            (Decimal("50") - pl.col("fear_greed_index")).abs().alias("fg_distance_from_neutral"),
            pl.col("fear_greed_index").alias("fg_distance_from_fear_extreme"),
            (Decimal("100") - pl.col("fear_greed_index")).alias("fg_distance_from_greed_extreme"),
        ])

        # Moving averages
        for window in self.smoothing_windows:
            result = result.with_columns([
                pl.col("fear_greed_index")
                .rolling_mean(window_size=window)
                .alias(f"fg_sma_{window}"),
            ])

            # Deviation from MA
            result = result.with_columns([
                (pl.col("fear_greed_index") - pl.col(f"fg_sma_{window}"))
                .alias(f"fg_deviation_{window}"),
            ])

        return result

    def _add_momentum_features(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add momentum-based features.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with momentum features added
        """
        result = data

        # Rate of change for different periods
        for period in self.momentum_periods:
            result = result.with_columns([
                # Fear & Greed momentum
                (
                    pl.col("fear_greed_index") -
                    pl.col("fear_greed_index").shift(period)
                ).alias(f"fg_roc_{period}d"),

                # Volatility momentum
                (
                    pl.col("volatility_index") -
                    pl.col("volatility_index").shift(period)
                ).alias(f"vix_roc_{period}d"),

                # Volume momentum
                (
                    (pl.col("volume") - pl.col("volume").shift(period)) /
                    pl.col("volume").shift(period)
                ).alias(f"volume_roc_{period}d"),
            ])

        # Momentum direction indicators
        result = result.with_columns([
            (pl.col("fg_roc_1d") > Decimal("0")).cast(pl.Int8).alias("fg_momentum_positive"),
            (pl.col("fg_roc_7d") > Decimal("0")).cast(pl.Int8).alias("fg_trend_positive"),
        ])

        # Acceleration (second derivative)
        result = result.with_columns([
            (pl.col("fg_roc_1d") - pl.col("fg_roc_1d").shift(1)).alias("fg_acceleration"),
        ])

        return result

    def _add_volatility_features(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add volatility-based features.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with volatility features added
        """
        result = data

        # Rolling volatility of Fear & Greed Index
        result = result.with_columns([
            pl.col("fear_greed_index")
            .rolling_std(window_size=self.volatility_window)
            .alias("fg_volatility"),
        ])

        # VIX features
        result = result.with_columns([
            # VIX normalized
            (pl.col("volatility_index") / pl.col("volatility_index").rolling_mean(window_size=30))
            .alias("vix_normalized"),

            # VIX percentile
            pl.col("volatility_index")
            .rank(method="average")
            .alias("vix_percentile") / len(data),
        ])

        # Relationship between Fear & Greed and VIX
        result = result.with_columns([
            (pl.col("fear_greed_index") * pl.col("volatility_index"))
            .alias("fg_vix_interaction"),

            (pl.col("fear_greed_index") - pl.col("vix_normalized") * Decimal("50"))
            .alias("fg_vix_divergence"),
        ])

        return result

    def _add_regime_features(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add market regime features.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with regime features added
        """
        result = data

        # Time spent in extreme zones
        for window in self.lookback_periods:
            result = result.with_columns([
                # Days in fear zone
                pl.col("fg_extreme_fear")
                .cast(pl.Int8)
                .rolling_sum(window_size=window)
                .alias(f"fg_fear_days_{window}d"),

                # Days in greed zone
                pl.col("fg_extreme_greed")
                .cast(pl.Int8)
                .rolling_sum(window_size=window)
                .alias(f"fg_greed_days_{window}d"),
            ])

        # Regime transitions
        result = result.with_columns([
            # Fear to greed transition
            (
                (pl.col("fg_extreme_fear").shift(1) == True) &
                (pl.col("fg_extreme_greed") == True)
            ).cast(pl.Int8).alias("fg_fear_to_greed_transition"),

            # Greed to fear transition
            (
                (pl.col("fg_extreme_greed").shift(1) == True) &
                (pl.col("fg_extreme_fear") == True)
            ).cast(pl.Int8).alias("fg_greed_to_fear_transition"),
        ])

        return result

    def _add_statistical_features(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add statistical features.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with statistical features added
        """
        result = data

        for window in self.lookback_periods:
            result = result.with_columns([
                # Z-score
                (
                    (pl.col("fear_greed_index") - pl.col("fear_greed_index").rolling_mean(window_size=window)) /
                    pl.col("fear_greed_index").rolling_std(window_size=window)
                ).alias(f"fg_zscore_{window}d"),

                # Percentile rank
                pl.col("fear_greed_index")
                .rolling_quantile(quantile=0.5, window_size=window)
                .alias(f"fg_median_{window}d"),

                # Range position
                (
                    (pl.col("fear_greed_index") - pl.col("fear_greed_index").rolling_min(window_size=window)) /
                    (
                        pl.col("fear_greed_index").rolling_max(window_size=window) -
                        pl.col("fear_greed_index").rolling_min(window_size=window)
                    )
                ).alias(f"fg_range_position_{window}d"),
            ])

        return result

    def _normalize_features(self, data: pl.DataFrame) -> pl.DataFrame:
        """Normalize extracted features.

        Args:
            data: DataFrame with features

        Returns:
            DataFrame with normalized features
        """
        # Identify feature columns (exclude original data columns)
        original_cols = ["timestamp", "fear_greed_index", "volatility_index", "volume", "price"]
        feature_cols = [col for col in data.columns if col not in original_cols]

        result = data

        if self.normalization_method == "zscore":
            for col in feature_cols:
                if data[col].dtype in [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
                    mean_val = data[col].mean()
                    std_val = data[col].std()
                    if std_val and std_val > 0:
                        result = result.with_columns([
                            ((pl.col(col) - mean_val) / std_val).alias(col)
                        ])

        elif self.normalization_method == "minmax":
            for col in feature_cols:
                if data[col].dtype in [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
                    min_val = data[col].min()
                    max_val = data[col].max()
                    if max_val and min_val is not None and max_val > min_val:
                        result = result.with_columns([
                            ((pl.col(col) - min_val) / (max_val - min_val)).alias(col)
                        ])

        return result

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names that will be extracted.

        Returns:
            List of feature column names
        """
        features = [
            "fg_normalized",
            "fg_extreme_fear",
            "fg_extreme_greed",
            "fg_neutral",
            "fg_distance_from_neutral",
            "fg_distance_from_fear_extreme",
            "fg_distance_from_greed_extreme",
            "fg_volatility",
            "vix_normalized",
            "vix_percentile",
            "fg_vix_interaction",
            "fg_vix_divergence",
            "fg_momentum_positive",
            "fg_trend_positive",
            "fg_acceleration",
            "fg_fear_to_greed_transition",
            "fg_greed_to_fear_transition",
        ]

        # Add windowed features
        for window in self.smoothing_windows:
            features.extend([
                f"fg_sma_{window}",
                f"fg_deviation_{window}",
            ])

        for period in self.momentum_periods:
            features.extend([
                f"fg_roc_{period}d",
                f"vix_roc_{period}d",
                f"volume_roc_{period}d",
            ])

        for window in self.lookback_periods:
            features.extend([
                f"fg_fear_days_{window}d",
                f"fg_greed_days_{window}d",
                f"fg_zscore_{window}d",
                f"fg_median_{window}d",
                f"fg_range_position_{window}d",
            ])

        return features

    def get_feature_importance_hints(self) -> Dict[str, str]:
        """Get hints about feature importance and interpretation.

        Returns:
            Dictionary mapping feature names to interpretation hints
        """
        return {
            "fg_normalized": "Normalized Fear & Greed value (0-1)",
            "fg_extreme_fear": "Market in extreme fear zone (<25)",
            "fg_extreme_greed": "Market in extreme greed zone (>75)",
            "fg_momentum_positive": "Fear & Greed momentum is positive",
            "fg_volatility": "Volatility of Fear & Greed Index",
            "fg_vix_interaction": "Interaction between Fear & Greed and VIX",
            "fg_fear_to_greed_transition": "Transition from fear to greed",
            "fg_greed_to_fear_transition": "Transition from greed to fear",
        }
