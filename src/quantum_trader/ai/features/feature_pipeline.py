"""Feature Processing Pipeline for ML Training and Inference.

This module implements a production-ready feature processing pipeline that handles
feature extraction, transformation, validation, and caching for trading ML models.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime
from pathlib import Path
import pickle
import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class FeatureTransformer:
    """Feature transformation component for normalization and encoding.

    Supports:
    - Standardization (z-score)
    - Min-Max normalization
    - Robust scaling
    - Log transformation
    - Polynomial features

    Attributes:
        config: Configuration dictionary
        method: Transformation method
        fitted: Whether transformer has been fitted
        stats: Fitted transformation statistics
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize feature transformer.

        Args:
            config: Configuration with keys:
                - method: 'standard', 'minmax', 'robust', or 'log'
                - columns: List of columns to transform (None for all)
                - clip_outliers: Whether to clip outliers
                - outlier_std: Number of std for outlier clipping

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self._validate_config()

        self.method: str = config.get("method", "standard")
        self.columns: Optional[List[str]] = config.get("columns", None)
        self.clip_outliers: bool = config.get("clip_outliers", False)
        self.outlier_std: Decimal = Decimal(str(config.get("outlier_std", 3.0)))

        # Fitted statistics
        self.fitted: bool = False
        self.stats: Dict[str, Dict[str, Decimal]] = {}

        logger.debug("feature_transformer_initialized", method=self.method)

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If config invalid
        """
        valid_methods = {"standard", "minmax", "robust", "log"}
        method = self.config.get("method", "standard")

        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {method}")

    def fit(self, data: pl.DataFrame) -> None:
        """Fit transformer to data.

        Args:
            data: Polars DataFrame with features

        Raises:
            ValueError: If data is invalid
        """
        try:
            if not isinstance(data, pl.DataFrame):
                raise ValueError(f"data must be Polars DataFrame, got {type(data)}")

            # Determine columns to transform
            cols_to_transform = self.columns or self._get_numeric_columns(data)

            logger.debug("fitting_transformer", n_columns=len(cols_to_transform))

            # Calculate statistics for each column
            for col in cols_to_transform:
                if col not in data.columns:
                    logger.warning("column_not_found", column=col)
                    continue

                col_data = data[col]

                if self.method == "standard":
                    mean = col_data.mean()
                    std = col_data.std()
                    self.stats[col] = {
                        "mean": Decimal(str(mean)) if mean is not None else Decimal("0"),
                        "std": Decimal(str(std)) if std is not None and std > 0 else Decimal("1")
                    }

                elif self.method == "minmax":
                    min_val = col_data.min()
                    max_val = col_data.max()
                    self.stats[col] = {
                        "min": Decimal(str(min_val)) if min_val is not None else Decimal("0"),
                        "max": Decimal(str(max_val)) if max_val is not None else Decimal("1")
                    }

                elif self.method == "robust":
                    median = col_data.median()
                    q75 = col_data.quantile(0.75)
                    q25 = col_data.quantile(0.25)
                    iqr = q75 - q25 if q75 and q25 else 1.0
                    self.stats[col] = {
                        "median": Decimal(str(median)) if median is not None else Decimal("0"),
                        "iqr": Decimal(str(iqr)) if iqr > 0 else Decimal("1")
                    }

            self.fitted = True
            logger.info("transformer_fitted", n_columns=len(self.stats))

        except Exception as e:
            logger.error("transformer_fit_failed", error=str(e))
            raise

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """Transform data using fitted statistics.

        Args:
            data: Polars DataFrame with features

        Returns:
            Transformed DataFrame

        Raises:
            ValueError: If transformer not fitted or data invalid
        """
        try:
            if not self.fitted:
                raise ValueError("Transformer must be fitted before transform")

            result = data.clone()

            for col, stats in self.stats.items():
                if col not in result.columns:
                    logger.warning("column_not_in_data", column=col)
                    continue

                if self.method == "standard":
                    mean = stats["mean"]
                    std = stats["std"]
                    result = result.with_columns([
                        ((pl.col(col) - float(mean)) / float(std)).alias(col)
                    ])

                elif self.method == "minmax":
                    min_val = stats["min"]
                    max_val = stats["max"]
                    range_val = max_val - min_val
                    if range_val > 0:
                        result = result.with_columns([
                            ((pl.col(col) - float(min_val)) / float(range_val)).alias(col)
                        ])

                elif self.method == "robust":
                    median = stats["median"]
                    iqr = stats["iqr"]
                    result = result.with_columns([
                        ((pl.col(col) - float(median)) / float(iqr)).alias(col)
                    ])

                elif self.method == "log":
                    # Log transform (adding 1 to handle zeros)
                    result = result.with_columns([
                        (pl.col(col) + 1).log().alias(col)
                    ])

                # Clip outliers if configured
                if self.clip_outliers and self.method == "standard":
                    clip_val = float(self.outlier_std)
                    result = result.with_columns([
                        pl.col(col).clip(-clip_val, clip_val).alias(col)
                    ])

            logger.debug("data_transformed", n_columns=len(self.stats))

            return result

        except Exception as e:
            logger.error("transform_failed", error=str(e))
            raise

    def fit_transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """Fit transformer and transform data in one step.

        Args:
            data: Polars DataFrame with features

        Returns:
            Transformed DataFrame
        """
        self.fit(data)
        return self.transform(data)

    def _get_numeric_columns(self, data: pl.DataFrame) -> List[str]:
        """Get list of numeric columns from DataFrame.

        Args:
            data: Polars DataFrame

        Returns:
            List of numeric column names
        """
        numeric_types = [
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64
        ]

        return [
            col for col in data.columns
            if data[col].dtype in numeric_types
        ]


class FeaturePipeline:
    """Production-ready feature processing pipeline.

    Orchestrates the complete feature processing workflow:
    1. Feature extraction
    2. Feature validation
    3. Feature transformation
    4. Feature selection
    5. Caching and persistence

    Attributes:
        config: Configuration dictionary
        extractors: List of feature extractors
        transformer: Feature transformer
        selector: Feature selector
        cache_enabled: Whether to cache features
        cache_dir: Directory for feature cache
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize feature pipeline.

        Args:
            config: Configuration with keys:
                - extractors: List of extractor configs
                - transformer: Transformer config
                - selector: Selector config (optional)
                - validation: Validation rules
                - cache_enabled: Enable feature caching
                - cache_dir: Cache directory path
                - handle_missing: How to handle missing values

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self._validate_config()

        # Initialize components
        self.extractors: List[Any] = []  # Will be populated with extractors
        self.transformer: Optional[FeatureTransformer] = None
        self.selector: Optional[Any] = None

        if "transformer" in config:
            self.transformer = FeatureTransformer(config["transformer"])

        # Validation rules
        self.validation_rules: Dict[str, Any] = config.get("validation", {})

        # Caching
        self.cache_enabled: bool = config.get("cache_enabled", False)
        self.cache_dir: Optional[Path] = None
        if self.cache_enabled:
            cache_path = config.get("cache_dir", "/tmp/feature_cache")
            self.cache_dir = Path(cache_path)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Missing value handling
        self.handle_missing: str = config.get("handle_missing", "drop")

        # Pipeline statistics
        self.processing_count: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0

        logger.info(
            "feature_pipeline_initialized",
            transformer=self.transformer is not None,
            cache_enabled=self.cache_enabled
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If config invalid
        """
        if "handle_missing" in self.config:
            valid_methods = {"drop", "fill_zero", "fill_mean", "fill_forward"}
            method = self.config["handle_missing"]
            if method not in valid_methods:
                raise ValueError(f"handle_missing must be one of {valid_methods}")

    def process(
        self,
        data: pl.DataFrame,
        training: bool = False,
        cache_key: Optional[str] = None
    ) -> pl.DataFrame:
        """Process features through the complete pipeline.

        Args:
            data: Input Polars DataFrame
            training: Whether this is training data (fits transformers)
            cache_key: Optional key for caching features

        Returns:
            Processed DataFrame with features

        Raises:
            ValueError: If data is invalid
        """
        try:
            logger.debug(
                "processing_features",
                rows=len(data),
                training=training,
                cached=cache_key is not None
            )

            # Check cache first
            if cache_key and self.cache_enabled and not training:
                cached = self._load_from_cache(cache_key)
                if cached is not None:
                    self.cache_hits += 1
                    logger.debug("features_loaded_from_cache", key=cache_key)
                    return cached
                self.cache_misses += 1

            # Step 1: Extract features (if extractors configured)
            result = data
            for extractor in self.extractors:
                result = extractor.extract(result)

            # Step 2: Validate features
            result = self._validate_features(result)

            # Step 3: Handle missing values
            result = self._handle_missing_values(result)

            # Step 4: Transform features
            if self.transformer:
                if training:
                    result = self.transformer.fit_transform(result)
                else:
                    result = self.transformer.transform(result)

            # Step 5: Select features (if selector configured)
            if self.selector:
                if training:
                    self.selector.fit(result)
                result = self.selector.transform(result)

            # Cache results
            if cache_key and self.cache_enabled:
                self._save_to_cache(cache_key, result)

            self.processing_count += 1

            logger.debug(
                "features_processed",
                input_cols=len(data.columns),
                output_cols=len(result.columns),
                rows=len(result)
            )

            return result

        except Exception as e:
            logger.error("feature_processing_failed", error=str(e))
            raise

    def _validate_features(self, data: pl.DataFrame) -> pl.DataFrame:
        """Validate features against configured rules.

        Args:
            data: DataFrame to validate

        Returns:
            Validated DataFrame

        Raises:
            ValueError: If validation fails
        """
        if not self.validation_rules:
            return data

        # Check for required columns
        if "required_columns" in self.validation_rules:
            required = self.validation_rules["required_columns"]
            missing = [col for col in required if col not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

        # Check value ranges
        if "value_ranges" in self.validation_rules:
            for col, (min_val, max_val) in self.validation_rules["value_ranges"].items():
                if col in data.columns:
                    col_min = data[col].min()
                    col_max = data[col].max()

                    if col_min is not None and col_min < min_val:
                        logger.warning(
                            "value_below_range",
                            column=col,
                            value=col_min,
                            min=min_val
                        )

                    if col_max is not None and col_max > max_val:
                        logger.warning(
                            "value_above_range",
                            column=col,
                            value=col_max,
                            max=max_val
                        )

        # Check for infinite values
        for col in data.columns:
            if data[col].dtype in [pl.Float32, pl.Float64]:
                if data[col].is_infinite().any():
                    logger.warning("infinite_values_detected", column=col)

        return data

    def _handle_missing_values(self, data: pl.DataFrame) -> pl.DataFrame:
        """Handle missing values according to configured strategy.

        Args:
            data: DataFrame with potential missing values

        Returns:
            DataFrame with missing values handled

        Raises:
            ValueError: If invalid strategy
        """
        if self.handle_missing == "drop":
            # Drop rows with any null values
            return data.drop_nulls()

        elif self.handle_missing == "fill_zero":
            # Fill nulls with zero
            return data.fill_null(0)

        elif self.handle_missing == "fill_mean":
            # Fill nulls with column mean
            result = data
            for col in data.columns:
                if data[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    mean_val = data[col].mean()
                    if mean_val is not None:
                        result = result.with_columns([
                            pl.col(col).fill_null(mean_val).alias(col)
                        ])
            return result

        elif self.handle_missing == "fill_forward":
            # Forward fill
            return data.fill_null(strategy="forward")

        return data

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a key.

        Args:
            cache_key: Cache key

        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{cache_key}.pkl"

    def _load_from_cache(self, cache_key: str) -> Optional[pl.DataFrame]:
        """Load features from cache.

        Args:
            cache_key: Cache key

        Returns:
            Cached DataFrame or None if not found
        """
        try:
            cache_path = self._get_cache_path(cache_key)

            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)

                # Validate cached data
                if isinstance(cached_data, pl.DataFrame):
                    return cached_data

                logger.warning("invalid_cache_data", key=cache_key)

        except Exception as e:
            logger.warning("cache_load_failed", key=cache_key, error=str(e))

        return None

    def _save_to_cache(self, cache_key: str, data: pl.DataFrame) -> None:
        """Save features to cache.

        Args:
            cache_key: Cache key
            data: DataFrame to cache
        """
        try:
            cache_path = self._get_cache_path(cache_key)

            with open(cache_path, "wb") as f:
                pickle.dump(data, f)

            logger.debug("features_cached", key=cache_key, path=str(cache_path))

        except Exception as e:
            logger.warning("cache_save_failed", key=cache_key, error=str(e))

    def add_extractor(self, extractor: Any) -> None:
        """Add a feature extractor to the pipeline.

        Args:
            extractor: Feature extractor instance
        """
        self.extractors.append(extractor)
        logger.debug("extractor_added", total=len(self.extractors))

    def set_selector(self, selector: Any) -> None:
        """Set feature selector for the pipeline.

        Args:
            selector: Feature selector instance
        """
        self.selector = selector
        logger.debug("selector_set")

    def clear_cache(self) -> None:
        """Clear all cached features."""
        if not self.cache_enabled or not self.cache_dir:
            return

        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()

            logger.info("cache_cleared", dir=str(self.cache_dir))

        except Exception as e:
            logger.error("cache_clear_failed", error=str(e))

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline processing statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "processing_count": self.processing_count,
            "cache_enabled": self.cache_enabled,
            "n_extractors": len(self.extractors),
            "has_transformer": self.transformer is not None,
            "has_selector": self.selector is not None,
        }

        if self.cache_enabled:
            stats.update({
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": (
                    self.cache_hits / (self.cache_hits + self.cache_misses)
                    if (self.cache_hits + self.cache_misses) > 0
                    else 0.0
                ),
            })

        return stats
