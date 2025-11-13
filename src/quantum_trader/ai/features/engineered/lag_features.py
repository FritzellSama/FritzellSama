"""
Lag Feature Engineering for Time Series Data.

This module creates lag features for time series analysis, including
lagged values, rolling statistics, and temporal patterns.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import polars as pl
from structlog import get_logger

from quantum_trader.exceptions import ValidationError, FeatureError

logger = get_logger(__name__)


@dataclass
class LagConfig:
    """Configuration for lag feature generation.

    Attributes:
        lag_periods: List of lag periods to create
        rolling_windows: List of rolling window sizes
        rolling_functions: List of rolling functions to apply
        columns: Columns to create lags for
        diff_periods: Periods for difference features
        pct_change_periods: Periods for percentage change features
        prefix: Prefix for generated feature names
    """
    lag_periods: List[int]
    rolling_windows: List[int] = None
    rolling_functions: List[str] = None
    columns: List[str] = None
    diff_periods: List[int] = None
    pct_change_periods: List[int] = None
    prefix: str = "lag"

    def __post_init__(self) -> None:
        """Initialize default values."""
        if self.rolling_windows is None:
            self.rolling_windows = []
        if self.rolling_functions is None:
            self.rolling_functions = ["mean", "std", "min", "max"]
        if self.diff_periods is None:
            self.diff_periods = []
        if self.pct_change_periods is None:
            self.pct_change_periods = []


class LagFeatureGenerator:
    """Generate lag features from time series data.

    Creates various types of lag features including simple lags,
    rolling statistics, differences, and percentage changes.

    Attributes:
        config: Feature generation configuration
        generated_features: Names of generated features

    Example:
        >>> config = LagConfig(
        ...     lag_periods=[1, 2, 3, 5, 10],
        ...     rolling_windows=[5, 10, 20],
        ...     rolling_functions=["mean", "std", "max", "min"],
        ...     columns=["price", "volume"],
        ...     diff_periods=[1, 5],
        ...     pct_change_periods=[1, 5]
        ... )
        >>> generator = LagFeatureGenerator(config)
        >>> df_with_lags = generator.generate_features(df)
    """

    def __init__(self, config: LagConfig) -> None:
        """Initialize lag feature generator.

        Args:
            config: Feature generation configuration

        Raises:
            ValidationError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.generated_features: List[str] = []

        logger.info(
            "Lag feature generator initialized",
            lag_periods=config.lag_periods,
            rolling_windows=config.rolling_windows,
            num_columns=len(config.columns) if config.columns else 0
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValidationError: If configuration is invalid
        """
        if not self.config.lag_periods:
            raise ValidationError("lag_periods cannot be empty")

        for period in self.config.lag_periods:
            if period < 1:
                raise ValidationError("lag_periods must be >= 1")

        for window in self.config.rolling_windows:
            if window < 2:
                raise ValidationError("rolling_windows must be >= 2")

        valid_functions = ["mean", "std", "min", "max", "median", "sum", "var"]
        for func in self.config.rolling_functions:
            if func not in valid_functions:
                raise ValidationError(f"Invalid rolling function: {func}")

    def generate_features(
        self,
        data: pl.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """Generate lag features.

        Args:
            data: Input dataframe
            columns: Columns to create lags for (overrides config)

        Returns:
            Dataframe with lag features added

        Raises:
            ValidationError: If data is invalid
            FeatureError: If feature generation fails
        """
        try:
            self._validate_data(data)

            logger.info("Generating lag features", num_rows=len(data))

            # Determine columns to process
            if columns is None:
                columns = self.config.columns

            if columns is None:
                # Use all numeric columns
                columns = [
                    col for col in data.columns
                    if data[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
                ]

            result = data.clone()

            # Generate lag features
            for col in columns:
                if col not in data.columns:
                    logger.warning(f"Column not found: {col}")
                    continue

                # Simple lags
                result = self._generate_simple_lags(result, col)

                # Rolling statistics
                if self.config.rolling_windows:
                    result = self._generate_rolling_features(result, col)

                # Difference features
                if self.config.diff_periods:
                    result = self._generate_diff_features(result, col)

                # Percentage change features
                if self.config.pct_change_periods:
                    result = self._generate_pct_change_features(result, col)

            # Fill null values (created by lagging) with 0
            result = result.fill_null(0)
            result = result.fill_nan(0)

            logger.info(
                "Lag features generated",
                num_features=len(self.generated_features),
                total_columns=len(result.columns)
            )

            return result

        except Exception as e:
            logger.error("Feature generation failed", error=str(e))
            raise FeatureError(f"Feature generation failed: {e}") from e

    def _generate_simple_lags(
        self,
        data: pl.DataFrame,
        column: str
    ) -> pl.DataFrame:
        """Generate simple lag features.

        Args:
            data: Input dataframe
            column: Column to create lags for

        Returns:
            Dataframe with lag features added
        """
        result = data.clone()

        for lag in self.config.lag_periods:
            try:
                feature_name = f"{self.config.prefix}_{column}_lag{lag}"
                result = result.with_columns(
                    pl.col(column).shift(lag).alias(feature_name)
                )
                self.generated_features.append(feature_name)

            except Exception as e:
                logger.warning(
                    f"Failed to generate lag feature",
                    column=column,
                    lag=lag,
                    error=str(e)
                )

        return result

    def _generate_rolling_features(
        self,
        data: pl.DataFrame,
        column: str
    ) -> pl.DataFrame:
        """Generate rolling window features.

        Args:
            data: Input dataframe
            column: Column to create rolling features for

        Returns:
            Dataframe with rolling features added
        """
        result = data.clone()

        for window in self.config.rolling_windows:
            for func in self.config.rolling_functions:
                try:
                    feature_name = f"{self.config.prefix}_{column}_roll{window}_{func}"

                    if func == "mean":
                        result = result.with_columns(
                            pl.col(column).rolling_mean(window).alias(feature_name)
                        )
                    elif func == "std":
                        result = result.with_columns(
                            pl.col(column).rolling_std(window).alias(feature_name)
                        )
                    elif func == "min":
                        result = result.with_columns(
                            pl.col(column).rolling_min(window).alias(feature_name)
                        )
                    elif func == "max":
                        result = result.with_columns(
                            pl.col(column).rolling_max(window).alias(feature_name)
                        )
                    elif func == "median":
                        result = result.with_columns(
                            pl.col(column).rolling_median(window).alias(feature_name)
                        )
                    elif func == "sum":
                        result = result.with_columns(
                            pl.col(column).rolling_sum(window).alias(feature_name)
                        )
                    elif func == "var":
                        result = result.with_columns(
                            pl.col(column).rolling_var(window).alias(feature_name)
                        )

                    self.generated_features.append(feature_name)

                except Exception as e:
                    logger.warning(
                        f"Failed to generate rolling feature",
                        column=column,
                        window=window,
                        func=func,
                        error=str(e)
                    )

        return result

    def _generate_diff_features(
        self,
        data: pl.DataFrame,
        column: str
    ) -> pl.DataFrame:
        """Generate difference features.

        Args:
            data: Input dataframe
            column: Column to create difference features for

        Returns:
            Dataframe with difference features added
        """
        result = data.clone()

        for period in self.config.diff_periods:
            try:
                feature_name = f"{self.config.prefix}_{column}_diff{period}"
                result = result.with_columns(
                    pl.col(column).diff(period).alias(feature_name)
                )
                self.generated_features.append(feature_name)

            except Exception as e:
                logger.warning(
                    f"Failed to generate diff feature",
                    column=column,
                    period=period,
                    error=str(e)
                )

        return result

    def _generate_pct_change_features(
        self,
        data: pl.DataFrame,
        column: str
    ) -> pl.DataFrame:
        """Generate percentage change features.

        Args:
            data: Input dataframe
            column: Column to create percentage change features for

        Returns:
            Dataframe with percentage change features added
        """
        result = data.clone()

        for period in self.config.pct_change_periods:
            try:
                feature_name = f"{self.config.prefix}_{column}_pct{period}"

                # Calculate percentage change: (current - previous) / previous
                lagged = pl.col(column).shift(period)
                result = result.with_columns(
                    ((pl.col(column) - lagged) / (lagged + 1e-10)).alias(feature_name)
                )

                self.generated_features.append(feature_name)

            except Exception as e:
                logger.warning(
                    f"Failed to generate pct change feature",
                    column=column,
                    period=period,
                    error=str(e)
                )

        return result

    def _validate_data(self, data: pl.DataFrame) -> None:
        """Validate input data.

        Args:
            data: Input dataframe

        Raises:
            ValidationError: If data is invalid
        """
        if data is None or len(data) == 0:
            raise ValidationError("Data cannot be empty")

        if len(data.columns) == 0:
            raise ValidationError("Data must have at least one column")

        # Check if we have enough rows for the largest lag
        max_lag = max(self.config.lag_periods)
        max_window = max(self.config.rolling_windows) if self.config.rolling_windows else 0
        max_period = max(max_lag, max_window)

        if len(data) < max_period + 1:
            raise ValidationError(
                f"Data must have at least {max_period + 1} rows for specified lags"
            )

    async def generate_features_async(
        self,
        data: pl.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """Generate features asynchronously.

        Args:
            data: Input dataframe
            columns: Columns to create lags for

        Returns:
            Dataframe with lag features added
        """
        return await asyncio.to_thread(self.generate_features, data, columns)

    def get_feature_names(self) -> List[str]:
        """Get names of generated features.

        Returns:
            List of feature names
        """
        return self.generated_features.copy()

    def get_feature_stats(
        self,
        data: pl.DataFrame
    ) -> Dict[str, Dict[str, Decimal]]:
        """Calculate statistics for generated features.

        Args:
            data: Dataframe with features

        Returns:
            Dictionary of feature statistics

        Raises:
            ValidationError: If data is invalid
        """
        try:
            stats = {}

            for feature in self.generated_features:
                if feature not in data.columns:
                    continue

                feature_data = data[feature]

                stats[feature] = {
                    "mean": Decimal(str(feature_data.mean())),
                    "std": Decimal(str(feature_data.std())),
                    "min": Decimal(str(feature_data.min())),
                    "max": Decimal(str(feature_data.max())),
                    "null_count": Decimal(str(feature_data.null_count()))
                }

            logger.info(
                "Feature statistics calculated",
                num_features=len(stats)
            )

            return stats

        except Exception as e:
            logger.error("Feature statistics calculation failed", error=str(e))
            raise FeatureError(f"Feature statistics calculation failed: {e}") from e

    def remove_null_rows(
        self,
        data: pl.DataFrame
    ) -> pl.DataFrame:
        """Remove rows with null values in lag features.

        Args:
            data: Input dataframe

        Returns:
            Dataframe with null rows removed

        Raises:
            ValidationError: If data is invalid
        """
        try:
            # Get maximum lag period
            max_lag = max(self.config.lag_periods)
            max_window = max(self.config.rolling_windows) if self.config.rolling_windows else 0
            max_period = max(max_lag, max_window)

            # Remove first N rows where lags would be null
            result = data.slice(max_period, len(data) - max_period)

            logger.info(
                "Null rows removed",
                original_rows=len(data),
                final_rows=len(result),
                removed_rows=max_period
            )

            return result

        except Exception as e:
            logger.error("Null row removal failed", error=str(e))
            raise FeatureError(f"Null row removal failed: {e}") from e

    def create_sequences(
        self,
        data: pl.DataFrame,
        sequence_length: int,
        target_col: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sequences for time series modeling.

        Args:
            data: Input dataframe
            sequence_length: Length of each sequence
            target_col: Target column name (optional)

        Returns:
            Tuple of (sequences, targets)

        Raises:
            ValidationError: If parameters are invalid
        """
        try:
            if sequence_length < 1:
                raise ValidationError("sequence_length must be >= 1")

            if len(data) < sequence_length:
                raise ValidationError("Data too short for sequence length")

            # Get feature columns (exclude target if specified)
            feature_cols = [
                col for col in data.columns
                if col != target_col
            ]

            # Convert to numpy
            feature_data = data.select(feature_cols).to_numpy()

            # Create sequences
            sequences = []
            for i in range(len(feature_data) - sequence_length + 1):
                sequences.append(feature_data[i:i + sequence_length])

            sequences = np.array(sequences)

            # Create targets if specified
            targets = None
            if target_col is not None:
                if target_col not in data.columns:
                    raise ValidationError(f"Target column not found: {target_col}")

                target_data = data[target_col].to_numpy()
                targets = target_data[sequence_length - 1:]

            logger.info(
                "Sequences created",
                num_sequences=len(sequences),
                sequence_length=sequence_length,
                num_features=len(feature_cols)
            )

            return sequences, targets

        except Exception as e:
            logger.error("Sequence creation failed", error=str(e))
            raise FeatureError(f"Sequence creation failed: {e}") from e

    def reset(self) -> None:
        """Reset generated features list."""
        self.generated_features.clear()
        logger.info("Feature generator reset")
