"""Lag feature engineering for time series.

This module creates lagged features, rolling statistics, and temporal
aggregations for time series forecasting.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any
import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class LagFeatureEngineer:
    """Engineer lag-based features for time series.

    Creates lagged values, rolling windows, and temporal statistics
    to capture historical patterns and trends.

    Attributes:
        config: Configuration dictionary
        lag_periods: List of lag periods to create
        rolling_windows: List of rolling window sizes

    Example:
        >>> config = {
        ...     "lag_periods": [1, 2, 3, 5, 10],
        ...     "rolling_windows": [5, 10, 20],
        ...     "statistics": ["mean", "std", "min", "max"]
        ... }
        >>> engineer = LagFeatureEngineer(config)
        >>> features = await engineer.create_lag_features(data, ["price", "volume"])
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize lag feature engineer.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self._validate_config()

        self.lag_periods = config.get("lag_periods", [1, 2, 3, 5, 10, 20])
        self.rolling_windows = config.get("rolling_windows", [5, 10, 20, 50])
        self.statistics = config.get("statistics", ["mean", "std", "min", "max"])
        self.diff_periods = config.get("diff_periods", [1, 5, 10])

        logger.info(
            "Lag feature engineer initialized",
            lag_periods=self.lag_periods,
            rolling_windows=self.rolling_windows,
            statistics=self.statistics
        )

    def _validate_config(self) -> None:
        """Validate configuration."""
        valid_stats = ["mean", "std", "min", "max", "sum", "median", "quantile"]

        stats = self.config.get("statistics", [])
        invalid = set(stats) - set(valid_stats)

        if invalid:
            raise ValueError(f"Invalid statistics: {invalid}")

    async def create_lag_features(
        self,
        data: pl.DataFrame,
        feature_cols: List[str],
        group_col: Optional[str] = None
    ) -> pl.DataFrame:
        """Create lag-based features.

        Args:
            data: Input DataFrame
            feature_cols: Columns to create lags for
            group_col: Optional grouping column (e.g., symbol)

        Returns:
            DataFrame with lag features

        Raises:
            ValueError: If feature columns not found
        """
        try:
            missing = set(feature_cols) - set(data.columns)
            if missing:
                raise ValueError(f"Missing feature columns: {missing}")

            logger.info(
                "Creating lag features",
                num_features=len(feature_cols),
                num_lags=len(self.lag_periods),
                num_windows=len(self.rolling_windows)
            )

            result = data.clone()

            # Create simple lags
            result = await self._create_simple_lags(result, feature_cols, group_col)

            # Create rolling statistics
            result = await self._create_rolling_features(result, feature_cols, group_col)

            # Create differencing features
            result = await self._create_diff_features(result, feature_cols, group_col)

            # Create expanding features
            result = await self._create_expanding_features(result, feature_cols, group_col)

            logger.info(
                "Lag features created",
                original_features=len(feature_cols),
                total_features=len(result.columns)
            )

            return result

        except Exception as e:
            logger.error("Failed to create lag features", error=str(e))
            raise

    async def _create_simple_lags(
        self,
        data: pl.DataFrame,
        feature_cols: List[str],
        group_col: Optional[str]
    ) -> pl.DataFrame:
        """Create simple lag features.

        Args:
            data: Input DataFrame
            feature_cols: Feature columns
            group_col: Optional grouping column

        Returns:
            DataFrame with lag features
        """
        try:
            result = data.clone()

            for col in feature_cols:
                for lag in self.lag_periods:
                    if group_col:
                        result = result.with_columns(
                            pl.col(col).shift(lag).over(group_col)
                            .alias(f"{col}_lag{lag}")
                        )
                    else:
                        result = result.with_columns(
                            pl.col(col).shift(lag)
                            .alias(f"{col}_lag{lag}")
                        )

            return result

        except Exception as e:
            logger.error("Failed to create simple lags", error=str(e))
            raise

    async def _create_rolling_features(
        self,
        data: pl.DataFrame,
        feature_cols: List[str],
        group_col: Optional[str]
    ) -> pl.DataFrame:
        """Create rolling window statistics.

        Args:
            data: Input DataFrame
            feature_cols: Feature columns
            group_col: Optional grouping column

        Returns:
            DataFrame with rolling features
        """
        try:
            result = data.clone()

            for col in feature_cols:
                for window in self.rolling_windows:
                    # Mean
                    if "mean" in self.statistics:
                        if group_col:
                            result = result.with_columns(
                                pl.col(col).rolling_mean(window).over(group_col)
                                .alias(f"{col}_rolling_mean_{window}")
                            )
                        else:
                            result = result.with_columns(
                                pl.col(col).rolling_mean(window)
                                .alias(f"{col}_rolling_mean_{window}")
                            )

                    # Std
                    if "std" in self.statistics:
                        if group_col:
                            result = result.with_columns(
                                pl.col(col).rolling_std(window).over(group_col)
                                .alias(f"{col}_rolling_std_{window}")
                            )
                        else:
                            result = result.with_columns(
                                pl.col(col).rolling_std(window)
                                .alias(f"{col}_rolling_std_{window}")
                            )

                    # Min
                    if "min" in self.statistics:
                        if group_col:
                            result = result.with_columns(
                                pl.col(col).rolling_min(window).over(group_col)
                                .alias(f"{col}_rolling_min_{window}")
                            )
                        else:
                            result = result.with_columns(
                                pl.col(col).rolling_min(window)
                                .alias(f"{col}_rolling_min_{window}")
                            )

                    # Max
                    if "max" in self.statistics:
                        if group_col:
                            result = result.with_columns(
                                pl.col(col).rolling_max(window).over(group_col)
                                .alias(f"{col}_rolling_max_{window}")
                            )
                        else:
                            result = result.with_columns(
                                pl.col(col).rolling_max(window)
                                .alias(f"{col}_rolling_max_{window}")
                            )

                    # Sum
                    if "sum" in self.statistics:
                        if group_col:
                            result = result.with_columns(
                                pl.col(col).rolling_sum(window).over(group_col)
                                .alias(f"{col}_rolling_sum_{window}")
                            )
                        else:
                            result = result.with_columns(
                                pl.col(col).rolling_sum(window)
                                .alias(f"{col}_rolling_sum_{window}")
                            )

                    # Median
                    if "median" in self.statistics:
                        if group_col:
                            result = result.with_columns(
                                pl.col(col).rolling_median(window).over(group_col)
                                .alias(f"{col}_rolling_median_{window}")
                            )
                        else:
                            result = result.with_columns(
                                pl.col(col).rolling_median(window)
                                .alias(f"{col}_rolling_median_{window}")
                            )

            return result

        except Exception as e:
            logger.error("Failed to create rolling features", error=str(e))
            raise

    async def _create_diff_features(
        self,
        data: pl.DataFrame,
        feature_cols: List[str],
        group_col: Optional[str]
    ) -> pl.DataFrame:
        """Create differencing features.

        Args:
            data: Input DataFrame
            feature_cols: Feature columns
            group_col: Optional grouping column

        Returns:
            DataFrame with differencing features
        """
        try:
            result = data.clone()

            for col in feature_cols:
                for period in self.diff_periods:
                    # Absolute difference
                    if group_col:
                        result = result.with_columns(
                            (pl.col(col) - pl.col(col).shift(period).over(group_col))
                            .alias(f"{col}_diff{period}")
                        )
                    else:
                        result = result.with_columns(
                            (pl.col(col) - pl.col(col).shift(period))
                            .alias(f"{col}_diff{period}")
                        )

                    # Percentage change
                    if group_col:
                        result = result.with_columns(
                            ((pl.col(col) - pl.col(col).shift(period).over(group_col)) /
                             (pl.col(col).shift(period).over(group_col) + Decimal("0.0001")))
                            .alias(f"{col}_pct_change{period}")
                        )
                    else:
                        result = result.with_columns(
                            ((pl.col(col) - pl.col(col).shift(period)) /
                             (pl.col(col).shift(period) + Decimal("0.0001")))
                            .alias(f"{col}_pct_change{period}")
                        )

            return result

        except Exception as e:
            logger.error("Failed to create diff features", error=str(e))
            raise

    async def _create_expanding_features(
        self,
        data: pl.DataFrame,
        feature_cols: List[str],
        group_col: Optional[str]
    ) -> pl.DataFrame:
        """Create expanding window features.

        Args:
            data: Input DataFrame
            feature_cols: Feature columns
            group_col: Optional grouping column

        Returns:
            DataFrame with expanding features
        """
        try:
            result = data.clone()

            for col in feature_cols:
                # Cumulative sum
                if group_col:
                    result = result.with_columns(
                        pl.col(col).cum_sum().over(group_col)
                        .alias(f"{col}_cumsum")
                    )
                else:
                    result = result.with_columns(
                        pl.col(col).cum_sum()
                        .alias(f"{col}_cumsum")
                    )

                # Cumulative mean
                if group_col:
                    result = result.with_columns(
                        pl.col(col).cum_mean().over(group_col)
                        .alias(f"{col}_cummean")
                    )
                else:
                    result = result.with_columns(
                        pl.col(col).cum_mean()
                        .alias(f"{col}_cummean")
                    )

                # Cumulative min/max
                if group_col:
                    result = result.with_columns([
                        pl.col(col).cum_min().over(group_col).alias(f"{col}_cummin"),
                        pl.col(col).cum_max().over(group_col).alias(f"{col}_cummax")
                    ])
                else:
                    result = result.with_columns([
                        pl.col(col).cum_min().alias(f"{col}_cummin"),
                        pl.col(col).cum_max().alias(f"{col}_cummax")
                    ])

            return result

        except Exception as e:
            logger.error("Failed to create expanding features", error=str(e))
            raise

    async def create_temporal_features(
        self,
        data: pl.DataFrame,
        timestamp_col: str
    ) -> pl.DataFrame:
        """Create time-based features from timestamp.

        Args:
            data: Input DataFrame
            timestamp_col: Timestamp column name

        Returns:
            DataFrame with temporal features
        """
        try:
            result = data.clone()

            # Extract temporal components
            result = result.with_columns([
                pl.col(timestamp_col).dt.hour().alias("hour"),
                pl.col(timestamp_col).dt.day().alias("day"),
                pl.col(timestamp_col).dt.weekday().alias("weekday"),
                pl.col(timestamp_col).dt.month().alias("month"),
                pl.col(timestamp_col).dt.quarter().alias("quarter")
            ])

            # Cyclical encoding for hour (0-23)
            result = result.with_columns([
                (pl.col("hour") * np.pi * 2 / 24).sin().alias("hour_sin"),
                (pl.col("hour") * np.pi * 2 / 24).cos().alias("hour_cos")
            ])

            # Cyclical encoding for day of week (0-6)
            result = result.with_columns([
                (pl.col("weekday") * np.pi * 2 / 7).sin().alias("weekday_sin"),
                (pl.col("weekday") * np.pi * 2 / 7).cos().alias("weekday_cos")
            ])

            # Cyclical encoding for month (1-12)
            result = result.with_columns([
                ((pl.col("month") - 1) * np.pi * 2 / 12).sin().alias("month_sin"),
                ((pl.col("month") - 1) * np.pi * 2 / 12).cos().alias("month_cos")
            ])

            logger.info("Temporal features created")

            return result

        except Exception as e:
            logger.error("Failed to create temporal features", error=str(e))
            raise
