"""Funding rate features for sentiment analysis.

This module extracts funding rate-based features for trading sentiment analysis.
Funding rates indicate market sentiment and positioning in perpetual futures markets.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import polars as pl
import numpy as np
from structlog import get_logger

logger = get_logger(__name__)


class FundingFeatureExtractor:
    """Extract funding rate features for sentiment analysis.

    Funding rates are critical indicators of market sentiment in perpetual futures.
    Positive funding indicates long bias, negative indicates short bias.

    Attributes:
        config: Configuration dictionary
        lookback_periods: Periods for feature calculation
        aggregation_windows: Time windows for aggregation

    Example:
        >>> config = {
        ...     "lookback_periods": [8, 24, 72],
        ...     "aggregation_windows": ["1h", "4h", "1d"],
        ...     "extreme_threshold": "0.001"
        ... }
        >>> extractor = FundingFeatureExtractor(config)
        >>> features = await extractor.extract_features(funding_data)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize funding feature extractor.

        Args:
            config: Configuration with lookback periods and thresholds

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self._validate_config()

        self.lookback_periods = config.get("lookback_periods", [8, 24, 72])
        self.aggregation_windows = config.get("aggregation_windows", ["1h", "4h", "1d"])
        self.extreme_threshold = Decimal(str(config.get("extreme_threshold", "0.001")))

        logger.info(
            "Funding feature extractor initialized",
            lookback_periods=self.lookback_periods,
            windows=self.aggregation_windows
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required config missing or invalid
        """
        required_keys = ["lookback_periods", "aggregation_windows", "extreme_threshold"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        if not isinstance(self.config["lookback_periods"], list):
            raise ValueError("lookback_periods must be a list")

        if not isinstance(self.config["aggregation_windows"], list):
            raise ValueError("aggregation_windows must be a list")

    async def extract_features(
        self,
        funding_data: pl.DataFrame,
        symbols: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """Extract funding rate features from historical data.

        Args:
            funding_data: DataFrame with columns [symbol, timestamp, funding_rate]
            symbols: Optional list of symbols to process

        Returns:
            DataFrame with extracted features

        Raises:
            ValueError: If input data is invalid
        """
        try:
            self._validate_input_data(funding_data)

            if symbols:
                funding_data = funding_data.filter(pl.col("symbol").is_in(symbols))

            logger.info(
                "Extracting funding features",
                rows=len(funding_data),
                symbols=symbols
            )

            # Calculate base features
            features = await self._calculate_statistical_features(funding_data)

            # Add momentum features
            features = await self._calculate_momentum_features(features)

            # Add extreme event features
            features = await self._calculate_extreme_features(features)

            # Add cross-symbol features
            features = await self._calculate_cross_symbol_features(features)

            logger.info(
                "Funding features extracted",
                feature_count=len(features.columns),
                rows=len(features)
            )

            return features

        except Exception as e:
            logger.error("Failed to extract funding features", error=str(e))
            raise

    def _validate_input_data(self, data: pl.DataFrame) -> None:
        """Validate input DataFrame structure.

        Args:
            data: Input DataFrame to validate

        Raises:
            ValueError: If data structure is invalid
        """
        required_columns = ["symbol", "timestamp", "funding_rate"]
        missing = set(required_columns) - set(data.columns)

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if len(data) == 0:
            raise ValueError("Empty DataFrame provided")

    async def _calculate_statistical_features(
        self,
        data: pl.DataFrame
    ) -> pl.DataFrame:
        """Calculate statistical features for each lookback period.

        Args:
            data: Input funding rate data

        Returns:
            DataFrame with statistical features
        """
        try:
            features_list = []

            for period in self.lookback_periods:
                # Sort by symbol and timestamp
                sorted_data = data.sort(["symbol", "timestamp"])

                # Calculate rolling statistics
                feature_df = sorted_data.group_by("symbol").agg([
                    # Mean funding rate
                    pl.col("funding_rate")
                    .tail(period)
                    .mean()
                    .alias(f"funding_mean_{period}h"),

                    # Standard deviation
                    pl.col("funding_rate")
                    .tail(period)
                    .std()
                    .alias(f"funding_std_{period}h"),

                    # Min/Max
                    pl.col("funding_rate")
                    .tail(period)
                    .min()
                    .alias(f"funding_min_{period}h"),

                    pl.col("funding_rate")
                    .tail(period)
                    .max()
                    .alias(f"funding_max_{period}h"),

                    # Median
                    pl.col("funding_rate")
                    .tail(period)
                    .median()
                    .alias(f"funding_median_{period}h"),

                    # Latest timestamp
                    pl.col("timestamp").max().alias("timestamp")
                ])

                features_list.append(feature_df)

            # Join all features
            result = features_list[0]
            for df in features_list[1:]:
                result = result.join(
                    df.drop("timestamp"),
                    on="symbol",
                    how="left"
                )

            return result

        except Exception as e:
            logger.error("Failed to calculate statistical features", error=str(e))
            raise

    async def _calculate_momentum_features(
        self,
        data: pl.DataFrame
    ) -> pl.DataFrame:
        """Calculate momentum-based features.

        Args:
            data: DataFrame with statistical features

        Returns:
            DataFrame with added momentum features
        """
        try:
            # Calculate rate of change between periods
            result = data.clone()

            for i in range(len(self.lookback_periods) - 1):
                short_period = self.lookback_periods[i]
                long_period = self.lookback_periods[i + 1]

                # Momentum as difference in means
                momentum_col = f"funding_momentum_{short_period}_{long_period}h"
                result = result.with_columns(
                    (pl.col(f"funding_mean_{short_period}h") -
                     pl.col(f"funding_mean_{long_period}h"))
                    .alias(momentum_col)
                )

                # Volatility ratio
                vol_ratio_col = f"funding_vol_ratio_{short_period}_{long_period}h"
                result = result.with_columns(
                    (pl.col(f"funding_std_{short_period}h") /
                     (pl.col(f"funding_std_{long_period}h") + Decimal("0.0000001")))
                    .alias(vol_ratio_col)
                )

            return result

        except Exception as e:
            logger.error("Failed to calculate momentum features", error=str(e))
            raise

    async def _calculate_extreme_features(
        self,
        data: pl.DataFrame
    ) -> pl.DataFrame:
        """Calculate features based on extreme funding events.

        Args:
            data: DataFrame with base features

        Returns:
            DataFrame with extreme event features
        """
        try:
            result = data.clone()

            for period in self.lookback_periods:
                mean_col = f"funding_mean_{period}h"
                max_col = f"funding_max_{period}h"
                min_col = f"funding_min_{period}h"

                # Extreme positive funding events
                extreme_pos_col = f"funding_extreme_pos_{period}h"
                result = result.with_columns(
                    (pl.col(max_col) > float(self.extreme_threshold))
                    .cast(pl.Int8)
                    .alias(extreme_pos_col)
                )

                # Extreme negative funding events
                extreme_neg_col = f"funding_extreme_neg_{period}h"
                result = result.with_columns(
                    (pl.col(min_col) < -float(self.extreme_threshold))
                    .cast(pl.Int8)
                    .alias(extreme_neg_col)
                )

                # Funding range
                range_col = f"funding_range_{period}h"
                result = result.with_columns(
                    (pl.col(max_col) - pl.col(min_col))
                    .alias(range_col)
                )

            return result

        except Exception as e:
            logger.error("Failed to calculate extreme features", error=str(e))
            raise

    async def _calculate_cross_symbol_features(
        self,
        data: pl.DataFrame
    ) -> pl.DataFrame:
        """Calculate cross-symbol relative features.

        Args:
            data: DataFrame with per-symbol features

        Returns:
            DataFrame with cross-symbol features
        """
        try:
            result = data.clone()

            for period in self.lookback_periods:
                mean_col = f"funding_mean_{period}h"

                # Calculate market-wide statistics
                market_mean = result[mean_col].mean()
                market_std = result[mean_col].std()

                # Z-score relative to market
                zscore_col = f"funding_zscore_{period}h"
                result = result.with_columns(
                    ((pl.col(mean_col) - market_mean) /
                     (market_std + Decimal("0.0000001")))
                    .alias(zscore_col)
                )

                # Percentile rank
                rank_col = f"funding_rank_{period}h"
                result = result.with_columns(
                    pl.col(mean_col)
                    .rank(method="average")
                    .alias(rank_col)
                )

            return result

        except Exception as e:
            logger.error("Failed to calculate cross-symbol features", error=str(e))
            raise

    async def calculate_sentiment_score(
        self,
        features: pl.DataFrame,
        symbol: str
    ) -> Decimal:
        """Calculate aggregate sentiment score from funding features.

        Args:
            features: DataFrame with extracted features
            symbol: Symbol to calculate score for

        Returns:
            Sentiment score between -1 (bearish) and 1 (bullish)

        Raises:
            ValueError: If symbol not found
        """
        try:
            symbol_data = features.filter(pl.col("symbol") == symbol)

            if len(symbol_data) == 0:
                raise ValueError(f"Symbol {symbol} not found in features")

            # Weight different components
            weights = {
                "mean": Decimal(str(self.config.get("weight_mean", "0.4"))),
                "momentum": Decimal(str(self.config.get("weight_momentum", "0.3"))),
                "extreme": Decimal(str(self.config.get("weight_extreme", "0.3")))
            }

            # Get primary period
            primary_period = self.lookback_periods[0]

            # Mean component (normalized)
            mean_value = Decimal(str(symbol_data[f"funding_mean_{primary_period}h"][0]))
            mean_score = self._normalize_score(mean_value, Decimal("-0.001"), Decimal("0.001"))

            # Momentum component
            if len(self.lookback_periods) > 1:
                momentum_col = f"funding_momentum_{self.lookback_periods[0]}_{self.lookback_periods[1]}h"
                momentum_value = Decimal(str(symbol_data[momentum_col][0]))
                momentum_score = self._normalize_score(
                    momentum_value,
                    Decimal("-0.0005"),
                    Decimal("0.0005")
                )
            else:
                momentum_score = Decimal("0")

            # Extreme component (binary)
            extreme_pos = int(symbol_data[f"funding_extreme_pos_{primary_period}h"][0])
            extreme_neg = int(symbol_data[f"funding_extreme_neg_{primary_period}h"][0])
            extreme_score = Decimal(str(extreme_pos - extreme_neg))

            # Weighted combination
            total_score = (
                weights["mean"] * mean_score +
                weights["momentum"] * momentum_score +
                weights["extreme"] * extreme_score
            )

            # Clamp to [-1, 1]
            final_score = max(Decimal("-1"), min(Decimal("1"), total_score))

            logger.info(
                "Calculated sentiment score",
                symbol=symbol,
                score=float(final_score),
                components={
                    "mean": float(mean_score),
                    "momentum": float(momentum_score),
                    "extreme": float(extreme_score)
                }
            )

            return final_score

        except Exception as e:
            logger.error(
                "Failed to calculate sentiment score",
                symbol=symbol,
                error=str(e)
            )
            raise

    def _normalize_score(
        self,
        value: Decimal,
        min_val: Decimal,
        max_val: Decimal
    ) -> Decimal:
        """Normalize value to [-1, 1] range.

        Args:
            value: Value to normalize
            min_val: Minimum expected value
            max_val: Maximum expected value

        Returns:
            Normalized score in [-1, 1]
        """
        if value <= min_val:
            return Decimal("-1")
        elif value >= max_val:
            return Decimal("1")
        else:
            # Linear interpolation
            range_val = max_val - min_val
            normalized = (value - min_val) / range_val
            return (normalized * Decimal("2")) - Decimal("1")
