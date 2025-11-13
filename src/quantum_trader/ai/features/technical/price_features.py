"""
Price-based Technical Features for Trading.

This module calculates price-based features including returns, volatility,
price levels, and statistical measures for trading signal generation.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class PriceFeatures:
    """Production-ready price feature calculator.

    Calculates comprehensive price-based features including returns,
    volatility, support/resistance levels, and statistical measures.

    Attributes:
        config: Configuration dictionary
        return_windows: Windows for return calculations
        volatility_windows: Windows for volatility calculations
        percentile_windows: Windows for percentile calculations
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize price feature calculator.

        Args:
            config: Configuration dictionary containing:
                - features.price.return_windows
                - features.price.volatility_windows
                - features.price.percentile_windows
                - features.price.support_resistance_window
        """
        self.config = config
        self._validate_config()

        price_config = self.config["features"]["price"]
        self.return_windows = price_config["return_windows"]
        self.volatility_windows = price_config["volatility_windows"]
        self.percentile_windows = price_config["percentile_windows"]
        self.support_resistance_window = price_config["support_resistance_window"]

        logger.info("price_features_initialized",
                   return_windows=self.return_windows,
                   volatility_windows=self.volatility_windows)

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required config parameters are missing
        """
        required_keys = [
            "features.price.return_windows",
            "features.price.volatility_windows",
            "features.price.percentile_windows",
            "features.price.support_resistance_window"
        ]

        for key in required_keys:
            parts = key.split(".")
            current = self.config
            for part in parts:
                if part not in current:
                    raise ValueError(f"Missing required config key: {key}")
                current = current[part]

    async def calculate_returns(
        self,
        df: pl.DataFrame,
        price_column: str = "close"
    ) -> pl.DataFrame:
        """Calculate returns for multiple windows.

        Args:
            df: Polars DataFrame with price data
            price_column: Name of price column to use

        Returns:
            DataFrame with return columns added

        Raises:
            ValueError: If required columns are missing
        """
        try:
            if price_column not in df.columns:
                raise ValueError(f"Column {price_column} not found in DataFrame")

            logger.debug("calculating_returns", windows=self.return_windows)

            result_df = df.clone()

            # Calculate returns for each window
            for window in self.return_windows:
                if len(df) < window:
                    logger.warning("insufficient_data_for_return_window",
                                 rows=len(df),
                                 window=window)
                    result_df = result_df.with_columns(
                        pl.lit(None).alias(f"return_{window}")
                    )
                    continue

                # Simple return: (price_t - price_t-n) / price_t-n
                result_df = result_df.with_columns([
                    (
                        (pl.col(price_column) - pl.col(price_column).shift(window)) /
                        pl.col(price_column).shift(window)
                    ).alias(f"return_{window}")
                ])

            # Log returns for window 1
            if 1 in self.return_windows:
                result_df = result_df.with_columns([
                    pl.when(pl.col(price_column) > 0)
                      .then((pl.col(price_column) / pl.col(price_column).shift(1)).log())
                      .otherwise(0)
                      .alias("log_return_1")
                ])

            logger.debug("returns_calculated", feature_count=len(self.return_windows))
            return result_df

        except Exception as e:
            logger.error("return_calculation_failed", error=str(e))
            raise

    async def calculate_volatility(
        self,
        df: pl.DataFrame,
        return_column: str = "return_1"
    ) -> pl.DataFrame:
        """Calculate volatility for multiple windows.

        Args:
            df: Polars DataFrame with return data
            return_column: Name of return column to use

        Returns:
            DataFrame with volatility columns added

        Raises:
            ValueError: If required columns are missing
        """
        try:
            # Ensure returns are calculated
            if return_column not in df.columns:
                df = await self.calculate_returns(df)

            logger.debug("calculating_volatility", windows=self.volatility_windows)

            result_df = df.clone()

            # Calculate volatility (rolling standard deviation) for each window
            for window in self.volatility_windows:
                if len(df) < window:
                    logger.warning("insufficient_data_for_volatility_window",
                                 rows=len(df),
                                 window=window)
                    result_df = result_df.with_columns(
                        pl.lit(None).alias(f"volatility_{window}")
                    )
                    continue

                result_df = result_df.with_columns([
                    pl.col(return_column).rolling_std(window_size=window).alias(f"volatility_{window}")
                ])

            # Annualized volatility (assuming daily data)
            if "volatility_20" in result_df.columns:  # 20-day volatility
                result_df = result_df.with_columns([
                    (pl.col("volatility_20") * np.sqrt(252)).alias("volatility_annualized")
                ])

            logger.debug("volatility_calculated", feature_count=len(self.volatility_windows))
            return result_df

        except Exception as e:
            logger.error("volatility_calculation_failed", error=str(e))
            raise

    async def calculate_price_percentiles(
        self,
        df: pl.DataFrame,
        price_column: str = "close"
    ) -> pl.DataFrame:
        """Calculate price percentiles over windows.

        Args:
            df: Polars DataFrame with price data
            price_column: Name of price column to use

        Returns:
            DataFrame with percentile columns added

        Raises:
            ValueError: If required columns are missing
        """
        try:
            if price_column not in df.columns:
                raise ValueError(f"Column {price_column} not found in DataFrame")

            logger.debug("calculating_percentiles", windows=self.percentile_windows)

            result_df = df.clone()

            # Calculate percentile rank for each window
            for window in self.percentile_windows:
                if len(df) < window:
                    logger.warning("insufficient_data_for_percentile_window",
                                 rows=len(df),
                                 window=window)
                    result_df = result_df.with_columns(
                        pl.lit(None).alias(f"percentile_{window}")
                    )
                    continue

                # Calculate rolling min and max
                result_df = result_df.with_columns([
                    pl.col(price_column).rolling_min(window_size=window).alias(f"_min_{window}"),
                    pl.col(price_column).rolling_max(window_size=window).alias(f"_max_{window}")
                ])

                # Calculate percentile rank (0 to 1)
                result_df = result_df.with_columns([
                    (
                        (pl.col(price_column) - pl.col(f"_min_{window}")) /
                        (pl.col(f"_max_{window}") - pl.col(f"_min_{window}"))
                    ).alias(f"percentile_{window}")
                ])

                # Drop temporary columns
                result_df = result_df.drop([f"_min_{window}", f"_max_{window}"])

            logger.debug("percentiles_calculated", feature_count=len(self.percentile_windows))
            return result_df

        except Exception as e:
            logger.error("percentile_calculation_failed", error=str(e))
            raise

    async def calculate_support_resistance(
        self,
        df: pl.DataFrame,
        price_column: str = "close"
    ) -> pl.DataFrame:
        """Calculate support and resistance levels.

        Args:
            df: Polars DataFrame with price data
            price_column: Name of price column to use

        Returns:
            DataFrame with support/resistance columns added

        Raises:
            ValueError: If required columns are missing
        """
        try:
            if price_column not in df.columns:
                raise ValueError(f"Column {price_column} not found in DataFrame")

            logger.debug("calculating_support_resistance",
                        window=self.support_resistance_window)

            result_df = df.clone()

            window = self.support_resistance_window

            if len(df) < window:
                logger.warning("insufficient_data_for_support_resistance",
                             rows=len(df),
                             required=window)
                return result_df.with_columns([
                    pl.lit(None).alias("support_level"),
                    pl.lit(None).alias("resistance_level"),
                    pl.lit(None).alias("distance_to_support"),
                    pl.lit(None).alias("distance_to_resistance")
                ])

            # Support: rolling minimum (low)
            # Resistance: rolling maximum (high)
            result_df = result_df.with_columns([
                pl.col("low").rolling_min(window_size=window).alias("support_level"),
                pl.col("high").rolling_max(window_size=window).alias("resistance_level")
            ])

            # Calculate distance to support/resistance as percentage
            result_df = result_df.with_columns([
                (
                    (pl.col(price_column) - pl.col("support_level")) /
                    pl.col(price_column) * 100
                ).alias("distance_to_support"),
                (
                    (pl.col("resistance_level") - pl.col(price_column)) /
                    pl.col(price_column) * 100
                ).alias("distance_to_resistance")
            ])

            logger.debug("support_resistance_calculated")
            return result_df

        except Exception as e:
            logger.error("support_resistance_calculation_failed", error=str(e))
            raise

    async def calculate_price_momentum(
        self,
        df: pl.DataFrame,
        price_column: str = "close"
    ) -> pl.DataFrame:
        """Calculate price momentum indicators.

        Args:
            df: Polars DataFrame with price data
            price_column: Name of price column to use

        Returns:
            DataFrame with momentum columns added

        Raises:
            ValueError: If required columns are missing
        """
        try:
            if price_column not in df.columns:
                raise ValueError(f"Column {price_column} not found in DataFrame")

            logger.debug("calculating_price_momentum")

            result_df = df.clone()

            # Price acceleration (second derivative)
            result_df = result_df.with_columns([
                (pl.col(price_column) - pl.col(price_column).shift(1)).alias("_price_change_1"),
                (pl.col(price_column) - pl.col(price_column).shift(2)).alias("_price_change_2")
            ])

            result_df = result_df.with_columns([
                (pl.col("_price_change_1") - pl.col("_price_change_2")).alias("price_acceleration")
            ])

            # Price velocity (rate of change)
            result_df = result_df.with_columns([
                (
                    (pl.col(price_column) - pl.col(price_column).shift(5)) /
                    pl.col(price_column).shift(5) * 100
                ).alias("price_velocity_5")
            ])

            # Moving average convergence
            result_df = result_df.with_columns([
                pl.col(price_column).rolling_mean(window_size=10).alias("_sma_10"),
                pl.col(price_column).rolling_mean(window_size=20).alias("_sma_20")
            ])

            result_df = result_df.with_columns([
                (pl.col("_sma_10") - pl.col("_sma_20")).alias("ma_convergence")
            ])

            # Drop temporary columns
            result_df = result_df.drop(["_price_change_1", "_price_change_2", "_sma_10", "_sma_20"])

            logger.debug("price_momentum_calculated")
            return result_df

        except Exception as e:
            logger.error("price_momentum_calculation_failed", error=str(e))
            raise

    async def calculate_price_levels(
        self,
        df: pl.DataFrame,
        price_column: str = "close"
    ) -> pl.DataFrame:
        """Calculate price levels and boundaries.

        Args:
            df: Polars DataFrame with price data
            price_column: Name of price column to use

        Returns:
            DataFrame with price level columns added

        Raises:
            ValueError: If required columns are missing
        """
        try:
            if price_column not in df.columns:
                raise ValueError(f"Column {price_column} not found in DataFrame")

            logger.debug("calculating_price_levels")

            result_df = df.clone()

            # Distance from recent high/low
            result_df = result_df.with_columns([
                pl.col(price_column).rolling_max(window_size=20).alias("_high_20"),
                pl.col(price_column).rolling_min(window_size=20).alias("_low_20"),
                pl.col(price_column).rolling_max(window_size=50).alias("_high_50"),
                pl.col(price_column).rolling_min(window_size=50).alias("_low_50")
            ])

            result_df = result_df.with_columns([
                (
                    (pl.col(price_column) - pl.col("_low_20")) /
                    (pl.col("_high_20") - pl.col("_low_20"))
                ).alias("price_position_20"),
                (
                    (pl.col(price_column) - pl.col("_low_50")) /
                    (pl.col("_high_50") - pl.col("_low_50"))
                ).alias("price_position_50")
            ])

            # Distance from moving averages
            result_df = result_df.with_columns([
                pl.col(price_column).rolling_mean(window_size=50).alias("_sma_50"),
                pl.col(price_column).rolling_mean(window_size=200).alias("_sma_200")
            ])

            result_df = result_df.with_columns([
                (
                    (pl.col(price_column) - pl.col("_sma_50")) /
                    pl.col("_sma_50") * 100
                ).alias("distance_from_sma_50"),
                (
                    (pl.col(price_column) - pl.col("_sma_200")) /
                    pl.col("_sma_200") * 100
                ).alias("distance_from_sma_200")
            ])

            # Drop temporary columns
            temp_cols = ["_high_20", "_low_20", "_high_50", "_low_50", "_sma_50", "_sma_200"]
            result_df = result_df.drop([c for c in temp_cols if c in result_df.columns])

            logger.debug("price_levels_calculated")
            return result_df

        except Exception as e:
            logger.error("price_levels_calculation_failed", error=str(e))
            raise

    async def calculate_all_features(
        self,
        df: pl.DataFrame
    ) -> pl.DataFrame:
        """Calculate all price features.

        Args:
            df: Polars DataFrame with OHLC data

        Returns:
            DataFrame with all price features added

        Raises:
            ValueError: If required columns are missing
        """
        try:
            logger.info("calculating_all_price_features", rows=len(df))

            # Validate required columns
            required_cols = ["open", "high", "low", "close", "volume"]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Required column {col} not found in DataFrame")

            # Calculate all feature sets
            df = await self.calculate_returns(df)
            df = await self.calculate_volatility(df)
            df = await self.calculate_price_percentiles(df)
            df = await self.calculate_support_resistance(df)
            df = await self.calculate_price_momentum(df)
            df = await self.calculate_price_levels(df)

            # Calculate high-low range
            df = df.with_columns([
                ((pl.col("high") - pl.col("low")) / pl.col("close") * 100).alias("hl_range_pct")
            ])

            # Calculate gap features
            df = df.with_columns([
                (
                    (pl.col("open") - pl.col("close").shift(1)) /
                    pl.col("close").shift(1) * 100
                ).alias("gap_pct")
            ])

            # Calculate intraday change
            df = df.with_columns([
                (
                    (pl.col("close") - pl.col("open")) /
                    pl.col("open") * 100
                ).alias("intraday_change_pct")
            ])

            logger.info("all_price_features_calculated",
                       rows=len(df),
                       total_features=len(df.columns))

            return df

        except Exception as e:
            logger.error("price_features_calculation_failed", error=str(e))
            raise

    async def calculate_volume_price_features(
        self,
        df: pl.DataFrame
    ) -> pl.DataFrame:
        """Calculate volume-weighted price features.

        Args:
            df: Polars DataFrame with OHLC data

        Returns:
            DataFrame with volume-price features added

        Raises:
            ValueError: If required columns are missing
        """
        try:
            required_cols = ["close", "volume"]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Required column {col} not found in DataFrame")

            logger.debug("calculating_volume_price_features")

            result_df = df.clone()

            # VWAP (Volume Weighted Average Price) for multiple windows
            for window in [10, 20, 50]:
                if len(df) < window:
                    continue

                result_df = result_df.with_columns([
                    (
                        (pl.col("close") * pl.col("volume")).rolling_sum(window_size=window) /
                        pl.col("volume").rolling_sum(window_size=window)
                    ).alias(f"vwap_{window}")
                ])

                # Distance from VWAP
                if f"vwap_{window}" in result_df.columns:
                    result_df = result_df.with_columns([
                        (
                            (pl.col("close") - pl.col(f"vwap_{window}")) /
                            pl.col(f"vwap_{window}") * 100
                        ).alias(f"distance_from_vwap_{window}")
                    ])

            logger.debug("volume_price_features_calculated")
            return result_df

        except Exception as e:
            logger.error("volume_price_feature_calculation_failed", error=str(e))
            raise
