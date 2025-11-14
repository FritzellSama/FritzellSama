"""
Momentum-based Technical Features for Trading.

This module calculates momentum indicators including RSI, MACD, Stochastic,
and other momentum-based features for trading signal generation.
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


class MomentumFeatures:
    """Production-ready momentum feature calculator.

    Calculates comprehensive momentum indicators for algorithmic trading
    using Polars for high-performance data processing.

    Attributes:
        config: Configuration dictionary from config files
        rsi_period: RSI calculation period
        macd_fast: MACD fast period
        macd_slow: MACD slow period
        macd_signal: MACD signal period
        stoch_k_period: Stochastic K period
        stoch_d_period: Stochastic D period
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize momentum feature calculator.

        Args:
            config: Configuration dictionary containing:
                - features.momentum.rsi_period
                - features.momentum.macd_fast_period
                - features.momentum.macd_slow_period
                - features.momentum.macd_signal_period
                - features.momentum.stochastic_k_period
                - features.momentum.stochastic_d_period
                - features.momentum.roc_period
                - features.momentum.williams_r_period
        """
        self.config = config
        self._validate_config()

        momentum_config = self.config["features"]["momentum"]
        self.rsi_period = momentum_config["rsi_period"]
        self.macd_fast = momentum_config["macd_fast_period"]
        self.macd_slow = momentum_config["macd_slow_period"]
        self.macd_signal = momentum_config["macd_signal_period"]
        self.stoch_k_period = momentum_config["stochastic_k_period"]
        self.stoch_d_period = momentum_config["stochastic_d_period"]
        self.roc_period = momentum_config["roc_period"]
        self.williams_r_period = momentum_config["williams_r_period"]

        logger.info("momentum_features_initialized",
                   rsi_period=self.rsi_period,
                   macd_fast=self.macd_fast,
                   macd_slow=self.macd_slow)

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required config parameters are missing or invalid
        """
        required_keys = [
            "features.momentum.rsi_period",
            "features.momentum.macd_fast_period",
            "features.momentum.macd_slow_period",
            "features.momentum.macd_signal_period",
            "features.momentum.stochastic_k_period",
            "features.momentum.stochastic_d_period",
            "features.momentum.roc_period",
            "features.momentum.williams_r_period"
        ]

        for key in required_keys:
            parts = key.split(".")
            current = self.config
            for part in parts:
                if part not in current:
                    raise ValueError(f"Missing required config key: {key}")
                current = current[part]

    async def calculate_rsi(
        self,
        df: pl.DataFrame,
        price_column: str = "close"
    ) -> pl.DataFrame:
        """Calculate Relative Strength Index (RSI).

        Args:
            df: Polars DataFrame with price data
            price_column: Name of price column to use

        Returns:
            DataFrame with RSI column added

        Raises:
            ValueError: If required columns are missing
        """
        try:
            if price_column not in df.columns:
                raise ValueError(f"Column {price_column} not found in DataFrame")

            if len(df) < self.rsi_period:
                logger.warning("insufficient_data_for_rsi",
                             rows=len(df),
                             required=self.rsi_period)
                return df.with_columns(pl.lit(None).alias("rsi"))

            # Calculate price changes
            df = df.with_columns([
                (pl.col(price_column) - pl.col(price_column).shift(1)).alias("price_change")
            ])

            # Separate gains and losses
            df = df.with_columns([
                pl.when(pl.col("price_change") > 0)
                  .then(pl.col("price_change"))
                  .otherwise(0)
                  .alias("gain"),
                pl.when(pl.col("price_change") < 0)
                  .then(pl.col("price_change").abs())
                  .otherwise(0)
                  .alias("loss")
            ])

            # Calculate average gains and losses using EMA
            df = df.with_columns([
                pl.col("gain").ewm_mean(span=self.rsi_period).alias("avg_gain"),
                pl.col("loss").ewm_mean(span=self.rsi_period).alias("avg_loss")
            ])

            # Calculate RS and RSI
            df = df.with_columns([
                (pl.col("avg_gain") / pl.col("avg_loss")).alias("rs")
            ])

            df = df.with_columns([
                (100 - (100 / (1 + pl.col("rs")))).alias("rsi")
            ])

            # Drop intermediate columns
            df = df.drop(["price_change", "gain", "loss", "avg_gain", "avg_loss", "rs"])

            logger.debug("rsi_calculated", rows=len(df))
            return df

        except Exception as e:
            logger.error("rsi_calculation_failed", error=str(e))
            raise

    async def calculate_macd(
        self,
        df: pl.DataFrame,
        price_column: str = "close"
    ) -> pl.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence).

        Args:
            df: Polars DataFrame with price data
            price_column: Name of price column to use

        Returns:
            DataFrame with MACD, signal, and histogram columns

        Raises:
            ValueError: If required columns are missing
        """
        try:
            if price_column not in df.columns:
                raise ValueError(f"Column {price_column} not found in DataFrame")

            min_required = max(self.macd_slow, self.macd_fast) + self.macd_signal
            if len(df) < min_required:
                logger.warning("insufficient_data_for_macd",
                             rows=len(df),
                             required=min_required)
                return df.with_columns([
                    pl.lit(None).alias("macd"),
                    pl.lit(None).alias("macd_signal"),
                    pl.lit(None).alias("macd_histogram")
                ])

            # Calculate fast and slow EMAs
            df = df.with_columns([
                pl.col(price_column).ewm_mean(span=self.macd_fast).alias("ema_fast"),
                pl.col(price_column).ewm_mean(span=self.macd_slow).alias("ema_slow")
            ])

            # Calculate MACD line
            df = df.with_columns([
                (pl.col("ema_fast") - pl.col("ema_slow")).alias("macd")
            ])

            # Calculate signal line
            df = df.with_columns([
                pl.col("macd").ewm_mean(span=self.macd_signal).alias("macd_signal")
            ])

            # Calculate histogram
            df = df.with_columns([
                (pl.col("macd") - pl.col("macd_signal")).alias("macd_histogram")
            ])

            # Drop intermediate columns
            df = df.drop(["ema_fast", "ema_slow"])

            logger.debug("macd_calculated", rows=len(df))
            return df

        except Exception as e:
            logger.error("macd_calculation_failed", error=str(e))
            raise

    async def calculate_stochastic(
        self,
        df: pl.DataFrame,
        high_column: str = "high",
        low_column: str = "low",
        close_column: str = "close"
    ) -> pl.DataFrame:
        """Calculate Stochastic Oscillator.

        Args:
            df: Polars DataFrame with OHLC data
            high_column: Name of high price column
            low_column: Name of low price column
            close_column: Name of close price column

        Returns:
            DataFrame with stochastic K and D columns

        Raises:
            ValueError: If required columns are missing
        """
        try:
            required_cols = [high_column, low_column, close_column]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Column {col} not found in DataFrame")

            if len(df) < self.stoch_k_period:
                logger.warning("insufficient_data_for_stochastic",
                             rows=len(df),
                             required=self.stoch_k_period)
                return df.with_columns([
                    pl.lit(None).alias("stoch_k"),
                    pl.lit(None).alias("stoch_d")
                ])

            # Calculate rolling high and low
            df = df.with_columns([
                pl.col(high_column).rolling_max(window_size=self.stoch_k_period).alias("highest_high"),
                pl.col(low_column).rolling_min(window_size=self.stoch_k_period).alias("lowest_low")
            ])

            # Calculate %K
            df = df.with_columns([
                (
                    (pl.col(close_column) - pl.col("lowest_low")) /
                    (pl.col("highest_high") - pl.col("lowest_low")) * 100
                ).alias("stoch_k")
            ])

            # Calculate %D (smoothed %K)
            df = df.with_columns([
                pl.col("stoch_k").rolling_mean(window_size=self.stoch_d_period).alias("stoch_d")
            ])

            # Drop intermediate columns
            df = df.drop(["highest_high", "lowest_low"])

            logger.debug("stochastic_calculated", rows=len(df))
            return df

        except Exception as e:
            logger.error("stochastic_calculation_failed", error=str(e))
            raise

    async def calculate_roc(
        self,
        df: pl.DataFrame,
        price_column: str = "close"
    ) -> pl.DataFrame:
        """Calculate Rate of Change (ROC).

        Args:
            df: Polars DataFrame with price data
            price_column: Name of price column to use

        Returns:
            DataFrame with ROC column added

        Raises:
            ValueError: If required columns are missing
        """
        try:
            if price_column not in df.columns:
                raise ValueError(f"Column {price_column} not found in DataFrame")

            if len(df) < self.roc_period:
                logger.warning("insufficient_data_for_roc",
                             rows=len(df),
                             required=self.roc_period)
                return df.with_columns(pl.lit(None).alias("roc"))

            # Calculate ROC
            df = df.with_columns([
                (
                    (pl.col(price_column) - pl.col(price_column).shift(self.roc_period)) /
                    pl.col(price_column).shift(self.roc_period) * 100
                ).alias("roc")
            ])

            logger.debug("roc_calculated", rows=len(df))
            return df

        except Exception as e:
            logger.error("roc_calculation_failed", error=str(e))
            raise

    async def calculate_williams_r(
        self,
        df: pl.DataFrame,
        high_column: str = "high",
        low_column: str = "low",
        close_column: str = "close"
    ) -> pl.DataFrame:
        """Calculate Williams %R.

        Args:
            df: Polars DataFrame with OHLC data
            high_column: Name of high price column
            low_column: Name of low price column
            close_column: Name of close price column

        Returns:
            DataFrame with Williams %R column

        Raises:
            ValueError: If required columns are missing
        """
        try:
            required_cols = [high_column, low_column, close_column]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Column {col} not found in DataFrame")

            if len(df) < self.williams_r_period:
                logger.warning("insufficient_data_for_williams_r",
                             rows=len(df),
                             required=self.williams_r_period)
                return df.with_columns(pl.lit(None).alias("williams_r"))

            # Calculate rolling high and low
            df = df.with_columns([
                pl.col(high_column).rolling_max(window_size=self.williams_r_period).alias("highest_high"),
                pl.col(low_column).rolling_min(window_size=self.williams_r_period).alias("lowest_low")
            ])

            # Calculate Williams %R
            df = df.with_columns([
                (
                    (pl.col("highest_high") - pl.col(close_column)) /
                    (pl.col("highest_high") - pl.col("lowest_low")) * -100
                ).alias("williams_r")
            ])

            # Drop intermediate columns
            df = df.drop(["highest_high", "lowest_low"])

            logger.debug("williams_r_calculated", rows=len(df))
            return df

        except Exception as e:
            logger.error("williams_r_calculation_failed", error=str(e))
            raise

    async def calculate_all_features(
        self,
        df: pl.DataFrame
    ) -> pl.DataFrame:
        """Calculate all momentum features.

        Args:
            df: Polars DataFrame with OHLC data

        Returns:
            DataFrame with all momentum features added

        Raises:
            ValueError: If required columns are missing
        """
        try:
            logger.info("calculating_all_momentum_features", rows=len(df))

            # Validate required columns
            required_cols = ["open", "high", "low", "close", "volume"]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Required column {col} not found in DataFrame")

            # Calculate all features
            df = await self.calculate_rsi(df)
            df = await self.calculate_macd(df)
            df = await self.calculate_stochastic(df)
            df = await self.calculate_roc(df)
            df = await self.calculate_williams_r(df)

            logger.info("all_momentum_features_calculated",
                       rows=len(df),
                       features=["rsi", "macd", "stochastic", "roc", "williams_r"])

            return df

        except Exception as e:
            logger.error("momentum_features_calculation_failed", error=str(e))
            raise
