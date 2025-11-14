"""Volatility-based features for technical analysis.

This module extracts volatility indicators and features from price data
for trading signal generation and risk management.
"""

from __future__ import annotations

import asyncio
import os
from decimal import Decimal
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class VolatilityFeatures:
    """Extract volatility-based technical features.

    Calculates various volatility metrics including historical volatility,
    ATR, Bollinger Bands, and volatility ratios.

    Attributes:
        config: Configuration dictionary
        window_sizes: List of window sizes for calculations
        atr_period: ATR calculation period

    Examples:
        >>> config = {"window_sizes": [14, 28], "atr_period": 14}
        >>> extractor = VolatilityFeatures(config)
        >>> features = await extractor.extract(
        ...     data=pl.DataFrame({"high": [...], "low": [...], "close": [...]})
        ... )
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize volatility feature extractor.

        Args:
            config: Configuration with window sizes, periods, etc.

        Raises:
            ValueError: If required config parameters missing
        """
        self.config = config
        self._validate_config()

        self.window_sizes: List[int] = config.get("window_sizes", [int(x) for x in os.getenv("VOL_WINDOW_SIZES", "14,28,50").split(",")])
        self.atr_period: int = int(config.get("atr_period", os.getenv("VOL_ATR_PERIOD", "14")))
        self.bb_period: int = int(config.get("bb_period", os.getenv("VOL_BB_PERIOD", "20")))
        self.bb_std: Decimal = Decimal(str(config.get("bb_std", os.getenv("VOL_BB_STD", "2.0"))))
        self.min_periods: int = int(config.get("min_periods", os.getenv("VOL_MIN_PERIODS", "20")))

        logger.info(
            "Volatility features extractor initialized",
            window_sizes=self.window_sizes,
            atr_period=self.atr_period,
            bb_period=self.bb_period
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If config validation fails
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        if "window_sizes" in self.config:
            if not self.config["window_sizes"]:
                raise ValueError("window_sizes must not be empty")
            if any(w <= 0 for w in self.config["window_sizes"]):
                raise ValueError("All window sizes must be positive")

    async def extract(
        self,
        data: pl.DataFrame,
        metadata: Optional[Dict[str, Any]] = None
    ) -> pl.DataFrame:
        """Extract volatility features from price data.

        Args:
            data: DataFrame with 'high', 'low', 'close' columns
            metadata: Optional metadata

        Returns:
            DataFrame with volatility features

        Raises:
            ValueError: If data DataFrame invalid
        """
        try:
            if data.is_empty():
                logger.warning("Empty data DataFrame provided")
                return self._empty_features()

            required_cols = ["high", "low", "close"]
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            if len(data) < self.min_periods:
                logger.warning(
                    "Insufficient data for volatility features",
                    data_length=len(data),
                    min_periods=self.min_periods
                )
                return self._empty_features()

            # Calculate various volatility features
            historical_vol = await self._calculate_historical_volatility(data)
            atr_features = await self._calculate_atr(data)
            bb_features = await self._calculate_bollinger_bands(data)
            parkinson_vol = await self._calculate_parkinson_volatility(data)
            garman_klass_vol = await self._calculate_garman_klass_volatility(data)

            # Combine all features
            features = self._combine_features(
                historical_vol,
                atr_features,
                bb_features,
                parkinson_vol,
                garman_klass_vol
            )

            logger.debug(
                "Volatility features extracted",
                num_observations=len(data),
                num_features=len(features)
            )

            return features

        except Exception as e:
            logger.error(
                "Volatility feature extraction failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def _calculate_historical_volatility(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate historical volatility for different windows.

        Args:
            data: DataFrame with close prices

        Returns:
            Dictionary of historical volatility metrics
        """
        features = {}

        # Calculate returns
        close_prices = data["close"].to_list()
        returns = []
        for i in range(1, len(close_prices)):
            if close_prices[i] is not None and close_prices[i-1] is not None:
                ret = (Decimal(str(close_prices[i])) - Decimal(str(close_prices[i-1]))) / Decimal(str(close_prices[i-1]))
                returns.append(ret)

        if not returns:
            return {f"hist_vol_{w}": Decimal("0") for w in self.window_sizes}

        returns_array = np.array([float(r) for r in returns], dtype=np.float64)

        for window in self.window_sizes:
            if len(returns_array) >= window:
                # Rolling volatility for last window periods
                window_returns = returns_array[-window:]
                vol = Decimal(str(np.std(window_returns, ddof=1)))

                # Annualize (assuming 365 periods per year)
                annualization_factor = Decimal(str(os.getenv("VOL_ANNUALIZATION", "252")))
                annualized_vol = vol * Decimal(str(np.sqrt(float(annualization_factor))))

                features[f"hist_vol_{window}"] = annualized_vol
            else:
                features[f"hist_vol_{window}"] = Decimal("0")

        return features

    async def _calculate_atr(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate Average True Range.

        Args:
            data: DataFrame with high, low, close prices

        Returns:
            Dictionary of ATR metrics
        """
        true_ranges = []

        for i in range(1, len(data)):
            high = Decimal(str(data["high"][i]))
            low = Decimal(str(data["low"][i]))
            prev_close = Decimal(str(data["close"][i-1]))

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        if len(true_ranges) < self.atr_period:
            return {
                "atr": Decimal("0"),
                "atr_percent": Decimal("0")
            }

        # Calculate ATR as simple moving average of true ranges
        recent_tr = true_ranges[-self.atr_period:]
        atr = sum(recent_tr) / Decimal(str(self.atr_period))

        # ATR as percentage of current price
        current_price = Decimal(str(data["close"][-1]))
        atr_percent = (atr / current_price) * Decimal("100") if current_price > 0 else Decimal("0")

        return {
            "atr": atr,
            "atr_percent": atr_percent
        }

    async def _calculate_bollinger_bands(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate Bollinger Bands.

        Args:
            data: DataFrame with close prices

        Returns:
            Dictionary of Bollinger Band metrics
        """
        if len(data) < self.bb_period:
            return {
                "bb_width": Decimal("0"),
                "bb_percent": Decimal("0")
            }

        # Calculate SMA and std dev
        close_prices = [Decimal(str(p)) for p in data["close"][-self.bb_period:].to_list()]
        sma = sum(close_prices) / Decimal(str(self.bb_period))

        # Calculate standard deviation
        variance = sum((p - sma) ** 2 for p in close_prices) / Decimal(str(self.bb_period))
        std_dev = Decimal(str(np.sqrt(float(variance))))

        # Bollinger Bands
        upper_band = sma + (self.bb_std * std_dev)
        lower_band = sma - (self.bb_std * std_dev)
        bb_width = upper_band - lower_band

        # Current price position in bands (0-100%)
        current_price = Decimal(str(data["close"][-1]))
        if bb_width > 0:
            bb_percent = ((current_price - lower_band) / bb_width) * Decimal("100")
        else:
            bb_percent = Decimal("50")

        return {
            "bb_width": bb_width,
            "bb_percent": bb_percent,
            "bb_upper": upper_band,
            "bb_lower": lower_band,
            "bb_sma": sma
        }

    async def _calculate_parkinson_volatility(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate Parkinson's historical volatility.

        Uses high and low prices for more efficient volatility estimation.

        Args:
            data: DataFrame with high and low prices

        Returns:
            Dictionary of Parkinson volatility metrics
        """
        if len(data) < 2:
            return {"parkinson_vol": Decimal("0")}

        # Calculate Parkinson volatility
        hl_ratios = []
        for high, low in zip(data["high"].to_list(), data["low"].to_list()):
            if high is not None and low is not None and low > 0:
                ratio = Decimal(str(high)) / Decimal(str(low))
                if ratio > 0:
                    hl_ratios.append(float(np.log(float(ratio))) ** 2)

        if not hl_ratios:
            return {"parkinson_vol": Decimal("0")}

        # Parkinson formula
        n = len(hl_ratios)
        sum_squared = sum(hl_ratios)
        parkinson = np.sqrt(sum_squared / (4 * n * np.log(2)))

        # Annualize
        annualization_factor = Decimal(str(os.getenv("VOL_ANNUALIZATION", "252")))
        annualized = Decimal(str(parkinson)) * Decimal(str(np.sqrt(float(annualization_factor))))

        return {"parkinson_vol": annualized}

    async def _calculate_garman_klass_volatility(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate Garman-Klass volatility estimator.

        Uses open, high, low, close for improved volatility estimation.

        Args:
            data: DataFrame with OHLC prices

        Returns:
            Dictionary of Garman-Klass volatility metrics
        """
        if "open" not in data.columns or len(data) < 2:
            return {"garman_klass_vol": Decimal("0")}

        gk_values = []

        for i in range(len(data)):
            open_price = data["open"][i]
            high = data["high"][i]
            low = data["low"][i]
            close = data["close"][i]

            if all(x is not None and x > 0 for x in [open_price, high, low, close]):
                hl = float(np.log(float(high) / float(low))) ** 2
                co = float(np.log(float(close) / float(open_price))) ** 2

                gk = 0.5 * hl - (2 * np.log(2) - 1) * co
                gk_values.append(gk)

        if not gk_values:
            return {"garman_klass_vol": Decimal("0")}

        # Calculate Garman-Klass volatility
        n = len(gk_values)
        gk_vol = np.sqrt(sum(gk_values) / n)

        # Annualize
        annualization_factor = Decimal(str(os.getenv("VOL_ANNUALIZATION", "252")))
        annualized = Decimal(str(gk_vol)) * Decimal(str(np.sqrt(float(annualization_factor))))

        return {"garman_klass_vol": annualized}

    def _combine_features(self, *feature_dicts: Dict[str, Decimal]) -> pl.DataFrame:
        """Combine multiple feature dictionaries.

        Args:
            *feature_dicts: Variable number of feature dictionaries

        Returns:
            Combined features DataFrame
        """
        all_features = {}
        for features in feature_dicts:
            all_features.update(features)

        if not all_features:
            return self._empty_features()

        return pl.DataFrame({
            "feature": list(all_features.keys()),
            "value": [str(v) for v in all_features.values()]
        })

    def _empty_features(self) -> pl.DataFrame:
        """Return empty features DataFrame.

        Returns:
            DataFrame with zero-valued features
        """
        zero_features = {
            "hist_vol_14": "0",
            "atr": "0",
            "atr_percent": "0",
            "bb_width": "0",
            "bb_percent": "0",
            "parkinson_vol": "0",
            "garman_klass_vol": "0"
        }

        return pl.DataFrame({
            "feature": list(zero_features.keys()),
            "value": list(zero_features.values())
        })

    async def extract_batch(
        self,
        data_batch: List[pl.DataFrame],
        metadata_batch: Optional[List[Dict[str, Any]]] = None
    ) -> List[pl.DataFrame]:
        """Extract features for batch of data.

        Args:
            data_batch: List of DataFrames
            metadata_batch: Optional list of metadata dicts

        Returns:
            List of feature DataFrames
        """
        try:
            metadata_batch = metadata_batch or [None] * len(data_batch)

            features_batch = []
            for data, metadata in zip(data_batch, metadata_batch):
                features = await self.extract(data, metadata)
                features_batch.append(features)

            logger.debug(
                "Batch volatility features extracted",
                batch_size=len(features_batch)
            )

            return features_batch

        except Exception as e:
            logger.error(
                "Batch volatility feature extraction failed",
                error=str(e),
                batch_size=len(data_batch)
            )
            raise
