"""Volume-based features for technical analysis.

This module extracts volume-related indicators and features from market data
for identifying accumulation/distribution patterns and volume trends.
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


class VolumeFeatures:
    """Extract volume-based technical features.

    Calculates volume indicators including OBV, volume ratios, VWAP,
    and volume-price relationships.

    Attributes:
        config: Configuration dictionary
        window_sizes: List of window sizes for calculations
        vwap_period: VWAP calculation period

    Examples:
        >>> config = {"window_sizes": [14, 28], "vwap_period": 20}
        >>> extractor = VolumeFeatures(config)
        >>> features = await extractor.extract(
        ...     data=pl.DataFrame({"close": [...], "volume": [...], "high": [...], "low": [...]})
        ... )
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize volume feature extractor.

        Args:
            config: Configuration with window sizes, periods, etc.

        Raises:
            ValueError: If required config parameters missing
        """
        self.config = config
        self._validate_config()

        self.window_sizes: List[int] = config.get("window_sizes", [int(x) for x in os.getenv("VOL_FEAT_WINDOW_SIZES", "14,28,50").split(",")])
        self.vwap_period: int = int(config.get("vwap_period", os.getenv("VOL_FEAT_VWAP_PERIOD", "20")))
        self.min_periods: int = int(config.get("min_periods", os.getenv("VOL_FEAT_MIN_PERIODS", "20")))

        logger.info(
            "Volume features extractor initialized",
            window_sizes=self.window_sizes,
            vwap_period=self.vwap_period
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
        """Extract volume features from market data.

        Args:
            data: DataFrame with 'close', 'volume', 'high', 'low' columns
            metadata: Optional metadata

        Returns:
            DataFrame with volume features

        Raises:
            ValueError: If data DataFrame invalid
        """
        try:
            if data.is_empty():
                logger.warning("Empty data DataFrame provided")
                return self._empty_features()

            required_cols = ["close", "volume"]
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            if len(data) < self.min_periods:
                logger.warning(
                    "Insufficient data for volume features",
                    data_length=len(data),
                    min_periods=self.min_periods
                )
                return self._empty_features()

            # Calculate various volume features
            obv_features = await self._calculate_obv(data)
            volume_ratios = await self._calculate_volume_ratios(data)
            vwap_features = await self._calculate_vwap(data)
            volume_price_features = await self._calculate_volume_price_correlation(data)
            mfi_features = await self._calculate_mfi(data)

            # Combine all features
            features = self._combine_features(
                obv_features,
                volume_ratios,
                vwap_features,
                volume_price_features,
                mfi_features
            )

            logger.debug(
                "Volume features extracted",
                num_observations=len(data),
                num_features=len(features)
            )

            return features

        except Exception as e:
            logger.error(
                "Volume feature extraction failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def _calculate_obv(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate On-Balance Volume (OBV).

        Args:
            data: DataFrame with close and volume data

        Returns:
            Dictionary of OBV metrics
        """
        obv_values = []
        obv = Decimal("0")

        close_prices = data["close"].to_list()
        volumes = data["volume"].to_list()

        for i in range(1, len(data)):
            if close_prices[i] is not None and close_prices[i-1] is not None:
                current_close = Decimal(str(close_prices[i]))
                prev_close = Decimal(str(close_prices[i-1]))
                volume = Decimal(str(volumes[i])) if volumes[i] is not None else Decimal("0")

                if current_close > prev_close:
                    obv += volume
                elif current_close < prev_close:
                    obv -= volume

                obv_values.append(obv)

        if not obv_values:
            return {
                "obv": Decimal("0"),
                "obv_sma_14": Decimal("0"),
                "obv_ema_14": Decimal("0")
            }

        # Current OBV
        current_obv = obv_values[-1]

        # OBV moving averages
        obv_sma_14 = Decimal("0")
        obv_ema_14 = Decimal("0")

        if len(obv_values) >= 14:
            obv_sma_14 = sum(obv_values[-14:]) / Decimal("14")

            # Simple EMA calculation
            multiplier = Decimal("2") / Decimal("15")  # 2/(period+1)
            obv_ema_14 = obv_values[-14]
            for obv_val in obv_values[-13:]:
                obv_ema_14 = (obv_val - obv_ema_14) * multiplier + obv_ema_14

        return {
            "obv": current_obv,
            "obv_sma_14": obv_sma_14,
            "obv_ema_14": obv_ema_14
        }

    async def _calculate_volume_ratios(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate volume ratios for different windows.

        Args:
            data: DataFrame with volume data

        Returns:
            Dictionary of volume ratio metrics
        """
        features = {}
        volumes = [Decimal(str(v)) if v is not None else Decimal("0") for v in data["volume"].to_list()]

        current_volume = volumes[-1]

        for window in self.window_sizes:
            if len(volumes) >= window:
                avg_volume = sum(volumes[-window:]) / Decimal(str(window))

                # Current volume vs average volume
                if avg_volume > 0:
                    vol_ratio = current_volume / avg_volume
                else:
                    vol_ratio = Decimal("1")

                features[f"vol_ratio_{window}"] = vol_ratio

                # Volume trend (recent avg vs older avg)
                if len(volumes) >= window * 2:
                    recent_avg = sum(volumes[-window:]) / Decimal(str(window))
                    older_avg = sum(volumes[-window*2:-window]) / Decimal(str(window))

                    if older_avg > 0:
                        vol_trend = (recent_avg - older_avg) / older_avg
                    else:
                        vol_trend = Decimal("0")

                    features[f"vol_trend_{window}"] = vol_trend
            else:
                features[f"vol_ratio_{window}"] = Decimal("1")
                features[f"vol_trend_{window}"] = Decimal("0")

        return features

    async def _calculate_vwap(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate Volume Weighted Average Price (VWAP).

        Args:
            data: DataFrame with high, low, close, volume data

        Returns:
            Dictionary of VWAP metrics
        """
        if not all(col in data.columns for col in ["high", "low", "close", "volume"]):
            return {
                "vwap": Decimal("0"),
                "price_vwap_ratio": Decimal("0")
            }

        if len(data) < self.vwap_period:
            return {
                "vwap": Decimal("0"),
                "price_vwap_ratio": Decimal("0")
            }

        # Calculate typical price for each period
        typical_prices = []
        volumes = []

        for i in range(len(data)):
            high = data["high"][i]
            low = data["low"][i]
            close = data["close"][i]
            volume = data["volume"][i]

            if all(x is not None for x in [high, low, close, volume]):
                typical_price = (Decimal(str(high)) + Decimal(str(low)) + Decimal(str(close))) / Decimal("3")
                typical_prices.append(typical_price)
                volumes.append(Decimal(str(volume)))

        if len(typical_prices) < self.vwap_period:
            return {
                "vwap": Decimal("0"),
                "price_vwap_ratio": Decimal("0")
            }

        # Calculate VWAP for the period
        recent_tp = typical_prices[-self.vwap_period:]
        recent_vol = volumes[-self.vwap_period:]

        total_volume = sum(recent_vol)
        if total_volume > 0:
            vwap = sum(tp * vol for tp, vol in zip(recent_tp, recent_vol)) / total_volume
        else:
            vwap = Decimal("0")

        # Current price vs VWAP
        current_price = Decimal(str(data["close"][-1]))
        if vwap > 0:
            price_vwap_ratio = current_price / vwap
        else:
            price_vwap_ratio = Decimal("1")

        return {
            "vwap": vwap,
            "price_vwap_ratio": price_vwap_ratio
        }

    async def _calculate_volume_price_correlation(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate correlation between volume and price changes.

        Args:
            data: DataFrame with close and volume data

        Returns:
            Dictionary of volume-price correlation metrics
        """
        if len(data) < 2:
            return {"vol_price_corr": Decimal("0")}

        # Calculate price changes and volume
        price_changes = []
        volumes = []

        close_prices = data["close"].to_list()
        volume_data = data["volume"].to_list()

        for i in range(1, len(data)):
            if close_prices[i] is not None and close_prices[i-1] is not None and volume_data[i] is not None:
                price_change = float((Decimal(str(close_prices[i])) - Decimal(str(close_prices[i-1]))) / Decimal(str(close_prices[i-1])))
                volume = float(volume_data[i])

                price_changes.append(price_change)
                volumes.append(volume)

        if len(price_changes) < 2:
            return {"vol_price_corr": Decimal("0")}

        # Calculate correlation
        price_array = np.array(price_changes)
        volume_array = np.array(volumes)

        correlation = np.corrcoef(price_array, volume_array)[0, 1]

        return {
            "vol_price_corr": Decimal(str(correlation)) if not np.isnan(correlation) else Decimal("0")
        }

    async def _calculate_mfi(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate Money Flow Index (MFI).

        Args:
            data: DataFrame with high, low, close, volume data

        Returns:
            Dictionary of MFI metrics
        """
        if not all(col in data.columns for col in ["high", "low", "close", "volume"]):
            return {"mfi": Decimal("50")}

        mfi_period = int(os.getenv("VOL_FEAT_MFI_PERIOD", "14"))

        if len(data) < mfi_period + 1:
            return {"mfi": Decimal("50")}

        # Calculate typical price and money flow
        typical_prices = []
        money_flows = []

        for i in range(len(data)):
            high = data["high"][i]
            low = data["low"][i]
            close = data["close"][i]
            volume = data["volume"][i]

            if all(x is not None for x in [high, low, close, volume]):
                typical_price = (Decimal(str(high)) + Decimal(str(low)) + Decimal(str(close))) / Decimal("3")
                money_flow = typical_price * Decimal(str(volume))

                typical_prices.append(typical_price)
                money_flows.append(money_flow)

        if len(typical_prices) < mfi_period + 1:
            return {"mfi": Decimal("50")}

        # Calculate positive and negative money flow
        positive_flow = Decimal("0")
        negative_flow = Decimal("0")

        for i in range(1, mfi_period + 1):
            idx = -(mfi_period - i + 1)
            if typical_prices[idx] > typical_prices[idx - 1]:
                positive_flow += money_flows[idx]
            elif typical_prices[idx] < typical_prices[idx - 1]:
                negative_flow += money_flows[idx]

        # Calculate MFI
        if negative_flow > 0:
            money_flow_ratio = positive_flow / negative_flow
            mfi = Decimal("100") - (Decimal("100") / (Decimal("1") + money_flow_ratio))
        else:
            mfi = Decimal("100")

        return {"mfi": mfi}

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
            "obv": "0",
            "obv_sma_14": "0",
            "vol_ratio_14": "1",
            "vwap": "0",
            "price_vwap_ratio": "1",
            "vol_price_corr": "0",
            "mfi": "50"
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
                "Batch volume features extracted",
                batch_size=len(features_batch)
            )

            return features_batch

        except Exception as e:
            logger.error(
                "Batch volume feature extraction failed",
                error=str(e),
                batch_size=len(data_batch)
            )
            raise
