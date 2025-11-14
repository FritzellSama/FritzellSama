"""
Volume Feature Extraction

Production-ready volume indicators and features for trading models.
Includes volume profiles, OBV, volume-price analysis, and flow metrics.
"""

import asyncio
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from collections import deque
import os

import polars as pl
import numpy as np
from structlog import get_logger

logger = get_logger(__name__)

# Set high precision
getcontext().prec = 28


class VolumeFeatures:
    """
    Extract volume-based features from trading data.

    Calculates volume indicators, price-volume relationships,
    and money flow metrics for trading signals.

    Attributes:
        config: Configuration dictionary
        windows: List of lookback windows
        volume_ma_type: Type of moving average for volume

    Example:
        >>> config = {"features": {"volume": {"windows": [10, 20, 50]}}}
        >>> vol_features = VolumeFeatures(config)
        >>> features = await vol_features.extract_features(ohlcv_df)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize volume features extractor.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        vol_config = self.config.get("features", {}).get("volume", {})

        # Calculation windows
        self.windows: List[int] = vol_config.get(
            "windows",
            [int(x) for x in os.getenv("VOLUME_WINDOWS", "10,20,50,100").split(",")]
        )

        # Volume moving average type
        self.volume_ma_type: str = vol_config.get(
            "volume_ma_type",
            os.getenv("VOLUME_MA_TYPE", "sma")
        )

        # MFI period
        self.mfi_period: int = vol_config.get("mfi_period", int(os.getenv("MFI_PERIOD", "14")))

        # Volume profile bins
        self.profile_bins: int = vol_config.get("profile_bins", int(os.getenv("VOLUME_PROFILE_BINS", "20")))

        # Thresholds
        self.high_volume_threshold: Decimal = Decimal(
            str(vol_config.get("high_volume_threshold", os.getenv("HIGH_VOLUME_THRESHOLD", "1.5")))
        )

        logger.info(
            "volume_features_initialized",
            windows=self.windows,
            ma_type=self.volume_ma_type,
            mfi_period=self.mfi_period
        )

    def _validate_config(self) -> None:
        """Validate configuration."""
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        vol_config = self.config.get("features", {}).get("volume", {})

        if vol_config:
            windows = vol_config.get("windows", [10, 20, 50])
            if not all(w > 0 for w in windows):
                raise ValueError("All windows must be positive")

    async def extract_features(
        self,
        ohlcv_data: pl.DataFrame,
        symbol: Optional[str] = None
    ) -> Dict[str, Decimal]:
        """
        Extract volume features from OHLCV data.

        Args:
            ohlcv_data: DataFrame with OHLCV data
                Required columns: timestamp, open, high, low, close, volume
            symbol: Trading symbol (for logging)

        Returns:
            Dictionary of volume features

        Raises:
            ValueError: If data schema is invalid

        Example:
            >>> features = await vol_features.extract_features(ohlcv_df, symbol="BTC/USDT")
        """
        try:
            # Validate schema
            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            if not all(col in ohlcv_data.columns for col in required_cols):
                raise ValueError(f"DataFrame missing required columns: {required_cols}")

            if len(ohlcv_data) < max(self.windows):
                logger.warning(
                    "insufficient_data_for_volume_features",
                    symbol=symbol,
                    data_length=len(ohlcv_data),
                    min_required=max(self.windows)
                )
                return self._get_default_features()

            # Sort by timestamp
            data = ohlcv_data.sort("timestamp")

            # Extract features
            features = {}

            # Basic volume metrics
            features.update(self._calculate_volume_metrics(data))

            # Volume moving averages
            features.update(self._calculate_volume_ma(data))

            # On-Balance Volume (OBV)
            features.update(self._calculate_obv(data))

            # Volume-Price Trend (VPT)
            features.update(self._calculate_vpt(data))

            # Money Flow Index (MFI)
            features.update(self._calculate_mfi(data))

            # Accumulation/Distribution
            features.update(self._calculate_accumulation_distribution(data))

            # Chaikin Money Flow
            features.update(self._calculate_chaikin_money_flow(data))

            # Volume Profile
            features.update(self._calculate_volume_profile(data))

            # Price-Volume divergence
            features.update(self._calculate_price_volume_divergence(data))

            # Volume trends
            features.update(self._calculate_volume_trends(data))

            logger.debug(
                "volume_features_extracted",
                symbol=symbol,
                num_features=len(features)
            )

            return features

        except Exception as e:
            logger.error("volume_feature_extraction_failed", symbol=symbol, error=str(e))
            raise

    def _calculate_volume_metrics(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate basic volume metrics."""
        features = {}

        volume = data.select("volume").to_numpy().flatten()
        current_volume = Decimal(str(volume[-1]))

        # Average volume (multiple windows)
        for window in self.windows:
            if len(volume) >= window:
                avg_volume = Decimal(str(np.mean(volume[-window:])))
                features[f"avg_volume_{window}"] = avg_volume

                # Volume ratio (current / average)
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else Decimal("1")
                features[f"volume_ratio_{window}"] = volume_ratio

        # Volume standard deviation
        if len(volume) >= min(self.windows):
            vol_std = Decimal(str(np.std(volume[-min(self.windows):])))
            features["volume_std"] = vol_std

        # Current volume
        features["current_volume"] = current_volume

        # High volume flag
        avg_vol_20 = features.get(f"avg_volume_20", current_volume)
        is_high_volume = Decimal("1") if current_volume > (avg_vol_20 * self.high_volume_threshold) else Decimal("0")
        features["is_high_volume"] = is_high_volume

        return features

    def _calculate_volume_ma(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate volume moving averages."""
        features = {}

        volume = data.select("volume").to_numpy().flatten()

        for window in self.windows:
            if len(volume) < window:
                continue

            if self.volume_ma_type == "ema":
                # Exponential moving average
                alpha = Decimal("2") / (Decimal(str(window)) + Decimal("1"))
                ema = Decimal(str(volume[-window]))

                for v in volume[-window + 1:]:
                    ema = alpha * Decimal(str(v)) + (Decimal("1") - alpha) * ema

                features[f"volume_ema_{window}"] = ema
            else:
                # Simple moving average
                sma = Decimal(str(np.mean(volume[-window:])))
                features[f"volume_sma_{window}"] = sma

        return features

    def _calculate_obv(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate On-Balance Volume."""
        features = {}

        close = data.select("close").to_numpy().flatten()
        volume = data.select("volume").to_numpy().flatten()

        # Calculate OBV
        obv = Decimal("0")
        obv_values = []

        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv += Decimal(str(volume[i]))
            elif close[i] < close[i - 1]:
                obv -= Decimal(str(volume[i]))

            obv_values.append(float(obv))

        if len(obv_values) > 0:
            features["obv"] = Decimal(str(obv_values[-1]))

            # OBV moving average
            for window in self.windows:
                if len(obv_values) >= window:
                    obv_ma = Decimal(str(np.mean(obv_values[-window:])))
                    features[f"obv_ma_{window}"] = obv_ma

        return features

    def _calculate_vpt(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate Volume-Price Trend."""
        features = {}

        close = data.select("close").to_numpy().flatten()
        volume = data.select("volume").to_numpy().flatten()

        # Calculate VPT
        vpt = Decimal("0")
        vpt_values = []

        for i in range(1, len(close)):
            if close[i - 1] != 0:
                price_change_pct = (close[i] - close[i - 1]) / close[i - 1]
                vpt += Decimal(str(volume[i] * price_change_pct))
                vpt_values.append(float(vpt))

        if len(vpt_values) > 0:
            features["vpt"] = Decimal(str(vpt_values[-1]))

        return features

    def _calculate_mfi(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate Money Flow Index."""
        features = {}

        high = data.select("high").to_numpy().flatten()
        low = data.select("low").to_numpy().flatten()
        close = data.select("close").to_numpy().flatten()
        volume = data.select("volume").to_numpy().flatten()

        if len(data) < self.mfi_period + 1:
            features["mfi"] = Decimal("50")
            return features

        # Typical price
        typical_price = (high + low + close) / 3

        # Raw money flow
        raw_money_flow = typical_price * volume

        # Positive and negative money flow
        positive_flow = []
        negative_flow = []

        for i in range(1, len(typical_price)):
            if typical_price[i] > typical_price[i - 1]:
                positive_flow.append(raw_money_flow[i])
                negative_flow.append(0)
            elif typical_price[i] < typical_price[i - 1]:
                positive_flow.append(0)
                negative_flow.append(raw_money_flow[i])
            else:
                positive_flow.append(0)
                negative_flow.append(0)

        positive_flow_array = np.array(positive_flow)
        negative_flow_array = np.array(negative_flow)

        # Calculate MFI
        if len(positive_flow_array) >= self.mfi_period:
            positive_mf = np.sum(positive_flow_array[-self.mfi_period:])
            negative_mf = np.sum(negative_flow_array[-self.mfi_period:])

            if negative_mf == 0:
                mfi = Decimal("100")
            else:
                money_ratio = positive_mf / negative_mf
                mfi = Decimal("100") - (Decimal("100") / (Decimal("1") + Decimal(str(money_ratio))))

            features["mfi"] = mfi

        return features

    def _calculate_accumulation_distribution(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate Accumulation/Distribution line."""
        features = {}

        high = data.select("high").to_numpy().flatten()
        low = data.select("low").to_numpy().flatten()
        close = data.select("close").to_numpy().flatten()
        volume = data.select("volume").to_numpy().flatten()

        # Calculate A/D
        ad = Decimal("0")
        ad_values = []

        for i in range(len(close)):
            if high[i] != low[i]:
                money_flow_multiplier = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            else:
                money_flow_multiplier = 0

            money_flow_volume = money_flow_multiplier * volume[i]
            ad += Decimal(str(money_flow_volume))
            ad_values.append(float(ad))

        if len(ad_values) > 0:
            features["accumulation_distribution"] = Decimal(str(ad_values[-1]))

        return features

    def _calculate_chaikin_money_flow(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate Chaikin Money Flow."""
        features = {}

        high = data.select("high").to_numpy().flatten()
        low = data.select("low").to_numpy().flatten()
        close = data.select("close").to_numpy().flatten()
        volume = data.select("volume").to_numpy().flatten()

        window = min(self.windows) if self.windows else 20

        if len(data) < window:
            features["chaikin_mf"] = Decimal("0")
            return features

        # Money flow volume
        mf_volume = []
        for i in range(len(close)):
            if high[i] != low[i]:
                mf_multiplier = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            else:
                mf_multiplier = 0

            mf_volume.append(mf_multiplier * volume[i])

        mf_volume_array = np.array(mf_volume)

        # CMF = sum(MF Volume) / sum(Volume)
        sum_mf_volume = np.sum(mf_volume_array[-window:])
        sum_volume = np.sum(volume[-window:])

        if sum_volume > 0:
            cmf = Decimal(str(sum_mf_volume / sum_volume))
        else:
            cmf = Decimal("0")

        features["chaikin_mf"] = cmf

        return features

    def _calculate_volume_profile(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate volume profile metrics."""
        features = {}

        high = data.select("high").to_numpy().flatten()
        low = data.select("low").to_numpy().flatten()
        volume = data.select("volume").to_numpy().flatten()

        window = max(self.windows) if self.windows else 50

        if len(data) < window:
            return features

        # Price range
        price_min = np.min(low[-window:])
        price_max = np.max(high[-window:])

        # Create price bins
        bins = np.linspace(price_min, price_max, self.profile_bins + 1)
        volume_at_price = np.zeros(self.profile_bins)

        # Accumulate volume in each price bin
        for i in range(-window, 0):
            idx = len(data) + i
            price = (high[idx] + low[idx]) / 2

            # Find bin
            bin_idx = np.digitize(price, bins) - 1
            bin_idx = max(0, min(bin_idx, self.profile_bins - 1))

            volume_at_price[bin_idx] += volume[idx]

        # Point of Control (price level with highest volume)
        poc_idx = np.argmax(volume_at_price)
        poc_price = Decimal(str((bins[poc_idx] + bins[poc_idx + 1]) / 2))

        # Value Area (70% of volume)
        sorted_indices = np.argsort(volume_at_price)[::-1]
        cumsum_volume = 0
        total_volume = np.sum(volume_at_price)
        value_area_indices = []

        for idx in sorted_indices:
            cumsum_volume += volume_at_price[idx]
            value_area_indices.append(idx)

            if cumsum_volume >= 0.7 * total_volume:
                break

        # Value Area High and Low
        vah_idx = max(value_area_indices)
        val_idx = min(value_area_indices)

        vah = Decimal(str(bins[vah_idx + 1]))
        val = Decimal(str(bins[val_idx]))

        features["volume_poc"] = poc_price
        features["volume_vah"] = vah
        features["volume_val"] = val

        # Current price position relative to POC
        current_price = Decimal(str(data["close"][-1]))
        price_vs_poc = (current_price - poc_price) / poc_price if poc_price > 0 else Decimal("0")
        features["price_vs_poc"] = price_vs_poc

        return features

    def _calculate_price_volume_divergence(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate price-volume divergence."""
        features = {}

        close = data.select("close").to_numpy().flatten()
        volume = data.select("volume").to_numpy().flatten()

        window = min(self.windows) if self.windows else 20

        if len(close) < window:
            return features

        # Price trend (simple linear regression)
        x = np.arange(window)
        price_coeffs = np.polyfit(x, close[-window:], 1)
        price_slope = Decimal(str(price_coeffs[0]))

        # Volume trend
        volume_coeffs = np.polyfit(x, volume[-window:], 1)
        volume_slope = Decimal(str(volume_coeffs[0]))

        # Divergence (opposite slopes)
        if (price_slope > 0 and volume_slope < 0) or (price_slope < 0 and volume_slope > 0):
            divergence = Decimal("1")
        else:
            divergence = Decimal("0")

        features["price_volume_divergence"] = divergence
        features["price_trend_slope"] = price_slope
        features["volume_trend_slope"] = volume_slope

        return features

    def _calculate_volume_trends(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate volume trend indicators."""
        features = {}

        volume = data.select("volume").to_numpy().flatten()

        for window in self.windows:
            if len(volume) < window * 2:
                continue

            # Recent vs historical volume
            recent_avg = Decimal(str(np.mean(volume[-window:])))
            historical_avg = Decimal(str(np.mean(volume[-window * 2:-window])))

            if historical_avg > 0:
                volume_change_pct = ((recent_avg - historical_avg) / historical_avg) * Decimal("100")
            else:
                volume_change_pct = Decimal("0")

            features[f"volume_change_pct_{window}"] = volume_change_pct

        return features

    def _get_default_features(self) -> Dict[str, Decimal]:
        """Return default features when insufficient data."""
        features = {}

        for window in self.windows:
            features[f"avg_volume_{window}"] = Decimal("0")
            features[f"volume_ratio_{window}"] = Decimal("1")
            features[f"volume_sma_{window}"] = Decimal("0")
            features[f"obv_ma_{window}"] = Decimal("0")
            features[f"volume_change_pct_{window}"] = Decimal("0")

        features["current_volume"] = Decimal("0")
        features["is_high_volume"] = Decimal("0")
        features["volume_std"] = Decimal("0")
        features["obv"] = Decimal("0")
        features["vpt"] = Decimal("0")
        features["mfi"] = Decimal("50")
        features["accumulation_distribution"] = Decimal("0")
        features["chaikin_mf"] = Decimal("0")
        features["price_volume_divergence"] = Decimal("0")

        return features
