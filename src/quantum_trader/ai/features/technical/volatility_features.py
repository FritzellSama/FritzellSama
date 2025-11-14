"""
Volatility Feature Extraction

Production-ready volatility indicators and features for trading models.
Includes historical volatility, implied volatility proxies, and volatility regimes.
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


class VolatilityFeatures:
    """
    Extract volatility-based features from price data.

    Calculates various volatility measures including historical volatility,
    ATR, Bollinger Bands, Keltner Channels, and volatility regimes.

    Attributes:
        config: Configuration dictionary
        windows: List of lookback windows for calculations
        volatility_percentiles: Percentiles for regime classification

    Example:
        >>> config = {"features": {"volatility": {"windows": [10, 20, 50]}}}
        >>> vol_features = VolatilityFeatures(config)
        >>> features = await vol_features.extract_features(price_data_df)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize volatility features extractor.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        vol_config = self.config.get("features", {}).get("volatility", {})

        # Calculation windows
        self.windows: List[int] = vol_config.get(
            "windows",
            [int(x) for x in os.getenv("VOLATILITY_WINDOWS", "10,20,50,100").split(",")]
        )

        # ATR multipliers
        self.atr_multipliers: List[float] = vol_config.get(
            "atr_multipliers",
            [float(x) for x in os.getenv("ATR_MULTIPLIERS", "1.5,2.0,3.0").split(",")]
        )

        # Bollinger Bands parameters
        self.bb_std_multiplier: Decimal = Decimal(
            str(vol_config.get("bb_std_multiplier", os.getenv("BB_STD_MULTIPLIER", "2.0")))
        )

        # Volatility regime percentiles
        self.regime_percentiles: List[int] = vol_config.get(
            "regime_percentiles",
            [int(x) for x in os.getenv("VOL_REGIME_PERCENTILES", "33,67").split(",")]
        )

        # Annualization factor (252 trading days)
        self.annualization_factor: Decimal = Decimal(
            str(vol_config.get("annualization_factor", os.getenv("VOL_ANNUALIZATION_FACTOR", "252")))
        )

        logger.info(
            "volatility_features_initialized",
            windows=self.windows,
            atr_multipliers=self.atr_multipliers
        )

    def _validate_config(self) -> None:
        """Validate configuration."""
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        vol_config = self.config.get("features", {}).get("volatility", {})

        if vol_config:
            windows = vol_config.get("windows", [10, 20, 50])
            if not all(w > 0 for w in windows):
                raise ValueError("All windows must be positive")

    async def extract_features(
        self,
        price_data: pl.DataFrame,
        symbol: Optional[str] = None
    ) -> Dict[str, Decimal]:
        """
        Extract volatility features from price data.

        Args:
            price_data: DataFrame with OHLC data
                Required columns: timestamp, open, high, low, close
            symbol: Trading symbol (for logging)

        Returns:
            Dictionary of volatility features

        Raises:
            ValueError: If data schema is invalid

        Example:
            >>> features = await vol_features.extract_features(ohlc_df, symbol="BTC/USDT")
        """
        try:
            # Validate schema
            required_cols = ["timestamp", "open", "high", "low", "close"]
            if not all(col in price_data.columns for col in required_cols):
                raise ValueError(f"DataFrame missing required columns: {required_cols}")

            if len(price_data) < max(self.windows):
                logger.warning(
                    "insufficient_data_for_volatility",
                    symbol=symbol,
                    data_length=len(price_data),
                    min_required=max(self.windows)
                )
                return self._get_default_features()

            # Sort by timestamp
            data = price_data.sort("timestamp")

            # Extract features
            features = {}

            # Historical volatility (multiple windows)
            features.update(self._calculate_historical_volatility(data))

            # ATR (Average True Range)
            features.update(self._calculate_atr(data))

            # Bollinger Bands
            features.update(self._calculate_bollinger_bands(data))

            # Keltner Channels
            features.update(self._calculate_keltner_channels(data))

            # Parkinson volatility (high-low)
            features.update(self._calculate_parkinson_volatility(data))

            # Garman-Klass volatility
            features.update(self._calculate_garman_klass_volatility(data))

            # Volatility regimes
            features.update(self._calculate_volatility_regimes(data))

            # Volatility metrics
            features.update(self._calculate_volatility_metrics(data))

            logger.debug(
                "volatility_features_extracted",
                symbol=symbol,
                num_features=len(features)
            )

            return features

        except Exception as e:
            logger.error("volatility_feature_extraction_failed", symbol=symbol, error=str(e))
            raise

    def _calculate_historical_volatility(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate historical volatility for multiple windows."""
        features = {}

        close_prices = data.select("close").to_numpy().flatten()
        returns = np.diff(np.log(close_prices))

        for window in self.windows:
            if len(returns) < window:
                features[f"hist_vol_{window}"] = Decimal("0")
                continue

            # Rolling volatility
            if len(returns) >= window:
                rolling_std = Decimal(str(np.std(returns[-window:])))
                annualized_vol = rolling_std * (self.annualization_factor ** Decimal("0.5"))
                features[f"hist_vol_{window}"] = annualized_vol

        return features

    def _calculate_atr(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate Average True Range."""
        features = {}

        high = data.select("high").to_numpy().flatten()
        low = data.select("low").to_numpy().flatten()
        close = data.select("close").to_numpy().flatten()

        # Calculate true range
        tr = []
        for i in range(1, len(high)):
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i - 1])
            l_pc = abs(low[i] - close[i - 1])
            tr.append(max(h_l, h_pc, l_pc))

        tr_array = np.array(tr)

        for window in self.windows:
            if len(tr_array) < window:
                features[f"atr_{window}"] = Decimal("0")
                features[f"atr_pct_{window}"] = Decimal("0")
                continue

            # ATR (Simple Moving Average of TR)
            atr = Decimal(str(np.mean(tr_array[-window:])))
            current_price = Decimal(str(close[-1]))

            # ATR as percentage of price
            atr_pct = (atr / current_price) * Decimal("100") if current_price > 0 else Decimal("0")

            features[f"atr_{window}"] = atr
            features[f"atr_pct_{window}"] = atr_pct

        return features

    def _calculate_bollinger_bands(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate Bollinger Bands features."""
        features = {}

        close_prices = data.select("close").to_numpy().flatten()
        current_price = Decimal(str(close_prices[-1]))

        for window in self.windows:
            if len(close_prices) < window:
                continue

            # Calculate middle band (SMA)
            sma = Decimal(str(np.mean(close_prices[-window:])))

            # Calculate standard deviation
            std = Decimal(str(np.std(close_prices[-window:])))

            # Upper and lower bands
            upper_band = sma + (std * self.bb_std_multiplier)
            lower_band = sma - (std * self.bb_std_multiplier)

            # Band width
            band_width = ((upper_band - lower_band) / sma) * Decimal("100") if sma > 0 else Decimal("0")

            # Price position in bands (0 = lower, 0.5 = middle, 1 = upper)
            if upper_band != lower_band:
                bb_position = (current_price - lower_band) / (upper_band - lower_band)
            else:
                bb_position = Decimal("0.5")

            features[f"bb_width_{window}"] = band_width
            features[f"bb_position_{window}"] = bb_position
            features[f"bb_upper_{window}"] = upper_band
            features[f"bb_lower_{window}"] = lower_band

        return features

    def _calculate_keltner_channels(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate Keltner Channels."""
        features = {}

        close_prices = data.select("close").to_numpy().flatten()
        high = data.select("high").to_numpy().flatten()
        low = data.select("low").to_numpy().flatten()

        current_price = Decimal(str(close_prices[-1]))

        # Calculate ATR for Keltner
        tr = []
        for i in range(1, len(high)):
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close_prices[i - 1])
            l_pc = abs(low[i] - close_prices[i - 1])
            tr.append(max(h_l, h_pc, l_pc))

        tr_array = np.array(tr)

        for window in self.windows:
            if len(close_prices) < window or len(tr_array) < window:
                continue

            # EMA of close
            ema = Decimal(str(np.mean(close_prices[-window:])))  # Simplified as SMA

            # ATR
            atr = Decimal(str(np.mean(tr_array[-window:])))

            # Keltner channels (typically 2 * ATR)
            upper_keltner = ema + (Decimal("2") * atr)
            lower_keltner = ema - (Decimal("2") * atr)

            # Channel width
            kc_width = ((upper_keltner - lower_keltner) / ema) * Decimal("100") if ema > 0 else Decimal("0")

            # Position in channel
            if upper_keltner != lower_keltner:
                kc_position = (current_price - lower_keltner) / (upper_keltner - lower_keltner)
            else:
                kc_position = Decimal("0.5")

            features[f"kc_width_{window}"] = kc_width
            features[f"kc_position_{window}"] = kc_position

        return features

    def _calculate_parkinson_volatility(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate Parkinson volatility (high-low estimator)."""
        features = {}

        high = data.select("high").to_numpy().flatten()
        low = data.select("low").to_numpy().flatten()

        for window in self.windows:
            if len(high) < window:
                continue

            # Parkinson volatility
            hl_ratios = np.log(high[-window:] / low[-window:])
            parkinson_var = np.mean(hl_ratios ** 2) / (4 * np.log(2))
            parkinson_vol = Decimal(str(np.sqrt(parkinson_var)))

            # Annualize
            annualized = parkinson_vol * (self.annualization_factor ** Decimal("0.5"))

            features[f"parkinson_vol_{window}"] = annualized

        return features

    def _calculate_garman_klass_volatility(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate Garman-Klass volatility estimator."""
        features = {}

        open_prices = data.select("open").to_numpy().flatten()
        high = data.select("high").to_numpy().flatten()
        low = data.select("low").to_numpy().flatten()
        close = data.select("close").to_numpy().flatten()

        for window in self.windows:
            if len(open_prices) < window:
                continue

            # Garman-Klass estimator
            hl_component = 0.5 * (np.log(high[-window:] / low[-window:]) ** 2)
            oc_component = (2 * np.log(2) - 1) * (np.log(close[-window:] / open_prices[-window:]) ** 2)

            gk_var = np.mean(hl_component - oc_component)
            gk_vol = Decimal(str(np.sqrt(gk_var)))

            # Annualize
            annualized = gk_vol * (self.annualization_factor ** Decimal("0.5"))

            features[f"gk_vol_{window}"] = annualized

        return features

    def _calculate_volatility_regimes(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Identify volatility regimes."""
        features = {}

        close_prices = data.select("close").to_numpy().flatten()
        returns = np.diff(np.log(close_prices))

        # Use longest window for regime classification
        window = max(self.windows)

        if len(returns) >= window:
            # Calculate rolling volatility
            rolling_vols = []
            for i in range(len(returns) - window + 1):
                vol = np.std(returns[i:i + window])
                rolling_vols.append(vol)

            rolling_vols_array = np.array(rolling_vols)

            # Calculate percentiles
            low_threshold = np.percentile(rolling_vols_array, self.regime_percentiles[0])
            high_threshold = np.percentile(rolling_vols_array, self.regime_percentiles[1])

            # Current volatility
            current_vol = Decimal(str(np.std(returns[-window:])))

            # Classify regime (0 = low, 1 = medium, 2 = high)
            if current_vol < Decimal(str(low_threshold)):
                regime = Decimal("0")
            elif current_vol > Decimal(str(high_threshold)):
                regime = Decimal("2")
            else:
                regime = Decimal("1")

            features["volatility_regime"] = regime
            features["vol_regime_low_threshold"] = Decimal(str(low_threshold))
            features["vol_regime_high_threshold"] = Decimal(str(high_threshold))

        return features

    def _calculate_volatility_metrics(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate additional volatility metrics."""
        features = {}

        close_prices = data.select("close").to_numpy().flatten()

        if len(close_prices) < 2:
            return features

        returns = np.diff(np.log(close_prices))

        # Volatility of volatility
        window = max(self.windows) if self.windows else 20

        if len(returns) >= window:
            rolling_vols = []
            for i in range(len(returns) - window + 1):
                vol = np.std(returns[i:i + window])
                rolling_vols.append(vol)

            if len(rolling_vols) > 1:
                vol_of_vol = Decimal(str(np.std(rolling_vols)))
                features["volatility_of_volatility"] = vol_of_vol

        # Upside vs downside volatility
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        if len(positive_returns) > 0:
            upside_vol = Decimal(str(np.std(positive_returns)))
            features["upside_volatility"] = upside_vol

        if len(negative_returns) > 0:
            downside_vol = Decimal(str(np.std(negative_returns)))
            features["downside_volatility"] = downside_vol

        # Volatility skew
        if len(positive_returns) > 0 and len(negative_returns) > 0:
            upside_vol_val = float(features.get("upside_volatility", Decimal("1")))
            downside_vol_val = float(features.get("downside_volatility", Decimal("1")))

            vol_skew = Decimal(str((downside_vol_val - upside_vol_val) / (downside_vol_val + upside_vol_val)))
            features["volatility_skew"] = vol_skew

        return features

    def _get_default_features(self) -> Dict[str, Decimal]:
        """Return default features when insufficient data."""
        features = {}

        for window in self.windows:
            features[f"hist_vol_{window}"] = Decimal("0")
            features[f"atr_{window}"] = Decimal("0")
            features[f"atr_pct_{window}"] = Decimal("0")
            features[f"bb_width_{window}"] = Decimal("0")
            features[f"bb_position_{window}"] = Decimal("0.5")
            features[f"kc_width_{window}"] = Decimal("0")
            features[f"kc_position_{window}"] = Decimal("0.5")
            features[f"parkinson_vol_{window}"] = Decimal("0")
            features[f"gk_vol_{window}"] = Decimal("0")

        features["volatility_regime"] = Decimal("1")  # Medium regime
        features["volatility_of_volatility"] = Decimal("0")
        features["upside_volatility"] = Decimal("0")
        features["downside_volatility"] = Decimal("0")
        features["volatility_skew"] = Decimal("0")

        return features
