"""Feature Extraction Engine for Trading Signals.

This module implements a comprehensive feature extraction system for trading
that combines technical indicators, market microstructure, and derived features.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Any, Set
from datetime import datetime
import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class FeatureExtractor:
    """Production-ready feature extraction for trading data.

    Extracts comprehensive features including:
    - Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
    - Price action features (returns, volatility, gaps)
    - Volume features (volume ratios, VWAP, OBV)
    - Market microstructure (bid-ask spread, order flow)
    - Time-based features (hour, day of week, month)
    - Lag features for temporal patterns

    Attributes:
        config: Configuration dictionary
        enabled_features: Set of feature groups to extract
        technical_indicators: Technical indicator configurations
        lag_periods: Periods for lag features
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize feature extractor.

        Args:
            config: Configuration with keys:
                - enabled_features: List of feature groups to extract
                - technical_indicators: Dict of indicator configs
                - lag_periods: List of lag periods
                - volume_features: Whether to include volume features
                - microstructure_features: Whether to include microstructure
                - time_features: Whether to include time-based features
                - return_periods: List of return calculation periods

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self._validate_config()

        self.enabled_features: Set[str] = set(
            config.get("enabled_features", ["technical", "price", "volume", "time"])
        )
        self.technical_indicators: Dict[str, Any] = config.get(
            "technical_indicators",
            {
                "sma_periods": [10, 20, 50, 200],
                "ema_periods": [9, 12, 26],
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "bb_period": 20,
                "bb_std": 2,
            }
        )
        self.lag_periods: List[int] = config.get("lag_periods", [1, 2, 3, 5, 10])
        self.return_periods: List[int] = config.get("return_periods", [1, 5, 10, 20])

        # Feature statistics for monitoring
        self._extraction_count: int = 0
        self._feature_stats: Dict[str, Any] = {}

        logger.info(
            "feature_extractor_initialized",
            enabled_features=list(self.enabled_features),
            indicators=list(self.technical_indicators.keys())
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required parameters missing or invalid
        """
        if "enabled_features" not in self.config:
            raise ValueError("enabled_features required in config")

        valid_feature_groups = {"technical", "price", "volume", "microstructure", "time", "lag"}
        enabled = set(self.config["enabled_features"])
        invalid = enabled - valid_feature_groups

        if invalid:
            raise ValueError(f"Invalid feature groups: {invalid}")

    def extract(self, data: pl.DataFrame) -> pl.DataFrame:
        """Extract all configured features from market data.

        Args:
            data: Polars DataFrame with columns:
                - timestamp: UTC datetime
                - open: Opening price (Decimal)
                - high: High price (Decimal)
                - low: Low price (Decimal)
                - close: Close price (Decimal)
                - volume: Trading volume (Decimal)
                Optional:
                - bid: Best bid price
                - ask: Best ask price
                - bid_volume: Bid volume
                - ask_volume: Ask volume

        Returns:
            DataFrame with original data plus extracted features

        Raises:
            ValueError: If data is invalid or missing required columns
        """
        try:
            self._validate_input_data(data)

            logger.debug("extracting_features", rows=len(data), groups=list(self.enabled_features))

            result = data.clone()

            # Extract feature groups
            if "price" in self.enabled_features:
                result = self._extract_price_features(result)

            if "technical" in self.enabled_features:
                result = self._extract_technical_indicators(result)

            if "volume" in self.enabled_features:
                result = self._extract_volume_features(result)

            if "microstructure" in self.enabled_features:
                result = self._extract_microstructure_features(result)

            if "time" in self.enabled_features:
                result = self._extract_time_features(result)

            if "lag" in self.enabled_features:
                result = self._extract_lag_features(result)

            # Track statistics
            self._extraction_count += 1
            feature_count = len(result.columns) - len(data.columns)

            logger.debug(
                "features_extracted",
                total_features=feature_count,
                extraction_count=self._extraction_count
            )

            return result

        except Exception as e:
            logger.error("feature_extraction_failed", error=str(e))
            raise

    def _validate_input_data(self, data: pl.DataFrame) -> None:
        """Validate input DataFrame.

        Args:
            data: Input DataFrame to validate

        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(data, pl.DataFrame):
            raise ValueError(f"data must be Polars DataFrame, got {type(data)}")

        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if len(data) == 0:
            raise ValueError("DataFrame is empty")

    def _extract_price_features(self, data: pl.DataFrame) -> pl.DataFrame:
        """Extract price-based features.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with price features added
        """
        result = data

        # Basic price features
        result = result.with_columns([
            # Price range
            (pl.col("high") - pl.col("low")).alias("price_range"),

            # Body size (close - open)
            (pl.col("close") - pl.col("open")).alias("candle_body"),

            # Upper shadow
            (pl.col("high") - pl.max_horizontal(["open", "close"])).alias("upper_shadow"),

            # Lower shadow
            (pl.min_horizontal(["open", "close"]) - pl.col("low")).alias("lower_shadow"),

            # Typical price
            ((pl.col("high") + pl.col("low") + pl.col("close")) / Decimal("3")).alias("typical_price"),

            # Weighted close
            ((pl.col("high") + pl.col("low") + pl.col("close") * Decimal("2")) / Decimal("4")).alias("weighted_close"),
        ])

        # Returns for different periods
        for period in self.return_periods:
            result = result.with_columns([
                # Simple return
                ((pl.col("close") - pl.col("close").shift(period)) / pl.col("close").shift(period))
                .alias(f"return_{period}"),

                # Log return
                (pl.col("close") / pl.col("close").shift(period)).log().alias(f"log_return_{period}"),
            ])

        # Price momentum
        result = result.with_columns([
            (pl.col("close") > pl.col("close").shift(1)).cast(pl.Int8).alias("price_up"),
            (pl.col("high") > pl.col("high").shift(1)).cast(pl.Int8).alias("higher_high"),
            (pl.col("low") > pl.col("low").shift(1)).cast(pl.Int8).alias("higher_low"),
        ])

        # Volatility (rolling std of returns)
        for window in [5, 10, 20]:
            result = result.with_columns([
                pl.col("return_1")
                .rolling_std(window_size=window)
                .alias(f"volatility_{window}"),
            ])

        return result

    def _extract_technical_indicators(self, data: pl.DataFrame) -> pl.DataFrame:
        """Extract technical indicator features.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with technical indicators added
        """
        result = data

        # Simple Moving Averages
        for period in self.technical_indicators["sma_periods"]:
            result = result.with_columns([
                pl.col("close").rolling_mean(window_size=period).alias(f"sma_{period}"),
            ])

            # Price relative to SMA
            result = result.with_columns([
                ((pl.col("close") - pl.col(f"sma_{period}")) / pl.col(f"sma_{period}"))
                .alias(f"close_sma_{period}_ratio"),
            ])

        # Exponential Moving Averages
        for period in self.technical_indicators["ema_periods"]:
            alpha = Decimal("2") / Decimal(str(period + 1))
            result = result.with_columns([
                pl.col("close").ewm_mean(alpha=float(alpha)).alias(f"ema_{period}"),
            ])

        # RSI (Relative Strength Index)
        rsi_period = self.technical_indicators["rsi_period"]
        result = self._calculate_rsi(result, rsi_period)

        # MACD
        macd_fast = self.technical_indicators["macd_fast"]
        macd_slow = self.technical_indicators["macd_slow"]
        macd_signal = self.technical_indicators["macd_signal"]
        result = self._calculate_macd(result, macd_fast, macd_slow, macd_signal)

        # Bollinger Bands
        bb_period = self.technical_indicators["bb_period"]
        bb_std = self.technical_indicators["bb_std"]
        result = self._calculate_bollinger_bands(result, bb_period, bb_std)

        return result

    def _calculate_rsi(self, data: pl.DataFrame, period: int) -> pl.DataFrame:
        """Calculate RSI indicator.

        Args:
            data: Input DataFrame
            period: RSI period

        Returns:
            DataFrame with RSI added
        """
        result = data

        # Calculate price changes
        result = result.with_columns([
            (pl.col("close") - pl.col("close").shift(1)).alias("price_change"),
        ])

        # Separate gains and losses
        result = result.with_columns([
            pl.when(pl.col("price_change") > 0)
            .then(pl.col("price_change"))
            .otherwise(Decimal("0"))
            .alias("gain"),

            pl.when(pl.col("price_change") < 0)
            .then(pl.col("price_change").abs())
            .otherwise(Decimal("0"))
            .alias("loss"),
        ])

        # Calculate average gain and loss
        result = result.with_columns([
            pl.col("gain").rolling_mean(window_size=period).alias("avg_gain"),
            pl.col("loss").rolling_mean(window_size=period).alias("avg_loss"),
        ])

        # Calculate RSI
        result = result.with_columns([
            (
                Decimal("100") -
                (Decimal("100") / (Decimal("1") + (pl.col("avg_gain") / pl.col("avg_loss"))))
            ).alias(f"rsi_{period}"),
        ])

        # Clean up temporary columns
        result = result.drop(["price_change", "gain", "loss", "avg_gain", "avg_loss"])

        return result

    def _calculate_macd(
        self, data: pl.DataFrame, fast: int, slow: int, signal: int
    ) -> pl.DataFrame:
        """Calculate MACD indicator.

        Args:
            data: Input DataFrame
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            DataFrame with MACD added
        """
        result = data

        # Calculate EMAs if not already present
        fast_alpha = float(Decimal("2") / Decimal(str(fast + 1)))
        slow_alpha = float(Decimal("2") / Decimal(str(slow + 1)))
        signal_alpha = float(Decimal("2") / Decimal(str(signal + 1)))

        if f"ema_{fast}" not in result.columns:
            result = result.with_columns([
                pl.col("close").ewm_mean(alpha=fast_alpha).alias(f"ema_{fast}"),
            ])

        if f"ema_{slow}" not in result.columns:
            result = result.with_columns([
                pl.col("close").ewm_mean(alpha=slow_alpha).alias(f"ema_{slow}"),
            ])

        # MACD line
        result = result.with_columns([
            (pl.col(f"ema_{fast}") - pl.col(f"ema_{slow}")).alias("macd_line"),
        ])

        # Signal line
        result = result.with_columns([
            pl.col("macd_line").ewm_mean(alpha=signal_alpha).alias("macd_signal"),
        ])

        # MACD histogram
        result = result.with_columns([
            (pl.col("macd_line") - pl.col("macd_signal")).alias("macd_histogram"),
        ])

        return result

    def _calculate_bollinger_bands(
        self, data: pl.DataFrame, period: int, std_dev: int
    ) -> pl.DataFrame:
        """Calculate Bollinger Bands.

        Args:
            data: Input DataFrame
            period: Moving average period
            std_dev: Number of standard deviations

        Returns:
            DataFrame with Bollinger Bands added
        """
        result = data

        # Middle band (SMA)
        if f"sma_{period}" not in result.columns:
            result = result.with_columns([
                pl.col("close").rolling_mean(window_size=period).alias(f"sma_{period}"),
            ])

        # Standard deviation
        result = result.with_columns([
            pl.col("close").rolling_std(window_size=period).alias("bb_std"),
        ])

        # Upper and lower bands
        result = result.with_columns([
            (pl.col(f"sma_{period}") + pl.col("bb_std") * Decimal(str(std_dev)))
            .alias(f"bb_upper_{period}"),

            (pl.col(f"sma_{period}") - pl.col("bb_std") * Decimal(str(std_dev)))
            .alias(f"bb_lower_{period}"),
        ])

        # %B (position within bands)
        result = result.with_columns([
            (
                (pl.col("close") - pl.col(f"bb_lower_{period}")) /
                (pl.col(f"bb_upper_{period}") - pl.col(f"bb_lower_{period}"))
            ).alias(f"bb_percent_{period}"),
        ])

        # Bandwidth
        result = result.with_columns([
            (
                (pl.col(f"bb_upper_{period}") - pl.col(f"bb_lower_{period}")) /
                pl.col(f"sma_{period}")
            ).alias(f"bb_width_{period}"),
        ])

        # Clean up
        result = result.drop(["bb_std"])

        return result

    def _extract_volume_features(self, data: pl.DataFrame) -> pl.DataFrame:
        """Extract volume-based features.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with volume features added
        """
        result = data

        # Volume statistics
        for window in [5, 10, 20]:
            result = result.with_columns([
                pl.col("volume").rolling_mean(window_size=window).alias(f"volume_sma_{window}"),
            ])

            # Volume ratio
            result = result.with_columns([
                (pl.col("volume") / pl.col(f"volume_sma_{window}"))
                .alias(f"volume_ratio_{window}"),
            ])

        # VWAP (Volume Weighted Average Price)
        result = result.with_columns([
            ((pl.col("typical_price") * pl.col("volume")).cumsum() / pl.col("volume").cumsum())
            .alias("vwap"),
        ])

        # OBV (On Balance Volume)
        result = result.with_columns([
            pl.when(pl.col("close") > pl.col("close").shift(1))
            .then(pl.col("volume"))
            .when(pl.col("close") < pl.col("close").shift(1))
            .then(-pl.col("volume"))
            .otherwise(Decimal("0"))
            .cumsum()
            .alias("obv"),
        ])

        # Volume momentum
        result = result.with_columns([
            (pl.col("volume") > pl.col("volume").shift(1)).cast(pl.Int8).alias("volume_increasing"),
        ])

        return result

    def _extract_microstructure_features(self, data: pl.DataFrame) -> pl.DataFrame:
        """Extract market microstructure features.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with microstructure features added
        """
        # Check if bid/ask data available
        if "bid" not in data.columns or "ask" not in data.columns:
            logger.debug("bid_ask_data_not_available_skipping_microstructure")
            return data

        result = data

        # Bid-Ask spread
        result = result.with_columns([
            (pl.col("ask") - pl.col("bid")).alias("spread_absolute"),
            ((pl.col("ask") - pl.col("bid")) / pl.col("bid")).alias("spread_relative"),
        ])

        # Mid price
        result = result.with_columns([
            ((pl.col("bid") + pl.col("ask")) / Decimal("2")).alias("mid_price"),
        ])

        # Price position in spread
        result = result.with_columns([
            (
                (pl.col("close") - pl.col("bid")) /
                (pl.col("ask") - pl.col("bid"))
            ).alias("price_position_in_spread"),
        ])

        # Order flow imbalance (if volume data available)
        if "bid_volume" in data.columns and "ask_volume" in data.columns:
            result = result.with_columns([
                (
                    (pl.col("bid_volume") - pl.col("ask_volume")) /
                    (pl.col("bid_volume") + pl.col("ask_volume"))
                ).alias("order_flow_imbalance"),
            ])

        return result

    def _extract_time_features(self, data: pl.DataFrame) -> pl.DataFrame:
        """Extract time-based features.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with time features added
        """
        result = data

        # Extract time components
        result = result.with_columns([
            pl.col("timestamp").dt.hour().alias("hour"),
            pl.col("timestamp").dt.weekday().alias("day_of_week"),
            pl.col("timestamp").dt.month().alias("month"),
            pl.col("timestamp").dt.day().alias("day_of_month"),
        ])

        # Trading session indicators
        result = result.with_columns([
            # Market open hours (approximate for crypto - 24/7, but lower volume certain hours)
            ((pl.col("hour") >= 9) & (pl.col("hour") <= 16)).cast(pl.Int8).alias("trading_hours"),

            # Weekend indicator
            (pl.col("day_of_week") >= 5).cast(pl.Int8).alias("is_weekend"),
        ])

        # Cyclical encoding for hour (sin/cos)
        result = result.with_columns([
            (pl.col("hour") * Decimal("2") * Decimal(str(np.pi)) / Decimal("24"))
            .sin()
            .alias("hour_sin"),

            (pl.col("hour") * Decimal("2") * Decimal(str(np.pi)) / Decimal("24"))
            .cos()
            .alias("hour_cos"),
        ])

        return result

    def _extract_lag_features(self, data: pl.DataFrame) -> pl.DataFrame:
        """Extract lagged features for temporal patterns.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with lag features added
        """
        result = data

        # Lag important features
        lag_columns = ["close", "volume", "return_1"]

        for col in lag_columns:
            if col in result.columns:
                for lag in self.lag_periods:
                    result = result.with_columns([
                        pl.col(col).shift(lag).alias(f"{col}_lag_{lag}"),
                    ])

        return result

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names that will be extracted.

        Returns:
            List of feature column names
        """
        # This is a comprehensive list - actual features depend on enabled_features
        features = []

        if "price" in self.enabled_features:
            features.extend([
                "price_range", "candle_body", "upper_shadow", "lower_shadow",
                "typical_price", "weighted_close", "price_up", "higher_high", "higher_low"
            ])

        if "technical" in self.enabled_features:
            features.extend([
                "macd_line", "macd_signal", "macd_histogram"
            ])

        if "volume" in self.enabled_features:
            features.extend(["vwap", "obv", "volume_increasing"])

        if "time" in self.enabled_features:
            features.extend([
                "hour", "day_of_week", "month", "day_of_month",
                "trading_hours", "is_weekend", "hour_sin", "hour_cos"
            ])

        return features

    def get_statistics(self) -> Dict[str, Any]:
        """Get feature extraction statistics.

        Returns:
            Dictionary with extraction statistics
        """
        return {
            "extraction_count": self._extraction_count,
            "enabled_features": list(self.enabled_features),
            "technical_indicators": list(self.technical_indicators.keys()),
            "lag_periods": self.lag_periods,
            "return_periods": self.return_periods,
        }
