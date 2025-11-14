"""
Technical Pattern Recognition Features.

This module detects and extracts features from common technical analysis
patterns such as candlestick patterns, chart patterns, and formations.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class CandlestickPattern(Enum):
    """Candlestick pattern types."""
    DOJI = "doji"
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    HARAMI_BULLISH = "harami_bullish"
    HARAMI_BEARISH = "harami_bearish"


class ChartPattern(Enum):
    """Chart pattern types."""
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    TRIANGLE_SYMMETRICAL = "triangle_symmetrical"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"


@dataclass
class PatternDetection:
    """Pattern detection result.

    Attributes:
        pattern_type: Type of pattern detected
        confidence: Detection confidence (0.0 to 1.0)
        start_idx: Start index in data
        end_idx: End index in data
        metadata: Additional pattern metadata
    """
    pattern_type: str
    confidence: Decimal
    start_idx: int
    end_idx: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class PatternFeatures:
    """Production-ready technical pattern feature extractor.

    Detects candlestick patterns, chart patterns, and other
    technical formations for trading signal generation.

    Attributes:
        config: Configuration dictionary
        min_confidence: Minimum confidence threshold for detection
        lookback_window: Lookback window for pattern detection
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize pattern feature extractor.

        Args:
            config: Configuration dictionary containing:
                - features.patterns.min_confidence
                - features.patterns.lookback_window
                - features.patterns.body_ratio_threshold
                - features.patterns.shadow_ratio_threshold
        """
        self.config = config
        self._validate_config()

        pattern_config = self.config["features"]["patterns"]
        self.min_confidence = Decimal(str(pattern_config["min_confidence"]))
        self.lookback_window = pattern_config["lookback_window"]
        self.body_ratio_threshold = Decimal(str(pattern_config["body_ratio_threshold"]))
        self.shadow_ratio_threshold = Decimal(str(pattern_config["shadow_ratio_threshold"]))

        logger.info("pattern_features_initialized",
                   min_confidence=str(self.min_confidence),
                   lookback_window=self.lookback_window)

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required config parameters are missing
        """
        required_keys = [
            "features.patterns.min_confidence",
            "features.patterns.lookback_window",
            "features.patterns.body_ratio_threshold",
            "features.patterns.shadow_ratio_threshold"
        ]

        for key in required_keys:
            parts = key.split(".")
            current = self.config
            for part in parts:
                if part not in current:
                    raise ValueError(f"Missing required config key: {key}")
                current = current[part]

    def _calculate_candle_metrics(
        self,
        open_price: Decimal,
        high_price: Decimal,
        low_price: Decimal,
        close_price: Decimal
    ) -> Dict[str, Decimal]:
        """Calculate candlestick metrics.

        Args:
            open_price: Open price
            high_price: High price
            low_price: Low price
            close_price: Close price

        Returns:
            Dictionary of candle metrics
        """
        body = abs(close_price - open_price)
        range_total = high_price - low_price

        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price

        body_ratio = body / range_total if range_total > 0 else Decimal("0")
        upper_shadow_ratio = upper_shadow / range_total if range_total > 0 else Decimal("0")
        lower_shadow_ratio = lower_shadow / range_total if range_total > 0 else Decimal("0")

        is_bullish = close_price > open_price
        is_bearish = close_price < open_price

        return {
            "body": body,
            "range": range_total,
            "upper_shadow": upper_shadow,
            "lower_shadow": lower_shadow,
            "body_ratio": body_ratio,
            "upper_shadow_ratio": upper_shadow_ratio,
            "lower_shadow_ratio": lower_shadow_ratio,
            "is_bullish": Decimal("1") if is_bullish else Decimal("0"),
            "is_bearish": Decimal("1") if is_bearish else Decimal("0")
        }

    async def detect_doji(
        self,
        df: pl.DataFrame,
        idx: int
    ) -> Optional[PatternDetection]:
        """Detect Doji pattern.

        Args:
            df: Polars DataFrame with OHLC data
            idx: Index to check

        Returns:
            Pattern detection or None
        """
        try:
            if idx >= len(df):
                return None

            row = df.row(idx, named=True)
            open_price = Decimal(str(row["open"]))
            high_price = Decimal(str(row["high"]))
            low_price = Decimal(str(row["low"]))
            close_price = Decimal(str(row["close"]))

            metrics = self._calculate_candle_metrics(
                open_price, high_price, low_price, close_price
            )

            # Doji: very small body relative to range
            if metrics["body_ratio"] < self.body_ratio_threshold:
                confidence = Decimal("1") - metrics["body_ratio"]

                if confidence >= self.min_confidence:
                    return PatternDetection(
                        pattern_type=CandlestickPattern.DOJI.value,
                        confidence=confidence,
                        start_idx=idx,
                        end_idx=idx,
                        metadata={"body_ratio": str(metrics["body_ratio"])}
                    )

            return None

        except Exception as e:
            logger.error("doji_detection_failed", idx=idx, error=str(e))
            return None

    async def detect_hammer(
        self,
        df: pl.DataFrame,
        idx: int
    ) -> Optional[PatternDetection]:
        """Detect Hammer pattern.

        Args:
            df: Polars DataFrame with OHLC data
            idx: Index to check

        Returns:
            Pattern detection or None
        """
        try:
            if idx >= len(df):
                return None

            row = df.row(idx, named=True)
            open_price = Decimal(str(row["open"]))
            high_price = Decimal(str(row["high"]))
            low_price = Decimal(str(row["low"]))
            close_price = Decimal(str(row["close"]))

            metrics = self._calculate_candle_metrics(
                open_price, high_price, low_price, close_price
            )

            # Hammer: long lower shadow, small body, small/no upper shadow
            if (metrics["lower_shadow_ratio"] > self.shadow_ratio_threshold and
                metrics["body_ratio"] < Decimal("0.3") and
                metrics["upper_shadow_ratio"] < Decimal("0.1")):

                confidence = metrics["lower_shadow_ratio"]

                if confidence >= self.min_confidence:
                    return PatternDetection(
                        pattern_type=CandlestickPattern.HAMMER.value,
                        confidence=confidence,
                        start_idx=idx,
                        end_idx=idx,
                        metadata={
                            "lower_shadow_ratio": str(metrics["lower_shadow_ratio"]),
                            "body_ratio": str(metrics["body_ratio"])
                        }
                    )

            return None

        except Exception as e:
            logger.error("hammer_detection_failed", idx=idx, error=str(e))
            return None

    async def detect_shooting_star(
        self,
        df: pl.DataFrame,
        idx: int
    ) -> Optional[PatternDetection]:
        """Detect Shooting Star pattern.

        Args:
            df: Polars DataFrame with OHLC data
            idx: Index to check

        Returns:
            Pattern detection or None
        """
        try:
            if idx >= len(df):
                return None

            row = df.row(idx, named=True)
            open_price = Decimal(str(row["open"]))
            high_price = Decimal(str(row["high"]))
            low_price = Decimal(str(row["low"]))
            close_price = Decimal(str(row["close"]))

            metrics = self._calculate_candle_metrics(
                open_price, high_price, low_price, close_price
            )

            # Shooting Star: long upper shadow, small body, small/no lower shadow
            if (metrics["upper_shadow_ratio"] > self.shadow_ratio_threshold and
                metrics["body_ratio"] < Decimal("0.3") and
                metrics["lower_shadow_ratio"] < Decimal("0.1")):

                confidence = metrics["upper_shadow_ratio"]

                if confidence >= self.min_confidence:
                    return PatternDetection(
                        pattern_type=CandlestickPattern.SHOOTING_STAR.value,
                        confidence=confidence,
                        start_idx=idx,
                        end_idx=idx,
                        metadata={
                            "upper_shadow_ratio": str(metrics["upper_shadow_ratio"]),
                            "body_ratio": str(metrics["body_ratio"])
                        }
                    )

            return None

        except Exception as e:
            logger.error("shooting_star_detection_failed", idx=idx, error=str(e))
            return None

    async def detect_engulfing(
        self,
        df: pl.DataFrame,
        idx: int
    ) -> Optional[PatternDetection]:
        """Detect Engulfing pattern (bullish or bearish).

        Args:
            df: Polars DataFrame with OHLC data
            idx: Index to check (current candle)

        Returns:
            Pattern detection or None
        """
        try:
            if idx < 1 or idx >= len(df):
                return None

            # Current candle
            curr = df.row(idx, named=True)
            curr_open = Decimal(str(curr["open"]))
            curr_close = Decimal(str(curr["close"]))

            # Previous candle
            prev = df.row(idx - 1, named=True)
            prev_open = Decimal(str(prev["open"]))
            prev_close = Decimal(str(prev["close"]))

            curr_body = abs(curr_close - curr_open)
            prev_body = abs(prev_close - prev_open)

            # Bullish Engulfing
            if (prev_close < prev_open and  # Previous is bearish
                curr_close > curr_open and  # Current is bullish
                curr_open < prev_close and  # Current opens below previous close
                curr_close > prev_open):    # Current closes above previous open

                confidence = min(Decimal("1"), curr_body / prev_body if prev_body > 0 else Decimal("1"))

                if confidence >= self.min_confidence:
                    return PatternDetection(
                        pattern_type=CandlestickPattern.ENGULFING_BULLISH.value,
                        confidence=confidence,
                        start_idx=idx - 1,
                        end_idx=idx,
                        metadata={"direction": "bullish"}
                    )

            # Bearish Engulfing
            if (prev_close > prev_open and  # Previous is bullish
                curr_close < curr_open and  # Current is bearish
                curr_open > prev_close and  # Current opens above previous close
                curr_close < prev_open):    # Current closes below previous open

                confidence = min(Decimal("1"), curr_body / prev_body if prev_body > 0 else Decimal("1"))

                if confidence >= self.min_confidence:
                    return PatternDetection(
                        pattern_type=CandlestickPattern.ENGULFING_BEARISH.value,
                        confidence=confidence,
                        start_idx=idx - 1,
                        end_idx=idx,
                        metadata={"direction": "bearish"}
                    )

            return None

        except Exception as e:
            logger.error("engulfing_detection_failed", idx=idx, error=str(e))
            return None

    async def detect_three_soldiers_crows(
        self,
        df: pl.DataFrame,
        idx: int
    ) -> Optional[PatternDetection]:
        """Detect Three White Soldiers or Three Black Crows pattern.

        Args:
            df: Polars DataFrame with OHLC data
            idx: Index to check (third candle)

        Returns:
            Pattern detection or None
        """
        try:
            if idx < 2 or idx >= len(df):
                return None

            # Get three candles
            candles = []
            for i in range(idx - 2, idx + 1):
                row = df.row(i, named=True)
                candles.append({
                    "open": Decimal(str(row["open"])),
                    "close": Decimal(str(row["close"])),
                    "high": Decimal(str(row["high"])),
                    "low": Decimal(str(row["low"]))
                })

            # Check Three White Soldiers (three consecutive bullish candles)
            all_bullish = all(c["close"] > c["open"] for c in candles)
            ascending = all(candles[i]["close"] > candles[i-1]["close"] for i in range(1, 3))

            if all_bullish and ascending:
                # Calculate confidence based on body sizes
                bodies = [abs(c["close"] - c["open"]) for c in candles]
                avg_body = sum(bodies) / Decimal("3")
                ranges = [c["high"] - c["low"] for c in candles]
                avg_range = sum(ranges) / Decimal("3")

                confidence = avg_body / avg_range if avg_range > 0 else Decimal("0")

                if confidence >= self.min_confidence:
                    return PatternDetection(
                        pattern_type=CandlestickPattern.THREE_WHITE_SOLDIERS.value,
                        confidence=confidence,
                        start_idx=idx - 2,
                        end_idx=idx,
                        metadata={"direction": "bullish"}
                    )

            # Check Three Black Crows (three consecutive bearish candles)
            all_bearish = all(c["close"] < c["open"] for c in candles)
            descending = all(candles[i]["close"] < candles[i-1]["close"] for i in range(1, 3))

            if all_bearish and descending:
                bodies = [abs(c["close"] - c["open"]) for c in candles]
                avg_body = sum(bodies) / Decimal("3")
                ranges = [c["high"] - c["low"] for c in candles]
                avg_range = sum(ranges) / Decimal("3")

                confidence = avg_body / avg_range if avg_range > 0 else Decimal("0")

                if confidence >= self.min_confidence:
                    return PatternDetection(
                        pattern_type=CandlestickPattern.THREE_BLACK_CROWS.value,
                        confidence=confidence,
                        start_idx=idx - 2,
                        end_idx=idx,
                        metadata={"direction": "bearish"}
                    )

            return None

        except Exception as e:
            logger.error("three_soldiers_crows_detection_failed", idx=idx, error=str(e))
            return None

    async def detect_all_candlestick_patterns(
        self,
        df: pl.DataFrame
    ) -> List[PatternDetection]:
        """Detect all candlestick patterns in DataFrame.

        Args:
            df: Polars DataFrame with OHLC data

        Returns:
            List of detected patterns

        Raises:
            ValueError: If required columns are missing
        """
        try:
            required_cols = ["open", "high", "low", "close"]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Required column {col} not found")

            logger.info("detecting_candlestick_patterns", rows=len(df))

            detections: List[PatternDetection] = []

            # Detect patterns for each candle
            for idx in range(len(df)):
                # Single candle patterns
                doji = await self.detect_doji(df, idx)
                if doji:
                    detections.append(doji)

                hammer = await self.detect_hammer(df, idx)
                if hammer:
                    detections.append(hammer)

                shooting_star = await self.detect_shooting_star(df, idx)
                if shooting_star:
                    detections.append(shooting_star)

                # Two candle patterns
                if idx >= 1:
                    engulfing = await self.detect_engulfing(df, idx)
                    if engulfing:
                        detections.append(engulfing)

                # Three candle patterns
                if idx >= 2:
                    soldiers_crows = await self.detect_three_soldiers_crows(df, idx)
                    if soldiers_crows:
                        detections.append(soldiers_crows)

            logger.info("candlestick_patterns_detected",
                       total_detections=len(detections))

            return detections

        except Exception as e:
            logger.error("pattern_detection_failed", error=str(e))
            raise

    async def extract_pattern_features(
        self,
        df: pl.DataFrame
    ) -> pl.DataFrame:
        """Extract pattern features as DataFrame.

        Args:
            df: Polars DataFrame with OHLC data

        Returns:
            DataFrame with pattern features

        Raises:
            ValueError: If input is invalid
        """
        try:
            logger.info("extracting_pattern_features", rows=len(df))

            # Detect all patterns
            detections = await self.detect_all_candlestick_patterns(df)

            # Create feature vector for each row
            pattern_features = []

            for idx in range(len(df)):
                features = {
                    "idx": idx,
                    "has_pattern": 0,
                    "pattern_count": 0,
                    "max_confidence": "0",
                    "bullish_signal": "0",
                    "bearish_signal": "0"
                }

                # Find patterns at this index
                patterns_at_idx = [d for d in detections if d.start_idx <= idx <= d.end_idx]

                if patterns_at_idx:
                    features["has_pattern"] = 1
                    features["pattern_count"] = len(patterns_at_idx)

                    max_conf = max(d.confidence for d in patterns_at_idx)
                    features["max_confidence"] = str(max_conf)

                    # Determine bullish/bearish signal
                    bullish_patterns = [
                        CandlestickPattern.HAMMER.value,
                        CandlestickPattern.ENGULFING_BULLISH.value,
                        CandlestickPattern.MORNING_STAR.value,
                        CandlestickPattern.THREE_WHITE_SOLDIERS.value,
                        CandlestickPattern.HARAMI_BULLISH.value
                    ]

                    bearish_patterns = [
                        CandlestickPattern.SHOOTING_STAR.value,
                        CandlestickPattern.ENGULFING_BEARISH.value,
                        CandlestickPattern.EVENING_STAR.value,
                        CandlestickPattern.THREE_BLACK_CROWS.value,
                        CandlestickPattern.HARAMI_BEARISH.value
                    ]

                    bullish_conf = sum(
                        d.confidence for d in patterns_at_idx
                        if d.pattern_type in bullish_patterns
                    )
                    bearish_conf = sum(
                        d.confidence for d in patterns_at_idx
                        if d.pattern_type in bearish_patterns
                    )

                    features["bullish_signal"] = str(bullish_conf)
                    features["bearish_signal"] = str(bearish_conf)

                pattern_features.append(features)

            # Create DataFrame
            result_df = pl.DataFrame(pattern_features)

            logger.info("pattern_features_extracted",
                       total_patterns=len(detections),
                       feature_count=len(result_df.columns))

            return result_df

        except Exception as e:
            logger.error("pattern_feature_extraction_failed", error=str(e))
            raise
