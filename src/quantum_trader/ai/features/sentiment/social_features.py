"""
Social Media Sentiment Feature Extraction

Production-ready sentiment analysis from social media sources for trading signals.
Processes Twitter, Reddit, and other social platforms for market sentiment.
"""

import asyncio
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import os
import re

import polars as pl
import numpy as np
from structlog import get_logger

logger = get_logger(__name__)

# Set high precision
getcontext().prec = 28


class SocialFeatures:
    """
    Extract sentiment features from social media data.

    Analyzes social media posts, comments, and engagement metrics
    to generate trading signals based on market sentiment.

    Attributes:
        config: Configuration dictionary
        sentiment_window: Time window for sentiment aggregation
        volume_threshold: Minimum mentions for valid signal
        weighted_scoring: Whether to weight by follower count
        platforms: Enabled social platforms

    Example:
        >>> config = {"features": {"sentiment": {"window_hours": 24}}}
        >>> social = SocialFeatures(config)
        >>> features = await social.extract_features(social_data_df, symbol="BTC/USDT")
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize social features extractor.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        features_config = self.config.get("features", {}).get("sentiment", {})

        # Load configuration
        self.sentiment_window_hours: int = features_config.get(
            "window_hours",
            int(os.getenv("SENTIMENT_WINDOW_HOURS", "24"))
        )
        self.volume_threshold: int = features_config.get(
            "volume_threshold",
            int(os.getenv("SENTIMENT_VOLUME_THRESHOLD", "10"))
        )
        self.weighted_scoring: bool = features_config.get(
            "weighted_scoring",
            os.getenv("SENTIMENT_WEIGHTED_SCORING", "true").lower() == "true"
        )

        # Platforms to analyze
        self.platforms: List[str] = features_config.get(
            "platforms",
            os.getenv("SENTIMENT_PLATFORMS", "twitter,reddit,telegram").split(",")
        )

        # Sentiment weights
        self.positive_weight: Decimal = Decimal(
            str(features_config.get("positive_weight", os.getenv("SENTIMENT_POSITIVE_WEIGHT", "1.0")))
        )
        self.negative_weight: Decimal = Decimal(
            str(features_config.get("negative_weight", os.getenv("SENTIMENT_NEGATIVE_WEIGHT", "1.5")))
        )
        self.neutral_weight: Decimal = Decimal(
            str(features_config.get("neutral_weight", os.getenv("SENTIMENT_NEUTRAL_WEIGHT", "0.5")))
        )

        # Influence multipliers
        self.influencer_threshold: int = features_config.get(
            "influencer_threshold",
            int(os.getenv("SENTIMENT_INFLUENCER_THRESHOLD", "10000"))
        )
        self.influencer_multiplier: Decimal = Decimal(
            str(features_config.get("influencer_multiplier", os.getenv("SENTIMENT_INFLUENCER_MULTIPLIER", "2.0")))
        )

        # Keyword patterns for filtering
        self.bullish_keywords: List[str] = features_config.get(
            "bullish_keywords",
            os.getenv("SENTIMENT_BULLISH_KEYWORDS", "moon,bullish,buy,pump,long").split(",")
        )
        self.bearish_keywords: List[str] = features_config.get(
            "bearish_keywords",
            os.getenv("SENTIMENT_BEARISH_KEYWORDS", "dump,bearish,sell,crash,short").split(",")
        )

        logger.info(
            "social_features_initialized",
            window_hours=self.sentiment_window_hours,
            platforms=self.platforms,
            volume_threshold=self.volume_threshold
        )

    def _validate_config(self) -> None:
        """
        Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        features_config = self.config.get("features", {}).get("sentiment", {})

        if features_config:
            window = features_config.get("window_hours", 24)
            if window <= 0:
                raise ValueError("window_hours must be positive")

    async def extract_features(
        self,
        social_data: pl.DataFrame,
        symbol: str,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Decimal]:
        """
        Extract sentiment features from social data.

        Args:
            social_data: DataFrame with social media posts
                Required columns: timestamp, platform, text, sentiment_score,
                                 engagement, follower_count
            symbol: Trading symbol to analyze
            timestamp: Reference timestamp (defaults to now)

        Returns:
            Dictionary of sentiment features

        Raises:
            ValueError: If data schema is invalid

        Example:
            >>> features = await social.extract_features(
            ...     social_data_df,
            ...     symbol="BTC/USDT",
            ...     timestamp=datetime.now(timezone.utc)
            ... )
        """
        try:
            # Validate schema
            required_cols = ["timestamp", "platform", "text", "sentiment_score"]
            if not all(col in social_data.columns for col in required_cols):
                raise ValueError(f"DataFrame missing required columns: {required_cols}")

            if timestamp is None:
                timestamp = datetime.now(timezone.utc)

            # Filter to time window
            cutoff_time = timestamp - timedelta(hours=self.sentiment_window_hours)
            windowed_data = social_data.filter(
                pl.col("timestamp") >= cutoff_time
            )

            if len(windowed_data) < self.volume_threshold:
                logger.warning(
                    "insufficient_social_volume",
                    symbol=symbol,
                    volume=len(windowed_data),
                    threshold=self.volume_threshold
                )
                return self._get_default_features()

            # Extract features
            features = {}

            # Basic sentiment metrics
            features.update(self._calculate_basic_sentiment(windowed_data))

            # Platform-specific metrics
            features.update(self._calculate_platform_metrics(windowed_data))

            # Temporal dynamics
            features.update(self._calculate_temporal_features(windowed_data, timestamp))

            # Engagement metrics
            if "engagement" in windowed_data.columns:
                features.update(self._calculate_engagement_features(windowed_data))

            # Influencer metrics
            if "follower_count" in windowed_data.columns:
                features.update(self._calculate_influencer_features(windowed_data))

            # Keyword analysis
            features.update(self._calculate_keyword_features(windowed_data))

            logger.info(
                "social_features_extracted",
                symbol=symbol,
                num_posts=len(windowed_data),
                sentiment_score=str(features.get("sentiment_score", 0))
            )

            return features

        except Exception as e:
            logger.error("feature_extraction_failed", symbol=symbol, error=str(e))
            raise

    def _calculate_basic_sentiment(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate basic sentiment metrics."""
        sentiment_values = data.select(pl.col("sentiment_score")).to_numpy().flatten()

        # Convert to Decimal for calculation
        sentiment_decimals = [Decimal(str(s)) for s in sentiment_values]

        if not sentiment_decimals:
            return {"sentiment_score": Decimal("0"), "sentiment_std": Decimal("0")}

        # Calculate mean sentiment
        mean_sentiment = sum(sentiment_decimals) / Decimal(str(len(sentiment_decimals)))

        # Calculate standard deviation
        if len(sentiment_decimals) > 1:
            variance = sum((s - mean_sentiment) ** 2 for s in sentiment_decimals) / Decimal(str(len(sentiment_decimals) - 1))
            std_sentiment = variance ** Decimal("0.5")
        else:
            std_sentiment = Decimal("0")

        # Sentiment distribution
        positive_count = sum(1 for s in sentiment_decimals if s > Decimal("0.1"))
        negative_count = sum(1 for s in sentiment_decimals if s < Decimal("-0.1"))
        neutral_count = len(sentiment_decimals) - positive_count - negative_count

        total = Decimal(str(len(sentiment_decimals)))
        positive_ratio = Decimal(str(positive_count)) / total if total > 0 else Decimal("0")
        negative_ratio = Decimal(str(negative_count)) / total if total > 0 else Decimal("0")
        neutral_ratio = Decimal(str(neutral_count)) / total if total > 0 else Decimal("0")

        return {
            "sentiment_score": mean_sentiment,
            "sentiment_std": std_sentiment,
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "neutral_ratio": neutral_ratio,
            "sentiment_volume": Decimal(str(len(sentiment_decimals)))
        }

    def _calculate_platform_metrics(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate platform-specific metrics."""
        features = {}

        for platform in self.platforms:
            platform_data = data.filter(pl.col("platform") == platform)

            if len(platform_data) == 0:
                features[f"{platform}_volume"] = Decimal("0")
                features[f"{platform}_sentiment"] = Decimal("0")
                continue

            sentiment_values = platform_data.select(pl.col("sentiment_score")).to_numpy().flatten()
            sentiment_decimals = [Decimal(str(s)) for s in sentiment_values]

            mean_sent = sum(sentiment_decimals) / Decimal(str(len(sentiment_decimals)))

            features[f"{platform}_volume"] = Decimal(str(len(platform_data)))
            features[f"{platform}_sentiment"] = mean_sent

        return features

    def _calculate_temporal_features(
        self,
        data: pl.DataFrame,
        reference_time: datetime
    ) -> Dict[str, Decimal]:
        """Calculate temporal dynamics features."""
        # Divide into time buckets
        hour_ago = reference_time - timedelta(hours=1)
        six_hours_ago = reference_time - timedelta(hours=6)

        recent_data = data.filter(pl.col("timestamp") >= hour_ago)
        mid_data = data.filter(
            (pl.col("timestamp") >= six_hours_ago) & (pl.col("timestamp") < hour_ago)
        )

        # Calculate sentiment trend
        recent_sentiment = self._get_mean_sentiment(recent_data)
        mid_sentiment = self._get_mean_sentiment(mid_data)

        sentiment_momentum = recent_sentiment - mid_sentiment

        # Volume trend
        recent_volume = Decimal(str(len(recent_data)))
        mid_volume = Decimal(str(len(mid_data)))

        volume_trend = (recent_volume - mid_volume) / (mid_volume + Decimal("1"))

        return {
            "sentiment_momentum": sentiment_momentum,
            "volume_trend": volume_trend,
            "recent_volume": recent_volume
        }

    def _calculate_engagement_features(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate engagement-based features."""
        engagement_values = data.select(pl.col("engagement")).to_numpy().flatten()
        engagement_decimals = [Decimal(str(e)) for e in engagement_values]

        if not engagement_decimals:
            return {"engagement_mean": Decimal("0"), "engagement_max": Decimal("0")}

        mean_engagement = sum(engagement_decimals) / Decimal(str(len(engagement_decimals)))
        max_engagement = max(engagement_decimals)

        # Weighted sentiment by engagement
        sentiment_values = data.select(pl.col("sentiment_score")).to_numpy().flatten()
        weighted_sentiments = []

        for sent, eng in zip(sentiment_values, engagement_values):
            weighted_sentiments.append(Decimal(str(sent)) * Decimal(str(eng)))

        total_engagement = sum(engagement_decimals)
        weighted_sentiment = (
            sum(weighted_sentiments) / total_engagement
            if total_engagement > 0
            else Decimal("0")
        )

        return {
            "engagement_mean": mean_engagement,
            "engagement_max": max_engagement,
            "engagement_weighted_sentiment": weighted_sentiment
        }

    def _calculate_influencer_features(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate influencer-based features."""
        # Filter high-influence accounts
        influencer_data = data.filter(
            pl.col("follower_count") >= self.influencer_threshold
        )

        influencer_sentiment = self._get_mean_sentiment(influencer_data)
        influencer_count = Decimal(str(len(influencer_data)))

        return {
            "influencer_sentiment": influencer_sentiment,
            "influencer_count": influencer_count
        }

    def _calculate_keyword_features(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate keyword-based features."""
        bullish_count = 0
        bearish_count = 0

        # Analyze text content
        for row in data.select("text").iter_rows():
            text = str(row[0]).lower()

            # Check for bullish keywords
            if any(keyword in text for keyword in self.bullish_keywords):
                bullish_count += 1

            # Check for bearish keywords
            if any(keyword in text for keyword in self.bearish_keywords):
                bearish_count += 1

        total = Decimal(str(len(data)))
        bullish_ratio = Decimal(str(bullish_count)) / total if total > 0 else Decimal("0")
        bearish_ratio = Decimal(str(bearish_count)) / total if total > 0 else Decimal("0")

        keyword_sentiment = bullish_ratio - bearish_ratio

        return {
            "keyword_bullish_ratio": bullish_ratio,
            "keyword_bearish_ratio": bearish_ratio,
            "keyword_sentiment": keyword_sentiment
        }

    def _get_mean_sentiment(self, data: pl.DataFrame) -> Decimal:
        """Helper to get mean sentiment from DataFrame."""
        if len(data) == 0:
            return Decimal("0")

        sentiment_values = data.select(pl.col("sentiment_score")).to_numpy().flatten()
        sentiment_decimals = [Decimal(str(s)) for s in sentiment_values]

        return sum(sentiment_decimals) / Decimal(str(len(sentiment_decimals)))

    def _get_default_features(self) -> Dict[str, Decimal]:
        """Return default features when insufficient data."""
        features = {
            "sentiment_score": Decimal("0"),
            "sentiment_std": Decimal("0"),
            "positive_ratio": Decimal("0"),
            "negative_ratio": Decimal("0"),
            "neutral_ratio": Decimal("0"),
            "sentiment_volume": Decimal("0"),
            "sentiment_momentum": Decimal("0"),
            "volume_trend": Decimal("0"),
            "recent_volume": Decimal("0")
        }

        # Add platform-specific defaults
        for platform in self.platforms:
            features[f"{platform}_volume"] = Decimal("0")
            features[f"{platform}_sentiment"] = Decimal("0")

        return features
