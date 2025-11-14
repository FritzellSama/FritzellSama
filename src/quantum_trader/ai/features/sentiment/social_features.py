"""Social sentiment feature extraction for trading signals.

This module extracts sentiment features from social media and news sources
to augment trading signals with market sentiment indicators.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timezone, timedelta
import os
import re
from collections import defaultdict

import numpy as np
import polars as pl
import aiohttp
from structlog import get_logger

logger = get_logger(__name__)


class SocialFeatures:
    """Extract sentiment features from social media and news.

    Processes text data from various sources to generate sentiment scores,
    volume metrics, and trend indicators for trading decisions.

    Attributes:
        config: Configuration dictionary
        sentiment_window: Time window for sentiment aggregation
        sources: Enabled social media sources
        cache: LRU cache for recent sentiment data

    Examples:
        >>> config = {"sentiment_window": "1h", "sources": ["twitter", "reddit"]}
        >>> extractor = SocialFeatures(config)
        >>> features = await extractor.extract(
        ...     data=pl.DataFrame({"text": ["Bullish on BTC"], "timestamp": [...]})
        ... )
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize social sentiment feature extractor.

        Args:
            config: Configuration with sources, window, API keys, etc.

        Raises:
            ValueError: If required config parameters missing
        """
        self.config = config
        self._validate_config()

        self.sentiment_window: str = config.get("sentiment_window", os.getenv("SENTIMENT_WINDOW", "1h"))
        self.sources: Set[str] = set(config.get("sources", os.getenv("SENTIMENT_SOURCES", "twitter,reddit,news").split(",")))
        self.min_posts: int = int(config.get("min_posts", os.getenv("SENTIMENT_MIN_POSTS", "10")))
        self.cache_size: int = int(config.get("cache_size", os.getenv("SENTIMENT_CACHE_SIZE", "1000")))

        # Sentiment keywords
        self.bullish_keywords: Set[str] = set(config.get("bullish_keywords", os.getenv("SENTIMENT_BULLISH", "bullish,moon,pump,buy").split(",")))
        self.bearish_keywords: Set[str] = set(config.get("bearish_keywords", os.getenv("SENTIMENT_BEARISH", "bearish,dump,sell,crash").split(",")))

        self.cache: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "Social features extractor initialized",
            sources=list(self.sources),
            window=self.sentiment_window,
            min_posts=self.min_posts
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If config validation fails
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        if "sources" in self.config:
            if not self.config["sources"]:
                raise ValueError("At least one source must be specified")

    async def extract(
        self,
        data: pl.DataFrame,
        symbol: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> pl.DataFrame:
        """Extract sentiment features from social data.

        Args:
            data: DataFrame with 'text', 'timestamp', 'source' columns
            symbol: Trading symbol to filter for
            metadata: Optional metadata

        Returns:
            DataFrame with sentiment features

        Raises:
            ValueError: If data DataFrame invalid
        """
        try:
            if data.is_empty():
                logger.warning("Empty data DataFrame provided")
                return self._empty_features()

            required_cols = ["text", "timestamp"]
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            # Filter by symbol if provided
            if symbol and "symbol" in data.columns:
                data = data.filter(pl.col("symbol") == symbol)

            # Calculate sentiment scores
            sentiment_scores = await self._calculate_sentiment(data)

            # Calculate volume metrics
            volume_metrics = await self._calculate_volume_metrics(data)

            # Calculate trend indicators
            trend_indicators = await self._calculate_trends(data)

            # Combine features
            features = self._combine_features(
                sentiment_scores,
                volume_metrics,
                trend_indicators
            )

            logger.debug(
                "Social features extracted",
                num_posts=len(data),
                symbol=symbol
            )

            return features

        except Exception as e:
            logger.error(
                "Social feature extraction failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def _calculate_sentiment(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate sentiment scores from text data.

        Args:
            data: DataFrame with text data

        Returns:
            Dictionary of sentiment metrics
        """
        scores = []

        for text in data["text"].to_list():
            if text is None:
                continue

            text_lower = text.lower()

            # Count bullish/bearish keywords
            bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in text_lower)
            bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in text_lower)

            # Calculate net sentiment (-1 to 1)
            total_keywords = bullish_count + bearish_count
            if total_keywords > 0:
                sentiment = Decimal(str((bullish_count - bearish_count) / total_keywords))
            else:
                sentiment = Decimal("0")

            scores.append(sentiment)

        if not scores:
            return {
                "mean_sentiment": Decimal("0"),
                "sentiment_std": Decimal("0"),
                "bullish_ratio": Decimal("0"),
                "bearish_ratio": Decimal("0")
            }

        scores_array = np.array([float(s) for s in scores], dtype=np.float64)

        return {
            "mean_sentiment": Decimal(str(np.mean(scores_array))),
            "sentiment_std": Decimal(str(np.std(scores_array))),
            "bullish_ratio": Decimal(str(np.sum(scores_array > 0) / len(scores_array))),
            "bearish_ratio": Decimal(str(np.sum(scores_array < 0) / len(scores_array)))
        }

    async def _calculate_volume_metrics(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate post volume metrics.

        Args:
            data: DataFrame with timestamp data

        Returns:
            Dictionary of volume metrics
        """
        if "source" not in data.columns:
            source_counts = {"total": len(data)}
        else:
            source_counts = data.group_by("source").agg(pl.count()).to_dict(as_series=False)
            source_dict = dict(zip(source_counts["source"], source_counts["count"]))
            source_counts = source_dict

        total_posts = len(data)

        return {
            "total_posts": Decimal(str(total_posts)),
            "posts_per_minute": Decimal(str(total_posts / 60)) if total_posts > 0 else Decimal("0"),
            "source_diversity": Decimal(str(len(source_counts))) if "source" in data.columns else Decimal("1")
        }

    async def _calculate_trends(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate sentiment trend indicators.

        Args:
            data: DataFrame with timestamp and text data

        Returns:
            Dictionary of trend indicators
        """
        if len(data) < 2:
            return {
                "sentiment_momentum": Decimal("0"),
                "volume_momentum": Decimal("0"),
                "trend_strength": Decimal("0")
            }

        # Sort by timestamp
        data = data.sort("timestamp")

        # Split into first and second half
        mid_point = len(data) // 2
        first_half = data[:mid_point]
        second_half = data[mid_point:]

        # Calculate sentiment for each half
        first_sentiment = await self._calculate_sentiment(first_half)
        second_sentiment = await self._calculate_sentiment(second_half)

        # Calculate momentum
        sentiment_momentum = second_sentiment["mean_sentiment"] - first_sentiment["mean_sentiment"]
        volume_momentum = Decimal(str(len(second_half))) - Decimal(str(len(first_half)))

        # Trend strength (combination of sentiment and volume change)
        trend_strength = abs(sentiment_momentum) * (Decimal("1") + abs(volume_momentum) / Decimal(str(len(data))))

        return {
            "sentiment_momentum": sentiment_momentum,
            "volume_momentum": volume_momentum,
            "trend_strength": trend_strength
        }

    def _combine_features(
        self,
        sentiment: Dict[str, Decimal],
        volume: Dict[str, Decimal],
        trends: Dict[str, Decimal]
    ) -> pl.DataFrame:
        """Combine all features into a single DataFrame.

        Args:
            sentiment: Sentiment metrics
            volume: Volume metrics
            trends: Trend indicators

        Returns:
            Combined features DataFrame
        """
        all_features = {**sentiment, **volume, **trends}

        # Convert to DataFrame
        features_df = pl.DataFrame({
            "feature": list(all_features.keys()),
            "value": [str(v) for v in all_features.values()]
        })

        return features_df

    def _empty_features(self) -> pl.DataFrame:
        """Return empty features DataFrame.

        Returns:
            DataFrame with zero-valued features
        """
        zero_features = {
            "mean_sentiment": "0",
            "sentiment_std": "0",
            "bullish_ratio": "0",
            "bearish_ratio": "0",
            "total_posts": "0",
            "posts_per_minute": "0",
            "source_diversity": "0",
            "sentiment_momentum": "0",
            "volume_momentum": "0",
            "trend_strength": "0"
        }

        return pl.DataFrame({
            "feature": list(zero_features.keys()),
            "value": list(zero_features.values())
        })

    async def extract_batch(
        self,
        data_batch: List[pl.DataFrame],
        symbols: Optional[List[str]] = None,
        metadata_batch: Optional[List[Dict[str, Any]]] = None
    ) -> List[pl.DataFrame]:
        """Extract features for batch of data.

        Args:
            data_batch: List of DataFrames
            symbols: Optional list of symbols
            metadata_batch: Optional list of metadata dicts

        Returns:
            List of feature DataFrames
        """
        try:
            symbols = symbols or [None] * len(data_batch)
            metadata_batch = metadata_batch or [None] * len(data_batch)

            features_batch = []
            for data, symbol, metadata in zip(data_batch, symbols, metadata_batch):
                features = await self.extract(data, symbol, metadata)
                features_batch.append(features)

            logger.debug(
                "Batch features extracted",
                batch_size=len(features_batch)
            )

            return features_batch

        except Exception as e:
            logger.error(
                "Batch feature extraction failed",
                error=str(e),
                batch_size=len(data_batch)
            )
            raise

    def clear_cache(self) -> None:
        """Clear the sentiment cache."""
        self.cache.clear()
        logger.debug("Sentiment cache cleared")
