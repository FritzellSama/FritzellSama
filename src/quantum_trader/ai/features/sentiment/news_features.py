"""
News Sentiment Feature Extraction for Trading.

This module processes news articles and social media to extract
sentiment features for trading signal generation.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re
from collections import Counter

import numpy as np
import polars as pl
import aiohttp
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class NewsArticle:
    """News article data structure.

    Attributes:
        title: Article title
        content: Article content
        source: News source
        timestamp: Publication timestamp
        symbols: Related trading symbols
        url: Article URL
    """
    title: str
    content: str
    source: str
    timestamp: datetime
    symbols: List[str]
    url: Optional[str] = None


@dataclass
class SentimentScore:
    """Sentiment analysis result.

    Attributes:
        polarity: Sentiment polarity (-1.0 to 1.0)
        subjectivity: Subjectivity score (0.0 to 1.0)
        positive_score: Positive sentiment score
        negative_score: Negative sentiment score
        neutral_score: Neutral sentiment score
        confidence: Confidence in prediction
    """
    polarity: Decimal
    subjectivity: Decimal
    positive_score: Decimal
    negative_score: Decimal
    neutral_score: Decimal
    confidence: Decimal


class NewsFeatures:
    """Production-ready news sentiment feature extractor.

    Processes news articles and social media to extract sentiment
    features for trading decisions.

    Attributes:
        config: Configuration dictionary
        sentiment_keywords: Dictionary of sentiment keywords
        max_articles_per_symbol: Maximum articles to process per symbol
        lookback_hours: Hours to look back for news
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize news feature extractor.

        Args:
            config: Configuration dictionary containing:
                - features.sentiment.news_api_key
                - features.sentiment.max_articles_per_symbol
                - features.sentiment.lookback_hours
                - features.sentiment.sentiment_threshold
                - features.sentiment.sources
        """
        self.config = config
        self._validate_config()

        sentiment_config = self.config["features"]["sentiment"]
        self.news_api_key = sentiment_config.get("news_api_key", "")
        self.max_articles = sentiment_config["max_articles_per_symbol"]
        self.lookback_hours = sentiment_config["lookback_hours"]
        self.sentiment_threshold = Decimal(str(sentiment_config["sentiment_threshold"]))
        self.sources = sentiment_config.get("sources", [])

        # Initialize sentiment lexicons
        self._initialize_sentiment_lexicons()

        logger.info("news_features_initialized",
                   max_articles=self.max_articles,
                   lookback_hours=self.lookback_hours)

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required config parameters are missing
        """
        required_keys = [
            "features.sentiment.max_articles_per_symbol",
            "features.sentiment.lookback_hours",
            "features.sentiment.sentiment_threshold"
        ]

        for key in required_keys:
            parts = key.split(".")
            current = self.config
            for part in parts:
                if part not in current:
                    raise ValueError(f"Missing required config key: {key}")
                current = current[part]

    def _initialize_sentiment_lexicons(self) -> None:
        """Initialize sentiment keyword lexicons."""
        # Positive financial keywords
        self.positive_keywords = {
            'bullish', 'gains', 'profit', 'surge', 'rally', 'upturn',
            'growth', 'positive', 'strong', 'beat', 'outperform',
            'upgrade', 'breakthrough', 'success', 'record', 'high',
            'optimistic', 'confidence', 'expansion', 'momentum', 'boost'
        }

        # Negative financial keywords
        self.negative_keywords = {
            'bearish', 'losses', 'loss', 'decline', 'crash', 'plunge',
            'downturn', 'negative', 'weak', 'miss', 'underperform',
            'downgrade', 'failure', 'bankruptcy', 'lawsuit', 'scandal',
            'pessimistic', 'concern', 'recession', 'slump', 'drop'
        }

        # Neutral keywords
        self.neutral_keywords = {
            'stable', 'unchanged', 'flat', 'neutral', 'hold', 'steady',
            'maintain', 'continue', 'persist', 'ongoing'
        }

        logger.debug("sentiment_lexicons_initialized",
                    positive_count=len(self.positive_keywords),
                    negative_count=len(self.negative_keywords))

    async def fetch_news_articles(
        self,
        symbols: List[str],
        start_time: datetime,
        end_time: datetime
    ) -> List[NewsArticle]:
        """Fetch news articles for symbols.

        Args:
            symbols: List of trading symbols
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            List of news articles

        Raises:
            aiohttp.ClientError: If API request fails
        """
        articles: List[NewsArticle] = []

        try:
            # In production, this would fetch from actual news APIs
            # For now, we simulate the structure
            logger.info("fetching_news_articles",
                       symbols=symbols,
                       start_time=start_time.isoformat(),
                       end_time=end_time.isoformat())

            if not self.news_api_key:
                logger.warning("no_news_api_key_configured")
                return articles

            # Fetch articles for each symbol
            for symbol in symbols[:self.max_articles]:
                try:
                    symbol_articles = await self._fetch_symbol_news(
                        symbol, start_time, end_time
                    )
                    articles.extend(symbol_articles)

                except Exception as e:
                    logger.error("symbol_news_fetch_failed",
                               symbol=symbol,
                               error=str(e))
                    continue

            logger.info("news_articles_fetched",
                       total_articles=len(articles),
                       symbols_processed=len(symbols))

            return articles

        except Exception as e:
            logger.error("news_fetch_failed", error=str(e))
            raise

    async def _fetch_symbol_news(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[NewsArticle]:
        """Fetch news for specific symbol.

        Args:
            symbol: Trading symbol
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            List of news articles for symbol
        """
        articles: List[NewsArticle] = []

        # Retry logic for API calls
        max_retries = self.config["features"]["sentiment"].get("api_max_retries", 3)
        retry_delay = self.config["features"]["sentiment"].get("api_retry_delay_seconds", 2)

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    # Build API request URL
                    api_url = self.config["features"]["sentiment"].get("news_api_url", "")

                    if not api_url:
                        logger.warning("news_api_url_not_configured")
                        return articles

                    params = {
                        "symbol": symbol,
                        "from": start_time.isoformat(),
                        "to": end_time.isoformat(),
                        "limit": self.max_articles,
                        "apiKey": self.news_api_key
                    }

                    async with session.get(api_url, params=params, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()

                            for item in data.get("articles", []):
                                article = NewsArticle(
                                    title=item.get("title", ""),
                                    content=item.get("description", ""),
                                    source=item.get("source", {}).get("name", "Unknown"),
                                    timestamp=datetime.fromisoformat(
                                        item.get("publishedAt", datetime.utcnow().isoformat())
                                    ),
                                    symbols=[symbol],
                                    url=item.get("url")
                                )
                                articles.append(article)

                            break  # Success, exit retry loop

                        elif response.status == 429:  # Rate limited
                            logger.warning("api_rate_limited",
                                         symbol=symbol,
                                         attempt=attempt + 1)
                            await asyncio.sleep(retry_delay * (2 ** attempt))
                            continue

                        else:
                            logger.error("api_request_failed",
                                       symbol=symbol,
                                       status=response.status)
                            break

            except asyncio.TimeoutError:
                logger.error("api_request_timeout",
                           symbol=symbol,
                           attempt=attempt + 1)
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue

            except Exception as e:
                logger.error("api_request_error",
                           symbol=symbol,
                           error=str(e))
                break

        return articles

    def analyze_sentiment(self, text: str) -> SentimentScore:
        """Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Sentiment score
        """
        if not text:
            return SentimentScore(
                polarity=Decimal("0"),
                subjectivity=Decimal("0"),
                positive_score=Decimal("0"),
                negative_score=Decimal("0"),
                neutral_score=Decimal("1"),
                confidence=Decimal("0")
            )

        # Tokenize and normalize text
        words = re.findall(r'\b\w+\b', text.lower())

        # Count sentiment keywords
        positive_count = sum(1 for word in words if word in self.positive_keywords)
        negative_count = sum(1 for word in words if word in self.negative_keywords)
        neutral_count = sum(1 for word in words if word in self.neutral_keywords)

        total_sentiment_words = positive_count + negative_count + neutral_count

        if total_sentiment_words == 0:
            return SentimentScore(
                polarity=Decimal("0"),
                subjectivity=Decimal("0"),
                positive_score=Decimal("0"),
                negative_score=Decimal("0"),
                neutral_score=Decimal("1"),
                confidence=Decimal("0")
            )

        # Calculate scores
        positive_score = Decimal(str(positive_count / len(words))) if words else Decimal("0")
        negative_score = Decimal(str(negative_count / len(words))) if words else Decimal("0")
        neutral_score = Decimal(str(neutral_count / len(words))) if words else Decimal("0")

        # Calculate polarity (-1 to 1)
        if positive_count + negative_count > 0:
            polarity = Decimal(str(
                (positive_count - negative_count) / (positive_count + negative_count)
            ))
        else:
            polarity = Decimal("0")

        # Calculate subjectivity (0 to 1)
        subjectivity = Decimal(str(total_sentiment_words / len(words))) if words else Decimal("0")

        # Calculate confidence based on sample size
        confidence = min(
            Decimal("1"),
            Decimal(str(total_sentiment_words)) / Decimal("10")
        )

        return SentimentScore(
            polarity=polarity,
            subjectivity=subjectivity,
            positive_score=positive_score,
            negative_score=negative_score,
            neutral_score=neutral_score,
            confidence=confidence
        )

    async def calculate_symbol_sentiment(
        self,
        symbol: str,
        articles: List[NewsArticle]
    ) -> Dict[str, Decimal]:
        """Calculate aggregate sentiment for symbol.

        Args:
            symbol: Trading symbol
            articles: List of news articles

        Returns:
            Dictionary of sentiment metrics
        """
        if not articles:
            return {
                "sentiment_score": Decimal("0"),
                "sentiment_polarity": Decimal("0"),
                "sentiment_confidence": Decimal("0"),
                "article_count": Decimal("0"),
                "positive_ratio": Decimal("0"),
                "negative_ratio": Decimal("0")
            }

        sentiments: List[SentimentScore] = []

        for article in articles:
            if symbol in article.symbols:
                combined_text = f"{article.title} {article.content}"
                sentiment = self.analyze_sentiment(combined_text)
                sentiments.append(sentiment)

        if not sentiments:
            return {
                "sentiment_score": Decimal("0"),
                "sentiment_polarity": Decimal("0"),
                "sentiment_confidence": Decimal("0"),
                "article_count": Decimal("0"),
                "positive_ratio": Decimal("0"),
                "negative_ratio": Decimal("0")
            }

        # Calculate weighted average sentiment
        total_weight = sum(s.confidence for s in sentiments)

        if total_weight > 0:
            avg_polarity = sum(
                s.polarity * s.confidence for s in sentiments
            ) / total_weight

            avg_confidence = sum(s.confidence for s in sentiments) / Decimal(str(len(sentiments)))
        else:
            avg_polarity = Decimal("0")
            avg_confidence = Decimal("0")

        # Calculate positive/negative ratios
        positive_count = sum(1 for s in sentiments if s.polarity > self.sentiment_threshold)
        negative_count = sum(1 for s in sentiments if s.polarity < -self.sentiment_threshold)

        positive_ratio = Decimal(str(positive_count / len(sentiments)))
        negative_ratio = Decimal(str(negative_count / len(sentiments)))

        return {
            "sentiment_score": avg_polarity,
            "sentiment_polarity": avg_polarity,
            "sentiment_confidence": avg_confidence,
            "article_count": Decimal(str(len(articles))),
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio
        }

    async def extract_sentiment_features(
        self,
        symbols: List[str],
        lookback_hours: Optional[int] = None
    ) -> pl.DataFrame:
        """Extract sentiment features for symbols.

        Args:
            symbols: List of trading symbols
            lookback_hours: Hours to look back (uses config default if None)

        Returns:
            Polars DataFrame with sentiment features

        Raises:
            ValueError: If symbols list is empty
        """
        if not symbols:
            raise ValueError("Symbols list cannot be empty")

        lookback = lookback_hours if lookback_hours is not None else self.lookback_hours

        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=lookback)

            logger.info("extracting_sentiment_features",
                       symbols=symbols,
                       lookback_hours=lookback)

            # Fetch news articles
            articles = await self.fetch_news_articles(symbols, start_time, end_time)

            # Calculate sentiment for each symbol
            results = []

            for symbol in symbols:
                sentiment_metrics = await self.calculate_symbol_sentiment(symbol, articles)

                results.append({
                    "symbol": symbol,
                    "timestamp": end_time,
                    "sentiment_score": str(sentiment_metrics["sentiment_score"]),
                    "sentiment_polarity": str(sentiment_metrics["sentiment_polarity"]),
                    "sentiment_confidence": str(sentiment_metrics["sentiment_confidence"]),
                    "article_count": str(sentiment_metrics["article_count"]),
                    "positive_ratio": str(sentiment_metrics["positive_ratio"]),
                    "negative_ratio": str(sentiment_metrics["negative_ratio"])
                })

            # Create Polars DataFrame
            df = pl.DataFrame(results)

            logger.info("sentiment_features_extracted",
                       symbols_processed=len(symbols),
                       total_articles=len(articles))

            return df

        except Exception as e:
            logger.error("sentiment_feature_extraction_failed", error=str(e))
            raise
