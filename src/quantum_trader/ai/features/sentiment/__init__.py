"""Sentiment analysis features module.

This module provides sentiment analysis-based features for trading signals,
including market sentiment extraction, social media sentiment, and news sentiment analysis.

Attributes:
    __all__: Public API exports for the sentiment analysis module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__ = [
    "SentimentAnalyzer",
    "NewssentimentExtractor",
    "SocialMediaSentiment",
    "MarketSentimentIndicator",
]

__version__ = "1.0.0"
__author__ = "Quantum Trader AI"


def __getattr__(name: str):
    """Lazy import for sentiment analysis features."""
    if name == "SentimentAnalyzer":
        from .analyzer import SentimentAnalyzer
        return SentimentAnalyzer
    elif name == "NewsSentimentExtractor":
        from .news import NewsSentimentExtractor
        return NewsSentimentExtractor
    elif name == "SocialMediaSentiment":
        from .social_media import SocialMediaSentiment
        return SocialMediaSentiment
    elif name == "MarketSentimentIndicator":
        from .indicators import MarketSentimentIndicator
        return MarketSentimentIndicator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
